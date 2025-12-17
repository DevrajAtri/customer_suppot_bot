from bot_tools.common import register_tool, run_with_retries_and_timeout
from bot_tools.schemas import UpgradePlanIn, UpgradePlanOut
# We no longer need get_db_conn, only safe_error_id
from core.db_utils import get_db_conn, safe_error_id
import time
import logging
from typing import Any # Import Any for type hinting

Logger = logging.getLogger(__name__)
ALLOWED_PLANS = ("free", "plus", "pro") # [cite: 283]

@register_tool("upgrade_plan", input_schema=UpgradePlanIn, output_schema=UpgradePlanOut)
def upgrade_plan(
    user_id: int, 
    new_plan: str, 
    db_connection: Any # New injected argument
    # db_path: str = None <-- REMOVED
):
    """
    Upgrades or downgrades a user's subscription plan.
    The db_connection is injected by the agent.
    """
    start = time.perf_counter()
    try:
        # Pydantic validation no longer includes db_path
        inp = UpgradePlanIn(user_id=user_id, new_plan=new_plan)
    except Exception as e:
        return {"error": "invalid_input", "hint": str(e)}

    # Validation logic from original file [cite: 293-295]
    if inp.new_plan not in ALLOWED_PLANS:
        return {"error": "invalid_plan", "hint": f"allowed: {ALLOWED_PLANS}"}

    # _work now accepts the connection
    def _work(conn: Any):
        # PROBLEM 4 SOLVED: We no longer call get_db_conn or conn.close()
        # conn = get_db_conn(db_path) <-- REMOVED
        try:
            cur = conn.cursor()
            
            # Start transaction [cite: 300]
            cur.execute("BEGIN IMMEDIATE")
            
            # Check user and old plan [cite: 301-303]
            cur.execute("SELECT plan FROM users WHERE id = ?", (inp.user_id,))
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return {"error": "user_not_found"}

            old_plan = row["plan"]
            
            # Check if no change [cite: 308-311]
            if old_plan == inp.new_plan:
                conn.rollback()
                return {"user_id": inp.user_id, "old_plan": old_plan, "new_plan": inp.new_plan, "status": "no_change", "duration_s": time.perf_counter() - start}

            # Update the plan [cite: 312-313]
            cur.execute("UPDATE users SET plan = ? WHERE id = ?", (inp.new_plan, inp.user_id))
            conn.commit()
            
            return {"user_id": inp.user_id, "old_plan": old_plan, "new_plan": inp.new_plan, "status": "updated", "duration_s": time.perf_counter() - start}
        
        finally:
            # conn.close() <-- REMOVED
            pass

    try:
        # Pass the db_connection as the first argument to _work
        return run_with_retries_and_timeout(_work, db_connection, retries=2, timeout_s=5.0)
    except Exception as exc:
        err = safe_error_id(exc)
        Logger.exception("upgrade_plan failed")
        # Ensure rollback is attempted on failure
        try:
            db_connection.rollback()
            Logger.info("Database transaction rolled back due to failure.")
        except Exception as rb_exc:
            Logger.error(f"Failed to rollback transaction: {rb_exc}")
        return {"error": err}