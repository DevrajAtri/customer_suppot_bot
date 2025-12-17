from bot_tools.common import register_tool, run_with_retries_and_timeout
from bot_tools.schemas import CheckStatusIn, CheckStatusOut, Invoice
from core.db_utils import safe_error_id
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

@register_tool("check_account_status", input_schema=CheckStatusIn, output_schema=CheckStatusOut)
def check_account_status(user_id: int, db_connection: Any):
    """
    Checks the account status for a user.
    The db_connection is injected by the agent.
    """
    start = time.perf_counter()
    try:
        # Pydantic validation
        inp = CheckStatusIn(user_id=user_id)
    except Exception as e:
        return {"error": "invalid_input", "hint": str(e)}

    def _work(conn: Any):
        # REMOVED: conn = get_db_conn(db_path) 
        # We use the injected 'conn' directly
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, username, plan, status, created_at FROM users WHERE id = ?", (inp.user_id,))
            row = cur.fetchone()
            duration = time.perf_counter() - start

            if not row:
                return {"error": "not_found", "duration_s": duration}

            # Fetch last invoice
            cur.execute("SELECT date, amount FROM invoices WHERE user_id = ? ORDER BY date DESC LIMIT 1", (inp.user_id,))
            inv = cur.fetchone()
            
            last_invoice = None
            if inv:
                last_invoice = Invoice(date=inv["date"], amount=float(inv["amount"]))

            return {
                "user_id": row["id"],
                "username": row["username"],
                "plan": row["plan"],
                "status": row["status"],
                "last_invoice": last_invoice.dict() if last_invoice else None,
                "duration_s": duration
            }
        finally:
            # REMOVED: conn.close() 
            # The agent owns the connection now
            pass

    try:
        return run_with_retries_and_timeout(_work, db_connection, retries=2, timeout_s=5.0)
    except Exception as exc:
        err = safe_error_id(exc)
        logger.exception("check_account_status failed")
        return {"error": err}