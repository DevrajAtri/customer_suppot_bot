from bot_tools.common import register_tool, run_with_retries_and_timeout
from bot_tools.schemas import HandoffIn, HandoffOut
# We no longer need get_db_conn, only safe_error_id
from core.db_utils import get_db_conn, safe_error_id
import time
import logging
from typing import Any, List # Import Any and List

logger = logging.getLogger(__name__)

@register_tool("handoff_to_human", input_schema=HandoffIn, output_schema=HandoffOut)
def handoff_to_human(
    conversation_id: str, 
    summary: str, 
    tags: List[str], 
    db_connection: Any # New injected argument
    # db_path: str = None <-- REMOVED
):
    """
    Logs a request to hand off the conversation to a human agent.
    The db_connection is injected by the agent.
    """
    start = time.perf_counter()
    try:
        # Pydantic validation no longer includes db_path
        inp = HandoffIn(conversation_id=conversation_id, summary=summary, tags=tags)
    except Exception as e:
        return {"error": "invalid_input", "hint": str(e)}

    # _work now accepts the connection
    def _work(conn: Any):
        # PROBLEM 4 SOLVED: We no longer call get_db_conn or conn.close()
        # conn = get_db_conn(db_path) <-- REMOVED
        try:
            cur = conn.cursor()
            
            # Format tags [cite: 345]
            tags_str = ",".join(inp.tags)
            
            # Insert handoff request [cite: 342-345]
            cur.execute("INSERT INTO handoffs (conversation_id, summary, tags, status) VALUES (?, ?, ?, ?)",
                        (inp.conversation_id, inp.summary, tags_str, "queued"))
            hid = cur.lastrowid
            conn.commit()
            
            return {"handoff_id": hid, "status": "queued", "duration_s": time.perf_counter() - start}

        finally:
            # conn.close() <-- REMOVED
            pass

    try:
        # Pass the db_connection as the first argument to _work
        return run_with_retries_and_timeout(_work, db_connection, retries=2, timeout_s=5.0)
    except Exception as exc:
        err = safe_error_id(exc)
        logger.exception("handoff_to_human failed")
        # This tool's _work function is a simple commit, but
        # we'll add a rollback just in case.
        try:
            db_connection.rollback()
        except Exception:
            pass # Ignore rollback errors if connection is bad
        return {"error": err}