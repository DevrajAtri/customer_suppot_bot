from bot_tools.common import register_tool, run_with_retries_and_timeout
from bot_tools.schemas import GetOrderStatusIn, GetOrderStatusOut
from core.db_utils import safe_error_id
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

@register_tool("get_order_status", input_schema=GetOrderStatusIn, output_schema=GetOrderStatusOut)
def get_order_status(order_id: int, db_connection: Any):
    """
    Retrieves the current status of an order directly from the database.
    The db_connection is injected by the agent.
    """
    start = time.perf_counter()
    try:
        # Validate input using the schema
        inp = GetOrderStatusIn(order_id=order_id)
    except Exception as e:
        return {"error": "invalid_input", "hint": str(e)}

    def _work(conn: Any):
        try:
            cur = conn.cursor()
            
            # Query the billing_orders table for the specific status field
            # We also grab date and total to give the user a complete answer
            cur.execute("""
                SELECT id, status, order_date, order_total 
                FROM billing_orders 
                WHERE id = ?
            """, (inp.order_id,))
            
            row = cur.fetchone()
            
            # Handle case where order doesn't exist
            if not row:
                return {"error": "order_not_found", "duration_s": time.perf_counter() - start}

            # Map the database row to the output schema
            # Note: 'items' defaults to [] in your schema, which is perfect here 
            # since we are only looking up the status.
            return {
                "order_id": row["id"],
                "status": row["status"], # The field you specifically requested
                "order_date": row["order_date"],
                "total_amount": float(row["order_total"]),
                "items": [], 
                "duration_s": time.perf_counter() - start
            }
        finally:
            # Connection is managed by the caller
            pass

    try:
        # Execute the worker function with the injected connection
        return run_with_retries_and_timeout(_work, db_connection, retries=2, timeout_s=5.0)
    except Exception as exc:
        err = safe_error_id(exc)
        logger.exception("get_order_status failed")
        return {"error": err}