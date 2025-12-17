from bot_tools.common import register_tool, run_with_retries_and_timeout
from bot_tools.schemas import ProcessRefundIn, ProcessRefundOut
from core.db_utils import safe_error_id
import time
import logging
import webbrowser
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

@register_tool("process_refund", input_schema=ProcessRefundIn, output_schema=ProcessRefundOut)
def process_refund(
    order_id: int, 
    reason: str, 
    db_connection: Any,
    amount: float = None, 
    idempotency_key: str = None
):
    """
    Executes a refund for a specific order. 
    INCLUDES HITL (Human-in-the-Loop) VERIFICATION:
    Opens a browser tab for agent review before committing changes.
    """
    start = time.perf_counter()
    try:
        # Validate input
        inp = ProcessRefundIn(
            order_id=order_id, 
            amount=amount, 
            reason=reason, 
            idempotency_key=idempotency_key
        )
    except Exception as e:
        return {"error": "invalid_input", "hint": str(e)}

    # --- PHASE 1: PRE-CHECKS (Read-Only) ---
    # We check these BEFORE opening the UI to avoid unnecessary user interruption.
    try:
        cur = db_connection.cursor()
        
        # A. Idempotency Check (Optimization: Don't ask human again if already done)
        if inp.idempotency_key:
            cur.execute("SELECT id, amount FROM refunds WHERE idempotency_key = ?", (inp.idempotency_key,))
            existing_refund = cur.fetchone()
            if existing_refund:
                logger.info("Idempotency match found for key %s - Skipping HITL", inp.idempotency_key)
                return {
                    "refund_id": existing_refund["id"], 
                    "order_id": inp.order_id, 
                    "amount": float(existing_refund["amount"]), 
                    "status": "completed", 
                    "duration_s": time.perf_counter() - start
                }

        # B. Fetch Order Context for the UI
        cur.execute("SELECT id, order_date, order_total, status FROM billing_orders WHERE id = ?", (inp.order_id,))
        order_row = cur.fetchone()
        
        if not order_row:
            return {"error": "order_not_found"}

    except Exception as e:
        logger.error(f"Pre-check failed: {e}")
        return {"error": "db_read_failed", "hint": str(e)}

    # --- PHASE 2: HUMAN-IN-THE-LOOP (Browser Simulation) ---
    
    # 1. Generate the HTML Ticket
    html_content = f"""
    <html>
    <head>
        <title>Refund Approval: #{inp.order_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; background: #f0f2f5; }}
            .card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }}
            h2 {{ margin-top: 0; color: #1a1a1a; border-bottom: 2px solid #eee; padding-bottom: 15px; }}
            .row {{ display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 16px; }}
            .label {{ color: #666; font-weight: 600; }}
            .value {{ color: #333; }}
            .reason-box {{ background: #fff5f5; border-left: 5px solid #ff4d4f; padding: 15px; margin-top: 20px; border-radius: 4px; }}
            .footer {{ margin-top: 30px; text-align: center; color: #888; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🛡️ Refund Request Review</h2>
            <div class="row"><span class="label">Order ID:</span> <span class="value">#{order_row['id']}</span></div>
            <div class="row"><span class="label">Date:</span> <span class="value">{order_row['order_date']}</span></div>
            <div class="row"><span class="label">Total Amount:</span> <span class="value">${order_row['order_total']}</span></div>
            <div class="row"><span class="label">Status:</span> <span class="value">{order_row['status']}</span></div>
            
            <div class="reason-box">
                <div class="label" style="color: #d32f2f; margin-bottom: 5px;">Customer Reason:</div>
                <div style="font-style: italic;">"{inp.reason}"</div>
            </div>

            <div class="footer">
                ⚠️ Action Required: Return to your terminal to <b>Approve</b> or <b>Reject</b>.
            </div>
        </div>
    </body>
    </html>
    """
    
    # 2. Open in Browser
    try:
        fd, path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
            tmp.write(html_content)
        webbrowser.open('file://' + path)
    except Exception as e:
        logger.warning(f"Could not open browser for HITL: {e}")

    # 3. Block & Wait for Terminal Input
    print(f"\n" + "="*60)
    print(f" 👮 HITL INTERVENTION REQUIRED")
    print(f"="*60)
    print(f"I have opened a ticket for Order #{inp.order_id} in your browser.")
    print(f"Reason: {inp.reason}")
    print(f"-"*60)
    
    decision = input(">> TYPE 'approve' to proceed, or enter a REJECTION REASON: ").strip()
    
    # Cleanup temp file
    try:
        os.remove(path)
    except:
        pass

    # 4. Handle Rejection (No DB writes happen here)
    if decision.lower() not in ['approve', 'yes', 'y', 'confirm', 'ok']:
        print(f"❌ Request Rejected. Reason: {decision}")
        return {
            "status": "rejected", 
            "reason": decision,
            "order_id": inp.order_id,
            "duration_s": time.perf_counter() - start
        }

    print(f"✅ Request Approved. Executing database transaction...")

    # --- PHASE 3: EXECUTION (Original Logic) ---
    # This runs ONLY if approved. We keep the original _work logic 
    # (including the transaction lock) to ensure data integrity.
    def _work(conn: Any):
        try:
            cur = conn.cursor()
            
            # 2. Transaction Lock
            cur.execute("BEGIN IMMEDIATE")
            
            # 3. Validate Order Existence & Status (Re-check inside lock for safety)
            cur.execute("SELECT id, order_total, status FROM billing_orders WHERE id = ?", (inp.order_id,))
            row = cur.fetchone()
            
            if not row:
                conn.rollback()
                return {"error": "order_not_found"}

            # 4. Calculate & Validate Amount
            order_total = float(row["order_total"])
            refund_amount = float(inp.amount) if inp.amount is not None else order_total
            
            if refund_amount > order_total:
                conn.rollback()
                return {
                    "error": "amount_exceeds_order_total", 
                    "hint": f"Requested {refund_amount} but order total is {order_total}"
                }

            # 5. Execute Writes
            cur.execute(
                "INSERT INTO refunds (order_id, amount, reason, idempotency_key) VALUES (?, ?, ?, ?)",
                (inp.order_id, refund_amount, inp.reason, inp.idempotency_key)
            )
            refund_id = cur.lastrowid
            
            cur.execute("UPDATE billing_orders SET status = ? WHERE id = ?", ("refunded", inp.order_id))
            
            # 6. Commit
            conn.commit()
            
            return {
                "refund_id": refund_id, 
                "order_id": inp.order_id, 
                "amount": refund_amount, 
                "status": "completed", 
                "duration_s": time.perf_counter() - start
            }
        
        finally:
            pass

    try:
        return run_with_retries_and_timeout(_work, db_connection, retries=2, timeout_s=5.0)
    except Exception as exc:
        err = safe_error_id(exc)
        logger.exception("process_refund failed")
        try:
            db_connection.rollback()
        except Exception:
            pass
        return {"error": err}