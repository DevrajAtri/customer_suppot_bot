import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FOLDER = Path("data")
DB_PATH = DB_FOLDER / "demo.db"

def get_db_connection():
    if not DB_PATH.exists():
        # Fallback if folder doesn't exist, though setup_db usually handles this
        print(f"Database not found at {DB_PATH}. Creating empty DB...")
        os.makedirs(DB_FOLDER, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def clear_tables(conn):
    """
    Wipes all data to ensure a clean state for testing.
    Deletes in specific order to avoid Foreign Key constraint errors.
    """
    cur = conn.cursor()
    print("--- 🧹 Clearing Old Data ---")
    
    # Order matters due to Foreign Keys!
    # Delete children first, then parents.
    tables = ["handoffs", "tickets", "refunds", "invoices", "billing_orders", "users"]
    
    for table in tables:
        try:
            cur.execute(f"DELETE FROM {table}")
            # Reset the auto-increment counter so IDs start from 1 (or 1001) again
            cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
        except sqlite3.OperationalError as e:
            print(f"Warning: Issue clearing {table} (might not exist): {e}")
            
    conn.commit()

def generate_data(conn):
    cur = conn.cursor()
    print("--- 📝 Inserting Rich Test Data ---")

    # =========================================================================
    # 1. USERS (The Personas)
    # =========================================================================
    # We create 6 distinct users to test every account state.
    users = [
        # (ID, Username, Email, Plan, Status)
        (1, "alice_active", "alice@example.com", "pro", "active"),        # The Perfect Customer
        (2, "bob_paused", "bob@example.com", "plus", "paused"),           # Payment failed context
        (3, "charlie_trial", "charlie@example.com", "free", "trial"),     # Upsell target (Upgrade Tool)
        (4, "dave_churned", "dave@example.com", "free", "canceled"),      # Win-back context
        (5, "eve_refunds", "eve@example.com", "plus", "active"),          # Refund heavy user
        (6, "frank_new", "frank@example.com", "free", "active"),          # Brand new, no orders yet
    ]
    
    cur.executemany("""
        INSERT INTO users (id, username, email, plan, status) 
        VALUES (?, ?, ?, ?, ?)
    """, users)
    print(f"✅ Inserted {len(users)} Users")

    # =========================================================================
    # 2. BILLING_ORDERS (Varied Statuses)
    # =========================================================================
    # We need orders in every state: placed, paid, shipped, delivered, canceled, refunded.
    # Note: We use specific IDs (1001, 1002...) to make testing easier.
    
    orders = [
        # --- Alice (User 1): Good history ---
        # ID, UserID, Date, Total, Status
        (1001, 1, (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), 299.99, 'shipped'),   # "Where is my order?"
        (1002, 1, (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'), 49.50, 'delivered'), # Old successful order
        
        # --- Bob (User 2): Paused account, but has orders ---
        (2001, 2, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 19.99, 'placed'),     # Just placed, maybe stuck?
        
        # --- Charlie (User 3): Trial user ---
        (3001, 3, (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d'), 150.00, 'paid'),      # Paid today, not shipped
        
        # --- Dave (User 4): Canceled user ---
        (4001, 4, (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'), 80.00, 'canceled'),  # Old history
        
        # --- Eve (User 5): The Refund Candidate ---
        (5001, 5, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 120.00, 'delivered'), # <--- TARGET FOR REFUND TOOL
        (5002, 5, (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'), 50.00, 'refunded'),  # Already refunded (Test logic)
        (5003, 5, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), 200.00, 'delivered'), # Another eligible one
    ]

    cur.executemany("""
        INSERT INTO billing_orders (id, user_id, order_date, order_total, status) 
        VALUES (?, ?, ?, ?, ?)
    """, orders)
    print(f"✅ Inserted {len(orders)} Orders")

    # =========================================================================
    # 3. INVOICES (Linked to Orders)
    # =========================================================================
    # Invoices confirm payment status.
    invoices = [
        # (ID, OrderID, UserID, Date, Amount, Status)
        (901, 1001, 1, (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), 299.99, 'paid'),
        (902, 1002, 1, (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'), 49.50, 'paid'),
        
        (903, 2001, 2, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 19.99, 'pending'), # Bob's payment pending?
        
        (904, 3001, 3, (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d'), 150.00, 'paid'),
        
        # Eve's orders
        (905, 5001, 5, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 120.00, 'paid'),
        (906, 5002, 5, (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'), 50.00, 'refunded'),
        (907, 5003, 5, (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), 200.00, 'paid'),
    ]
    
    cur.executemany("""
        INSERT INTO invoices (id, order_id, user_id, date, amount, status) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, invoices)
    print(f"✅ Inserted {len(invoices)} Invoices")

    # =========================================================================
    # 4. REFUNDS (Historical Data)
    # =========================================================================
    # Order 5002 was already refunded. We log it here.
    refunds = [
        # (OrderID, Amount, Reason, Key)
        (5002, 50.00, "Item didn't fit", "ref_5002_legacy")
    ]
    
    cur.executemany("""
        INSERT INTO refunds (order_id, amount, reason, idempotency_key, status) 
        VALUES (?, ?, ?, ?, 'completed')
    """, refunds)
    print(f"✅ Inserted {len(refunds)} Historical Refunds")

    # =========================================================================
    # 5. TICKETS (Support Context)
    # =========================================================================
    # Test if the bot can see open issues.
    tickets = [
        # (UserID, Subject, Description, Priority, Status, Key)
        (2, "Payment Failing", "I tried to pay for order 2001 but it failed.", "high", "open", "tkt_bob_1"),
        (4, "Why canceled?", "I want to reactivate my account.", "medium", "open", "tkt_dave_1"),
        (1, "Thanks", "Just saying I love the service.", "low", "closed", "tkt_alice_1"),
    ]
    
    cur.executemany("""
        INSERT INTO tickets (user_id, subject, description, priority, status, idempotency_key) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, tickets)
    print(f"✅ Inserted {len(tickets)} Support Tickets")

    conn.commit()

if __name__ == "__main__":
    try:
        connection = get_db_connection()
        clear_tables(connection)
        generate_data(connection)
        connection.close()
        print("\n🎉 SUCCESS: Database populated with rich test scenarios!")
        print("-------------------------------------------------------")
        print("User 1 (Alice): Active, Pro, Order #1001 (Shipped)")
        print("User 2 (Bob):   Paused, Plus, Order #2001 (Placed/Pending)")
        print("User 3 (Charlie): Trial, Free, Order #3001 (Paid)")
        print("User 5 (Eve):   Active, Plus, Order #5001 (Delivered -> REFUNDABLE)")
        print("-------------------------------------------------------")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")