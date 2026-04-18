import sqlite3
import os
from pathlib import Path

# Define the path to match core/db_utils.py
DB_FOLDER = Path("data")
DB_PATH = DB_FOLDER / "demo.db"

SCHEMA_SQL = """
-- Use settings for better performance and foreign key support
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys = ON;

-- USERS: Customers
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    plan TEXT NOT NULL CHECK (plan IN ('free', 'plus', 'pro')),
    status TEXT NOT NULL CHECK (status IN ('active', 'trial', 'paused', 'canceled')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- BILLING_ORDERS: Renamed from 'orders' to match process_refund.py 
CREATE TABLE IF NOT EXISTS billing_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    order_date TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    order_total REAL NOT NULL CHECK (order_total >= 0),
    status TEXT NOT NULL CHECK (status IN ('placed', 'paid', 'shipped', 'delivered', 'canceled', 'refunded')),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- INVOICES: Used by check_account_status.py
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER UNIQUE NOT NULL,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount >= 0),
    status TEXT NOT NULL CHECK (status IN ('paid', 'pending', 'refunded')),
    FOREIGN KEY (order_id) REFERENCES billing_orders (id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- REFUNDS: Used by process_refund.py
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    reason TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now')),
    status TEXT NOT NULL DEFAULT 'completed',
    FOREIGN KEY (order_id) REFERENCES billing_orders (id)
);

-- TICKETS: Used by create_support_ticket.py
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    subject TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'open',
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- HANDOFFS: Used by handoff_to_human.py
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    tags TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);

-- Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON billing_orders (user_id);
CREATE INDEX IF NOT EXISTS idx_invoices_user_id ON invoices (user_id);
CREATE INDEX IF NOT EXISTS idx_refunds_idempotency_key ON refunds (idempotency_key);
CREATE INDEX IF NOT EXISTS idx_tickets_idempotency_key ON tickets (idempotency_key);
CREATE INDEX IF NOT EXISTS idx_handoffs_conversation_id ON handoffs (conversation_id);
"""

def init_db():
    # Ensure the folder exists
    if DB_PATH.exists():
        return  # DB already exists → skip

    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Execute the Schema
    # Note: executescript can run multiple statements separated by semicolons
    cur.executescript(SCHEMA_SQL)

    # --- SEED DUMMY DATA FOR TESTING ---
    print("Seeding dummy data...")
    
    # 1. User: Alice (ID 1)
    cur.execute("""
        INSERT OR IGNORE INTO users (id, username, email, plan, status) 
        VALUES (1, 'alice_test', 'alice@example.com', 'plus', 'active')
    """)
    
    # 2. Order: Completed Order for Alice (ID 101)
    cur.execute("""
        INSERT OR IGNORE INTO billing_orders (id, user_id, order_total, status) 
        VALUES (101, 1, 50.00, 'delivered')
    """)
    
    # 3. Invoice: For Order 101
    cur.execute("""
        INSERT OR IGNORE INTO invoices (id, order_id, user_id, date, amount, status) 
        VALUES (900, 101, 1, '2023-10-01', 50.00, 'paid')
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {DB_PATH}")

if __name__ == "__main__":
    init_db()