# core/db_utils.py
import sqlite3
from pathlib import Path
import logging
import time
import traceback
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- THE FIX ---
# 1. Get the absolute path of THIS file (core/db_utils.py)
CURRENT_FILE = Path(__file__).resolve()
# 2. Go up two levels: core/db_utils.py -> core -> ROOT
ROOT_DIR = CURRENT_FILE.parent.parent
# 3. Define the absolute path to the data folder
DEFAULT_DB = ROOT_DIR / "data" / "demo.db"

print(f"🔍 DEBUG: Expecting Database at: {DEFAULT_DB}")

def get_db_path(db_path: Optional[str]) -> str:
    path = str(Path(db_path) if db_path else DEFAULT_DB)
    return path

def get_db_conn(db_path: Optional[str] = None, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Returns a sqlite3.Connection configured for the app.
    Caller is responsible for closing it.
    """
    path = get_db_path(db_path)
    
    # CRITICAL CHECK: Does the file actually exist?
    if not Path(path).exists():
        print(f"❌ ERROR: Database file NOT found at: {path}")
        print(f"   (Python is looking in: {Path.cwd()})")
    else:
        print(f"✅ CONNECTING: Found database at {path}")

    conn = sqlite3.connect(path, timeout=timeout, detect_types=sqlite3.PARSE_DECLTYPES)
    # --- INSERT THIS BLOCK ---
    # This forces Python to list every table it can see immediately
    check_cur = conn.cursor()
    tables = [row[0] for row in check_cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print(f"👀 DIAGNOSTIC: Python sees these tables: {tables}")
    # -------------------------
    conn.row_factory = sqlite3.Row
    
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute(f"PRAGMA busy_timeout={int(timeout*1000)};")
    except Exception:
        logger.exception("Failed to set pragmas for DB %s", path)
    return conn

def safe_error_id(exc: Exception) -> str:
    """Log full trace and return opaque short id for user-facing errors."""
    eid = hex(int(time.time() * 1000))[-8:]
    logger.error("Error id %s: %s", eid, exc)
    logger.debug(traceback.format_exc())
    return f"err_{eid}"