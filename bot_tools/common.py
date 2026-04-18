from typing import Callable, Dict, Any, Optional, Type
from xxx import BaseModel
import logging
import time
import functools
import sqlite3  # <--- Added for error handling

logger = logging.getLogger(__name__)

# Global registry to hold all tool definitions
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_tool(name: str, input_schema: Optional[Type[BaseModel]] = None, 
                  output_schema: Optional[Type[BaseModel]] = None):
    """
    Decorator to register a tool with the global registry.
    """
    def deco(fn: Callable):
        TOOL_REGISTRY[name] = {
            "fn": fn, 
            "input_schema": input_schema,
            "output_schema": output_schema
        }
        logger.info("Registered tool -> %s", name)
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return deco

def get_registered_tools() -> Dict[str, Dict[str, Any]]:
    return TOOL_REGISTRY

def run_with_retries_and_timeout(fn, *args, retries: int = 2, 
                                 timeout_s: float = 5.0, backoff: float = 0.5, **kwargs):
    """
    Executes a function with retries and handles DB transaction rollbacks.
    """
    last_exc = None
    
    # attempt 1 + retries
    for attempt in range(1, retries + 2):
        start = time.perf_counter()
        try:
            # Execute synchronously to preserve the DB connection thread context
            result = fn(*args, **kwargs)
            
            duration = time.perf_counter() - start
            logger.info("Success calling %s in %.3fs (attempt %d)", 
                        getattr(fn, "__name__", str(fn)), duration, attempt)
            return result
            
        except Exception as e:
            # --- FIX: Transaction Cleanup ---
            # If the tool crashed, the transaction might still be open.
            # We must verify if any argument is a DB connection and roll it back.
            conn = None
            
            # Check positional args for a connection
            for arg in args:
                if isinstance(arg, sqlite3.Connection):
                    conn = arg
                    break
            
            # Check keyword args if not found yet
            if not conn:
                for v in kwargs.values():
                    if isinstance(v, sqlite3.Connection):
                        conn = v
                        break
            
            # Perform Rollback
            if conn:
                try:
                    conn.rollback()
                    logger.warning("♻️ Rolled back DB transaction after error in %s", getattr(fn, "__name__", str(fn)))
                except Exception as rb_e:
                    logger.error("Failed to rollback transaction: %s", rb_e)
            # --------------------------------
            
            last_exc = e
            duration = time.perf_counter() - start
            logger.warning("Exception calling %s (attempt %d in %.2fs): %s",
                           getattr(fn, "__name__", str(fn)), attempt, duration, e)
            
            # If we have retries left, wait and continue
            if attempt <= retries:
                time.sleep(backoff * attempt)
    
    # If we exit the loop, we failed all attempts
    raise last_exc