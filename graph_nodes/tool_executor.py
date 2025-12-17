import logging
import sqlite3
import json
from graph_nodes.state import AgentState
from core.db_utils import get_db_conn

# Import the actual tool functions
# Make sure these import paths match your folder structure
from bot_tools.check_account_status import check_account_status
from bot_tools.upgrade_plan import upgrade_plan
from bot_tools.get_order_status import get_order_status
from bot_tools.process_refund import process_refund

# Setup Logger
logger = logging.getLogger(__name__)

# Map string names (from LLM) to actual Python functions
TOOL_MAP = {
    "check_account_status": check_account_status,
    "upgrade_plan": upgrade_plan,
    "get_order_status": get_order_status,
    "process_refund": process_refund,
}


def tool_executor_node(state: AgentState):
    """
    Executes the specific tool decided by the Planner.
    Injects the database connection automatically.
    """
    print("--- NODE: Tool Executor ---")
    
    # 1. Get the pending tool call
    tool_call = state.get("pending_tool_call")
    if not tool_call:
        logger.warning("Executor called but 'pending_tool_call' is empty.")
        return {"next_node": "router"}

    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    
    print(f"Executing: {tool_name} with args: {tool_args}")

    # 2. Locate the function
    func = TOOL_MAP.get(tool_name)
    if not func:
        error_msg = f"Tool '{tool_name}' not found in registry."
        logger.error(error_msg)
        return {
            "tool_outputs": [{"error": error_msg}],
            "pending_tool_call": None,
            "next_node": "synthesizer" # Let synthesizer explain the error
        }

    # 3. Execution (With Safety & Injection)
    conn = None
    result = None
    try:
        # Create DB connection specifically for this operation
        conn = get_db_conn()
        
        # INJECTION: We manually add 'db_connection' to the args
        # because the LLM (Planner) doesn't know about it.
        tool_args["db_connection"] = conn
        
        # EXECUTE THE TOOL
        # We pass **tool_args to unpack the dictionary into arguments
        result = func(**tool_args)
        
        # Tag the result with the tool name for the Synthesizer
        if isinstance(result, dict):
            result["tool"] = tool_name
        else:
            # Fallback if a tool returns a raw string/number
            result = {"tool": tool_name, "output": str(result)}

    except Exception as e:
        logger.exception(f"Tool execution failed for {tool_name}")
        result = {
            "tool": tool_name,
            "error": str(e),
            "hint": "Internal tool error. Please notify admin."
        }
    
    finally:
        # ALWAYS close the connection to prevent database locks
        if conn:
            conn.close()

    # 4. Update State
    # We append the result to 'tool_outputs' so Synthesizer can read it.
    # We Clear 'pending_tool_call' because it's done.
    # We direct the flow to the Synthesizer.
    return {
        "tool_outputs": [result], 
        "pending_tool_call": None,
        "next_node": "synthesizer"
    }