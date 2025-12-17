import operator
from typing import TypedDict, List, Optional, Dict, Any, Annotated

class AgentState(TypedDict):
    """
    The shared state graph for the support bot.
    """
    
    # --- 1. Conversation History ---
    # Holds the full conversation (User, Assistant, System).
    # We use 'operator.add' (if using LangGraph) to append updates automatically.
    # Otherwise, nodes should append manually.
    messages: Annotated[List[Dict[str, str]], operator.add]

    # --- 2. The "Active" Query ---
    # The Router (Refiner) writes this. The Retriever and Tools read this.
    # This prevents "Context Pollution" by isolating the current request.
    current_task_query: str

    # --- 3. Knowledge Context ---
    # "Retriever appends the chunks in the state not replaces them"
    # We store the raw text chunks here.
    retrieved_chunks: Annotated[List[str], operator.add]

    # --- 4. Tool Execution Context ---
    # Stores outputs from tools to be used by the Synthesizer.
    # Format: [{"tool": "get_order", "input": "123", "output": "Shipped"}]
    tool_outputs: Annotated[List[Dict[str, Any]], operator.add]

    # --- 5. Control Flow & Flags ---
    
    # FOR TOOL EXECUTOR:
    # If Tool Planner selects a tool, it puts the schema here.
    # Example: {"name": "get_order_status", "args": {"order_id": "12345"}}
    pending_tool_call: Optional[Dict[str, Any]]

    # FOR ASK USER:
    # If a node needs info, it sets this reason (e.g., "Missing Order ID").
    # The 'ask_user' node reads this to generate the question.
    missing_info_reason: Optional[str]

    # --- 6. Safety Guards (Loop Limits) ---
    # "Retriever can be called only 3 times"
    retrieval_count: int
    
    # "Router can loop only 6 times"
    router_loop_count: int

    # (Optional) Tracks which node to go to next (useful for debugging)
    next_node: Optional[str]