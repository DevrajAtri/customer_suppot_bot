import os
import logging
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

from langgraph.graph import StateGraph, END
from graph_nodes.state import AgentState

# Import all the nodes we created
from graph_nodes.router import router_node
from graph_nodes.chit_chat import chit_chat_node
from graph_nodes.ask_user import ask_user_node
from graph_nodes.retriever import retriever_node
from graph_nodes.tool_planner import tool_planner_node
from graph_nodes.tool_executor import tool_executor_node
from graph_nodes.synthesizer import synthesizer_node

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_next_node(state: AgentState):
    """
    Traffic Controller Function.
    Reads the 'next_node' key set by the previous node and determines where to go.
    """
    next_node = state.get("next_node")
    
    # Safety fallback
    if not next_node:
        logger.warning("No 'next_node' found in state. Defaulting to END.")
        return END
        
    return next_node

def build_graph():
    """
    Constructs the StateGraph for the E-commerce Bot.
    """
    # 1. Initialize the Graph with our State
    workflow = StateGraph(AgentState)

    # 2. Add All Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("chit_chat", chit_chat_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tool_planner", tool_planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 3. Define Entry Point
    workflow.set_entry_point("router")

    # 4. Define Edges (The Logic Flow)
    
    # --- ROUTER EDGES ---
    # The Router decides everything. It can go to:
    # chit_chat, ask_user, retriever, tool_planner, or synthesizer.
    workflow.add_conditional_edges(
        "router",
        get_next_node,
        {
            "chit_chat": "chit_chat",
            "ask_user": "ask_user",
            "retriever": "retriever",
            "tool_planner": "tool_planner",
            "synthesizer": "synthesizer"
        }
    )

    # --- TOOL PLANNER EDGES ---
    # Can go to: tool_executor (success), ask_user (missing info), or router (failure)
    workflow.add_conditional_edges(
        "tool_planner",
        get_next_node,
        {
            "tool_executor": "tool_executor",
            "ask_user": "ask_user",
            "router": "router"
        }
    )

    # --- RETRIEVER EDGE ---
    # Always goes back to Router to re-evaluate with new info
    workflow.add_edge("retriever", "router")

    # --- TOOL EXECUTOR EDGE ---
    # Based on our code, it currently points to 'synthesizer'.
    # If you change logic to go back to router, this dynamic check handles it automatically.
    workflow.add_conditional_edges(
        "tool_executor",
        get_next_node,
        {
            "synthesizer": "synthesizer",
            "router": "router"
        }
    )

    # --- TERMINAL EDGES (Conversational Stops) ---
    # These nodes output a message and then stop to wait for the user.
    workflow.add_edge("chit_chat", END)
    workflow.add_edge("ask_user", END)
    workflow.add_edge("synthesizer", END)

    # 5. Compile
    app = workflow.compile()
    return app

# --- Execution Entry Point (for testing) ---
if __name__ == "__main__":
    app = build_graph()
    print("Graph compiled successfully.")
    
    # Optional: Generate a diagram if you have graphviz installed
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("bot_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph diagram saved to bot_graph.png")
    except Exception as e:
        print(f"Could not draw graph: {e}")