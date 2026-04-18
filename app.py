import streamlit as st
import sqlite3
import os
from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from core.db_utils import get_db_conn
from core.setup_db import init_db

init_db()
# Load environment variables (API Keys)
load_dotenv()

# Verify the key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

# --- Import your Graph ---
from graph_nodes.main import build_graph

# --- Configuration ---
st.set_page_config(page_title="Support Chat", page_icon="💬")

# --- 1. Load the Bot Icon ---
try:
    BOT_AVATAR = Image.open("images.jpeg")
except FileNotFoundError:
    BOT_AVATAR = "🤖"

# --- 2. Initialize the Agent (Singleton) ---
@st.cache_resource
def get_agent():
    """
    Builds the LangGraph agent once and caches it.
    """
    # 1. Setup DB Connection (Ensure folder exists)
    db_path = os.path.join("data", "demo.db")
    os.makedirs("data", exist_ok=True)
    conn = get_db_conn()
    
    # 2. Build the Graph
    bot = build_graph()
    
    print("--- Agent & DB Initialized ---")
    return bot, conn

# Load the agent
agent_graph, db_conn = get_agent()

# --- 3. Custom CSS ---
st.markdown("""
<style>
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessage"]:has(img) {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
    }
    [data-testid="stChatMessage"]:has(img) * {
        color: #000000 !important;
    }
    [data-testid="stChatMessage"]:not(:has(img)) {
        background-color: #0078FF;
        color: #ffffff;
    }
    [data-testid="stChatMessage"]:not(:has(img)) p {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"]:not(:has(img)) svg {
        fill: #ffffff !important;
    }
    /* Style the sidebar button to look urgent */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 I'm Finch. How can I help you today?"}
    ]

# --- Sidebar Actions ---
with st.sidebar:
    st.header("Support Actions")
    st.write("Stuck? Request a human agent directly.")
    
    # THE HANDOFF BUTTON
    if st.button("👨‍💼 Talk to a Human", type="primary"):
        # We inject a specific trigger phrase that the Router recognizes
        st.session_state.handoff_trigger = "I want to speak to a human agent immediately."

# --- Header ---
st.title("Support Bot")
st.divider()

# --- Display Chat History ---
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message("user"):
            st.markdown(message["content"])

# --- Input Handling Logic ---
# We check if a button triggered the input OR if the user typed something
user_input = None

if "handoff_trigger" in st.session_state:
    user_input = st.session_state.handoff_trigger
    del st.session_state.handoff_trigger  # Clear it so it doesn't loop
elif prompt := st.chat_input("Type a message..."):
    user_input = prompt

# --- Agent Execution ---
if user_input:
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Prepare Agent State
    langchain_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_history.append(AIMessage(content=msg["content"]))

    current_state = {
        "messages": langchain_history,
        "router_loop_count": 0,
        "retrieval_count": 0,
    }

    # 3. Stream Agent Response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking..."):
                # Run the graph
                # Using invoke() for simplicity, but you can switch to stream() if desired
                final_state = agent_graph.invoke(current_state)
                
                # Extract response
                if final_state.get("messages"):
                    last_message = final_state["messages"][-1]
                    full_response = last_message.content
                    message_placeholder.markdown(full_response)
                else:
                    full_response = "I'm having trouble connecting. Please try again."
                    message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ An error occurred: {str(e)}"
            message_placeholder.error(full_response)
    
    # 4. Save Bot Response
    st.session_state.messages.append({"role": "assistant", "content": full_response})