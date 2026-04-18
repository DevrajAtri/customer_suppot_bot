import os
import logging
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from xxx import BaseModel, Field

from graph_nodes.state import AgentState

# Setup Logger
logger = logging.getLogger(__name__)

# --- 1. LLM SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 2. STRUCTURED OUTPUT DEFINITION ---
class RouterDecision(BaseModel):
    # We enforce a 'thought_process' to match the XML prompt's logic
    thought_process: str = Field(
        description="Internal reasoning about the context, tools, and user intent."
    )
    route: Literal["chit_chat", "ask_user", "retriever", "tool_planner", "synthesizer"] = Field(
        description="The selected next node."
    )
    refined_query: str = Field(
        description="A specific, standalone query (e.g., 'status of order 123')."
    )

# --- 3. HELPER: TRUNCATE TEXT ---
def format_knowledge_chunks(chunks: list) -> str:
    if not chunks:
        return "No knowledge retrieved yet."
    formatted = []
    for i, chunk in enumerate(chunks):
        words = chunk.split()
        truncated = " ".join(words[:150]) # Limit to 150 words per chunk
        formatted.append(f"Chunk {i+1}: {truncated}...")
    return "\n".join(formatted)

# --- 4. MAIN NODE ---
def router_node(state: AgentState):
    print("--- NODE: Router ---")
    
    # --- A. READ STATE & CALCULATE CONSTRAINTS ---
    current_loop_count = state.get("router_loop_count", 0)
    retrieval_count = state.get("retrieval_count", 0)
    messages = state.get("messages", [])
    
    # Constraint 1: Max Total Loops (Circuit Breaker)
    if current_loop_count >= 6:
        logger.warning(f"Router hit max loops ({current_loop_count}). Forcing synthesis.")
        return {
            "next_node": "synthesizer",
            "current_task_query": "Summarize the conversation and apologize for being stuck.",
        }

    # Constraint 2: Retrieval Logic
    allow_retrieval = retrieval_count < 3

    # --- B. PREPARE PROMPT VARIABLES ---
    # 1. User Input
    user_input = messages[-1].content if messages else ""
    
    # 2. History (Serialized for context)
    # We take the last 5 messages to allow pronoun resolution (e.g., "it" -> "order 123")
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-5:]])

    # 3. Tool Outputs (Formatted)
    raw_tools = state.get("tool_outputs", [])
    context_tools = str(raw_tools) if raw_tools else "No tool outputs yet."

    # 4. Knowledge (Formatted & Truncated)
    raw_chunks = state.get("retrieved_chunks", [])
    context_knowledge = format_knowledge_chunks(raw_chunks)

    # --- C. CONSTRUCT XML SYSTEM PROMPT ---
    system_prompt = f"""
<system_role>
You are the Central Orchestrator (Router) for an E-commerce Support Bot.
Your goal is to strictly route the user's request to the correct node based on the specific capabilities defined below.
</system_role>

<tool_registry>
    <tool name="check_account_status">
        <triggers>my account status, why is my account paused, check subscription</triggers>
    </tool>
    
    <tool name="upgrade_plan">
        <triggers>upgrade to pro, change my plan, switch subscription</triggers>
    </tool>
    
    <tool name="get_order_status">
        <triggers>where is my order, shipping status, tracking, delivery date</triggers>
    </tool>
    
    <tool name="process_refund">
        <triggers>refund my order, I want my money back, cancel and refund</triggers>
        <anti_triggers>how do I return, return policy, can I return this</anti_triggers> </tool>
    
    <tool name="handoff_to_human">
        <triggers>speak to human, talk to agent, person please, escalate issue, supervisor, file a complaint, issue not resolved</triggers>
    </tool>
</tool_registry>

<context_state>
    <conversation_history>
{history_str}
    </conversation_history>

    <tool_outputs>
{context_tools}
    </tool_outputs>
    
    <retrieved_knowledge>
{context_knowledge}
    </retrieved_knowledge>

    <retrieval_status>
        Allowed: {allow_retrieval}
        Current Count: {retrieval_count}/3
    </retrieval_status>
</context_state>

<routing_options>
    1. 'chit_chat': For generic small talk (Hi, Thanks, Bye).
    2. 'ask_user': If intent is unclear or details are missing (e.g., "It's broken" -> "What is broken?").
    3. 'retriever': For Policy/Terms questions. *RESTRICTION: Only pick if <retrieval_status> is True.*
    4. 'tool_planner': If the user implies an Action (Refund, Status) AND that action exists in <tool_registry>.
    5. 'synthesizer': If <tool_outputs> or <retrieved_knowledge> contains the answer to the user's question.
</routing_options>

<examples>
    <example>
        <input>User: "Where is order #1001?"</input>
        <context>Tools: [], Knowledge: []</context>
        <thought>Matches 'get_order_status' trigger. Tool hasn't run yet.</thought>
        <decision>{{ "route": "tool_planner", "refined_query": "get status for order 1001" }}</decision>
    </example>

    <example>
        <input>User: "Can I return swimwear?"</input>
        <context>Tools: [], Knowledge: []</context>
        <thought>User is asking about policy (anti-trigger for refund). This is a knowledge retrieval question.</thought>
        <decision>{{ "route": "retriever", "refined_query": "return policy for swimwear" }}</decision>
    </example>

    <example>
        <input>User: "Refund it."</input>
        <context>Tools: [{{'status': 'success', 'refund_id': 999}}], Knowledge: []</context>
        <thought>The user is confirming, BUT the tool output shows the refund was already processed successfully just now. I should not run it again.</thought>
        <decision>{{ "route": "synthesizer", "refined_query": "confirm refund success" }}</decision>
    </example>
    
    <example>
        <input>User: "Delete my account history."</input>
        <context>Tools: [], Knowledge: []</context>
        <thought>User wants action, BUT 'delete account' is NOT in <tool_registry>. Sending to planner will fail. Must synthesize apology.</thought>
        <decision>{{ "route": "synthesizer", "refined_query": "explain cannot delete account history" }}</decision>
    </example>

    <example>
        <input>User: "What about the other one?" (Context: History discusses Order #123 and #456)</input>
        <context>Tools: [], Knowledge: []</context>
        
        <decision>{{ "route": "tool_planner", "refined_query": "check status for order #456" }}</decision>
    </example>
</examples>

<instructions>
Analyze the user's latest input against the <tool_registry> and <context_state>.

First, think step-by-step inside 'thought_process' field:
1. **Existing Data Check:** Is the answer already in <tool_outputs> or <retrieved_knowledge>? If yes -> 'synthesizer'.
2. **Policy Check:** Is the user asking for a policy? -> 'retriever'.
3. **Action Check (Strict):** Is the user asking for an action found in <tool_registry>?
   - IF YES -> 'tool_planner'.
   - IF NO (Action not supported) -> 'synthesizer' (to apologize).
4. **Ambiguity:** Is the input unclear? -> 'ask_user'.

Then, produce the final JSON.
</instructions>
"""

    # --- D. INVOKE LLM ---
    router = llm.with_structured_output(RouterDecision)
    
    try:
        decision = router.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"<latest_user_input>\n{user_input}\n</latest_user_input>")
        ])
    except Exception as e:
        logger.error(f"Router LLM Failed: {e}")
        return {"next_node": "synthesizer"}

    print(f"Router Thought: {decision.thought_process}")
    print(f"Router Route: {decision.route} | Query: {decision.refined_query}")

    # --- E. UPDATE STATE & RETURN ---
    updates = {
        "current_task_query": decision.refined_query,
        "router_loop_count": current_loop_count + 1,
        "next_node": decision.route
    }
    
    # Increment retrieval count only if we actually go to retriever
    if decision.route == "retriever":
        updates["retrieval_count"] = retrieval_count + 1
    
    # If routing to 'ask_user', pass the specific question so it knows what to ask
    if decision.route == "ask_user":
        updates["missing_info_reason"] = decision.refined_query
        
    return updates