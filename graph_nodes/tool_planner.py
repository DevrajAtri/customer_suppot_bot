import os
import json
import logging
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from xxx import BaseModel, Field

from graph_nodes.state import AgentState
from bot_tools.schemas import (
    CheckStatusIn, 
    UpgradePlanIn, 
    GetOrderStatusIn, 
    ProcessRefundIn,
    HandoffIn
)

# Setup Logger
logger = logging.getLogger(__name__)

# --- 1. LLM SETUP ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, 
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# --- 2. STRUCTURED OUTPUT ---
class ToolDecision(BaseModel):
    thought_process: str = Field(
        description="Internal monologue. 1. Identify User Intent. 2. Match to Tool. 3. Check for ALL required arguments in history."
    )
    action: Literal["call_tool", "ask_user", "no_tool"] = Field(
        description="The decision: run a tool, ask for missing info, or give up."
    )
    tool_name: Optional[str] = Field(
        default=None, 
        description="The exact name of the tool to call."
    )
    args_json: Optional[str] = Field(
        default="{}", 
        description="Valid JSON string of arguments. Example: '{\"order_id\": 123}'"
    )
    missing_info: Optional[str] = Field(
        default=None, 
        description="If action is 'ask_user', explain precisely what is missing."
    )

# --- 3. IMPROVED PROMPT FUNCTION ---
def get_system_prompt():
    schemas = {
        "check_account_status": json.dumps(CheckStatusIn.model_json_schema()),
        "upgrade_plan": json.dumps(UpgradePlanIn.model_json_schema()),
        "get_order_status": json.dumps(GetOrderStatusIn.model_json_schema()),
        "process_refund": json.dumps(ProcessRefundIn.model_json_schema()),
        "handoff_to_human": json.dumps(HandoffIn.model_json_schema()),
    }
    
    formatted_schemas = "\n".join([f'<tool name="{k}">\n{v}\n</tool>' for k, v in schemas.items()])

    return f"""
<system_role>
You are the expert Tool Planner for an E-commerce Support Bot.
Your goal is to map user requests to tools and extract arguments with 100% precision.
</system_role>

<tool_definitions>
{formatted_schemas}
</tool_definitions>

<examples>
    <example>
        <input>User: "Refund order 1001 because it is broken."</input>
        <thought_process>
            1. Intent: Refund. Matches 'process_refund'.
            2. Schema Check: Requires 'order_id' (found 1001) and 'reason' (found "broken").
            3. Decision: All args present. Call tool.
        </thought_process>
        <output>action="call_tool", tool_name="process_refund", args_json='{{ "order_id": 1001, "reason": "broken" }}'</output>
    </example>

    <example>
        <input>User: "I want a refund."</input>
        <thought_process>
            1. Intent: Refund. Matches 'process_refund'.
            2. Schema Check: Requires 'order_id' and 'reason'.
            3. Analysis: History does not contain order ID or reason.
            4. Decision: Missing critical info. Must ask user.
        </thought_process>
        <output>action="ask_user", missing_info="I need the Order ID and the reason for the return."</output>
    </example>

    <example>
        <input>User: "Where is it?" (History: "Order #555 shipped?")</input>
        <thought_process>
            1. Intent: Status. Matches 'get_order_status'.
            2. Schema Check: Requires 'order_id'.
            3. Analysis: User said "it", referring to "Order #555" in history.
            4. Decision: Arg found in history. Call tool.
        </thought_process>
        <output>action="call_tool", tool_name="get_order_status", args_json='{{ "order_id": 555 }}'</output>
    </example>
</examples>

<instructions>
1. **Analyze History:** Look at <conversation_history> to find parameter values. Resolve pronouns (e.g., "it" -> previous order ID).
2. **Strict Matching:** based strictly on the <tool_definitions> provided above.
   - If ALL required arguments are found -> action: "call_tool"
   - If the tool is known but arguments are missing -> action: "ask_user"
   - If no tool matches -> action: "no_tool"
3. **Safety:** Do NOT invent values. If an ID is missing, you must ask for it.
4. **Format:** Output the arguments as a valid JSON string in 'args_json'.
</instructions>
"""

# --- 4. MAIN NODE ---
def tool_planner_node(state: AgentState):
    print("--- NODE: Tool Planner ---")
    
    query = state.get("current_task_query")
    messages = state.get("messages", [])
    
    if not query:
        return {"next_node": "router"}

    planner = llm.with_structured_output(ToolDecision)
    
    # Format History with XML tags
    history_text = "\n".join([f"{m.type}: {m.content}" for m in messages[-5:]])
    
    system_msg = SystemMessage(content=get_system_prompt())
    
    # We wrap the dynamic input in XML tags too
    user_content = f"""
<current_task>
{query}
</current_task>

<conversation_history>
{history_text}
</conversation_history>
"""
    user_msg = HumanMessage(content=user_content)

    try:
        decision = planner.invoke([system_msg, user_msg])
        
        print(f"Planner Thought: {decision.thought_process}")
        print(f"Planner Action: {decision.action}")

        # --- BRANCH A: CALL TOOL ---
        if decision.action == "call_tool" and decision.tool_name:
            try:
                parsed_args = json.loads(decision.args_json)
            except json.JSONDecodeError:
                return {
                    "missing_info_reason": "Failed to parse parameters.",
                    "next_node": "ask_user"
                }

            return {
                "pending_tool_call": {
                    "name": decision.tool_name,
                    "args": parsed_args
                },
                "missing_info_reason": None,
                "next_node": "tool_executor"
            }

        # --- BRANCH B: ASK USER ---
        elif decision.action == "ask_user":
            return {
                "missing_info_reason": decision.missing_info,
                "pending_tool_call": None,
                "next_node": "ask_user"
            }

        # --- BRANCH C: FALLBACK ---
        else:
            return {"next_node": "router"}

    except Exception as e:
        logger.error(f"Tool Planner Error: {e}")
        return {"next_node": "router"}