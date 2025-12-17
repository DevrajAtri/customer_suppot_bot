import os
import logging
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from graph_nodes.state import AgentState

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=os.getenv("GOOGLE_API_KEY")
)

def format_context(state: AgentState) -> str:
    context_parts = []
    
    # 1. Tool Outputs
    tool_outputs = state.get("tool_outputs", [])
    if tool_outputs:
        context_parts.append("--- DATA FROM DATABASE ---")
        for t in tool_outputs:
            if "error" in t:
                context_parts.append(f"Tool Error: {t['error']} (Hint: {t.get('hint', 'None')})")
            else:
                clean = {k: v for k, v in t.items() if k not in ["duration_s", "tool", "query"]}
                context_parts.append(f"Tool Result: {clean}")
    
    # 2. Retrieved Docs
    docs = state.get("retrieved_chunks", [])
    if docs:
        context_parts.append("\n--- NEW SEARCH RESULTS ---")
        for i, doc in enumerate(docs):
            context_parts.append(f"Chunk {i+1}: {doc}")
            
    return "\n".join(context_parts)

def synthesizer_node(state: AgentState):
    print("--- NODE: Synthesizer ---")
    
    # 1. Prepare Variables
    query = state.get("current_task_query", "User Request")
    context_str = format_context(state)
    messages = state["messages"]
    
    # 2. Prepare History
    history_buffer = messages[:-1] if len(messages) > 1 else []
    chat_history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in history_buffer[-10:]])
    
    # 3. XML Prompt (Double Braces for JSON examples!)
    prompt_template = """
<system_role>
You are Finch, a helpful and precise Customer Support Agent for an E-commerce platform.
Your goal is to answer the user's question based STRICTLY on the provided data sources.
</system_role>

<data_sources>
    <chat_history>
    {chat_history}
    </chat_history>

    <fresh_context>
    {context}
    </fresh_context>
</data_sources>

<examples>
    <example_1>
        <type>Tool Result (Success)</type>
        <user_query>Where is order #1001?</user_query>
        <fresh_context>Tool Result: {{'status': 'shipped', 'delivery_date': '2023-10-25'}}</fresh_context>
        <chat_history>...</chat_history>
        <thought_process>
            1. Check Fresh Context: Found tool output for order #1001. Status is shipped.
            2. Formulate Answer: State status and date.
        </thought_process>
        <response>Order #1001 has been shipped and is expected to arrive on 2023-10-25.</response>
    </example_1>

    <example_2>
        <type>History Retrieval (The "Memory" Edge Case)</type>
        <user_query>Can I return swimwear?</user_query>
        <fresh_context>No new search results.</fresh_context>
        <chat_history>
            User: What is the return policy?
            Bot: Returns are allowed within 30 days. Exceptions: Swimwear and Lingerie cannot be returned.
        </chat_history>
        <thought_process>
            1. Check Fresh Context: Empty.
            2. Check Chat History: Found previous discussion about return policy. It explicitly states swimwear is an exception.
            3. Formulate Answer: Refer back to the previous information.
        </thought_process>
        <response>As mentioned in our earlier conversation regarding the return policy, swimwear is listed as an exception and cannot be returned.</response>
    </example_2>

    <example_3>
        <type>Ambiguous/Missing Info</type>
        <user_query>Do you sell laptops?</user_query>
        <fresh_context>Chunk 1: We sell clothing and accessories...</fresh_context>
        <chat_history>...</chat_history>
        <thought_process>
            1. Check Fresh Context: Context mentions clothing, doesn't mention electronics/laptops.
            2. Check History: No mention.
            3. Constraint Check: If info is missing, admit it.
        </thought_process>
        <response>I apologize, but I don't see any information about laptops in our catalog. We primarily specialize in clothing and accessories.</response>
    </example_3>

    <example_4>
        <type>Tool Error</type>
        <user_query>Check order #999</user_query>
        <fresh_context>Tool Error: Order ID not found.</fresh_context>
        <chat_history>...</chat_history>
        <thought_process>
            1. Check Fresh Context: The tool returned an error "Order ID not found".
            2. Formulate Answer: Inform the user the ID is invalid.
        </thought_process>
        <response>I couldn't find an order with ID #999. Could you please double-check the number?</response>
    </example_4>

    <example_5>
        <type>Unsupported Action / Directive</type>
        <user_query>explain cannot delete account history</user_query>
        <fresh_context>No new search results.</fresh_context>
        <chat_history>...</chat_history>
        <thought_process>
            1. Check Fresh Context: Empty.
            2. Analyze Query: The query is not asking a question, it is an INSTRUCTION from the Router to explain a limitation.
            3. Formulate Answer: Politely inform the user this action is not possible.
        </thought_process>
        <response>I apologize, but we cannot delete your account history as it is required for billing and audit purposes. This action is not currently supported.</response>
    </example_5>

    <example_6>
        <type>Tool Business Logic Rejection</type>
        <user_query>Refund order #5001</user_query>
        <fresh_context>Tool Result: {{'status': 'rejected', 'reason': 'Policy violation: Item used', 'order_id': 5001}}</fresh_context>
        <chat_history>...</chat_history>
        <thought_process>
            1. Check Fresh Context: Found tool output. Status is 'rejected'.
            2. Analyze Reason: The rejection reason is "Policy violation: Item used".
            3. Formulate Answer: clearly explain WHY the request was denied using the provided reason.
        </thought_process>
        <response>I cannot process the refund for order #5001 because it violates our policy: The item has been used.</response>
    </example_6>
</examples>

<instructions>
1. **Analyze Sources & Query:** Look at <fresh_context> and <chat_history>. ALSO look at the <user_query>—if it is an instruction (e.g., "explain cannot do X"), follow it directly even if context is empty.
2. **Reasoning:** Before answering, write a brief <thought_process> to confirm where you found the answer or why you are declining.
3. **Tone:** Professional, direct, and polite.
4. **Constraint:** If the user is asking a QUESTION and the answer is NOT in the history or fresh context, say "I don't have that information." Do not hallucinate facts, but you MAY explain system limitations if instructed by the query.
</instructions>

<user_query>
{query}
</user_query>
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    # RESTORED: Error Handling
    try:
        response_msg = chain.invoke({
            "context": context_str,
            "chat_history": chat_history_str,
            "query": query
        })
        
        # --- NEW: CLEANING LOGIC ---
        # 1. Remove the <thought_process> block
        raw_content = response_msg.content
        clean_content = re.sub(r'<thought_process>.*?</thought_process>', '', raw_content, flags=re.DOTALL).strip()
        
        # 2. Remove <response> tags if the model used them (just in case)
        clean_content = clean_content.replace('<response>', '').replace('</response>', '').strip()
        
        # 3. Update the message content with the clean version
        response_msg.content = clean_content
        
    except Exception as e:
        logger.error(f"Synthesizer LLM Error: {e}")
        response_msg = AIMessage(content="I apologize, but I am having trouble connecting to my brain right now.")

    # 4. Return & Flush State
    return {
        "messages": [response_msg],
        "current_task_query": None,
        "tool_outputs": [],
        "retrieved_chunks": [],
        "pending_tool_call": None,
        "missing_info_reason": None,
        "router_loop_count": 0,
        "retrieval_count": 0
    }