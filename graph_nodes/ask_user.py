import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from graph_nodes.state import AgentState

# Initialize the "Small" LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0, # Strict and precise for formulating questions
    api_key=os.getenv("GOOGLE_API_KEY")
)

def ask_user_node(state: AgentState):
    """
    Generates a polite follow-up question based on missing information.
    """
    print("--- NODE: Ask User ---")
    
    reason = state.get("missing_info_reason", "I need more information.")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a support agent. You need to ask the user for specific information to proceed.
        
        Reason/Missing Info: {reason}
        
        Write a polite, concise question asking the user for this information.
        """
    )
    
    chain = prompt | llm
    
    # Generate the question
    response_msg = chain.invoke({"reason": reason})
    
    # 1. Append the AI's question to the history
    # 2. CLEAR the 'missing_info_reason' so we don't get stuck in a loop
    return {
        "messages": [response_msg],
        "missing_info_reason": None 
    }