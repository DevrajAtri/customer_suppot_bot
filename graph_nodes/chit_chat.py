import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from graph_nodes.state import AgentState

# Initialize the "Small" LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7, # Slightly creative for greetings
    api_key=os.getenv("GOOGLE_API_KEY")
)

def chit_chat_node(state: AgentState):
    """
    Handles simple greetings and closures.
    """
    print("--- NODE: Chit Chat ---")
    
    # We provide a very simple system instruction to keep it focused.
    messages = [
        ("system", "You are a helpful customer support assistant for an E-commerce platform. Reply to the user's greeting or closing politely. Do not try to solve technical issues here, just be social."),
    ] + state['messages']
    
    # Generate response
    response = llm.invoke(messages)
    
    # Return the update (LangGraph automatically appends this to the message history)
    return {"messages": [response]}