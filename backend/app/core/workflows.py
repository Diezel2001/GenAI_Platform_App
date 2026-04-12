"""
LangGraph Workflow - Minimal Complete Example

A single workflow demonstrating all core LangGraph patterns:
- State schema
- Tools
- Nodes  
- Routers (conditional edges)
- START → END flow

Example Usage:
    from backend.app.core.workflows import create_workflow

    app = create_workflow().compile()
    result = app.invoke({"messages": [HumanMessage(content="Your input")]})
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool, ToolRegistry


# =============================================================================
# STATE SCHEMA
# =============================================================================

class WorkflowState(TypedDict, total=False):
    """State schema for the workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    confidence: float


# =============================================================================
# TOOLS
# =============================================================================

# Create tool registry
registry = ToolRegistry()

@tool
def search_documents(query: str) -> str:
    """Search internal documents."""
    return f"Found: {query} content"

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web results for: {query}"

# Add tools to registry
registry.add_tool(search_documents)
registry.add_tool(web_search)

# Get compiled tools
tools = registry.get_tools()

def get_compiled_tools():
    """Get the compiled tools from the registry."""
    return tools


# =============================================================================
# NODES
# =============================================================================

def classify_node(state: WorkflowState) -> dict:
    """Classify user intent."""
    messages = state.get("messages", [])
    content = messages[-1].content if messages else ""
    
    intent = "web" if "web" in content.lower() else "internal"
    return {"intent": intent}

def process_internal(state: WorkflowState) -> dict:
    """Process using internal knowledge."""
    return {"messages": [AIMessage(content="Internal knowledge response")]}

def process_web(state: WorkflowState) -> dict:
    """Process using web search."""
    return {"messages": [AIMessage(content="Web search response")]}

def route(state: WorkflowState) -> dict:
    """Simple processing node."""
    return {}


# =============================================================================
# ROUTER (Conditional Edge Logic)
# =============================================================================

def intent_router(state: WorkflowState) -> Literal["process_internal", "process_web"]:
    """Route based on classified intent."""
    intent = state.get("intent", "internal")
    return "process_web" if intent == "web" else "process_internal"


# =============================================================================
# WORKFLOW
# =============================================================================

def create_workflow() -> StateGraph:
    """Creates the workflow graph."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("process_internal", process_internal)
    workflow.add_node("process_web", process_web)

    # Define edges
    workflow.add_edge(START, "classify")
    
    # Conditional edge: router directs to appropriate processing node
    workflow.add_conditional_edges(
        "classify",
        intent_router,
        {
            "process_internal": "process_internal",
            "process_web": "process_web",
        }
    )
    
    # Both paths lead to END
    workflow.add_edge("process_internal", END)
    workflow.add_edge("process_web", END)

    return workflow


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Running workflow...")
    app = create_workflow().compile()
    
    # Test with internal query
    result = app.invoke({
        "messages": [HumanMessage(content="Tell me about AI")],
        "intent": "",
        "confidence": 0.0
    })
    print(f"Result: {result}")
