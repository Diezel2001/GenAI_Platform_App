from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import app.core.prompts as prompts
from app.services.llm import Llmwrapper
from langgraph.prebuilt import ToolNode
import json
import uuid

def parse_json(input_str):
    try:
        raw = input_str.content if hasattr(input_str, "content") else str(input_str)
        raw = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)
        return {"success": True, "data": data, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}

def format_messages(messages):
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "unknown"
        formatted.append(f"{role}: {m.content}")
    return "\n".join(formatted)

def _make_human_message(content: str) -> HumanMessage:
    """Return a HumanMessage with the given content."""
    return HumanMessage(content=content)

# =============================================================================
# STATE SCHEMA
# =============================================================================

class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    analysis: str
    route: str
    steps: str
    step_count: int
    max_steps: int
    is_final: bool


# =============================================================================
# TOOLS
# =============================================================================


@tool
def search_documents(query: str) -> str:
    """Search internal documents."""
    return f"Found: {query} content"

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web results for: {query}"

# Get compiled tools
tools_ = [search_documents, web_search]
tool_node = ToolNode(tools_)

def format_tool(tool):
        args = ", ".join([
            f"{name}: {getattr(param.annotation, "__name__", "any")}"
            for name, param in tool.args_schema.__annotations__.items()
        ])
        return f"- {tool.name}({args}): {tool.description}"

# =============================================================================
# NODES
# =============================================================================

def analyze_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    
    if not messages:
        return state  # nothing to process
    last_msg = messages[-1]
    content = last_msg.content

    # Call LLM
    analysis_raw = Llmwrapper.llm.invoke(
        prompts.message_analyzer_p.format(message=content)
    )

    # Parse JSON safely
    parsed = parse_json(analysis_raw)
    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Parsing failed")],
            "is_final": True
        }

    data = parsed["data"]
    analysis_result = data.get("analysis", content)
    if isinstance(analysis_result, dict) and "message" in analysis_result:
        analysis_text = analysis_result["message"]
    else:
        analysis_text = analysis_result

    return {
        "messages": [AIMessage(content=analysis_text)]
    }
    

def route_node(state: WorkflowState) -> dict:
    """Route based on analysis using task_router_p."""
    messages = state.get("messages", [])
    
    route = Llmwrapper.llm.invoke(prompts.router_p.format(messages=format_messages(messages[-10:])))
    # Parse JSON safely
    parsed = parse_json(route)
    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Parsing failed")],
            "is_final": True
        }

    data = parsed["data"]
    return {"route": data.get("route", "DIRECT")}

def direct_llm_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    messages_f = format_messages(messages[-10:])
    response = Llmwrapper.llm.invoke(messages_f)

    return {
        "messages": [AIMessage(content=response.content if hasattr(response, "content") else str(response))],
        "is_final": True,
    }

def plan_node(state: WorkflowState) -> dict:
    """Plan node for breaking down into tasks."""
    return {
        "messages": [AIMessage(content="Failed to plan")]
    }

def breakdown_node(state: WorkflowState) -> dict:
    """Break down task into steps if needed."""
    return {
        "messages": [AIMessage(content="Failed to Breakdown")]
    }

def agent_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    messages_f = format_messages(messages[-10:])
    tool_desc = "\n".join([format_tool(t) for t in tools_])

    prompt_text = prompts.agent_template_p.format(
        tools=tool_desc,
        messages=messages_f)

    response = Llmwrapper.llm.invoke(prompt_text)
    
    try:
        parsed = parse_json(response)
        if not parsed["success"]:
            return {
                "messages": [AIMessage(content="Failed to parse agent output.")],
                "is_final": True,
            }
        data = parsed["data"]
        
        is_final = data.get("is_final", False)
        if is_final:
            return {
                "messages": [AIMessage(content=str(data.get("final_answer", "No answer found.")))],
                "is_final": True,
            }
        
        tool_name = data.get("action")
        tool_args = data.get("args", {})

        valid_tool_names = {t.name for t in tools_}
        if tool_name not in valid_tool_names:
            return {
                "messages": [
                    AIMessage(content=f"Invalid tool: {tool_name}")
                ],
                "is_final": True,
            }

        tool_call = {
            "name": tool_name,
            "args": tool_args,
            "id": f"call_{uuid.uuid4().hex}",
        }
        
        ai_msg = AIMessage(
            content=data.get("thought", ""),
            tool_calls=[tool_call]
        )
        
        return {
            "messages": [ai_msg],
            "is_final": False,
        }

    except Exception as e:
        # Fallback for parsing errors
        return {
            "messages": [AIMessage(content=f"Error parsing agent output: {str(e)}")],
            "is_final": True
        }

# =============================================================================
# Conditional Edges
# =============================================================================

def should_continue(state: WorkflowState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if not getattr(last_message, "tool_calls", None):
        return "end"
        
    return "tools"

def route_router(state: WorkflowState) -> Literal["direct", "react", "plan"]:
    return state.get("route", "DIRECT").lower()



# =============================================================================
# WORKFLOW
# =============================================================================

def create_agent_workflow() -> StateGraph:
    """Creates the workflow graph."""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("route", route_node)
    workflow.add_node("direct", direct_llm_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)  # Use built-in ToolNode

    # Define edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "route")  # Add edge from analyze to route

    # Conditional edge: router directs to appropriate path
    workflow.add_conditional_edges(
        "route",
        route_router,
        {
            "direct": "direct",
            "react": "agent",
            "plan": "plan",
        }
    )

    # Plan path edges
    workflow.add_edge("plan", "breakdown")
    workflow.add_edge("breakdown", END)

    # edges for the agent loop
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    # Both paths lead to END
    workflow.add_edge("direct", END)

    return workflow


