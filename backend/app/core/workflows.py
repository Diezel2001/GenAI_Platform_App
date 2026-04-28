from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import app.core.prompts as prompts
from app.services.llm import Llmwrapper
from langgraph.prebuilt import ToolNode
import json, uuid, re

# =============================================================================
# HELPERS
# =============================================================================

def parse_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return {"success": True, "data": json.loads(match.group())}
    except Exception as e:
        return {"success": False, "error": str(e)}

def format_messages(messages):
    formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            formatted.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            formatted.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, ToolMessage):
            formatted.append(f"TOOL RESULT:\n{m.content}")
        else:
            formatted.append(f"UNKNOWN: {m.content}")
    return "\n".join(formatted)


def _make_human_message(content: str) -> HumanMessage:
    return HumanMessage(content=content)

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)

# =============================================================================
# STATE
# =============================================================================

class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    analysis: str
    route: str
    steps: str
    step_count: int
    max_steps: int
    observations: list[str]
    tool_used: str
    is_final: bool

# =============================================================================
# TOOLS
# =============================================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    if b == 0:
        return "Error: division by zero"
    return a / b

@tool
def search_knowledge(query: str) -> str:
    """Search internal knowledge base."""
    kb = {
        "python": "Python is a programming language.",
        "langgraph": "LangGraph is used for building LLM workflows.",
        "dspy": "DSPy is a framework for programming LLMs."
    }
    return kb.get(query.lower(), "No info found.")

@tool
def score_exam(correct: int, total: int) -> str:
    """Compute exam percentage score."""
    if total == 0:
        return "invalid"
    percent = (correct / total) * 100
    return f"{percent:.2f}%"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (mock tool)."""
    weather_db = {
        "manila": "hot and humid, 33°C",
        "tokyo": "cool, 18°C and cloudy",
        "london": "rainy, 12°C"
    }
    return weather_db.get(city.lower(), "weather data not available")

tools_ = [add, multiply, divide, search_knowledge, score_exam, get_weather]
# tool_node = ToolNode(tools_)


def format_tool(tool):
    args = []
    for name, param in tool.args_schema.__annotations__.items():
        args.append(f"{name}: {getattr(param, '__name__', 'any')}")

    return f"""
Tool: {tool.name}
Description: {tool.description}
Arguments: {", ".join(args)}
"""

# =============================================================================
# NODES
# =============================================================================

def analyze_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return state

    last_msg = messages[-1]

    analysis_raw = Llmwrapper.llm.invoke(
        prompts.message_analyzer_p.format(message=last_msg.content, convo=format_messages(messages[-5:]))
    )

    parsed = parse_json(analysis_raw)
    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Parsing failed in analyze node")],
            "is_final": True
        }

    data = parsed["data"]
    analysis_text = data.get("analysis", last_msg.content)

    if isinstance(analysis_text, dict):
        analysis_text = analysis_text.get("message", str(analysis_text))
    
    
    return {"analysis": analysis_text}


def route_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    analysis = state.get("analysis", "")

    route = Llmwrapper.llm.invoke(
        prompts.router_p.format(messages=format_messages(messages[-10:]), analysis=analysis, user_message=format_messages([messages[-1]]))
    )

    parsed = parse_json(route)
    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Routing parse failed")],
            "is_final": True
        }

    return {"route": parsed["data"].get("route", "DIRECT")}


def direct_llm_node(state: WorkflowState) -> dict:
    messages = state.get("messages", [])
    response = Llmwrapper.llm.invoke(format_messages(messages[-10:]))

    return {
        "messages": [AIMessage(content=str(response))],
        "is_final": True,
    }


def plan_node(state: WorkflowState) -> dict:
    return {"messages": [AIMessage(content="Planning not implemented")]}


def breakdown_node(state: WorkflowState) -> dict:
    return {"messages": [AIMessage(content="Breakdown not implemented")]}


def build_agent_context(messages):
    last_user_idx = None

    for i in reversed(range(len(messages))):
        if isinstance(messages[i], HumanMessage):
            last_user_idx = i
            break

    if last_user_idx is None:
        return None, []

    user_msg = messages[last_user_idx].content

    tool_msgs = [
        m for m in messages[last_user_idx+1:]
        if isinstance(m, ToolMessage)
    ]

    return user_msg, tool_msgs

def format_agent_context(user_msg, tool_msgs):
    context = f"USER QUESTION:\n{user_msg}\n\n"

    if tool_msgs:
        context += "Observation:\n"
        for i, t in enumerate(tool_msgs, 1):
            context += f"{i}. {t.content}\n"

    return context

def reason_node(state: WorkflowState):
    messages = state["messages"]
    observations = state.get("observations")
    analysis = state.get("analysis", "")

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 5)

    if step_count >= max_steps:
        return {
            "messages": [AIMessage(content="Max steps reached.")]
        }

    user_msg, _ = build_agent_context(messages)
    tool_desc = "\n".join([format_tool(t) for t in tools_])
    obs_list = state.get("observations", [])
    obs_text = "\n".join(obs_list) if obs_list else "None"

    prompt = prompts.reason_prompt.format(
        tools=tool_desc,
        user_message=user_msg,
        observations=obs_text,
        analysis=analysis
    )

    response = Llmwrapper.llm.invoke(prompt)
    parsed = parse_json(response)

    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Parse error")]
        }

    data = parsed["data"]

    if data.get("type") == "final":
        return {
            "messages": [
                AIMessage(content=data.get("final_answer", "No answer"))
            ]
        }

    if data.get("type") != "tool":
        return {
            "messages": [AIMessage(content="Invalid response format")]
        }

    tool_name = data.get("action")
    tool_args = data.get("args", {})

    tool_map = {t.name: t for t in tools_}

    if tool_name not in tool_map:
        return {
            "messages": [AIMessage(content=f"Invalid tool: {tool_name}")]
        }

    tool = tool_map[tool_name]

    expected_args = tool.args_schema.__annotations__.keys()

    missing = [arg for arg in expected_args if arg not in tool_args]
    if missing:
        return {
            "messages": [
                AIMessage(content=f"Missing args for {tool_name}: {missing}")
            ]
        }

    tool_call = {
        "name": tool_name,
        "args": tool_args,
        "id": f"call_{uuid.uuid4().hex}",
    }
    print(tool_name)

    return {
        "messages": [
            AIMessage(
                content=f"THOUGHT: {data.get('thought','')}",
                tool_calls=[tool_call]
            )
        ],
        "step_count": step_count + 1
    }

def act_node(state: WorkflowState):
    messages = state.get("messages", [])
    if not messages:
        return {}
    last = messages[-1]

    tool_calls = getattr(last, "tool_calls", [])
    tool_map = {t.name: t for t in tools_}

    results = []
    raw_results = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name not in tool_map:
            results.append(
                ToolMessage(content=f"Invalid tool: {tool_name}", tool_call_id=call["id"])
            )
            continue

        result = tool_map[tool_name].invoke(tool_args)
        raw_results.append(f"{tool_name}({tool_args}) -> {result}")

        results.append(
            ToolMessage(
                content=f"{tool_name}({tool_args}) -> {result}",
                tool_call_id=call["id"]
            )
        )

    return {"messages": results,
            "raw_tool_results": raw_results
    }

def observe_node(state: WorkflowState):
    messages = state["messages"]
    analysis = state.get("analysis", "")
    user_msg, tool_msgs = build_agent_context(messages)

    prompt = prompts.observe_prompt.format(
        user_message=user_msg,
        result=tool_msgs,
        analysis=analysis
    )

    response = Llmwrapper.llm.invoke(prompt)

    parsed = parse_json(response)

    if not parsed["success"]:
        return {
            "messages": [AIMessage(content="Parse error")],
            "is_final": True
        }

    data = parsed["data"]
    prev_obs = state.get("observations", [])

    obs = data.get("obs", "No useful observation")

    return {
        "observations": prev_obs + [obs]
    }

# =============================================================================
# ROUTING
# =============================================================================

def should_continue(state: WorkflowState) -> Literal["act", "end"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "act"
    return "end"


def route_router(state: WorkflowState) -> Literal["direct", "react", "plan"]:
    return state.get("route", "DIRECT").lower()

# =============================================================================
# WORKFLOW
# =============================================================================

def create_agent_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("route", route_node)
    workflow.add_node("direct", direct_llm_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("breakdown", breakdown_node)

    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("observe", observe_node)

    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "route")

    workflow.add_conditional_edges(
        "route",
        route_router,
        {
            "direct": "direct",
            "react": "reason",
            "plan": "plan",
        }
    )

    workflow.add_edge("plan", "breakdown")
    workflow.add_edge("breakdown", END)

    workflow.add_conditional_edges(
        "reason",
        should_continue,
        {
            "act": "act",
            "end": END
        }
    )

    workflow.add_edge("act", "observe")
    workflow.add_edge("observe", "reason")
    workflow.add_edge("direct", END)

    return workflow