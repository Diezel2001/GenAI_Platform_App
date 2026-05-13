from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal

from pydantic import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage


logger = logging.getLogger(__name__)


# =========================================================
# SCHEMAS
# =========================================================

class ObservationSchema(BaseModel):
    user_goal: str
    known_facts: List[Any] = Field(default_factory=list)
    missing_information: List[Any] = Field(default_factory=list)
    constraints: List[Any] = Field(default_factory=list)


class ReasoningSchema(BaseModel):
    thought: str
    action: Literal["TOOL", "FINAL"]
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


class FinalResponseSchema(BaseModel):
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources_used: List[str] = Field(default_factory=list)


# =========================================================
# STATE
# =========================================================

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

    observation: Optional[dict]
    reasoning: Optional[dict]
    final_response: Optional[dict]

    tool_result: Optional[Any]

    iteration_count: int
    max_iterations: int

    metadata: dict


# =========================================================
# AGENT
# =========================================================

class ReActAgent:

    def __init__(
        self,
        llm: Any,
        *,
        checkpoint: Optional[BaseCheckpointSaver] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        enable_logging: bool = True,
        allowed_tools: Optional[List[str]] = None,
    ):

        self.llm = llm
        self.checkpoint = checkpoint
        self.max_iterations = max_iterations
        self.allowed_tools = set(allowed_tools or [])

        self.tools: Dict[str, Any] = {}

        self.system_prompt = system_prompt or self._default_system_prompt()

        if enable_logging:
            logging.basicConfig(level=logging.INFO)

        self.graph = self._build_graph()

    # =====================================================
    # TOOLS
    # =====================================================

    def add_tool(self, tool: Any) -> None:
        if self.allowed_tools and tool.name not in self.allowed_tools:
            raise ValueError(f"Tool not allowed: {tool.name}")
        self.tools[tool.name] = tool

    def add_tools(self, tools: List[Any]) -> None:
        for t in tools:
            self.add_tool(t)

    # =====================================================
    # INVOKE
    # =====================================================

    def invoke(
        self,
        user_input: str,
        *,
        thread_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):

        config = {"configurable": {"thread_id": thread_id}}

        state: GraphState = {
            "messages": [HumanMessage(content=user_input)],
            "metadata": metadata or {},
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
        }

        result = self.graph.invoke(state, config=config)

        return FinalResponseSchema.model_validate(result["final_response"])

    # =====================================================
    # GRAPH
    # =====================================================

    def _build_graph(self):

        g = StateGraph(GraphState)

        g.add_node("observe", self._observe)
        g.add_node("reason", self._reason)
        g.add_node("act", self._act)
        g.add_node("finalize", self._finalize)

        g.set_entry_point("observe")

        g.add_edge("observe", "reason")

        g.add_conditional_edges(
            "reason",
            self._router,
            {"TOOL": "act", "FINAL": "finalize"},
        )

        g.add_edge("act", "observe")
        g.add_edge("finalize", END)

        return g.compile(checkpointer=self.checkpoint)

    # =====================================================
    # ROUTER
    # =====================================================

    def _router(self, state: GraphState) -> str:
        r = ReasoningSchema.model_validate(state["reasoning"])
        return r.action

    # =====================================================
    # OBSERVE
    # =====================================================

    def _observe(self, state: GraphState) -> Dict[str, Any]:

        prompt = self._observe_prompt(state)

        raw = self.llm.invoke(prompt)
        raw = raw.content if hasattr(raw, "content") else raw

        data = self._parse_json_safe(raw)

        obs = ObservationSchema.model_validate(data)

        # ✅ NORMALIZATION FIX (prevents dict-in-list crash)
        obs.known_facts = self._normalize_list(obs.known_facts)
        obs.missing_information = self._normalize_list(obs.missing_information)
        obs.constraints = self._normalize_list(obs.constraints)

        logger.info("observe completed")

        return {"observation": obs.model_dump()}

    # =====================================================
    # REASON
    # =====================================================

    def _reason(self, state: GraphState) -> Dict[str, Any]:

        prompt = self._reason_prompt(state)

        for attempt in range(2):

            raw = self.llm.invoke(prompt)
            raw = raw.content if hasattr(raw, "content") else raw

            try:
                data = self._parse_json_safe(raw)
                r = ReasoningSchema.model_validate(data)

                logger.info("reason completed | action=%s", r.action)
                return {"reasoning": r.model_dump()}

            except Exception as e:
                logger.warning("reason retry %s failed: %s", attempt + 1, str(e))
                prompt += "\n\nReturn STRICT VALID JSON only."

        fallback = ReasoningSchema(
            thought="fallback",
            action="FINAL",
            final_answer="Failed to generate valid reasoning.",
        )

        return {"reasoning": fallback.model_dump()}

    # =====================================================
    # ACT
    # =====================================================

    def _act(self, state: GraphState) -> Dict[str, Any]:

        r = ReasoningSchema.model_validate(state["reasoning"])

        tool = self.tools.get(r.tool_name or "")
        if not tool:
            raise ValueError(f"Unknown tool: {r.tool_name}")

        logger.info("executing tool=%s", tool.name)

        result = tool.invoke(r.tool_input or {})

        return {
            "tool_result": result,
            "messages": [ToolMessage(content=str(result), tool_call_id=tool.name)],
            "iteration_count": state["iteration_count"] + 1,
        }

    # =====================================================
    # FINALIZE
    # =====================================================

    def _finalize(self, state: GraphState) -> Dict[str, Any]:

        prompt = self._finalize_prompt(state)

        raw = self.llm.invoke(prompt)
        raw = raw.content if hasattr(raw, "content") else raw

        data = self._parse_json_safe(raw)
        final = FinalResponseSchema.model_validate(data)

        logger.info("finalize completed")

        return {"final_response": final.model_dump()}

    # =====================================================
    # PROMPTS
    # =====================================================

    def _observe_prompt(self, state: GraphState) -> str:
        return f"""
Return ONLY valid JSON.

USER:
{self._latest_user_message(state)}

TOOLS:
{self._tool_descriptions()}

FORMAT:
{{
  "user_goal": "",
  "known_facts": [],
  "missing_information": [],
  "constraints": []
}}
""".strip()

    def _reason_prompt(self, state: GraphState) -> str:
        return f"""
Return ONLY valid JSON.

OBSERVATION:
{state.get("observation", {})}

TOOLS:
{self._tool_descriptions()}

FORMAT:
{{
  "thought": "",
  "action": "TOOL | FINAL",
  "tool_name": null,
  "tool_input": null,
  "final_answer": null
}}
""".strip()

    def _finalize_prompt(self, state: GraphState) -> str:
        return f"""
You are producing FINAL ANSWER.

Use ONLY:
- conversation
- reasoning
- tool_result

TOOL_RESULT:
{state.get("tool_result", {})}

REASONING:
{state.get("reasoning", {})}

CONVERSATION:
{[m.content for m in state.get("messages", []) if hasattr(m, "content")]}

OUTPUT JSON ONLY:
{{
  "response": "",
  "confidence": 0.0,
  "sources_used": []
}}
""".strip()

    # =====================================================
    # SAFE JSON PARSER
    # =====================================================

    def _parse_json_safe(self, text: str) -> Dict[str, Any]:

        if not text or not str(text).strip():
            raise ValueError("Empty LLM output")

        text = str(text).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found: {text[:200]}")

        return json.loads(match.group(0))

    # =====================================================
    # NORMALIZER
    # =====================================================

    def _normalize_list(self, items: List[Any]) -> List[str]:
        out = []
        for x in items:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(next(iter(x.values()), str(x)))
            else:
                out.append(str(x))
        return out

    # =====================================================
    # HELPERS
    # =====================================================

    def _latest_user_message(self, state: GraphState) -> str:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                return m.content
        return ""

    def _tool_descriptions(self) -> str:
        if not self.tools:
            return "None"
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def _default_system_prompt(self) -> str:
        return "You are a deterministic ReAct agent. Output JSON only."