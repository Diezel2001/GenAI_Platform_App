"""Integration tests for the LangGraph workflow (backend/app/core/workflows.py)."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.core.workflows import create_workflow
from app.services.llm.Llmwrapper import llm


def _make_human_message(content: str) -> HumanMessage:
    """Return a HumanMessage with the given content."""
    return HumanMessage(content=content)


def _make_ai_message(content: str, tool_calls: list | None = None) -> AIMessage:
    """Return an AIMessage, optionally with tool_calls."""
    if tool_calls:
        return AIMessage(content=content, tool_calls=tool_calls)
    return AIMessage(content=content)


class TestWorkflowIntegration:
    """Integration tests that run the full compiled graph with a real LLM."""

    @pytest.fixture
    def compiled_workflow(self):
        """Return a compiled LangGraph app."""
        return create_workflow().compile()

    def test_simple_direct_question(self, compiled_workflow):
        """
        Test a simple direct question that should be answered immediately.
        Expected path: analyze → route (DIRECT) → direct_llm → END
        """
        # Test with a simple question
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("What is the capital of France?")],
            "intent": "internal",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "Paris" in last_msg.content

    def test_complex_task_with_planning(self, compiled_workflow):
        """
        Test a complex task that requires planning and breakdown.
        Expected path: analyze → route (PLAN) → plan → breakdown → agent → tools → agent → ... → END
        """
        # Test with a complex task
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("Write a Python script to analyze customer feedback and generate insights.")],
            "intent": "internal",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "Python script" in last_msg.content or "customer feedback" in last_msg.content

    def test_web_search_intent(self, compiled_workflow):
        """
        Test a query that requires web search.
        Expected path: analyze → route (DIRECT/PLAN) → direct_llm/agent → tools (web_search) → END
        """
        # Test with a web search intent
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("What's the latest news about AI developments?")],
            "intent": "web",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "AI developments" in last_msg.content or "latest news" in last_msg.content

    def test_tool_calling_with_search(self, compiled_workflow):
        """
        Test a query that requires internal document search.
        Expected path: analyze → route (DIRECT/PLAN) → direct_llm/agent → tools (search_documents) → END
        """
        # Test with a document search query
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("Find information about our company's privacy policy.")],
            "intent": "internal",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "privacy policy" in last_msg.content or "company information" in last_msg.content

    def test_empty_message_handling(self, compiled_workflow):
        """
        Test how the workflow handles empty messages.
        """
        # Test with empty message
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("")],
            "intent": "internal",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "I'm sorry" in last_msg.content or "could you please" in last_msg.content

    def test_multiple_messages_history(self, compiled_workflow):
        """
        Test how the workflow handles message history.
        """
        # Test with message history
        result = compiled_workflow.invoke({
            "messages": [
                _make_human_message("What's the weather like today?"),
                _make_human_message("And what about tomorrow?")
            ],
            "intent": "internal",
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "tomorrow" in last_msg.content or "weather" in last_msg.content

    def test_workflow_compiles_without_error(self, compiled_workflow):
        """Test that the workflow compiles without errors."""
        assert compiled_workflow is not None

    def test_workflow_accepts_initial_state(self, compiled_workflow):
        """Test that the workflow accepts a minimal valid initial state."""
        result = compiled_workflow.invoke({
            "messages": [_make_human_message("Hello")],
        })

        assert result is not None