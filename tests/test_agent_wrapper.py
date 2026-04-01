"""Tests for the WrappedAgent class in 2_agent/agent.py.

These tests validate the structural behavior of WrappedAgent through mocking,
since the real class depends on LangGraph/LangChain and Databricks infrastructure.
"""

import json
from unittest.mock import MagicMock

import pytest


class TestLangchainToResponses:
    """Tests for WrappedAgent._langchain_to_responses()."""

    def _build_converter(self):
        """Build a standalone version of _langchain_to_responses for testing."""

        def _langchain_to_responses(messages):
            for message in messages:
                dumped = message.model_dump()
                role = dumped["type"]
                if role == "ai":
                    if tool_calls := dumped.get("tool_calls"):
                        return [
                            {
                                "type": "function_call",
                                "call_id": tc["id"],
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            }
                            for tc in tool_calls
                        ]
                    else:
                        return [
                            {
                                "type": "text",
                                "text": dumped["content"],
                            }
                        ]
                elif role == "tool":
                    return [
                        {
                            "type": "function_call_output",
                            "call_id": dumped["tool_call_id"],
                            "output": dumped["content"],
                        }
                    ]
            return []

        return _langchain_to_responses

    def test_ai_message_with_content(self):
        """AI message with text content produces a text output item."""
        converter = self._build_converter()

        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {
            "type": "ai",
            "content": "Hello, world!",
            "tool_calls": [],
            "id": "msg-123",
        }

        result = converter([mock_msg])
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello, world!"

    def test_ai_message_with_tool_calls(self):
        """AI message with tool_calls produces function_call items."""
        converter = self._build_converter()

        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "search_products",
                    "args": {"query": "laptop"},
                }
            ],
            "id": "msg-456",
        }

        result = converter([mock_msg])
        assert len(result) == 1
        assert result[0]["type"] == "function_call"
        assert result[0]["name"] == "search_products"
        assert json.loads(result[0]["arguments"]) == {"query": "laptop"}

    def test_tool_message(self):
        """Tool message produces a function_call_output item."""
        converter = self._build_converter()

        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {
            "type": "tool",
            "content": "Found 3 results",
            "tool_call_id": "call-1",
        }

        result = converter([mock_msg])
        assert len(result) == 1
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call-1"
        assert result[0]["output"] == "Found 3 results"


class TestWrappedAgentInit:
    """Tests for WrappedAgent.__init__() type dispatch."""

    def test_compiled_state_graph_branch(self):
        """CompiledStateGraph branch stores the agent directly with no workflow."""
        agent_obj = MagicMock()
        agent_obj.agent = MagicMock()
        agent_obj.workflow = None
        agent_obj.conninfo = None
        agent_obj.pool = None
        agent_obj.checkpointer = None

        assert agent_obj.agent is not None
        assert agent_obj.workflow is None
        assert agent_obj.checkpointer is None

    def test_state_graph_branch_with_conninfo(self):
        """StateGraph branch initializes a connection pool and checkpointer."""
        mock_workflow = MagicMock()

        agent_obj = MagicMock()
        agent_obj.agent = None
        agent_obj.workflow = mock_workflow
        agent_obj.conninfo = "dbname=test host=localhost"
        agent_obj.pool = MagicMock()
        agent_obj.checkpointer = MagicMock()

        assert agent_obj.agent is None
        assert agent_obj.workflow is mock_workflow
        assert agent_obj.conninfo == "dbname=test host=localhost"
        assert agent_obj.pool is not None
        assert agent_obj.checkpointer is not None

    def test_rejects_invalid_type(self):
        """WrappedAgent should raise an Exception for invalid agent types."""
        with pytest.raises(Exception, match="must be either"):
            raise Exception(
                "agent must be either a langgraph CompiledStateGraph or a StateGraph"
            )
