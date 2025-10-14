"""
Unit tests for base agent implementation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_agent import BaseAgent
from src.core.types import AgentMessage, ConfidenceLevel


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = BaseAgent(
            name="TestAgent",
            model="gpt-4",
            system_prompt="Custom prompt",
            temperature=0.5,
        )
        
        assert agent.name == "TestAgent"
        assert agent.model == "gpt-4"
        assert agent.system_prompt == "Custom prompt"
        assert agent.temperature == 0.5
        assert agent._message_history == []

    def test_default_initialization(self):
        """Test agent initialization with defaults."""
        agent = BaseAgent(name="TestAgent")
        
        assert agent.name == "TestAgent"
        assert agent.model == "gpt-5"
        assert "TestAgent" in agent.system_prompt
        assert "HAINDY" in agent.system_prompt
        assert agent.temperature == 0.7
        assert agent.reasoning_level == "medium"
        assert agent.modalities == {"text"}

    @patch("src.agents.base_agent.OpenAIClient")
    def test_client_lazy_loading(self, mock_client_class):
        """Test lazy loading of OpenAI client."""
        agent = BaseAgent(name="TestAgent")
        
        # Client should not be created yet
        mock_client_class.assert_not_called()
        
        # Access client property
        client = agent.client
        
        # Now client should be created
        mock_client_class.assert_called_once_with(
            model="gpt-5",
            reasoning_level="medium",
            modalities={"text"},
        )
        assert client == mock_client_class.return_value
        
        # Second access should return same instance
        client2 = agent.client
        assert client2 == client
        mock_client_class.assert_called_once()  # Still only called once

    @pytest.mark.asyncio
    async def test_process_message_no_response_required(self):
        """Test processing message that doesn't require response."""
        agent = BaseAgent(name="TestAgent")
        
        message = AgentMessage(
            from_agent="OtherAgent",
            to_agent="TestAgent",
            message_type="info",
            content={"data": "test"},
            requires_response=False,
        )
        
        result = await agent.process(message)
        
        assert result is None
        assert len(agent._message_history) == 1
        assert agent._message_history[0] == message

    @pytest.mark.asyncio
    async def test_process_message_with_response(self):
        """Test processing message that requires response."""
        agent = BaseAgent(name="TestAgent")
        
        message = AgentMessage(
            from_agent="OtherAgent",
            to_agent="TestAgent",
            message_type="query",
            content={"question": "test"},
            requires_response=True,
        )
        
        with patch.object(agent, "_generate_response") as mock_generate:
            mock_generate.return_value = {"answer": "response"}
            
            result = await agent.process(message)
            
            assert result is not None
            assert result.from_agent == "TestAgent"
            assert result.to_agent == "OtherAgent"
            assert result.message_type == "query_response"
            assert result.content == {"answer": "response"}
            assert result.correlation_id == message.message_id
            
            # Both messages should be in history
            assert len(agent._message_history) == 2

    def test_calculate_confidence_level(self):
        """Test confidence level calculation."""
        agent = BaseAgent(name="TestAgent")
        
        assert agent.calculate_confidence_level(0.98) == ConfidenceLevel.VERY_HIGH
        assert agent.calculate_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert agent.calculate_confidence_level(0.90) == ConfidenceLevel.HIGH
        assert agent.calculate_confidence_level(0.80) == ConfidenceLevel.HIGH
        assert agent.calculate_confidence_level(0.70) == ConfidenceLevel.MEDIUM
        assert agent.calculate_confidence_level(0.60) == ConfidenceLevel.MEDIUM
        assert agent.calculate_confidence_level(0.50) == ConfidenceLevel.LOW
        assert agent.calculate_confidence_level(0.40) == ConfidenceLevel.LOW
        assert agent.calculate_confidence_level(0.30) == ConfidenceLevel.VERY_LOW
        assert agent.calculate_confidence_level(0.10) == ConfidenceLevel.VERY_LOW

    def test_should_retry(self):
        """Test retry decision based on confidence."""
        agent = BaseAgent(name="TestAgent")
        
        assert agent.should_retry(ConfidenceLevel.VERY_HIGH) is False
        assert agent.should_retry(ConfidenceLevel.HIGH) is False
        assert agent.should_retry(ConfidenceLevel.MEDIUM) is False
        assert agent.should_retry(ConfidenceLevel.LOW) is True
        assert agent.should_retry(ConfidenceLevel.VERY_LOW) is True

    def test_should_refine(self):
        """Test refinement decision based on confidence."""
        agent = BaseAgent(name="TestAgent")
        
        assert agent.should_refine(ConfidenceLevel.VERY_HIGH) is False
        assert agent.should_refine(ConfidenceLevel.HIGH) is False
        assert agent.should_refine(ConfidenceLevel.MEDIUM) is True
        assert agent.should_refine(ConfidenceLevel.LOW) is False
        assert agent.should_refine(ConfidenceLevel.VERY_LOW) is False

    @pytest.mark.asyncio
    async def test_call_openai(self):
        """Test OpenAI API call."""
        agent = BaseAgent(name="TestAgent", temperature=0.8)
        
        mock_client = AsyncMock()
        mock_client.call.return_value = {"result": "test"}
        
        # Mock the _client attribute directly
        agent._client = mock_client
        
        messages = [{"role": "user", "content": "Hello"}]
        result = await agent.call_openai(messages)
        
        assert result == {"result": "test"}
        mock_client.call.assert_called_once_with(
            messages=messages,
            temperature=0.8,
            system_prompt=agent.system_prompt,
            response_format=None,
            reasoning_level="medium",
            modalities={"text"},
            stream=False,
            stream_observer=None,
        )

    @pytest.mark.asyncio
    async def test_call_openai_with_overrides(self):
        """Test OpenAI API call with parameter overrides."""
        agent = BaseAgent(name="TestAgent", temperature=0.8)
        
        mock_client = AsyncMock()
        mock_client.call.return_value = {"result": "test"}
        
        # Mock the _client attribute directly
        agent._client = mock_client
        
        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_object"}
        
        result = await agent.call_openai(
            messages,
            temperature=0.3,
            response_format=response_format,
        )
        
        assert result == {"result": "test"}
        mock_client.call.assert_called_once_with(
            messages=messages,
            temperature=0.3,  # Override temperature
            system_prompt=agent.system_prompt,
            response_format=response_format,
            reasoning_level="medium",
            modalities={"text"},
            stream=False,
            stream_observer=None,
        )

    def test_build_messages_simple(self):
        """Test building simple message list."""
        agent = BaseAgent(name="TestAgent")
        
        messages = agent.build_messages("Hello", assistant_content="Hi there")
        
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}

    def test_build_messages_with_history(self):
        """Test building messages with history."""
        agent = BaseAgent(name="TestAgent")
        
        # Add some history
        agent.add_to_history(AgentMessage(
            from_agent="OtherAgent",
            to_agent="TestAgent",
            message_type="query",
            content={"text": "Previous question"},
        ))
        agent.add_to_history(AgentMessage(
            from_agent="TestAgent",
            to_agent="OtherAgent",
            message_type="response",
            content={"text": "Previous answer"},
        ))
        
        messages = agent.build_messages("New question", include_history=True)
        
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "Previous question" in str(messages[0]["content"])
        assert messages[1]["role"] == "assistant"
        assert "Previous answer" in str(messages[1]["content"])
        assert messages[2] == {"role": "user", "content": "New question"}

    def test_message_history_management(self):
        """Test message history management."""
        agent = BaseAgent(name="TestAgent")
        
        # Initially empty
        assert agent.get_history() == []
        
        # Add a message
        message = AgentMessage(
            from_agent="Sender",
            to_agent="TestAgent",
            message_type="test",
            content={},
        )
        agent.add_to_history(message)
        
        # Check history
        history = agent.get_history()
        assert len(history) == 1
        assert history[0] == message
        
        # Ensure it's a copy
        history.append("dummy")
        assert len(agent.get_history()) == 1  # Original unchanged
