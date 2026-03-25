"""Unit tests for base agent implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haindy.agents.base_agent import BaseAgent
from haindy.core.types import AgentMessage, ConfidenceLevel
from haindy.models.anthropic_client import AnthropicClient
from haindy.models.google_client import GoogleClient
from haindy.models.openai_client import OpenAIClient


def _make_settings(
    provider: str = "openai",
    openai_model: str = "gpt-5.4",
    openai_codex_model: str = "gpt-5.4",
    anthropic_model: str = "claude-sonnet-4-6",
    google_model: str = "gemini-3-flash-preview",
) -> MagicMock:
    s = MagicMock()
    s.agent_provider = provider
    s.openai_model = openai_model
    s.openai_codex_model = openai_codex_model
    s.anthropic_model = anthropic_model
    s.google_model = google_model
    provider_models = {
        "openai": openai_model,
        "openai-codex": openai_codex_model,
        "anthropic": anthropic_model,
        "google": google_model,
    }
    s.get_provider_model.side_effect = lambda provider_name, computer_use=False: (
        provider_models[provider_name]
    )
    return s


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
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("openai"),
        ):
            agent = BaseAgent(name="TestAgent")

        assert agent.name == "TestAgent"
        assert agent.model == "gpt-5.4"
        assert "TestAgent" in agent.system_prompt
        assert "HAINDY" in agent.system_prompt
        assert agent.temperature == 0.7
        assert agent.reasoning_level == "medium"
        assert agent.modalities == {"text"}

    # -----------------------------------------------------------------------
    # Provider dispatch
    # -----------------------------------------------------------------------

    def test_client_dispatches_openai_by_default(self):
        """Default provider should produce an OpenAIClient."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("openai"),
        ):
            agent = BaseAgent(name="TestAgent")
            assert isinstance(agent.client, OpenAIClient)

    def test_client_dispatches_anthropic(self):
        """agent_provider='anthropic' should produce an AnthropicClient."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("anthropic"),
        ):
            agent = BaseAgent(name="TestAgent")
            assert isinstance(agent.client, AnthropicClient)

    def test_client_dispatches_google(self):
        """agent_provider='google' should produce a GoogleClient."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("google"),
        ):
            agent = BaseAgent(name="TestAgent")
            assert isinstance(agent.client, GoogleClient)

    def test_client_dispatches_openai_codex(self):
        """agent_provider='openai-codex' should still produce an OpenAIClient."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("openai-codex"),
        ):
            agent = BaseAgent(name="TestAgent")
            assert isinstance(agent.client, OpenAIClient)

    def test_client_uses_model_override(self):
        """Explicit model passed to BaseAgent should override settings default."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("anthropic"),
        ):
            agent = BaseAgent(name="TestAgent", model="claude-opus-4-5")
            client = agent.client
            assert isinstance(client, AnthropicClient)
            assert client.model == "claude-opus-4-5"

    def test_client_uses_settings_model_when_none(self):
        """When model=None, the provider's settings model is used."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings(
                "anthropic", anthropic_model="claude-haiku-3-5"
            ),
        ):
            agent = BaseAgent(name="TestAgent")
            client = agent.client
            assert isinstance(client, AnthropicClient)
            assert client.model == "claude-haiku-3-5"

    def test_client_is_cached(self):
        """The same client instance should be returned on repeated accesses."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("openai"),
        ):
            agent = BaseAgent(name="TestAgent")
            c1 = agent.client
            c2 = agent.client
            assert c1 is c2

    def test_client_is_recreated_when_provider_changes(self):
        """Changing provider setting should produce a new client."""
        openai_settings = _make_settings("openai")
        anthropic_settings = _make_settings("anthropic")

        with patch(
            "haindy.agents.base_agent.get_settings", return_value=openai_settings
        ):
            agent = BaseAgent(name="TestAgent")
            first = agent.client
            assert isinstance(first, OpenAIClient)

        with patch(
            "haindy.agents.base_agent.get_settings", return_value=anthropic_settings
        ):
            second = agent.client
            assert isinstance(second, AnthropicClient)
            assert second is not first

    # -----------------------------------------------------------------------
    # call_model
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_call_model(self):
        """call_model should delegate to the client with correct arguments."""
        settings = _make_settings("openai")
        with patch("haindy.agents.base_agent.get_settings", return_value=settings):
            agent = BaseAgent(name="TestAgent", temperature=0.8)
        mock_client = AsyncMock()
        mock_client.call.return_value = {"result": "test"}
        agent._client = mock_client
        agent._client_provider = "openai"

        messages = [{"role": "user", "content": "Hello"}]
        with patch("haindy.agents.base_agent.get_settings", return_value=settings):
            result = await agent.call_model(messages)

        assert result == {"result": "test"}
        mock_client.call.assert_called_once_with(
            messages=messages,
            temperature=0.8,
            max_tokens=None,
            system_prompt=agent.system_prompt,
            response_format=None,
            reasoning_level="medium",
            modalities={"text"},
            stream=False,
            stream_observer=None,
        )

    @pytest.mark.asyncio
    async def test_call_model_with_overrides(self):
        """call_model should honour per-call temperature and format overrides."""
        settings = _make_settings("openai")
        with patch("haindy.agents.base_agent.get_settings", return_value=settings):
            agent = BaseAgent(name="TestAgent", temperature=0.8)
        mock_client = AsyncMock()
        mock_client.call.return_value = {"result": "test"}
        agent._client = mock_client
        agent._client_provider = "openai"

        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_object"}

        with patch("haindy.agents.base_agent.get_settings", return_value=settings):
            result = await agent.call_model(
                messages,
                temperature=0.3,
                response_format=response_format,
            )

        assert result == {"result": "test"}
        mock_client.call.assert_called_once_with(
            messages=messages,
            temperature=0.3,
            max_tokens=None,
            system_prompt=agent.system_prompt,
            response_format=response_format,
            reasoning_level="medium",
            modalities={"text"},
            stream=False,
            stream_observer=None,
        )

    # -----------------------------------------------------------------------
    # process / message handling
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_message_no_response_required(self):
        """Test processing message that does not require a response."""
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
        """Test processing message that requires a response."""
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

            assert len(agent._message_history) == 2

    # -----------------------------------------------------------------------
    # Confidence helpers
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # build_messages
    # -----------------------------------------------------------------------

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

        agent.add_to_history(
            AgentMessage(
                from_agent="OtherAgent",
                to_agent="TestAgent",
                message_type="query",
                content={"text": "Previous question"},
            )
        )
        agent.add_to_history(
            AgentMessage(
                from_agent="TestAgent",
                to_agent="OtherAgent",
                message_type="response",
                content={"text": "Previous answer"},
            )
        )

        messages = agent.build_messages("New question", include_history=True)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "Previous question" in str(messages[0]["content"])
        assert messages[1]["role"] == "assistant"
        assert "Previous answer" in str(messages[1]["content"])
        assert messages[2] == {"role": "user", "content": "New question"}

    # -----------------------------------------------------------------------
    # update_reasoning_level
    # -----------------------------------------------------------------------

    def test_update_reasoning_level_invalidates_client_cache(self):
        """Updating reasoning level should clear the cached client."""
        with patch(
            "haindy.agents.base_agent.get_settings",
            return_value=_make_settings("openai"),
        ):
            agent = BaseAgent(name="TestAgent")
            _ = agent.client  # populate cache

        assert agent._client is not None
        agent.update_reasoning_level("high")
        assert agent._client is None
        assert agent._client_provider is None
        assert agent.reasoning_level == "high"

    # -----------------------------------------------------------------------
    # message history
    # -----------------------------------------------------------------------

    def test_message_history_management(self):
        """Test message history management."""
        agent = BaseAgent(name="TestAgent")

        assert agent.get_history() == []

        message = AgentMessage(
            from_agent="Sender",
            to_agent="TestAgent",
            message_type="test",
            content={},
        )
        agent.add_to_history(message)

        history = agent.get_history()
        assert len(history) == 1
        assert history[0] == message

        history.append("dummy")
        assert len(agent.get_history()) == 1
