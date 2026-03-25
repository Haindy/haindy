"""Base implementation for AI agents in the HAINDY framework."""

import logging
from typing import Any

from haindy.config.settings import get_settings
from haindy.core.interfaces import Agent
from haindy.core.types import AgentMessage, ConfidenceLevel
from haindy.models.openai_client import OpenAIClient, ResponseStreamObserver


class BaseAgent(Agent):
    """Base implementation of an AI agent with multi-provider support."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        reasoning_level: str = "medium",
        modalities: set[str] | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Name identifier for the agent
            model: Model override (provider-specific). If None, uses provider default.
            system_prompt: System prompt for the agent
            temperature: Temperature for model responses
            reasoning_level: Reasoning effort level
            modalities: Set of modalities to use
        """
        effective_model = model if model is not None else "gpt-5.4"
        super().__init__(name, effective_model)
        self.logger = logging.getLogger(f"agent.{name}")
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.temperature = temperature
        self.reasoning_level = reasoning_level
        self.modalities = modalities or {"text"}
        self._client: Any | None = None

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        return (
            f"You are {self.name}, an AI agent in the HAINDY autonomous testing system. "
            f"You collaborate with other agents to plan, execute, and evaluate tests. "
            f"Always be precise, factual, and focused on your specific role."
        )

    @property
    def client(self) -> Any:
        """Lazy-load the appropriate LLM client based on agent_provider setting."""
        if self._client is None:
            settings = get_settings()
            provider = str(settings.agent_provider).strip().lower()
            if provider == "anthropic":
                from haindy.models.anthropic_client import AnthropicClient

                self._client = AnthropicClient()
            elif provider == "google":
                from haindy.models.google_client import GoogleClient

                self._client = GoogleClient()
            else:
                self._client = OpenAIClient(
                    model=self.model,
                    reasoning_level=self.reasoning_level,
                    modalities=self.modalities,
                )
        return self._client

    async def process(self, message: AgentMessage) -> AgentMessage | None:
        """
        Process an incoming message.

        Args:
            message: Incoming message to process

        Returns:
            Optional response message
        """
        self.add_to_history(message)
        self.logger.info(
            f"Processing message from {message.from_agent}: {message.message_type}"
        )

        # Default implementation - can be overridden by subclasses
        if not message.requires_response:
            return None

        # Generate response based on message type
        response_content = await self._generate_response(message)

        if response_content:
            response = AgentMessage(
                from_agent=self.name,
                to_agent=message.from_agent,
                message_type=f"{message.message_type}_response",
                content=response_content,
                correlation_id=message.message_id,
            )
            self.add_to_history(response)
            return response

        return None

    async def _generate_response(self, message: AgentMessage) -> dict[str, Any] | None:
        """
        Generate response content for a message.

        Args:
            message: Message to respond to

        Returns:
            Response content dictionary
        """
        # This is a placeholder - subclasses should implement specific logic
        return {"status": "acknowledged", "agent": self.name}

    def calculate_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Convert numeric confidence score to confidence level.

        Args:
            confidence_score: Score between 0.0 and 1.0

        Returns:
            Corresponding confidence level
        """
        if confidence_score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def should_retry(self, confidence_level: ConfidenceLevel) -> bool:
        """
        Determine if an action should be retried based on confidence.

        Args:
            confidence_level: Current confidence level

        Returns:
            True if retry is recommended
        """
        return confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]

    def should_refine(self, confidence_level: ConfidenceLevel) -> bool:
        """
        Determine if coordinates should be refined based on confidence.

        Args:
            confidence_level: Current confidence level

        Returns:
            True if refinement is recommended
        """
        return confidence_level == ConfidenceLevel.MEDIUM

    async def call_model(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_level: str | None = None,
        modalities: set[str] | None = None,
        stream: bool = False,
        stream_observer: ResponseStreamObserver | None = None,
    ) -> dict[str, Any]:
        """
        Make a call to the configured model provider.

        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            response_format: Optional response format specification
            stream: Enable streaming API integration
            stream_observer: Optional observer for streaming events

        Returns:
            API response
        """
        return await self.client.call(
            messages=messages,
            temperature=temperature or self.temperature,
            system_prompt=self.system_prompt,
            response_format=response_format,
            reasoning_level=reasoning_level or self.reasoning_level,
            modalities=modalities or self.modalities,
            stream=stream,
            stream_observer=stream_observer,
        )

    def update_reasoning_level(self, level: str) -> None:
        """Update reasoning level for future calls."""
        self.reasoning_level = level
        if self._client:
            self._client.reasoning_level = level

    def build_messages(
        self,
        user_content: str,
        assistant_content: str | None = None,
        include_history: bool = False,
    ) -> list[dict[str, str]]:
        """
        Build message list for model API calls.

        Args:
            user_content: User message content
            assistant_content: Optional assistant message
            include_history: Whether to include message history

        Returns:
            List of message dictionaries
        """
        messages = []

        if include_history:
            # Add recent history (last 10 messages)
            recent_history = self._message_history[-10:]
            for msg in recent_history:
                role = "assistant" if msg.from_agent == self.name else "user"
                messages.append({"role": role, "content": str(msg.content)})

        messages.append({"role": "user", "content": user_content})

        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        return messages
