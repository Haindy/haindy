"""
Base implementation for AI agents in the HAINDY framework.
"""

import logging
from typing import Any, Dict, List, Optional

from src.core.interfaces import Agent
from src.core.types import AgentMessage, ConfidenceLevel
from src.models.openai_client import OpenAIClient


class BaseAgent(Agent):
    """Base implementation of an AI agent with OpenAI integration."""

    def __init__(
        self,
        name: str,
        model: str = "o4-mini",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Name identifier for the agent
            model: OpenAI model to use
            system_prompt: System prompt for the agent
            temperature: Temperature for model responses
        """
        super().__init__(name, model)
        self.logger = logging.getLogger(f"agent.{name}")
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.temperature = temperature
        self._client: Optional[OpenAIClient] = None

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        return (
            f"You are {self.name}, an AI agent in the HAINDY autonomous testing system. "
            f"You collaborate with other agents to plan, execute, and evaluate tests. "
            f"Always be precise, factual, and focused on your specific role."
        )

    @property
    def client(self) -> OpenAIClient:
        """Lazy-load OpenAI client."""
        if self._client is None:
            self._client = OpenAIClient(model=self.model)
        return self._client

    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
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

    async def _generate_response(
        self, message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
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

    async def call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a call to OpenAI API.

        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            response_format: Optional response format specification

        Returns:
            API response
        """
        return await self.client.call(
            messages=messages,
            temperature=temperature or self.temperature,
            system_prompt=self.system_prompt,
            response_format=response_format,
        )

    def build_messages(
        self,
        user_content: str,
        assistant_content: Optional[str] = None,
        include_history: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Build message list for OpenAI API.

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