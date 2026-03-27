"""Base implementation for AI agents in the HAINDY framework."""

import base64
import logging
from pathlib import Path
from typing import Any, cast

from haindy.config.settings import get_settings
from haindy.core.interfaces import Agent
from haindy.core.types import AgentMessage, ConfidenceLevel
from haindy.models.errors import ModelCallError
from haindy.models.openai_client import OpenAIClient, ResponseStreamObserver
from haindy.utils.model_logging import get_model_logger, log_model_call_failure


class _CapturedStreamObserver(ResponseStreamObserver):
    """Wrap a stream observer while capturing partial output for failure logs."""

    def __init__(self, delegate: ResponseStreamObserver | None) -> None:
        self._delegate = delegate
        self.partial_text = ""
        self.usage_totals: dict[str, int] = {}
        self.started = False
        self.ended = False
        self.last_error: Any | None = None

    def _dispatch(self, method_name: str, *args: Any) -> None:
        if self._delegate is None:
            return
        method = getattr(self._delegate, method_name, None)
        if method is None:
            return
        try:
            method(*args)
        except Exception:
            pass

    def on_stream_start(self) -> Any:
        self.started = True
        self._dispatch("on_stream_start")

    def on_text_delta(self, delta: str) -> Any:
        self.partial_text += delta
        self._dispatch("on_text_delta", delta)

    def on_usage_delta(self, delta: dict[str, int]) -> Any:
        for key, value in delta.items():
            try:
                self.usage_totals[key] = int(value)
            except Exception:
                continue
        self._dispatch("on_usage_delta", delta)

    def on_usage_total(self, totals: dict[str, int]) -> Any:
        normalized: dict[str, int] = {}
        for key, value in totals.items():
            try:
                normalized[key] = int(value)
            except Exception:
                continue
        self.usage_totals.update(normalized)
        self._dispatch("on_usage_total", totals)

    def on_token_progress(
        self, total_tokens: int, delta_tokens: int, delta_chars: int
    ) -> Any:
        self._dispatch("on_token_progress", total_tokens, delta_tokens, delta_chars)

    def on_error(self, error: Any) -> Any:
        self.last_error = error
        self._dispatch("on_error", error)

    def on_stream_end(self) -> Any:
        self.ended = True
        self._dispatch("on_stream_end")

    def snapshot_response(self) -> dict[str, Any] | None:
        if (
            not self.started
            and not self.partial_text
            and not self.usage_totals
            and self.last_error is None
        ):
            return None
        payload: dict[str, Any] = {
            "stream_started": self.started,
            "stream_ended": self.ended,
        }
        if self.partial_text:
            payload["content_text"] = self.partial_text
        if self.usage_totals:
            payload["usage"] = dict(self.usage_totals)
        if self.last_error is not None:
            payload["stream_error"] = str(self.last_error)
        return payload


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
        if model is not None:
            effective_model = model
        else:
            settings = get_settings()
            effective_model = settings.get_provider_model(settings.agent_provider)
        super().__init__(name, effective_model)
        self.logger = logging.getLogger(f"agent.{name}")
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.temperature = temperature
        self.reasoning_level = reasoning_level
        self.modalities = modalities or {"text"}
        self._client: Any | None = None
        self._client_provider: str | None = None
        self._model_override = model is not None
        self._model_logger: Any | None = None

    def _resolve_model_logger(self) -> Any | None:
        candidate = getattr(self, "_model_logger", None)
        if candidate is not None:
            return candidate

        settings = get_settings()
        log_path = getattr(settings, "model_log_path", None)
        if not isinstance(log_path, (Path, str)):
            return None

        max_screenshots = getattr(settings, "max_screenshots", None)
        self._model_logger = get_model_logger(
            Path(log_path),
            max_screenshots=max_screenshots,
        )
        return self._model_logger

    @staticmethod
    def _extract_data_url_bytes(value: str) -> bytes | None:
        if not value.startswith("data:image/") or "," not in value:
            return None
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            return None
        try:
            return base64.b64decode(encoded)
        except Exception:
            return None

    @classmethod
    def _scrub_content_item_for_log(
        cls,
        item: Any,
        *,
        screenshots: list[tuple[str, bytes]],
        label: str,
    ) -> Any:
        if not isinstance(item, dict):
            return item

        sanitized = dict(item)
        item_type = str(sanitized.get("type") or "").strip().lower()
        if item_type in {"input_image", "image_url"}:
            raw_image = sanitized.get("image_url")
            if isinstance(raw_image, dict):
                image_url = raw_image.get("url") or raw_image.get("image_url")
                image_bytes = (
                    cls._extract_data_url_bytes(image_url)
                    if isinstance(image_url, str)
                    else None
                )
                if image_bytes:
                    screenshots.append((label, image_bytes))
                    normalized_image = dict(raw_image)
                    if "url" in normalized_image:
                        normalized_image["url"] = "<<attached screenshot>>"
                    if "image_url" in normalized_image:
                        normalized_image["image_url"] = "<<attached screenshot>>"
                    sanitized["image_url"] = normalized_image
            elif isinstance(raw_image, str):
                image_bytes = cls._extract_data_url_bytes(raw_image)
                if image_bytes:
                    screenshots.append((label, image_bytes))
                    sanitized["image_url"] = "<<attached screenshot>>"
            return sanitized

        if item_type == "image":
            source = sanitized.get("source")
            if isinstance(source, dict) and source.get("type") == "base64":
                encoded = source.get("data")
                if isinstance(encoded, str):
                    try:
                        screenshots.append((label, base64.b64decode(encoded)))
                        normalized_source = dict(source)
                        normalized_source["data"] = "<<attached screenshot>>"
                        sanitized["source"] = normalized_source
                    except Exception:
                        return sanitized
            data = sanitized.get("data")
            if isinstance(data, str):
                try:
                    screenshots.append((label, base64.b64decode(data)))
                    sanitized["data"] = "<<attached screenshot>>"
                except Exception:
                    return sanitized
            return sanitized

        return sanitized

    @classmethod
    def _prepare_request_payload_for_log(
        cls,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        system_prompt: str | None,
        response_format: dict[str, Any] | None,
        reasoning_level: str | None,
        modalities: set[str] | None,
        stream: bool,
    ) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        screenshots: list[tuple[str, bytes]] = []
        sanitized_messages: list[dict[str, Any]] = []
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                sanitized_messages.append({"content": str(message)})
                continue
            sanitized_message = dict(message)
            content = sanitized_message.get("content")
            if isinstance(content, list):
                sanitized_message["content"] = [
                    cls._scrub_content_item_for_log(
                        item,
                        screenshots=screenshots,
                        label=f"model_input_{message_index}_{item_index}",
                    )
                    for item_index, item in enumerate(content, start=1)
                ]
            sanitized_messages.append(sanitized_message)

        payload = {
            "messages": sanitized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "response_format": response_format,
            "reasoning_level": reasoning_level,
            "modalities": sorted(modalities or []),
            "stream": stream,
        }
        return payload, screenshots

    @classmethod
    def _flatten_message_content(cls, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                        continue
                    image_url = item.get("image_url")
                    if isinstance(image_url, str) and image_url.startswith(
                        "data:image/"
                    ):
                        parts.append("<<attached screenshot>>")
                        continue
                    if isinstance(image_url, dict):
                        url_value = image_url.get("url") or image_url.get("image_url")
                        if isinstance(url_value, str) and url_value.startswith(
                            "data:image/"
                        ):
                            parts.append("<<attached screenshot>>")
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    @classmethod
    def _build_prompt_for_log(cls, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip() or "user"
            content_text = cls._flatten_message_content(message.get("content"))
            if not content_text.strip():
                continue
            lines.append(f"[{role}] {content_text}")
        if lines:
            return "\n\n".join(lines)
        return str(messages)

    async def _log_model_success(
        self,
        *,
        model_logger: Any,
        agent: str,
        model: str,
        prompt: str,
        request_payload: Any,
        response: Any,
        screenshots: list[tuple[str, bytes]] | None,
        metadata: dict[str, Any],
    ) -> None:
        log_outcome = getattr(model_logger, "log_outcome", None)
        if callable(log_outcome):
            await log_outcome(
                agent=agent,
                model=model,
                prompt=prompt,
                request_payload=request_payload,
                response=response,
                screenshots=screenshots,
                metadata=metadata,
                outcome="success",
            )
            return

        log_call = getattr(model_logger, "log_call", None)
        if callable(log_call):
            await log_call(
                agent=agent,
                model=model,
                prompt=prompt,
                request_payload=request_payload,
                response=response,
                screenshots=screenshots,
                metadata=metadata,
            )

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
        settings = get_settings()
        provider = str(settings.agent_provider).strip().lower()

        if self._client is None or self._client_provider != provider:
            if not self._model_override:
                self.model = settings.get_provider_model(provider)
            if provider == "anthropic":
                from haindy.models.anthropic_client import AnthropicClient

                self._client = AnthropicClient(model=self.model)
            elif provider == "google":
                from haindy.models.google_client import GoogleClient

                self._client = GoogleClient(model=self.model)
            else:
                self._client = OpenAIClient(
                    model=self.model,
                    reasoning_level=self.reasoning_level,
                    modalities=self.modalities,
                )
            self._client_provider = provider
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
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_level: str | None = None,
        modalities: set[str] | None = None,
        stream: bool = False,
        stream_observer: ResponseStreamObserver | None = None,
        log_agent: str | None = None,
        log_metadata: dict[str, Any] | None = None,
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
        resolved_temperature = temperature or self.temperature
        resolved_reasoning = reasoning_level or self.reasoning_level
        resolved_modalities = modalities or self.modalities
        should_stream = stream or stream_observer is not None
        call_metadata = {
            "provider": self._client_provider
            or str(get_settings().agent_provider).strip().lower(),
            "stream": should_stream,
        }
        if log_metadata:
            call_metadata.update(log_metadata)

        request_payload, screenshots = self._prepare_request_payload_for_log(
            messages=messages,
            temperature=resolved_temperature,
            max_tokens=max_tokens,
            system_prompt=self.system_prompt,
            response_format=response_format,
            reasoning_level=resolved_reasoning,
            modalities=resolved_modalities,
            stream=should_stream,
        )
        prompt = self._build_prompt_for_log(messages)
        model_logger = self._resolve_model_logger()
        capture = _CapturedStreamObserver(stream_observer) if should_stream else None

        try:
            response = await self.client.call(
                messages=messages,
                temperature=resolved_temperature,
                max_tokens=max_tokens,
                system_prompt=self.system_prompt,
                response_format=response_format,
                reasoning_level=resolved_reasoning,
                modalities=resolved_modalities,
                stream=stream,
                stream_observer=capture or stream_observer,
            )
        except Exception as exc:
            if model_logger is not None and callable(
                getattr(model_logger, "log_outcome", None)
            ):
                failure_kind = None
                failure_response: Any | None = None
                logged_exception: BaseException = exc
                if isinstance(exc, ModelCallError):
                    failure_kind = exc.failure_kind
                    failure_response = exc.response_payload
                    if exc.__cause__ is not None and isinstance(
                        exc.__cause__, BaseException
                    ):
                        logged_exception = exc.__cause__

                partial_stream = capture.snapshot_response() if capture else None
                if partial_stream is not None:
                    if failure_response is None:
                        failure_response = partial_stream
                    else:
                        failure_response = {
                            "provider_response": failure_response,
                            "stream": partial_stream,
                        }

                await log_model_call_failure(
                    model_logger,
                    agent=log_agent or self.name,
                    model=self.model,
                    prompt=prompt,
                    request_payload=request_payload,
                    exception=logged_exception,
                    response=failure_response,
                    screenshots=screenshots or None,
                    metadata=call_metadata,
                    failure_kind=failure_kind,
                )
            raise

        if model_logger is not None:
            await self._log_model_success(
                model_logger=model_logger,
                agent=log_agent or self.name,
                model=self.model,
                prompt=prompt,
                request_payload=request_payload,
                response=response,
                screenshots=screenshots or None,
                metadata=call_metadata,
            )
        return cast(dict[str, Any], response)

    def update_reasoning_level(self, level: str) -> None:
        """Update reasoning level for future calls."""
        self.reasoning_level = level
        self._client = None
        self._client_provider = None

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
