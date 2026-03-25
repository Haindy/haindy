"""Anthropic API client wrapper for non-CU agent calls."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from anthropic import AsyncAnthropic as _AsyncAnthropic
except Exception:  # pragma: no cover
    _AsyncAnthropic = None  # type: ignore[assignment]

from haindy.models.openai_client import ResponseStreamObserver

logger = logging.getLogger("anthropic_client")


class AnthropicClient:
    """Wrapper for Anthropic API interactions used by non-CU agents."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_retries: int = 3,
        reasoning_level: str = "medium",
        modalities: set[str] | None = None,
    ) -> None:
        if _AsyncAnthropic is None:
            raise RuntimeError(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )
        self.model = model
        self.max_retries = max_retries
        self.reasoning_level = reasoning_level
        self.modalities = modalities or {"text"}
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from haindy.config.settings import get_settings

            settings = get_settings()
            key = self._api_key or settings.anthropic_api_key
            self._client = _AsyncAnthropic(
                api_key=key,
                max_retries=self.max_retries,
            )
        return self._client

    async def call(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_level: str | None = None,
        modalities: set[str] | None = None,
        stream: bool = False,
        stream_observer: ResponseStreamObserver | None = None,
    ) -> dict[str, Any]:
        """Make a call to the Anthropic API."""
        # Separate system messages from the conversation history.
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in {"system", "developer"}:
                system_parts.append(content if isinstance(content, str) else str(content))
                continue
            anthropic_messages.append({"role": role, "content": content})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        system_text = "\n\n".join(system_parts) if system_parts else None

        # Handle JSON mode by hinting in the system prompt.
        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"} and system_text is not None:
            if "json" not in system_text.lower():
                system_text = system_text + "\n\nRespond with valid JSON."
        elif format_type in {"json_object", "json_schema"} and system_text is None:
            system_text = "Respond with valid JSON."

        effective_max_tokens = max_tokens or 8192

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            kwargs["system"] = system_text

        logger.debug(
            "Anthropic API call: model=%s, messages=%d, temperature=%s",
            self.model,
            len(anthropic_messages),
            temperature,
        )

        client = self._get_client()

        if stream or stream_observer is not None:
            return await self._call_streaming(
                client=client,
                kwargs=kwargs,
                format_type=format_type,
                stream_observer=stream_observer,
            )

        response = await client.messages.create(**kwargs)
        return self._normalize_response(response, format_type)

    @staticmethod
    def _notify_observer(observer: ResponseStreamObserver | None, method: str, *args: Any) -> None:
        """Call an observer method, silently ignoring any exception it raises."""
        if observer is None:
            return
        try:
            getattr(observer, method)(*args)
        except Exception:
            pass

    async def _call_streaming(
        self,
        *,
        client: Any,
        kwargs: dict[str, Any],
        format_type: str | None,
        stream_observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]:
        """Stream a response and gather the final result."""
        self._notify_observer(stream_observer, "on_stream_start")

        full_text = ""
        usage_in: int = 0
        usage_out: int = 0
        finish_reason: str | None = None
        model_used = self.model

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                full_text += text
                self._notify_observer(stream_observer, "on_text_delta", text)

            final = await stream.get_final_message()
            model_used = getattr(final, "model", self.model)
            usage = getattr(final, "usage", None)
            if usage is not None:
                usage_in = int(getattr(usage, "input_tokens", 0) or 0)
                usage_out = int(getattr(usage, "output_tokens", 0) or 0)
            stop_reason = getattr(final, "stop_reason", None)
            finish_reason = str(stop_reason) if stop_reason is not None else None

        self._notify_observer(stream_observer, "on_stream_end")

        content_value: Any = full_text
        if format_type in {"json_object", "json_schema"}:
            if not full_text:
                content_value = {}
            else:
                try:
                    content_value = json.loads(full_text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON streaming response", exc_info=True)
                    raise

        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": usage_in,
                "completion_tokens": usage_out,
                "total_tokens": usage_in + usage_out,
            },
            "model": model_used,
            "finish_reason": finish_reason,
        }

    def _normalize_response(
        self, response: Any, format_type: str | None
    ) -> dict[str, Any]:
        """Extract text content and usage from an Anthropic messages response."""
        content_blocks = getattr(response, "content", []) or []
        text_parts: list[str] = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")

        raw_text = "\n".join(text_parts)
        content_value: Any = raw_text

        if format_type in {"json_object", "json_schema"}:
            if not raw_text:
                content_value = {}
            else:
                try:
                    content_value = json.loads(raw_text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response", exc_info=True)
                    raise

        usage = getattr(response, "usage", None)
        usage_in = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        usage_out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        stop_reason = getattr(response, "stop_reason", None)
        finish_reason = str(stop_reason) if stop_reason is not None else None

        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": usage_in,
                "completion_tokens": usage_out,
                "total_tokens": usage_in + usage_out,
            },
            "model": getattr(response, "model", self.model),
            "finish_reason": finish_reason,
        }
