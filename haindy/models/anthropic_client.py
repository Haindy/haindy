"""Anthropic API client wrapper for non-CU agent calls."""
from __future__ import annotations

import json
import logging
from typing import Any

from haindy.config.settings import get_settings
from haindy.models.llm_client import dispatch_observer
from haindy.models.openai_client import ResponseStreamObserver

logger = logging.getLogger("anthropic_client")


class AnthropicClient:
    """Wrapper for Anthropic API interactions (non-computer-use)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.anthropic_api_key
        self._model = settings.anthropic_model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
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
        anthropic_messages = []
        extracted_system: str | None = system_prompt

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text = content if isinstance(content, str) else str(content)
                if extracted_system:
                    extracted_system = f"{extracted_system}\n\n{text}"
                else:
                    extracted_system = text
                continue
            anthropic_messages.append({"role": role, "content": content})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        # JSON mode: inject instruction into system prompt
        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"}:
            json_instruction = "Respond with valid JSON only."
            if extracted_system:
                extracted_system = f"{extracted_system}\n\n{json_instruction}"
            else:
                extracted_system = json_instruction

        effective_max_tokens = max_tokens or 8192

        client = self._get_client()

        if stream or stream_observer is not None:
            return await self._call_streaming(
                messages=anthropic_messages,
                system=extracted_system,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                response_format=response_format,
                stream_observer=stream_observer,
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if extracted_system:
            kwargs["system"] = extracted_system

        response = await client.messages.create(**kwargs)

        content_text = ""
        if response.content:
            first = response.content[0]
            content_text = getattr(first, "text", "") or ""

        content_value: Any = content_text
        if format_type in {"json_object", "json_schema"}:
            if content_text:
                try:
                    content_value = json.loads(content_text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response from Anthropic")
                    raise

        usage = response.usage
        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "input_tokens", 0)
                + getattr(usage, "output_tokens", 0),
            },
            "model": getattr(response, "model", self._model),
            "finish_reason": getattr(response, "stop_reason", None),
        }

    async def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        stream_observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]:
        """Make a streaming call to the Anthropic API."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        if stream_observer is not None:
            dispatch_observer(stream_observer,"on_stream_start")

        full_text = ""
        input_tokens = 0
        output_tokens = 0
        model_name = self._model
        stop_reason: str | None = None

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    if stream_observer is not None:
                        dispatch_observer(stream_observer,"on_text_delta", text_chunk)

                final_message = await stream.get_final_message()
                model_name = getattr(final_message, "model", self._model)
                stop_reason = getattr(final_message, "stop_reason", None)
                usage = getattr(final_message, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
        except Exception as exc:
            if stream_observer is not None:
                dispatch_observer(stream_observer,"on_error", exc)
            if stream_observer is not None:
                dispatch_observer(stream_observer,"on_stream_end")
            raise

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if stream_observer is not None:
            dispatch_observer(stream_observer,"on_usage_total", usage_dict)
            dispatch_observer(stream_observer,"on_stream_end")

        format_type = response_format.get("type") if response_format else None
        content_value: Any = full_text
        if format_type in {"json_object", "json_schema"} and full_text:
            try:
                content_value = json.loads(full_text)
            except json.JSONDecodeError:
                logger.error("Failed to parse streaming JSON response from Anthropic")
                raise

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": model_name,
            "finish_reason": stop_reason,
        }

