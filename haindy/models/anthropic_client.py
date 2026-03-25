"""Anthropic API client wrapper for the HAINDY framework."""

from __future__ import annotations

import inspect
import json
import logging
import math
from typing import Any

import anthropic

from haindy.config.settings import get_settings
from haindy.models.openai_client import ResponseStreamObserver


async def _dispatch_observer(
    observer: ResponseStreamObserver | None,
    method_name: str,
    *args: Any,
    logger: logging.Logger,
) -> None:
    if observer is None:
        return
    method = getattr(observer, method_name, None)
    if method is None:
        return
    try:
        result = method(*args)
    except Exception as error:
        logger.debug("Streaming observer %s raised: %s", method_name, error)
        return
    if inspect.isawaitable(result):
        await result


def _get_usage_int(obj: Any, *keys: str) -> int:
    for key in keys:
        val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return 0


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("output_text")
                    or item.get("input_text")
                )
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


class AnthropicClient:
    """Wrapper for Anthropic API interactions using the messages API."""

    _DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self.model = (
            model or getattr(settings, "anthropic_model", None) or "claude-sonnet-4-6"
        )
        self.max_retries = max_retries
        self.logger = logging.getLogger("anthropic_client")
        self._client: anthropic.AsyncAnthropic | None = None
        self._api_key = settings.anthropic_api_key

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
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
        # TODO: map to extended thinking for high/xhigh
        modalities: set[str] | None = None,
        stream: bool = False,
        stream_observer: ResponseStreamObserver | None = None,
    ) -> dict[str, Any]:
        """Make a call to the Anthropic messages API.

        Args:
            messages: Conversation messages with role/content keys.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens (defaults to 8192).
            system_prompt: System instructions.
            response_format: If {"type": "json_object"}, requests JSON output.
            reasoning_level: Ignored (no native equivalent in messages API).
            modalities: Ignored (Anthropic messages API is text-only).
            stream: Whether to stream the response.
            stream_observer: Optional observer for streaming events.

        Returns:
            Dict with keys: content, usage, model, finish_reason.
        """
        effective_max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS

        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        json_mode = response_format is not None and response_format.get("type") in {
            "json_object",
            "json_schema",
        }
        if json_mode:
            system_parts.append("Respond with valid JSON.")

        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in {"system", "developer"}:
                text = content if isinstance(content, str) else str(content)
                if text.strip():
                    system_parts.append(text)
                continue
            anthropic_role = "assistant" if role == "assistant" else "user"
            anthropic_messages.append(
                {"role": anthropic_role, "content": _flatten_message_content(content)}
            )

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        system_text = "\n\n".join(part for part in system_parts if part.strip())

        self.logger.debug(
            "Anthropic API call: model=%s, messages=%d, temperature=%s",
            self.model,
            len(anthropic_messages),
            temperature,
        )

        client = self._get_client()

        if stream or stream_observer is not None:
            return await self._call_streaming(
                client=client,
                messages=anthropic_messages,
                system_text=system_text,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                observer=stream_observer,
                json_mode=json_mode,
            )

        return await self._call_blocking(
            client=client,
            messages=anthropic_messages,
            system_text=system_text,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            json_mode=json_mode,
        )

    async def _call_blocking(
        self,
        client: anthropic.AsyncAnthropic,
        messages: list[dict[str, Any]],
        system_text: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        response = await client.messages.create(**kwargs)

        content_text = self._extract_text(response)
        content_value: Any = content_text
        if json_mode and content_text:
            try:
                content_value = json.loads(content_text)
            except json.JSONDecodeError:
                self.logger.error(
                    "Failed to parse JSON response from Anthropic", exc_info=True
                )
                raise

        usage = getattr(response, "usage", None)
        return {
            "content": content_value,
            "usage": self._normalize_usage(usage),
            "model": getattr(response, "model", self.model),
            "finish_reason": self._extract_stop_reason(response),
        }

    async def _call_streaming(
        self,
        client: anthropic.AsyncAnthropic,
        messages: list[dict[str, Any]],
        system_text: str,
        temperature: float,
        max_tokens: int,
        observer: ResponseStreamObserver | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        await _dispatch_observer(observer, "on_stream_start", logger=self.logger)

        collected_text = ""
        final_message: Any = None

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text_delta in stream.text_stream:
                    collected_text += text_delta
                    await _dispatch_observer(
                        observer, "on_text_delta", text_delta, logger=self.logger
                    )
                    delta_chars = len(text_delta)
                    delta_tokens = max(1, math.ceil(delta_chars / 4))
                    await _dispatch_observer(
                        observer,
                        "on_token_progress",
                        max(1, math.ceil(len(collected_text) / 4)),
                        delta_tokens,
                        delta_chars,
                        logger=self.logger,
                    )
                final_message = await stream.get_final_message()
        except Exception as error:
            await _dispatch_observer(observer, "on_error", error, logger=self.logger)
            await _dispatch_observer(observer, "on_stream_end", logger=self.logger)
            raise

        usage = getattr(final_message, "usage", None) if final_message else None
        usage_dict = self._normalize_usage(usage)
        await _dispatch_observer(
            observer,
            "on_usage_total",
            {
                "input_tokens": usage_dict["prompt_tokens"],
                "output_tokens": usage_dict["completion_tokens"],
                "total_tokens": usage_dict["total_tokens"],
            },
            logger=self.logger,
        )
        await _dispatch_observer(observer, "on_stream_end", logger=self.logger)

        content_value: Any = collected_text
        if json_mode and collected_text:
            try:
                content_value = json.loads(collected_text)
            except json.JSONDecodeError:
                self.logger.error(
                    "Failed to parse JSON response from Anthropic stream", exc_info=True
                )
                raise

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": getattr(final_message, "model", self.model)
            if final_message
            else self.model,
            "finish_reason": self._extract_stop_reason(final_message),
        }

    def _extract_text(self, response: Any) -> str:
        content_blocks = getattr(response, "content", None)
        if not content_blocks:
            return ""
        texts: list[str] = []
        for block in content_blocks:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
        return "\n".join(texts)

    def _extract_stop_reason(self, response: Any) -> str | None:
        if response is None:
            return None
        return getattr(response, "stop_reason", None)

    def _normalize_usage(self, usage: Any) -> dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = _get_usage_int(usage, "input_tokens")
        completion = _get_usage_int(usage, "output_tokens")
        total = _get_usage_int(usage, "total_tokens") or (prompt + completion)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
