"""Anthropic API client wrapper for non-CU agent calls."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from typing import Any

from haindy.config.settings import get_settings
from haindy.models.errors import ModelCallError
from haindy.models.llm_client import dispatch_observer
from haindy.models.openai_client import ResponseStreamObserver
from haindy.models.structured_output import (
    extract_json_schema_definition,
    response_format_expects_json,
)

logger = logging.getLogger("anthropic_client")

_DATA_URL_PATTERN = re.compile(
    r"^data:(image/[A-Za-z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\n\r]+)$",
    re.DOTALL,
)
_SUPPORTED_OUTPUT_MODALITIES = {"text"}
_EFFORT_LEVEL_MAP = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}
_EFFORT_MODEL_PREFIXES = (
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)


def _flatten_message_content(content: Any) -> str:
    """Collapse mixed message content into plain text for diagnostics."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _flatten_response_text_blocks(content_blocks: Any) -> str:
    """Collect all text blocks from an Anthropic message response."""
    if isinstance(content_blocks, str):
        return content_blocks

    parts: list[str] = []
    for item in content_blocks or []:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


class AnthropicClient:
    """Wrapper for Anthropic API interactions (non-computer-use)."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.anthropic_api_key
        self._model = model or settings.anthropic_model or "claude-sonnet-4-6"
        self.model = self._model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _append_system_text(current: str | None, text: str | None) -> str | None:
        normalized = str(text or "").strip()
        if not normalized:
            return current
        if current:
            return f"{current}\n\n{normalized}"
        return normalized

    @staticmethod
    def _supports_effort(model: str) -> bool:
        normalized = str(model or "").strip().lower()
        return any(normalized.startswith(prefix) for prefix in _EFFORT_MODEL_PREFIXES)

    def _map_effort(self, reasoning_level: str | None) -> str | None:
        if not reasoning_level or not self._supports_effort(self._model):
            return None
        normalized = str(reasoning_level).strip().lower()
        return _EFFORT_LEVEL_MAP.get(normalized)

    @staticmethod
    def _invalid_request_error(
        message: str,
        *,
        payload: Any | None = None,
    ) -> ModelCallError:
        return ModelCallError(
            message,
            failure_kind="request_validation_error",
            response_payload=payload,
        )

    def _validate_modalities(self, modalities: set[str] | None) -> None:
        requested = {str(item).strip().lower() for item in (modalities or {"text"})}
        unsupported = sorted(
            item for item in requested if item not in _SUPPORTED_OUTPUT_MODALITIES
        )
        if unsupported:
            raise self._invalid_request_error(
                "Anthropic non-CU client supports text responses only.",
                payload={"unsupported_modalities": unsupported},
            )

    def _normalize_image_source(self, source: dict[str, Any]) -> dict[str, Any]:
        source_type = str(source.get("type") or "").strip().lower()
        if source_type == "base64":
            media_type = str(source.get("media_type") or "").strip()
            data = str(source.get("data") or "").strip()
            if not media_type.startswith("image/") or not data:
                raise self._invalid_request_error(
                    "Anthropic image blocks require base64 image data and media_type.",
                    payload={"source": source},
                )
            try:
                base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise self._invalid_request_error(
                    "Anthropic image blocks require valid base64 data.",
                    payload={"source": source},
                ) from exc
            return {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }

        if source_type == "url":
            url = str(source.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                raise self._invalid_request_error(
                    "Anthropic image URL sources must use http or https URLs.",
                    payload={"source": source},
                )
            return {"type": "url", "url": url}

        if source_type == "file":
            return dict(source)

        raise self._invalid_request_error(
            "Unsupported Anthropic image source type.",
            payload={"source": source},
        )

    def _normalize_image_reference(self, reference: Any) -> dict[str, Any]:
        if isinstance(reference, dict):
            if isinstance(reference.get("source"), dict):
                return self._normalize_image_source(reference["source"])
            if isinstance(reference.get("url"), str):
                return self._normalize_image_reference(reference.get("url"))
            if isinstance(reference.get("image_url"), str):
                return self._normalize_image_reference(reference.get("image_url"))

        if not isinstance(reference, str):
            raise self._invalid_request_error(
                "Unsupported image reference for Anthropic message content.",
                payload={"image_reference": reference},
            )

        normalized = reference.strip()
        data_url_match = _DATA_URL_PATTERN.match(normalized)
        if data_url_match:
            media_type = normalized.split(";", 1)[0].split(":", 1)[1]
            data = data_url_match.group("data")
            try:
                base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise self._invalid_request_error(
                    "Anthropic image data URLs must contain valid base64 image data.",
                    payload={"image_reference": "<data-url>"},
                ) from exc
            return {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }

        if normalized.startswith(("http://", "https://")):
            return {"type": "url", "url": normalized}

        raise self._invalid_request_error(
            "Anthropic image inputs must be http(s) URLs or image data URLs.",
            payload={"image_reference": reference},
        )

    def _normalize_content_item(self, item: Any) -> dict[str, Any]:
        if isinstance(item, str):
            return {"type": "text", "text": item}
        if not isinstance(item, dict):
            return {"type": "text", "text": str(item)}

        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"text", "input_text"}:
            return {"type": "text", "text": str(item.get("text") or "")}

        if item_type == "image" and isinstance(item.get("source"), dict):
            return {
                "type": "image",
                "source": self._normalize_image_source(item["source"]),
            }

        if item_type in {"image_url", "input_image"} or "image_url" in item:
            image_reference = item.get("image_url")
            if isinstance(image_reference, dict):
                image_reference = image_reference.get("url") or image_reference.get(
                    "image_url"
                )
            if image_reference is None:
                image_reference = item.get("url")
            return {
                "type": "image",
                "source": self._normalize_image_reference(image_reference),
            }

        if "text" in item and item_type == "":
            return {"type": "text", "text": str(item.get("text") or "")}

        raise self._invalid_request_error(
            "Unsupported non-CU content block for Anthropic.",
            payload={"content_item": item},
        )

    def _normalize_message_content(self, content: Any) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            blocks = [self._normalize_content_item(item) for item in content]
            return blocks or ""

        if isinstance(content, dict):
            return [self._normalize_content_item(content)]

        return str(content)

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        expects_json: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        anthropic_messages: list[dict[str, Any]] = []
        extracted_system: str | None = system_prompt

        for msg in messages:
            role = str(msg.get("role", "user") or "user").strip().lower()
            content = msg.get("content", "")
            if role == "system":
                extracted_system = self._append_system_text(
                    extracted_system,
                    _flatten_message_content(content),
                )
                continue

            normalized_content = self._normalize_message_content(content)
            anthropic_messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": normalized_content,
                }
            )

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        if expects_json and anthropic_messages[-1]["role"] == "assistant":
            raise self._invalid_request_error(
                "Anthropic structured outputs do not support assistant message prefills.",
                payload={"messages": anthropic_messages},
            )

        return anthropic_messages, extracted_system

    def _build_output_config(
        self,
        response_format: dict[str, Any] | None,
        reasoning_level: str | None,
    ) -> tuple[dict[str, Any] | None, bool]:
        expects_json = response_format_expects_json(response_format)
        json_schema = extract_json_schema_definition(response_format)
        if expects_json and json_schema is None:
            raise self._invalid_request_error(
                "Anthropic json_schema response_format requires a schema definition.",
                payload={"response_format": response_format},
            )

        output_config: dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": json_schema["schema"],
            }

        effort = self._map_effort(reasoning_level)
        if effort:
            output_config["effort"] = effort

        return (output_config or None), expects_json

    def _parse_structured_content(
        self,
        *,
        content_text: str,
        stop_reason: str | None,
        provider_response: Any,
        usage: dict[str, int] | None = None,
        streaming: bool = False,
    ) -> Any:
        payload = {
            "provider_response": provider_response,
            "content_text": content_text,
            "stop_reason": stop_reason,
        }
        if usage is not None:
            payload["usage"] = usage

        if stop_reason == "refusal":
            raise ModelCallError(
                "Anthropic refused the structured-output request.",
                failure_kind="response_refusal",
                response_payload=payload,
            )
        if stop_reason == "max_tokens":
            raise ModelCallError(
                "Anthropic structured output was truncated at max_tokens.",
                failure_kind="response_truncated",
                response_payload=payload,
            )
        if not content_text.strip():
            raise ModelCallError(
                "Anthropic returned empty structured output.",
                failure_kind="response_parse_error",
                response_payload=payload,
            )

        try:
            return json.loads(content_text)
        except json.JSONDecodeError as exc:
            log_message = (
                "Failed to parse streaming JSON response from Anthropic"
                if streaming
                else "Failed to parse JSON response from Anthropic"
            )
            logger.error(log_message)
            raise ModelCallError(
                "Failed to parse JSON response from Anthropic.",
                failure_kind="response_parse_error",
                response_payload=payload,
            ) from exc

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
        self._validate_modalities(modalities)
        output_config, expects_json = self._build_output_config(
            response_format,
            reasoning_level,
        )
        anthropic_messages, extracted_system = self._normalize_messages(
            messages,
            system_prompt=system_prompt,
            expects_json=expects_json,
        )
        effective_max_tokens = max_tokens or 8192

        client = self._get_client()

        if stream or stream_observer is not None:
            return await self._call_streaming(
                messages=anthropic_messages,
                system=extracted_system,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                output_config=output_config,
                expects_json=expects_json,
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
        if output_config:
            kwargs["output_config"] = output_config

        response = await client.messages.create(**kwargs)

        content_text = _flatten_response_text_blocks(getattr(response, "content", None))
        stop_reason = getattr(response, "stop_reason", None)
        content_value: Any = content_text
        if expects_json:
            content_value = self._parse_structured_content(
                content_text=content_text,
                stop_reason=stop_reason,
                provider_response=response,
            )

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
            "finish_reason": stop_reason,
        }

    async def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        temperature: float,
        max_tokens: int,
        output_config: dict[str, Any] | None,
        expects_json: bool,
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
        if output_config:
            kwargs["output_config"] = output_config

        if stream_observer is not None:
            dispatch_observer(stream_observer, "on_stream_start")

        full_text = ""
        input_tokens = 0
        output_tokens = 0
        model_name = self._model
        stop_reason: str | None = None
        final_message: Any | None = None

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    if stream_observer is not None:
                        dispatch_observer(stream_observer, "on_text_delta", text_chunk)

                final_message = await stream.get_final_message()
                model_name = getattr(final_message, "model", self._model)
                stop_reason = getattr(final_message, "stop_reason", None)
                usage = getattr(final_message, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
        except Exception as exc:
            if stream_observer is not None:
                dispatch_observer(stream_observer, "on_error", exc)
                dispatch_observer(stream_observer, "on_stream_end")
            raise

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if stream_observer is not None:
            dispatch_observer(stream_observer, "on_usage_total", usage_dict)
            dispatch_observer(stream_observer, "on_stream_end")

        content_value: Any = full_text
        if expects_json:
            content_value = self._parse_structured_content(
                content_text=full_text,
                stop_reason=stop_reason,
                provider_response=final_message,
                usage=usage_dict,
                streaming=True,
            )

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": model_name,
            "finish_reason": stop_reason,
        }
