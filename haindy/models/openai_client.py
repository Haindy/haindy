"""OpenAI API client wrapper for the HAINDY framework."""

from __future__ import annotations

import inspect
import json
import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol

import openai
from openai import AsyncOpenAI

from haindy.auth import (
    CODEX_SYSTEM_INSTRUCTIONS,
    OpenAIAuthManager,
    ResolvedOpenAIAuth,
)
from haindy.config.settings import get_settings
from haindy.models.errors import ModelCallError
from haindy.models.structured_output import extract_json_schema_definition

SUPPORTED_OPENAI_MODEL = "gpt-5.4"


class ResponseStreamObserver(Protocol):
    """Observer that receives streaming updates from the Responses API."""

    def on_stream_start(self) -> Any: ...

    def on_text_delta(self, delta: str) -> Any: ...

    def on_usage_delta(self, delta: dict[str, int]) -> Any: ...

    def on_usage_total(self, totals: dict[str, int]) -> Any: ...

    def on_token_progress(
        self, total_tokens: int, delta_tokens: int, delta_chars: int
    ) -> Any: ...

    def on_error(self, error: Any) -> Any: ...

    def on_stream_end(self) -> Any: ...


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        max_retries: int = 3,
        reasoning_level: str = "medium",
        modalities: set[str] | None = None,
        request_timeout: float | None = None,
        auth_manager: OpenAIAuthManager | None = None,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            model: Model to use for completions
            api_key: Optional API key (defaults to env/config)
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
        self.reasoning_level = reasoning_level
        self.modalities = modalities or {"text"}
        self.logger = logging.getLogger("openai_client")
        self._token_encoder_cache: dict[str, Any] = {}
        self._client: AsyncOpenAI | None = None
        self._client_signature: tuple[Any, ...] | None = None
        self._resolved_auth: ResolvedOpenAIAuth | None = None

        settings = get_settings()
        self._settings = settings
        self.api_key = api_key or settings.openai_api_key
        self._api_key_override = api_key
        self._auth_manager = auth_manager or OpenAIAuthManager(settings=settings)
        self.request_timeout = request_timeout or float(
            settings.openai_request_timeout_seconds
        )

        if model != SUPPORTED_OPENAI_MODEL:
            raise ValueError(
                f"Unsupported OpenAI model '{model}'. "
                f"Supported model is '{SUPPORTED_OPENAI_MODEL}'."
            )
        self.model = model

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
        """Make a call to the OpenAI API."""

        final_messages: list[dict[str, Any]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        self.logger.debug(
            f"OpenAI API call: model={self.model}, "
            f"messages={len(final_messages)}, temperature={temperature}"
        )
        auth = await self._resolve_auth()

        try:
            if stream or stream_observer is not None or auth.mode == "codex_oauth":
                try:
                    return await self._call_responses_api_streaming(
                        final_messages=final_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        reasoning_level=reasoning_level or self.reasoning_level,
                        system_prompt=system_prompt,
                        observer=stream_observer,
                    )
                except Exception:
                    if auth.mode == "codex_oauth":
                        raise
                    self.logger.warning(
                        "Streaming responses call failed; falling back to standard execution",
                        exc_info=True,
                    )

            return await self._call_responses_api(
                final_messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                reasoning_level=reasoning_level or self.reasoning_level,
                system_prompt=system_prompt,
            )

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            if (
                self._resolved_auth is not None
                and self._resolved_auth.mode == "codex_oauth"
                and isinstance(
                    e, (openai.AuthenticationError, openai.PermissionDeniedError)
                )
            ):
                raise RuntimeError(
                    "OpenAI Codex OAuth authentication failed. Re-run "
                    "--codex-auth login or --codex-auth logout."
                ) from e
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        temperature: float = 0.7,
        detail: str = "high",
    ) -> dict[str, Any]:
        """
        Analyze an image using vision capabilities.

        Args:
            image_data: Image data as bytes
            prompt: Analysis prompt
            temperature: Temperature for response
            detail: Image detail level ('low', 'high', 'auto')

        Returns:
            Analysis response
        """
        import base64

        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": detail,
                        },
                    },
                ],
            }
        ]

        return await self.call(messages=messages, temperature=temperature)

    async def create_structured_output(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        temperature: float = 0.7,
        examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Create structured output with schema validation.

        Args:
            prompt: User prompt
            response_schema: JSON schema for response
            temperature: Temperature for response
            examples: Optional examples to include

        Returns:
            Structured response matching schema
        """
        # Build the system prompt with schema
        system_prompt = (
            "You are a precise AI that always returns valid JSON matching the provided schema. "
            f"Schema: {json.dumps(response_schema, indent=2)}"
        )

        messages: list[dict[str, Any]] = []

        # Add examples if provided
        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(example["output"]),
                    }
                )

        # Add the actual prompt
        messages.append({"role": "user", "content": prompt})

        response = await self.call(
            messages=messages,
            temperature=temperature,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
        )

        content = response.get("content", {})
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Structured output response was not a JSON object")

    def estimate_cost(self, usage: dict[str, int]) -> float:
        """
        Estimate cost based on token usage.

        Args:
            usage: Token usage dictionary

        Returns:
            Estimated cost in USD
        """
        model_pricing = {
            "prompt": 1.25 / 1_000_000,
            "completion": 5.0 / 1_000_000,
        }

        prompt_cost = usage.get("prompt_tokens", 0) * model_pricing["prompt"]
        completion_cost = (
            usage.get("completion_tokens", 0) * model_pricing["completion"]
        )

        return prompt_cost + completion_cost

    def _supports_responses_temperature(self, reasoning_level: str | None) -> bool:
        """Return True if the Responses model accepts the temperature parameter."""
        return (reasoning_level or "medium") == "none"

    async def _call_responses_api(
        self,
        final_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        reasoning_level: str | None,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        instructions, input_items = self._prepare_responses_input(
            final_messages, system_prompt
        )
        auth = await self._resolve_auth()
        instructions = self._apply_codex_instruction_defaults(
            instructions=instructions,
            auth=auth,
        )
        instructions = self._ensure_json_keyword_for_response_format(
            instructions=instructions,
            input_items=input_items,
            response_format=response_format,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            kwargs["instructions"] = instructions
        if auth.mode == "codex_oauth":
            kwargs["store"] = False

        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        text_config = self._map_response_format_to_text_config(response_format)
        if text_config:
            kwargs["text"] = {"format": text_config}

        if reasoning_level:
            kwargs["reasoning"] = {"effort": reasoning_level}

        if self._supports_responses_temperature(reasoning_level or "medium"):
            kwargs["temperature"] = temperature

        client = await self._get_client(auth)
        response = await client.responses.create(
            timeout=self.request_timeout,
            **kwargs,
        )

        content_text = self._extract_output_text(response)
        content_value: Any = content_text

        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"}:
            if not content_text:
                content_value = {}
            else:
                try:
                    content_value = json.loads(content_text)
                except json.JSONDecodeError as exc:
                    self.logger.error(
                        "Failed to parse JSON response",
                        exc_info=True,
                    )
                    raise ModelCallError(
                        "Failed to parse JSON response from OpenAI.",
                        failure_kind="response_parse_error",
                        response_payload={
                            "provider_response": response,
                            "content_text": content_text,
                        },
                    ) from exc
        usage = getattr(response, "usage", None)

        usage_dict = {
            "prompt_tokens": self._safe_usage_lookup(usage, "input_tokens"),
            "completion_tokens": self._safe_usage_lookup(usage, "output_tokens"),
            "total_tokens": self._safe_usage_lookup(usage, "total_tokens"),
        }

        finish_reason = getattr(response, "status", None)

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": getattr(response, "model", self.model),
            "finish_reason": finish_reason,
        }

    async def _call_responses_api_streaming(
        self,
        final_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        reasoning_level: str | None,
        system_prompt: str | None,
        observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]:
        instructions, input_items = self._prepare_responses_input(
            final_messages, system_prompt
        )
        auth = await self._resolve_auth()
        instructions = self._apply_codex_instruction_defaults(
            instructions=instructions,
            auth=auth,
        )
        instructions = self._ensure_json_keyword_for_response_format(
            instructions=instructions,
            input_items=input_items,
            response_format=response_format,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            kwargs["instructions"] = instructions
        if auth.mode == "codex_oauth":
            kwargs["store"] = False

        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        text_config = self._map_response_format_to_text_config(response_format)
        if text_config:
            kwargs["text"] = {"format": text_config}

        if reasoning_level:
            kwargs["reasoning"] = {"effort": reasoning_level}

        if self._supports_responses_temperature(reasoning_level or "medium"):
            kwargs["temperature"] = temperature

        usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        last_reported_usage = dict(usage_totals)
        token_encoder = self._get_token_encoder(self.model)
        estimated_output_tokens = 0

        await self._dispatch_observer(observer, "on_stream_start")

        final_response: Any = None
        client = await self._get_client(auth)

        try:
            async with client.responses.stream(
                timeout=self.request_timeout,
                **kwargs,
            ) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        delta_text = self._resolve_text_delta(
                            getattr(event, "delta", None)
                        )
                        if delta_text:
                            await self._dispatch_observer(
                                observer, "on_text_delta", delta_text
                            )
                            delta_chars = len(delta_text)
                            delta_tokens = self._estimate_token_count(
                                delta_text, token_encoder
                            )
                            if delta_tokens:
                                estimated_output_tokens += delta_tokens
                            await self._dispatch_observer(
                                observer,
                                "on_token_progress",
                                estimated_output_tokens,
                                delta_tokens,
                                delta_chars,
                            )
                    elif event_type in {"response.in_progress", "response.completed"}:
                        usage_payload = self._extract_usage_from_event(event)
                        if usage_payload:
                            delta_payload = self._calculate_usage_delta(
                                last_reported_usage, usage_payload
                            )
                            usage_totals.update(usage_payload)
                            last_reported_usage.update(usage_payload)
                            if delta_payload:
                                await self._dispatch_observer(
                                    observer, "on_usage_delta", delta_payload
                                )
                    elif event_type == "response.error":
                        error = getattr(event, "error", None)
                        if error is not None:
                            await self._dispatch_observer(observer, "on_error", error)

                final_response = await stream.get_final_response()
        except Exception as error:
            await self._dispatch_observer(observer, "on_error", error)
            await self._dispatch_observer(observer, "on_stream_end")
            raise

        if final_response is None:
            raise RuntimeError("Streaming response did not return a final result")

        content_text = self._extract_output_text(final_response)
        content_value: Any = content_text
        usage = getattr(final_response, "usage", None)

        combined_usage = dict(usage_totals)
        for key in list(combined_usage.keys()):
            final_value = self._safe_usage_lookup(usage, key)
            if final_value:
                combined_usage[key] = max(combined_usage[key], final_value)

        if estimated_output_tokens:
            combined_usage["output_tokens"] = max(
                combined_usage.get("output_tokens", 0), estimated_output_tokens
            )
            combined_usage["total_tokens"] = max(
                combined_usage.get("total_tokens", 0), estimated_output_tokens
            )

        await self._dispatch_observer(observer, "on_usage_total", combined_usage)
        await self._dispatch_observer(observer, "on_stream_end")

        usage_dict = {
            "prompt_tokens": combined_usage.get("input_tokens", 0),
            "completion_tokens": combined_usage.get("output_tokens", 0),
            "total_tokens": combined_usage.get("total_tokens", 0),
        }

        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"}:
            if not content_text:
                content_value = {}
            else:
                try:
                    content_value = json.loads(content_text)
                except json.JSONDecodeError as exc:
                    self.logger.error("Failed to parse JSON response", exc_info=True)
                    raise ModelCallError(
                        "Failed to parse streaming JSON response from OpenAI.",
                        failure_kind="response_parse_error",
                        response_payload={
                            "provider_response": final_response,
                            "content_text": content_text,
                            "usage": combined_usage,
                        },
                    ) from exc

        finish_reason = getattr(final_response, "status", None)

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": getattr(final_response, "model", self.model),
            "finish_reason": finish_reason,
        }

    def _prepare_responses_input(
        self,
        final_messages: Sequence[dict[str, Any]],
        system_prompt: str | None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions = system_prompt or None
        input_items: list[dict[str, Any]] = []

        for message in final_messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role in {"system", "developer"}:
                text = content if isinstance(content, str) else str(content)
                if instructions:
                    if text.strip() and text.strip() != instructions.strip():
                        instructions = f"{instructions}\n\n{text}"
                else:
                    instructions = text
                continue

            content_items = self._convert_content_for_role(role, content)
            input_items.append({"role": role, "content": content_items})

        if not input_items:
            input_items.append(
                {"role": "user", "content": [{"type": "input_text", "text": ""}]}
            )

        return instructions, input_items

    async def _resolve_auth(self) -> ResolvedOpenAIAuth:
        auth = await self._auth_manager.resolve_openai_auth(
            api_key_override=self._api_key_override
        )
        self._resolved_auth = auth
        return auth

    async def _get_client(self, auth: ResolvedOpenAIAuth) -> AsyncOpenAI:
        signature = (
            auth.mode,
            auth.base_url or "",
            tuple(sorted(auth.default_headers.items())),
        )
        if self._client is not None and self._client_signature == signature:
            return self._client

        kwargs: dict[str, Any] = {
            "max_retries": self.max_retries,
        }

        if auth.mode == "codex_oauth":
            kwargs["api_key"] = self._oauth_token_provider
            kwargs["base_url"] = auth.base_url
            kwargs["default_headers"] = auth.default_headers
        else:
            kwargs["api_key"] = auth.token

        self._client = AsyncOpenAI(**kwargs)
        self._client_signature = signature
        return self._client

    async def _oauth_token_provider(self) -> str:
        auth = await self._auth_manager.resolve_openai_auth(
            api_key_override=self._api_key_override
        )
        if auth.mode != "codex_oauth":
            raise RuntimeError(
                "Codex OAuth session is no longer active. Recreate the OpenAI client."
            )
        self._resolved_auth = auth
        return auth.token

    def _apply_codex_instruction_defaults(
        self,
        *,
        instructions: str | None,
        auth: ResolvedOpenAIAuth,
    ) -> str | None:
        if auth.mode != "codex_oauth":
            return instructions
        if instructions and instructions.strip():
            return instructions
        return CODEX_SYSTEM_INSTRUCTIONS

    def _convert_content_for_role(
        self, role: str, content: Any
    ) -> list[dict[str, Any]]:
        content_items: list[dict[str, Any]] = []
        default_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text", "output_text"}:
                        normalized = dict(item)
                        if normalized["type"] == "text":
                            normalized["type"] = default_type
                        elif (
                            role != "assistant" and normalized["type"] == "output_text"
                        ):
                            normalized["type"] = default_type
                        elif role == "assistant" and normalized["type"] == "input_text":
                            normalized["type"] = default_type
                        content_items.append(normalized)
                    elif item_type in {"image_url", "input_image"}:
                        image_data = item.get("image_url")
                        detail = item.get("detail")

                        if isinstance(image_data, dict):
                            detail = detail or image_data.get("detail")
                            image_url = image_data.get("url") or image_data.get(
                                "image_url"
                            )
                        else:
                            image_url = image_data

                        if not image_url:
                            image_url = item.get("url")

                        if isinstance(image_url, dict):
                            detail = detail or image_url.get("detail")
                            image_url = image_url.get("url")

                        if image_url:
                            image_item = {"type": "input_image", "image_url": image_url}
                            if detail:
                                image_item["detail"] = detail
                            content_items.append(image_item)
                    else:
                        content_items.append(
                            {"type": default_type, "text": json.dumps(item)}
                        )
                else:
                    content_items.append({"type": default_type, "text": str(item)})
        else:
            content_items.append({"type": default_type, "text": str(content)})

        return content_items

    def _extract_output_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        output_segments = getattr(response, "output", None)
        if not output_segments:
            return ""

        texts: list[str] = []
        for segment in output_segments:
            for piece in segment.get("content", []):
                text_value = piece.get("text")
                if text_value:
                    texts.append(text_value)

        return "\n".join(texts)

    def _safe_usage_lookup(self, usage: Any, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(key, 0))
        return int(getattr(usage, key, 0))

    def _get_token_encoder(self, model: str) -> Any | None:
        if model in self._token_encoder_cache:
            return self._token_encoder_cache[model]

        try:
            import tiktoken  # type: ignore
        except ImportError:
            self._token_encoder_cache[model] = None
            return None

        try:
            encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoder = None

        self._token_encoder_cache[model] = encoder
        return encoder

    def _estimate_token_count(self, text: str, encoder: Any | None) -> int:
        if not text:
            return 0

        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass

        # Fallback heuristic: roughly 4 characters per token.
        return max(1, math.ceil(len(text) / 4))

    def _map_response_format_to_text_config(
        self, response_format: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Translate legacy response_format into Responses API text config."""
        if not response_format:
            return None

        format_type = response_format.get("type")
        if format_type == "json_object":
            return {"type": "json_object"}

        if format_type == "json_schema":
            schema_payload = extract_json_schema_definition(response_format)
            if schema_payload:
                return {
                    "type": "json_schema",
                    "name": schema_payload["name"],
                    "schema": schema_payload["schema"],
                    "strict": schema_payload["strict"],
                }

        if format_type == "text":
            return {"type": "text"}

        return None

    def _ensure_json_keyword_for_response_format(
        self,
        *,
        instructions: str | None,
        input_items: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str | None:
        """Ensure `input` includes 'json' when JSON text formats are requested.

        The Responses API rejects `text.format=json_object` payloads when none of the
        input messages include the word `json`.
        """
        format_type = response_format.get("type") if response_format else None
        if format_type not in {"json_object", "json_schema"}:
            return instructions

        input_text_fragments: list[str] = []
        for item in input_items:
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    input_text_fragments.append(text)

        if any("json" in fragment.lower() for fragment in input_text_fragments):
            return instructions

        suffix = "Respond with valid json."
        input_items.append(
            {"role": "user", "content": [{"type": "input_text", "text": suffix}]}
        )
        if instructions and instructions.strip():
            return instructions
        return suffix

    async def _dispatch_observer(
        self,
        observer: ResponseStreamObserver | None,
        method_name: str,
        *args: Any,
    ) -> None:
        if observer is None:
            return

        method = getattr(observer, method_name, None)
        if method is None:
            return

        try:
            result = method(*args)
        except Exception as observer_error:  # pragma: no cover - defensive logging
            self.logger.debug(
                "Streaming observer %s raised an exception: %s",
                method_name,
                observer_error,
            )
            return

        if inspect.isawaitable(result):
            await result

    def _extract_usage_from_event(self, event: Any) -> dict[str, int]:
        response_obj = getattr(event, "response", None)
        if response_obj is None:
            return {}

        usage_obj = getattr(response_obj, "usage", None)
        return self._normalize_usage_dict(usage_obj)

    def _calculate_usage_delta(
        self, previous: dict[str, int], current: dict[str, int]
    ) -> dict[str, int]:
        delta: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            new_value = int(current.get(key, previous.get(key, 0)) or 0)
            old_value = int(previous.get(key, 0) or 0)
            difference = new_value - old_value
            if difference > 0:
                delta[key] = difference
        return delta

    def _normalize_usage_dict(self, usage: Any) -> dict[str, int]:
        payload: dict[str, int] = {}
        if usage is None:
            return payload

        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value: int | None
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)

            if value is None:
                continue

            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                continue

        return payload

    def _normalize_chat_usage(self, usage: Any) -> dict[str, int]:
        """
        Normalize chat completion usage payloads across SDK versions.

        OpenAI 1.x returns prompt/completion tokens, while 2.x may expose
        input/output tokens. This helper maps both to a consistent schema.
        """

        prompt_tokens = self._lookup_usage_with_aliases(
            usage, ("prompt_tokens", "input_tokens")
        )
        completion_tokens = self._lookup_usage_with_aliases(
            usage, ("completion_tokens", "output_tokens")
        )
        total_tokens = self._lookup_usage_with_aliases(usage, ("total_tokens",))

        # Derive totals if missing
        if total_tokens == 0 and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _lookup_usage_with_aliases(self, usage: Any, aliases: Sequence[str]) -> int:
        for key in aliases:
            value = self._extract_usage_value(usage, key)
            if value is not None:
                return value
        return 0

    def _extract_usage_value(self, usage: Any, key: str) -> int | None:
        if usage is None:
            return None

        raw_value: Any | None
        if isinstance(usage, dict):
            if key not in usage:
                return None
            raw_value = usage.get(key)
        else:
            if not hasattr(usage, key):
                return None
            raw_value = getattr(usage, key)

        if raw_value is None:
            return None

        try:
            return int(raw_value)
        except (TypeError, ValueError):
            self.logger.debug(
                "Unable to coerce usage value '%s' for key '%s' to int", raw_value, key
            )
            return None

    def _resolve_text_delta(self, delta: Any) -> str:
        if delta is None:
            return ""
        if isinstance(delta, str):
            return delta
        if isinstance(delta, dict):
            text_value = delta.get("text") or delta.get("output_text")
            return str(text_value) if text_value else ""

        text_attr = getattr(delta, "text", None)
        if text_attr:
            return str(text_attr)

        output_attr = getattr(delta, "output_text", None)
        if output_attr:
            return str(output_attr)

        return ""
