"""OpenAI API client wrapper for the HAINDY framework."""

import json
import logging
import math
import inspect
from typing import Any, Dict, List, Optional, Sequence, Set, Protocol

import openai
from openai import AsyncOpenAI

from src.config.settings import get_settings


class ResponseStreamObserver(Protocol):
    """Observer that receives streaming updates from the Responses API."""

    def on_stream_start(self) -> Any:
        ...

    def on_text_delta(self, delta: str) -> Any:
        ...

    def on_usage_delta(self, delta: Dict[str, int]) -> Any:
        ...

    def on_usage_total(self, totals: Dict[str, int]) -> Any:
        ...

    def on_token_progress(self, total_tokens: int, delta_tokens: int, delta_chars: int) -> Any:
        ...

    def on_error(self, error: Any) -> Any:
        ...

    def on_stream_end(self) -> Any:
        ...


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        reasoning_level: str = "medium",
        modalities: Optional[Set[str]] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            model: Model to use for completions
            api_key: Optional API key (defaults to env/config)
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.max_retries = max_retries
        self.reasoning_level = reasoning_level
        self.modalities = modalities or {"text"}
        self.logger = logging.getLogger("openai_client")
        self._token_encoder_cache: Dict[str, Any] = {}

        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.request_timeout = request_timeout or float(
            settings.openai_request_timeout_seconds
        )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_level: Optional[str] = None,
        modalities: Optional[Set[str]] = None,
        stream: bool = False,
        stream_observer: Optional["ResponseStreamObserver"] = None,
    ) -> Dict[str, Any]:
        """Make a call to the OpenAI API."""

        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        self.logger.debug(
            f"OpenAI API call: model={self.model}, "
            f"messages={len(final_messages)}, temperature={temperature}"
        )

        use_responses = self._should_use_responses_api(self.model)

        try:
            if use_responses:
                if stream or stream_observer is not None:
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
                    except Exception as stream_error:
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

            return await self._call_chat_completions(
                final_messages=final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                modalities=modalities,
            )

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
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
    ) -> Dict[str, Any]:
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
        response_schema: Dict[str, Any],
        temperature: float = 0.7,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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

        messages = []

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

        return response["content"]

    def estimate_cost(self, usage: Dict[str, int]) -> float:
        """
        Estimate cost based on token usage.

        Args:
            usage: Token usage dictionary

        Returns:
            Estimated cost in USD
        """
        # Pricing as of GPT-4o mini (adjust as needed)
        # These are example prices - update with actual pricing
        pricing = {
            "gpt-5": {
                "prompt": 1.25 / 1_000_000,
                "completion": 5.0 / 1_000_000,
            },
            "gpt-4.1": {
                "prompt": 0.75 / 1_000_000,
                "completion": 2.5 / 1_000_000,
            },
            "gpt-4.1-mini": {
                "prompt": 0.5 / 1_000_000,
                "completion": 1.2 / 1_000_000,
            },
            "gpt-4.1-mini-vision": {
                "prompt": 0.3 / 1_000_000,
                "completion": 0.9 / 1_000_000,
            },
            "gpt-4o-mini": {
                "prompt": 0.15 / 1_000_000,
                "completion": 0.6 / 1_000_000,
            },
            "gpt-4o": {
                "prompt": 5.0 / 1_000_000,
                "completion": 15.0 / 1_000_000,
            },
        }

        model_pricing = self._resolve_pricing(pricing)

        prompt_cost = usage.get("prompt_tokens", 0) * model_pricing["prompt"]
        completion_cost = (
            usage.get("completion_tokens", 0) * model_pricing["completion"]
        )

        return prompt_cost + completion_cost

    def _resolve_pricing(self, pricing: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Resolve pricing entry for the configured model."""
        if self.model in pricing:
            return pricing[self.model]

        # Attempt to match by family prefix (e.g., gpt-4.1-mini-high)
        for key in pricing:
            if self.model.startswith(key):
                return pricing[key]

        return pricing["gpt-4o-mini"]

    def _should_use_responses_api(self, model: str) -> bool:
        """Return True when the Responses API should be used."""
        return model.startswith("gpt-5") or model.startswith("gpt-4.1")

    def _supports_responses_temperature(self, model: str) -> bool:
        """Return True if the Responses model accepts temperature parameter."""
        # Reasoning models such as GPT-5 ignore temperature; omit it to avoid 400s.
        return not model.startswith("gpt-5")

    async def _call_chat_completions(
        self,
        final_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        modalities: Optional[Set[str]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": final_messages,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens

        if response_format:
            kwargs["response_format"] = response_format

        selected_modalities = modalities or self.modalities
        if selected_modalities:
            kwargs["modalities"] = sorted(selected_modalities)

        response = await self.client.chat.completions.create(
            timeout=self.request_timeout,
            **kwargs,
        )

        content = response.choices[0].message.content
        if response_format and response_format.get("type") == "json_object":
            try:
                content = json.loads(content)
            except json.JSONDecodeError as exc:
                self.logger.error(f"Failed to parse JSON response: {exc}")
                content = {"error": "Invalid JSON response", "raw": content}

        return {
            "content": content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }

    async def _call_responses_api(
        self,
        final_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        reasoning_level: Optional[str],
        system_prompt: Optional[str],
    ) -> Dict[str, Any]:
        instructions, input_items = self._prepare_responses_input(final_messages, system_prompt)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            kwargs["instructions"] = instructions

        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        text_config = self._map_response_format_to_text_config(response_format)
        if text_config:
            kwargs["text"] = text_config

        if reasoning_level:
            kwargs["reasoning"] = {"effort": reasoning_level}

        if self._supports_responses_temperature(self.model):
            kwargs["temperature"] = temperature

        response = await self.client.responses.create(
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
                except json.JSONDecodeError:
                    self.logger.error(
                        "Failed to parse JSON response",
                        exc_info=True,
                    )
                    raise
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
        final_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        reasoning_level: Optional[str],
        system_prompt: Optional[str],
        observer: Optional[ResponseStreamObserver],
    ) -> Dict[str, Any]:
        instructions, input_items = self._prepare_responses_input(final_messages, system_prompt)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            kwargs["instructions"] = instructions

        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens

        text_config = self._map_response_format_to_text_config(response_format)
        if text_config:
            kwargs["text"] = text_config

        if reasoning_level:
            kwargs["reasoning"] = {"effort": reasoning_level}

        if self._supports_responses_temperature(self.model):
            kwargs["temperature"] = temperature

        usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        last_reported_usage = dict(usage_totals)
        token_encoder = self._get_token_encoder(self.model)
        estimated_output_tokens = 0

        await self._dispatch_observer(observer, "on_stream_start")

        final_response: Any = None

        try:
            async with self.client.responses.stream(
                timeout=self.request_timeout,
                **kwargs,
            ) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        delta_text = self._resolve_text_delta(getattr(event, "delta", None))
                        if delta_text:
                            await self._dispatch_observer(observer, "on_text_delta", delta_text)
                            delta_chars = len(delta_text)
                            delta_tokens = self._estimate_token_count(delta_text, token_encoder)
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
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse JSON response", exc_info=True)
                    raise

        finish_reason = getattr(final_response, "status", None)

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": getattr(final_response, "model", self.model),
            "finish_reason": finish_reason,
        }

    def _prepare_responses_input(
        self,
        final_messages: Sequence[Dict[str, Any]],
        system_prompt: Optional[str],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        instructions = system_prompt or None
        input_items: List[Dict[str, Any]] = []

        for message in final_messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                text = content if isinstance(content, str) else str(content)
                if instructions:
                    if text.strip() and text.strip() != instructions.strip():
                        instructions = f"{instructions}\n{text}"
                else:
                    instructions = text
                continue

            content_items = self._convert_content_for_role(role, content)
            input_items.append({"role": role, "content": content_items})

        if not input_items:
            input_items.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})

        return instructions, input_items

    def _convert_content_for_role(self, role: str, content: Any) -> List[Dict[str, Any]]:
        content_items: List[Dict[str, Any]] = []
        default_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text", "output_text"}:
                        normalized = dict(item)
                        if normalized["type"] == "text":
                            normalized["type"] = default_type
                        elif role != "assistant" and normalized["type"] == "output_text":
                            normalized["type"] = default_type
                        elif role == "assistant" and normalized["type"] == "input_text":
                            normalized["type"] = default_type
                        content_items.append(normalized)
                    elif item_type in {"image_url", "input_image"}:
                        image_data = item.get("image_url")
                        detail = item.get("detail")

                        if isinstance(image_data, dict):
                            detail = detail or image_data.get("detail")
                            image_url = image_data.get("url") or image_data.get("image_url")
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
                        content_items.append({"type": default_type, "text": json.dumps(item)})
                else:
                    content_items.append({"type": default_type, "text": str(item)})
        else:
            content_items.append({"type": default_type, "text": str(content)})

        return content_items

    def _extract_output_text(self, response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        output_segments = getattr(response, "output", None)
        if not output_segments:
            return ""

        texts: List[str] = []
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

    def _get_token_encoder(self, model: str) -> Optional[Any]:
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

    def _estimate_token_count(self, text: str, encoder: Optional[Any]) -> int:
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
        self, response_format: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Translate legacy response_format into Responses API text config."""
        if not response_format:
            return None

        format_type = response_format.get("type")
        if format_type == "json_object":
            return {"format": {"type": "json_object"}}

        if format_type == "json_schema":
            schema = response_format.get("json_schema")
            if schema:
                return {"format": {"type": "json_schema", "json_schema": schema}}

        if format_type == "text":
            return {"format": {"type": "text"}}

        return None

    async def _dispatch_observer(
        self,
        observer: Optional[ResponseStreamObserver],
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

    def _extract_usage_from_event(self, event: Any) -> Dict[str, int]:
        response_obj = getattr(event, "response", None)
        if response_obj is None:
            return {}

        usage_obj = getattr(response_obj, "usage", None)
        return self._normalize_usage_dict(usage_obj)

    def _calculate_usage_delta(
        self, previous: Dict[str, int], current: Dict[str, int]
    ) -> Dict[str, int]:
        delta: Dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            new_value = int(current.get(key, previous.get(key, 0)) or 0)
            old_value = int(previous.get(key, 0) or 0)
            difference = new_value - old_value
            if difference > 0:
                delta[key] = difference
        return delta

    def _normalize_usage_dict(self, usage: Any) -> Dict[str, int]:
        payload: Dict[str, int] = {}
        if usage is None:
            return payload

        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value: Optional[int]
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
