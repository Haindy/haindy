"""Google Gemini API client wrapper for the HAINDY framework."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

try:
    import google.genai as genai
    from google.genai import types as genai_types
except Exception:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

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


class GoogleClient:
    """Wrapper for Google Gemini API interactions via Vertex AI backend."""

    _DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self.model = (
            model or getattr(settings, "google_model", None) or "gemini-3.1-pro-preview"
        )
        self.logger = logging.getLogger("google_client")
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if genai is None:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )

        settings = self._settings
        vertex_project = str(getattr(settings, "vertex_project", "") or "").strip()
        vertex_location = str(
            getattr(settings, "vertex_location", "us-central1") or ""
        ).strip()
        vertex_api_key = str(getattr(settings, "vertex_api_key", "") or "").strip()

        if vertex_project:
            if not vertex_location:
                raise RuntimeError(
                    "HAINDY_VERTEX_LOCATION is required when HAINDY_VERTEX_PROJECT is configured."
                )
            if vertex_api_key:
                self.logger.warning(
                    "Ignoring HAINDY_VERTEX_API_KEY because HAINDY_VERTEX_PROJECT is "
                    "configured; using Vertex project/location mode."
                )
            self._client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location,
            )
            self.logger.info(
                "Initialized Google client in Vertex mode (project=%s, location=%s)",
                vertex_project,
                vertex_location,
            )
            return self._client

        if not vertex_api_key:
            raise RuntimeError(
                "Google provider requires either "
                "HAINDY_VERTEX_PROJECT+HAINDY_VERTEX_LOCATION or HAINDY_VERTEX_API_KEY."
            )

        self._client = genai.Client(api_key=vertex_api_key)
        self.logger.debug("Initialized Google client in API key mode")
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
        """Make a call to the Google Gemini API.

        Args:
            messages: Conversation messages with role/content keys.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens (defaults to 8192).
            system_prompt: System instructions.
            response_format: If {"type": "json_object"}, requests JSON output.
            reasoning_level: Ignored (no native equivalent).
            modalities: Ignored.
            stream: Whether to stream the response.
            stream_observer: Optional observer for streaming events.

        Returns:
            Dict with keys: content, usage, model, finish_reason.
        """
        if genai is None or genai_types is None:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )

        effective_max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS
        json_mode = response_format is not None and response_format.get("type") in {
            "json_object",
            "json_schema",
        }

        client = self._get_client()
        contents, system_instruction = self._build_contents(messages, system_prompt)

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = genai_types.GenerateContentConfig(**config_kwargs)

        self.logger.debug(
            "Google Gemini API call: model=%s, contents=%d, temperature=%s",
            self.model,
            len(contents),
            temperature,
        )

        if stream or stream_observer is not None:
            return await self._call_streaming(
                client=client,
                contents=contents,
                config=config,
                observer=stream_observer,
                json_mode=json_mode,
            )

        return await self._call_blocking(
            client=client,
            contents=contents,
            config=config,
            json_mode=json_mode,
        )

    async def _call_blocking(
        self,
        client: Any,
        contents: list[Any],
        config: Any,
        json_mode: bool,
    ) -> dict[str, Any]:
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        content_text = self._extract_text(response)
        content_value: Any = content_text
        if json_mode and content_text:
            try:
                content_value = json.loads(content_text)
            except json.JSONDecodeError:
                self.logger.error(
                    "Failed to parse JSON response from Google", exc_info=True
                )
                raise

        return {
            "content": content_value,
            "usage": self._normalize_usage(response),
            "model": self.model,
            "finish_reason": self._extract_finish_reason(response),
        }

    async def _call_streaming(
        self,
        client: Any,
        contents: list[Any],
        config: Any,
        observer: ResponseStreamObserver | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        await _dispatch_observer(observer, "on_stream_start", logger=self.logger)

        collected_text = ""
        last_response: Any = None

        try:
            async for chunk in client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                last_response = chunk
                delta = self._extract_text(chunk)
                if delta:
                    collected_text += delta
                    await _dispatch_observer(
                        observer, "on_text_delta", delta, logger=self.logger
                    )
        except Exception as error:
            await _dispatch_observer(observer, "on_error", error, logger=self.logger)
            await _dispatch_observer(observer, "on_stream_end", logger=self.logger)
            raise

        usage_dict = self._normalize_usage(last_response)
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
                    "Failed to parse JSON response from Google stream", exc_info=True
                )
                raise

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": self.model,
            "finish_reason": self._extract_finish_reason(last_response),
        }

    def _build_contents(
        self, messages: list[dict[str, Any]], system_prompt: str | None
    ) -> tuple[list[Any], str | None]:
        """Convert messages to Google genai Content objects.

        System messages are extracted and returned separately as system_instruction.
        User/assistant messages are converted to Content objects.
        """
        system_parts: list[str] = []
        if system_prompt and system_prompt.strip():
            system_parts.append(system_prompt.strip())

        content_items: list[Any] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text = _flatten_message_content(content)

            if role in {"system", "developer"}:
                if text.strip():
                    system_parts.append(text.strip())
                continue

            google_role = "model" if role == "assistant" else "user"
            content_items.append(
                genai_types.Content(
                    role=google_role,
                    parts=[genai_types.Part.from_text(text=text)],
                )
            )

        if not content_items:
            content_items.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text="")],
                )
            )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return content_items, system_instruction

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        texts: list[str] = []
        for candidate in candidates:
            content_obj = getattr(candidate, "content", None)
            if content_obj is None:
                continue
            parts = getattr(content_obj, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    texts.append(part_text)
        return "\n".join(texts)

    def _extract_finish_reason(self, response: Any) -> str | None:
        if response is None:
            return None
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            if reason is not None:
                return str(reason)
        return None

    def _normalize_usage(self, response: Any) -> dict[str, int]:
        if response is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = _get_usage_int(usage, "prompt_token_count")
        completion = _get_usage_int(usage, "candidates_token_count")
        total = _get_usage_int(usage, "total_token_count") or (prompt + completion)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
