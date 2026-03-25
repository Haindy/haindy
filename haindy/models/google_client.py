"""Google Gemini API client wrapper for non-CU agent calls."""

from __future__ import annotations

import json
import logging
from inspect import isawaitable
from typing import Any

from haindy.config.settings import get_settings
from haindy.models.llm_client import dispatch_observer
from haindy.models.openai_client import ResponseStreamObserver

logger = logging.getLogger("google_client")

genai: Any | None
genai_types: Any | None

try:
    import google.genai as genai
    import google.genai.types as genai_types
except ImportError:  # pragma: no cover - exercised in tests via monkeypatching
    genai = None
    genai_types = None


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from a string, e.g. ```json\\n{...}\\n```."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (e.g. ```json or ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


class GoogleClient:
    """Wrapper for Google Gemini API interactions (non-computer-use)."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.vertex_api_key
        self._model = model or settings.google_model or "gemini-3-flash-preview"
        self.model = self._model
        self._vertex_project = getattr(settings, "vertex_project", "") or ""
        self._vertex_location = (
            getattr(settings, "vertex_location", "us-central1") or "us-central1"
        )
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if genai is None:
                raise RuntimeError(
                    "google-genai is required for the Google provider. "
                    'Install it with `pip install "google-genai"`.'
                )

            if self._vertex_project:
                if not self._vertex_location:
                    raise RuntimeError(
                        "HAINDY_VERTEX_LOCATION is required when using Vertex AI."
                    )
                self._client = genai.Client(
                    vertexai=True,
                    project=self._vertex_project,
                    location=self._vertex_location,
                )
            else:
                if not self._api_key:
                    raise RuntimeError(
                        "HAINDY_VERTEX_API_KEY is required when using the Google provider without Vertex AI."
                    )
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _append_distinct_instruction(
        self, current: str | None, text: str | None
    ) -> str | None:
        """Append a system instruction fragment only when it adds new content."""
        normalized = str(text or "").strip()
        if not normalized:
            return current
        if current and normalized == current.strip():
            return current
        if current:
            return f"{current}\n\n{normalized}"
        return normalized

    def _make_part(self, text: str) -> Any:
        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )
        from_text = getattr(genai_types.Part, "from_text", None)
        if callable(from_text):
            return from_text(text=text)
        return genai_types.Part(text=text)

    def _build_contents(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
    ) -> tuple[list[Any], str | None]:
        """Convert HAINDY messages into Gemini content blocks."""
        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )

        system_instruction: str | None = system_prompt or None
        google_contents: list[Any] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text = content if isinstance(content, str) else str(content)
                system_instruction = self._append_distinct_instruction(
                    system_instruction, text
                )
                continue

            google_role = "model" if role == "assistant" else "user"
            text_content = content if isinstance(content, str) else str(content)
            google_contents.append(
                genai_types.Content(
                    role=google_role,
                    parts=[self._make_part(text_content)],
                )
            )

        if not google_contents:
            google_contents.append(
                genai_types.Content(
                    role="user",
                    parts=[self._make_part("")],
                )
            )

        return google_contents, system_instruction

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
        """Make a call to the Google Gemini API."""
        google_contents, system_instruction = self._build_contents(
            messages,
            system_prompt=system_prompt,
        )

        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"}:
            system_instruction = self._append_distinct_instruction(
                system_instruction, "Respond with valid JSON only."
            )

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if format_type in {"json_object", "json_schema"}:
            config_kwargs["response_mime_type"] = "application/json"

        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )
        generation_config = genai_types.GenerateContentConfig(**config_kwargs)
        client = self._get_client()

        if stream or stream_observer is not None:
            return await self._call_streaming(
                client=client,
                contents=google_contents,
                config=generation_config,
                response_format=response_format,
                stream_observer=stream_observer,
            )

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=google_contents,
            config=generation_config,
        )

        content_text = ""
        if response.candidates:
            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", None) or []
            content_text = "".join(getattr(part, "text", "") or "" for part in parts)
            finish_reason = getattr(candidate, "finish_reason", None)
        else:
            finish_reason = None

        content_value: Any = content_text
        if format_type in {"json_object", "json_schema"} and content_text:
            try:
                content_value = json.loads(_strip_markdown_fences(content_text))
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response from Google")
                raise

        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": self._model,
            "finish_reason": finish_reason,
        }

    async def _call_streaming(
        self,
        client: Any,
        contents: list[Any],
        config: Any,
        response_format: dict[str, Any] | None,
        stream_observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]:
        """Make a streaming call to the Google Gemini API."""
        if stream_observer is not None:
            dispatch_observer(stream_observer, "on_stream_start")

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream_result = client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            if isawaitable(stream_result):
                stream_result = await stream_result

            async for chunk in stream_result:
                chunk_text = ""
                if chunk.candidates:
                    parts = getattr(chunk.candidates[0].content, "parts", None) or []
                    chunk_text = "".join(
                        getattr(part, "text", "") or "" for part in parts
                    )
                if chunk_text:
                    full_text += chunk_text
                    if stream_observer is not None:
                        dispatch_observer(stream_observer, "on_text_delta", chunk_text)

                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta is not None:
                    pt = getattr(usage_meta, "prompt_token_count", 0) or 0
                    ct = getattr(usage_meta, "candidates_token_count", 0) or 0
                    if pt:
                        prompt_tokens = pt
                    if ct:
                        completion_tokens = ct
        except Exception as exc:
            if stream_observer is not None:
                dispatch_observer(stream_observer, "on_error", exc)
            if stream_observer is not None:
                dispatch_observer(stream_observer, "on_stream_end")
            raise

        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if stream_observer is not None:
            dispatch_observer(stream_observer, "on_usage_total", usage_dict)
            dispatch_observer(stream_observer, "on_stream_end")

        format_type = response_format.get("type") if response_format else None
        content_value: Any = full_text
        if format_type in {"json_object", "json_schema"} and full_text:
            try:
                content_value = json.loads(_strip_markdown_fences(full_text))
            except json.JSONDecodeError:
                logger.error("Failed to parse streaming JSON response from Google")
                raise

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": self._model,
            "finish_reason": None,
        }
