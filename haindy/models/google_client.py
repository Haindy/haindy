"""Google Gemini API client wrapper for non-CU agent calls."""
from __future__ import annotations

import json
import logging
from typing import Any

from haindy.config.settings import get_settings
from haindy.models.llm_client import dispatch_observer
from haindy.models.openai_client import ResponseStreamObserver

logger = logging.getLogger("google_client")


class GoogleClient:
    """Wrapper for Google Gemini API interactions (non-computer-use)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.vertex_api_key
        self._model = settings.google_model
        self._vertex_project = getattr(settings, "vertex_project", "") or ""
        self._vertex_location = (
            getattr(settings, "vertex_location", "us-central1") or "us-central1"
        )
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import google.genai as genai

            if self._vertex_project:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._vertex_project,
                    location=self._vertex_location,
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
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
        """Make a call to the Google Gemini API."""
        import google.genai.types as genai_types

        system_instruction: str | None = system_prompt
        google_contents: list[Any] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text = content if isinstance(content, str) else str(content)
                if system_instruction:
                    system_instruction = f"{system_instruction}\n\n{text}"
                else:
                    system_instruction = text
                continue

            google_role = "model" if role == "assistant" else "user"
            text_content = content if isinstance(content, str) else str(content)
            google_contents.append(
                genai_types.Content(
                    role=google_role,
                    parts=[genai_types.Part(text=text_content)],
                )
            )

        if not google_contents:
            google_contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="")],
                )
            )

        format_type = response_format.get("type") if response_format else None
        if format_type in {"json_object", "json_schema"}:
            json_instruction = "Respond with valid JSON only."
            if system_instruction:
                system_instruction = f"{system_instruction}\n\n{json_instruction}"
            else:
                system_instruction = json_instruction

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

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
            content_text = "".join(
                getattr(part, "text", "") or "" for part in parts
            )

        content_value: Any = content_text
        if format_type in {"json_object", "json_schema"} and content_text:
            try:
                content_value = json.loads(content_text)
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
            "finish_reason": None,
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
            dispatch_observer(stream_observer,"on_stream_start")

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for chunk in await client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            ):
                chunk_text = ""
                if chunk.candidates:
                    parts = getattr(chunk.candidates[0].content, "parts", None) or []
                    chunk_text = "".join(
                        getattr(part, "text", "") or "" for part in parts
                    )
                if chunk_text:
                    full_text += chunk_text
                    if stream_observer is not None:
                        dispatch_observer(stream_observer,"on_text_delta", chunk_text)

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
                dispatch_observer(stream_observer,"on_error", exc)
            if stream_observer is not None:
                dispatch_observer(stream_observer,"on_stream_end")
            raise

        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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
                logger.error("Failed to parse streaming JSON response from Google")
                raise

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": self._model,
            "finish_reason": None,
        }

