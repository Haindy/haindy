"""Google Generative AI client wrapper for non-CU agent calls."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import google.genai as _genai
except Exception:  # pragma: no cover
    _genai = None  # type: ignore[assignment]

from haindy.models.openai_client import ResponseStreamObserver

logger = logging.getLogger("google_client")


class GoogleClient:
    """Wrapper for Google Generative AI API interactions used by non-CU agents."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro-preview-05-06",
        api_key: str | None = None,
    ) -> None:
        if _genai is None:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )
        self.model = model
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        from haindy.config.settings import get_settings

        settings = get_settings()
        vertex_project = str(settings.vertex_project or "").strip()
        vertex_location = str(settings.vertex_location or "").strip()
        vertex_api_key = self._api_key or str(settings.vertex_api_key or "").strip()

        if vertex_project:
            self._client = _genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location or "us-central1",
            )
            logger.info(
                "Initialized Google agent client in Vertex mode",
                extra={"vertex_project": vertex_project, "vertex_location": vertex_location},
            )
        elif vertex_api_key:
            self._client = _genai.Client(api_key=vertex_api_key)
            logger.debug("Initialized Google agent client in API key mode")
        else:
            raise RuntimeError(
                "Google provider requires either "
                "HAINDY_VERTEX_PROJECT+HAINDY_VERTEX_LOCATION or "
                "HAINDY_VERTEX_API_KEY."
            )

        return self._client

    def _build_contents(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
    ) -> tuple[str | None, list[Any]]:
        """Convert OpenAI-style messages to Google contents format."""
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)

        contents: list[Any] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in {"system", "developer"}:
                system_parts.append(
                    content if isinstance(content, str) else str(content)
                )
                continue
            # Google uses "model" for assistant role
            google_role = "model" if role == "assistant" else "user"
            text = content if isinstance(content, str) else str(content)
            contents.append({"role": google_role, "parts": [{"text": text}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        combined_system = "\n\n".join(system_parts) if system_parts else None
        return combined_system, contents

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
        """Make a call to the Google Generative AI API."""
        format_type = response_format.get("type") if response_format else None

        combined_system, contents = self._build_contents(messages, system_prompt)

        # Handle JSON mode.
        if format_type in {"json_object", "json_schema"}:
            json_hint = "Respond with valid JSON."
            if combined_system:
                if "json" not in combined_system.lower():
                    combined_system = combined_system + "\n\n" + json_hint
            else:
                combined_system = json_hint

        from google.genai import types as _types  # type: ignore[import-untyped]

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if combined_system:
            config_kwargs["system_instruction"] = combined_system

        generation_config = _types.GenerateContentConfig(**config_kwargs)

        client = self._get_client()

        logger.debug(
            "Google API call: model=%s, messages=%d, temperature=%s",
            self.model,
            len(contents),
            temperature,
        )

        if stream or stream_observer is not None:
            return await self._call_streaming(
                client=client,
                contents=contents,
                generation_config=generation_config,
                format_type=format_type,
                stream_observer=stream_observer,
            )

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=generation_config,
        )
        return self._normalize_response(response, format_type)

    async def _call_streaming(
        self,
        *,
        client: Any,
        contents: list[Any],
        generation_config: Any,
        format_type: str | None,
        stream_observer: ResponseStreamObserver | None,
    ) -> dict[str, Any]:
        """Stream a response and gather the final result."""
        if stream_observer is not None:
            try:
                stream_observer.on_stream_start()
            except Exception:
                pass

        full_text = ""
        usage_in: int = 0
        usage_out: int = 0

        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generation_config,
        ):
            chunk_text = ""
            for candidate in getattr(chunk, "candidates", []) or []:
                for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        chunk_text += t

            if chunk_text:
                full_text += chunk_text
                if stream_observer is not None:
                    try:
                        stream_observer.on_text_delta(chunk_text)
                    except Exception:
                        pass

            usage = getattr(chunk, "usage_metadata", None)
            if usage is not None:
                usage_in = int(getattr(usage, "prompt_token_count", usage_in) or usage_in)
                usage_out = int(
                    getattr(usage, "candidates_token_count", usage_out) or usage_out
                )

        if stream_observer is not None:
            try:
                stream_observer.on_stream_end()
            except Exception:
                pass

        content_value: Any = full_text
        if format_type in {"json_object", "json_schema"}:
            if not full_text:
                content_value = {}
            else:
                try:
                    content_value = json.loads(full_text)
                except json.JSONDecodeError:
                    logger.error(
                        "Failed to parse JSON streaming response", exc_info=True
                    )
                    raise

        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": usage_in,
                "completion_tokens": usage_out,
                "total_tokens": usage_in + usage_out,
            },
            "model": self.model,
            "finish_reason": None,
        }

    def _normalize_response(
        self, response: Any, format_type: str | None
    ) -> dict[str, Any]:
        """Extract text and usage from a Google GenerateContent response."""
        text_parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    text_parts.append(t)

        raw_text = "".join(text_parts)
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

        usage = getattr(response, "usage_metadata", None)
        usage_in = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        usage_out = (
            int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        )

        return {
            "content": content_value,
            "usage": {
                "prompt_tokens": usage_in,
                "completion_tokens": usage_out,
                "total_tokens": usage_in + usage_out,
            },
            "model": self.model,
            "finish_reason": None,
        }
