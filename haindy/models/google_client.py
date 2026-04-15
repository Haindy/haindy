"""Google Gemini API client wrapper for non-CU agent calls."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from inspect import isawaitable
from typing import Any

from haindy.config.settings import get_settings
from haindy.models.errors import ModelCallError
from haindy.models.llm_client import dispatch_observer
from haindy.models.openai_client import ResponseStreamObserver
from haindy.models.structured_output import extract_json_schema_definition

logger = logging.getLogger("google_client")

_DATA_URL_PATTERN = re.compile(
    r"^data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\n\r]+)$",
    re.DOTALL,
)

genai: Any | None
genai_types: Any | None

try:
    import google.genai as genai  # type: ignore[no-redef]
    import google.genai.types as genai_types  # type: ignore[no-redef]
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

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        """Collapse mixed multimodal content into plain text for logging/system text."""
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
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type in {"text", "input_text", "output_text"}:
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            parts.append(text)
                            continue
                    if item_type in {"image", "image_url", "input_image"}:
                        parts.append("<<attached image>>")
                        continue
                    image_url = item.get("image_url")
                    if isinstance(image_url, (str, dict)):
                        parts.append("<<attached image>>")
                        continue
                    if "text" in item and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                        continue
                parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _make_text_part(self, text: str) -> Any:
        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )
        from_text = getattr(genai_types.Part, "from_text", None)
        if callable(from_text):
            return from_text(text=text)
        return genai_types.Part(text=text)

    def _make_bytes_part(self, data: bytes, mime_type: str) -> Any:
        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )
        from_bytes = getattr(genai_types.Part, "from_bytes", None)
        if callable(from_bytes):
            return from_bytes(data=data, mime_type=mime_type)
        inline_data = getattr(genai_types, "Blob", None)
        if inline_data is None:
            raise RuntimeError(
                "google-genai Part.from_bytes is unavailable in this SDK build."
            )
        return genai_types.Part(inline_data=inline_data(data=data, mime_type=mime_type))

    def _make_uri_part(self, file_uri: str, mime_type: str | None = None) -> Any:
        if genai_types is None:
            raise RuntimeError(
                "google-genai is required for the Google provider. "
                'Install it with `pip install "google-genai"`.'
            )
        from_uri = getattr(genai_types.Part, "from_uri", None)
        if callable(from_uri):
            return from_uri(file_uri=file_uri, mime_type=mime_type)
        file_data = getattr(genai_types, "FileData", None)
        if file_data is None:
            raise RuntimeError(
                "google-genai Part.from_uri is unavailable in this SDK build."
            )
        return genai_types.Part(
            file_data=file_data(file_uri=file_uri, mime_type=mime_type)
        )

    def _normalize_image_reference(
        self,
        reference: Any,
        *,
        mime_type: str | None = None,
    ) -> Any:
        if isinstance(reference, dict):
            source = reference.get("source")
            if isinstance(source, dict):
                source_type = str(source.get("type") or "").strip().lower()
                if source_type == "base64":
                    return self._normalize_image_reference(
                        {
                            "data": source.get("data"),
                            "mime_type": source.get("media_type")
                            or source.get("mime_type")
                            or mime_type,
                        }
                    )
                if source_type in {"file", "uri", "url"}:
                    return self._normalize_image_reference(
                        source.get("file_uri")
                        or source.get("uri")
                        or source.get("url"),
                        mime_type=source.get("mime_type")
                        or source.get("media_type")
                        or mime_type,
                    )

            nested_reference = (
                reference.get("image_url")
                or reference.get("url")
                or reference.get("uri")
                or reference.get("file_uri")
            )
            if nested_reference is not None:
                return self._normalize_image_reference(
                    nested_reference,
                    mime_type=reference.get("mime_type")
                    or reference.get("media_type")
                    or mime_type,
                )

            raw_data = reference.get("data")
            resolved_mime = str(
                reference.get("mime_type")
                or reference.get("media_type")
                or mime_type
                or ""
            ).strip()
            if raw_data is not None:
                if not resolved_mime.startswith("image/"):
                    raise self._invalid_request_error(
                        "Google image bytes require an image/* mime_type.",
                        payload={"image_reference": "<bytes>"},
                    )
                if isinstance(raw_data, str):
                    try:
                        raw_bytes = base64.b64decode(raw_data, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise self._invalid_request_error(
                            "Google image base64 inputs must contain valid base64 data.",
                            payload={"image_reference": "<base64>"},
                        ) from exc
                    return self._make_bytes_part(raw_bytes, resolved_mime)
                if isinstance(raw_data, (bytes, bytearray, memoryview)):
                    return self._make_bytes_part(bytes(raw_data), resolved_mime)

        if isinstance(reference, (bytes, bytearray, memoryview)):
            resolved_mime = str(mime_type or "").strip()
            if not resolved_mime.startswith("image/"):
                raise self._invalid_request_error(
                    "Google inline image bytes require an image/* mime_type.",
                    payload={"image_reference": "<bytes>"},
                )
            return self._make_bytes_part(bytes(reference), resolved_mime)

        if not isinstance(reference, str):
            raise self._invalid_request_error(
                "Unsupported Google image reference.",
                payload={"image_reference": reference},
            )

        normalized = reference.strip()
        data_url_match = _DATA_URL_PATTERN.match(normalized)
        if data_url_match:
            try:
                image_bytes = base64.b64decode(
                    data_url_match.group("data"), validate=True
                )
            except (ValueError, binascii.Error) as exc:
                raise self._invalid_request_error(
                    "Google image data URLs must contain valid base64 data.",
                    payload={"image_reference": "<data-url>"},
                ) from exc
            return self._make_bytes_part(image_bytes, data_url_match.group("mime"))

        if normalized.startswith(("http://", "https://", "file://", "gs://")):
            return self._make_uri_part(normalized, mime_type)

        raise self._invalid_request_error(
            "Google image inputs must be image data URLs, image bytes, or URI-based file references.",
            payload={"image_reference": reference},
        )

    def _normalize_content_item(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._make_text_part(item)
        if not isinstance(item, dict):
            return self._make_text_part(str(item))

        item_type = str(item.get("type") or "").strip().lower()
        if item_type in {"text", "input_text", "output_text"}:
            return self._make_text_part(str(item.get("text") or ""))

        if item_type in {"image", "image_url", "input_image"} or "image_url" in item:
            image_reference = item.get("image_url")
            if image_reference is None:
                image_reference = (
                    item.get("url") or item.get("uri") or item.get("file_uri")
                )
            return self._normalize_image_reference(
                image_reference,
                mime_type=item.get("mime_type") or item.get("media_type"),
            )

        if item_type == "" and "text" in item:
            return self._make_text_part(str(item.get("text") or ""))

        raise self._invalid_request_error(
            "Unsupported non-CU content block for Google.",
            payload={"content_item": item},
        )

    def _normalize_message_content(self, content: Any) -> list[Any]:
        # Gemini expects typed Part objects for multimodal inputs. If we coerce the
        # mixed content list to str(), inline image data becomes plain text and can
        # explode the token count.
        if isinstance(content, list):
            parts = [self._normalize_content_item(item) for item in content]
            return parts or [self._make_text_part("")]
        if isinstance(content, dict):
            return [self._normalize_content_item(content)]
        return [
            self._make_text_part(content if isinstance(content, str) else str(content))
        ]

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
                text = self._flatten_message_content(content)
                system_instruction = self._append_distinct_instruction(
                    system_instruction, text
                )
                continue

            google_role = "model" if role == "assistant" else "user"
            google_contents.append(
                genai_types.Content(
                    role=google_role,
                    parts=self._normalize_message_content(content),
                )
            )

        if not google_contents:
            google_contents.append(
                genai_types.Content(
                    role="user",
                    parts=[self._make_text_part("")],
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
        if format_type == "json_schema":
            schema_def = extract_json_schema_definition(response_format)
            if schema_def and schema_def.get("schema"):
                config_kwargs["response_schema"] = schema_def["schema"]

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
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse JSON response from Google")
                raise ModelCallError(
                    "Failed to parse JSON response from Google.",
                    failure_kind="response_parse_error",
                    response_payload={
                        "provider_response": response,
                        "content_text": content_text,
                    },
                ) from exc

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
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse streaming JSON response from Google")
                raise ModelCallError(
                    "Failed to parse streaming JSON response from Google.",
                    failure_kind="response_parse_error",
                    response_payload={
                        "content_text": full_text,
                        "usage": usage_dict,
                    },
                ) from exc

        return {
            "content": content_value,
            "usage": usage_dict,
            "model": self._model,
            "finish_reason": None,
        }
