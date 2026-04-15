"""Tests for GoogleClient."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_settings(
    vertex_project: str = "",
    vertex_location: str = "us-central1",
    vertex_api_key: str = "test-vertex-key",
    google_model: str | None = None,
) -> Any:
    settings = MagicMock()
    settings.vertex_project = vertex_project
    settings.vertex_location = vertex_location
    settings.vertex_api_key = vertex_api_key
    if google_model is not None:
        type(settings).google_model = property(lambda self: google_model)
    else:
        type(settings).google_model = property(lambda self: None)
    return settings


@pytest.fixture()
def patched_settings() -> Generator[Any, None, None]:
    settings = _make_settings()
    with patch("haindy.models.google_client.get_settings", return_value=settings):
        yield settings


def _make_genai_response(text: str = "Hello") -> Any:
    part = SimpleNamespace(text=text)
    content_obj = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content_obj, finish_reason="STOP")
    usage = SimpleNamespace(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage,
        text=text,
    )


class TestGoogleClientInit:
    def test_default_model_falls_back_to_hardcoded(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        client = GoogleClient()
        assert client.model == "gemini-3-flash-preview"

    def test_custom_model_is_stored(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        client = GoogleClient(model="gemini-2.0-flash")
        assert client.model == "gemini-2.0-flash"


class TestGoogleClientGetClient:
    def test_raises_when_genai_not_installed(self, patched_settings: Any) -> None:
        from haindy.models import google_client as gc_module

        original = gc_module.genai
        try:
            gc_module.genai = None  # type: ignore[assignment]
            from haindy.models.google_client import GoogleClient

            client = GoogleClient()
            with pytest.raises(RuntimeError, match="google-genai"):
                client._get_client()
        finally:
            gc_module.genai = original

    def test_api_key_mode_when_no_project(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        with patch("haindy.models.google_client.genai") as mock_genai:
            mock_genai.Client = MagicMock(return_value=MagicMock())
            client = GoogleClient()
            result = client._get_client()

        mock_genai.Client.assert_called_once_with(api_key="test-vertex-key")
        assert result is not None

    def test_vertex_mode_when_project_configured(self) -> None:
        settings = _make_settings(
            vertex_project="my-project",
            vertex_location="us-east1",
            vertex_api_key="",
        )
        with patch("haindy.models.google_client.get_settings", return_value=settings):
            from haindy.models.google_client import GoogleClient

            with patch("haindy.models.google_client.genai") as mock_genai:
                mock_genai.Client = MagicMock(return_value=MagicMock())
                client = GoogleClient()
                client._get_client()

        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="my-project",
            location="us-east1",
        )

    def test_raises_when_no_key_and_no_project(self) -> None:
        settings = _make_settings(
            vertex_project="",
            vertex_location="us-central1",
            vertex_api_key="",
        )
        with patch("haindy.models.google_client.get_settings", return_value=settings):
            from haindy.models.google_client import GoogleClient

            with patch("haindy.models.google_client.genai") as mock_genai:
                mock_genai.Client = MagicMock()
                client = GoogleClient()
                with pytest.raises(RuntimeError, match="HAINDY_VERTEX"):
                    client._get_client()

    def test_vertex_mode_missing_location_falls_back_to_default(self) -> None:
        settings = _make_settings(
            vertex_project="my-project",
            vertex_location="",
            vertex_api_key="",
        )
        with patch("haindy.models.google_client.get_settings", return_value=settings):
            from haindy.models.google_client import GoogleClient

            with patch("haindy.models.google_client.genai") as mock_genai:
                mock_genai.Client = MagicMock(return_value=MagicMock())
                client = GoogleClient()
                client._get_client()

        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="my-project",
            location="us-central1",
        )


class TestGoogleClientCall:
    @pytest.mark.asyncio
    async def test_basic_call_returns_normalized_dict(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response("world")

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="hi")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            result = await client.call(messages=[{"role": "user", "content": "Hello"}])

        assert result["content"] == "world"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["finish_reason"] == "STOP"

    @pytest.mark.asyncio
    async def test_json_mode_parses_response(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response('{"key": "value"}')

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            result = await client.call(
                messages=[{"role": "user", "content": "give json"}],
                response_format={"type": "json_object"},
            )

        assert result["content"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_json_schema_mode_forwards_response_schema(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response('{"decision": "continue"}')
        captured_config: list[Any] = []

        def _capture_config(**kwargs: Any) -> Any:
            captured_config.append(kwargs)
            return MagicMock()

        schema = {
            "type": "object",
            "properties": {"decision": {"type": "string"}},
            "required": ["decision"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test_schema", "schema": schema, "strict": True},
        }

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(side_effect=_capture_config)
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            result = await client.call(
                messages=[{"role": "user", "content": "go"}],
                response_format=response_format,
            )

        assert result["content"] == {"decision": "continue"}
        assert captured_config
        config_kwargs = captured_config[0]
        assert config_kwargs.get("response_mime_type") == "application/json"
        assert config_kwargs.get("response_json_schema") == schema

    @pytest.mark.asyncio
    async def test_json_object_mode_does_not_forward_response_schema(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response('{"key": "value"}')
        captured_config: list[Any] = []

        def _capture_config(**kwargs: Any) -> Any:
            captured_config.append(kwargs)
            return MagicMock()

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(side_effect=_capture_config)
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            await client.call(
                messages=[{"role": "user", "content": "go"}],
                response_format={"type": "json_object"},
            )

        assert captured_config
        config_kwargs = captured_config[0]
        assert config_kwargs.get("response_mime_type") == "application/json"
        assert "response_json_schema" not in config_kwargs

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_config(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response("ok")
        captured_config: list[Any] = []

        def _capture_config(**kwargs: Any) -> Any:
            captured_config.append(kwargs)
            return MagicMock()

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(side_effect=_capture_config)
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            await client.call(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="Be helpful.",
            )

        assert captured_config
        config_kwargs = captured_config[0]
        assert config_kwargs.get("system_instruction") == "Be helpful."

    @pytest.mark.asyncio
    async def test_reasoning_level_silently_ignored(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        response = _make_genai_response("ok")

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=response)
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            result = await client.call(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_level="high",
            )

        assert result["content"] == "ok"

    @pytest.mark.asyncio
    async def test_streaming_dispatches_observer(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        chunk1 = _make_genai_response("Hello ")
        chunk2 = _make_genai_response("World")

        started = []
        ended = []
        deltas: list[str] = []

        class Obs:
            def on_stream_start(self) -> None:
                started.append(True)

            def on_stream_end(self) -> None:
                ended.append(True)

            def on_text_delta(self, delta: str) -> None:
                deltas.append(delta)

            def on_usage_total(self, totals: dict) -> None:
                pass

            def on_usage_delta(self, delta: dict) -> None:
                pass

            def on_token_progress(
                self, total: int, delta_tokens: int, delta_chars: int
            ) -> None:
                pass

            def on_error(self, error: Any) -> None:
                pass

        async def _fake_stream(**kwargs: Any) -> AsyncGenerator[Any, None]:
            yield chunk1
            yield chunk2

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content_stream = _fake_stream
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            obs = Obs()
            result = await client.call(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_observer=obs,
            )

        assert started == [True]
        assert ended == [True]
        assert "Hello " in deltas or "World" in deltas
        assert result["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_streaming_accepts_awaitable_stream(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        chunk = _make_genai_response("Hello")

        async def _stream() -> AsyncGenerator[Any, None]:
            yield chunk

        async def _fake_stream(**kwargs: Any) -> AsyncGenerator[Any, None]:
            return _stream()

        with (
            patch("haindy.models.google_client.genai") as mock_genai,
            patch("haindy.models.google_client.genai_types") as mock_types,
        ):
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content_stream = _fake_stream
            mock_genai.Client = MagicMock(return_value=mock_client)

            client = GoogleClient()
            result = await client.call(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )

        assert result["content"] == "Hello"


class TestGoogleClientBuildContents:
    def test_system_messages_extracted(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        with patch("haindy.models.google_client.genai_types") as mock_types:
            contents_created: list[Any] = []

            def _make_content(role: str, parts: Any) -> Any:
                obj = SimpleNamespace(role=role, parts=parts)
                contents_created.append(obj)
                return obj

            mock_types.Content = MagicMock(side_effect=_make_content)
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )

            client = GoogleClient()
            _, system_instruction = client._build_contents(
                [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                system_prompt=None,
            )

        assert system_instruction == "Be helpful."
        # Only the user message should create a Content object
        assert len(contents_created) == 1
        assert contents_created[0].role == "user"

    def test_system_prompt_and_system_message_combined(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        with patch("haindy.models.google_client.genai_types") as mock_types:
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )

            client = GoogleClient()
            _, system_instruction = client._build_contents(
                [{"role": "system", "content": "Extra."}],
                system_prompt="Base prompt.",
            )

        assert system_instruction is not None
        assert "Base prompt." in system_instruction
        assert "Extra." in system_instruction

    def test_duplicate_system_prompt_is_deduped(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        with patch("haindy.models.google_client.genai_types") as mock_types:
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )

            client = GoogleClient()
            _, system_instruction = client._build_contents(
                [{"role": "system", "content": "Base prompt."}],
                system_prompt="Base prompt.",
            )

        assert system_instruction == "Base prompt."

    def test_assistant_role_mapped_to_model(self, patched_settings: Any) -> None:
        from haindy.models.google_client import GoogleClient

        roles_created: list[str] = []

        with patch("haindy.models.google_client.genai_types") as mock_types:

            def _capture(role: str, parts: Any) -> Any:
                roles_created.append(role)
                return SimpleNamespace(role=role, parts=parts)

            mock_types.Content = MagicMock(side_effect=_capture)
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(text="x")
            )

            client = GoogleClient()
            client._build_contents(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                system_prompt=None,
            )

        assert "model" in roles_created
        assert "user" in roles_created

    def test_multimodal_content_uses_distinct_text_and_image_parts(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.google_client import GoogleClient

        image_bytes = b"fake-png-bytes"
        data_url = (
            f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        )

        with patch("haindy.models.google_client.genai_types") as mock_types:
            contents_created: list[Any] = []

            def _make_content(role: str, parts: Any) -> Any:
                obj = SimpleNamespace(role=role, parts=parts)
                contents_created.append(obj)
                return obj

            mock_types.Content = MagicMock(side_effect=_make_content)
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                side_effect=lambda *, text: SimpleNamespace(kind="text", text=text)
            )
            mock_types.Part.from_bytes = MagicMock(
                side_effect=lambda *, data, mime_type: SimpleNamespace(
                    kind="image", data=data, mime_type=mime_type
                )
            )

            client = GoogleClient()
            client._build_contents(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe the image."},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                system_prompt=None,
            )

        assert len(contents_created) == 1
        assert contents_created[0].role == "user"
        assert [part.kind for part in contents_created[0].parts] == ["text", "image"]
        assert contents_created[0].parts[0].text == "Describe the image."
        assert contents_created[0].parts[1].data == image_bytes
        assert contents_created[0].parts[1].mime_type == "image/png"
        mock_types.Part.from_text.assert_called_once_with(text="Describe the image.")
        mock_types.Part.from_bytes.assert_called_once_with(
            data=image_bytes,
            mime_type="image/png",
        )

    def test_invalid_data_url_raises_request_validation_error(
        self, patched_settings: Any
    ) -> None:
        from haindy.models.errors import ModelCallError
        from haindy.models.google_client import GoogleClient

        with patch("haindy.models.google_client.genai_types") as mock_types:
            mock_types.Content = MagicMock(
                side_effect=lambda role, parts: SimpleNamespace(role=role, parts=parts)
            )
            mock_types.Part = MagicMock()
            mock_types.Part.from_text = MagicMock(
                return_value=SimpleNamespace(kind="text", text="x")
            )
            mock_types.Part.from_bytes = MagicMock()

            client = GoogleClient()
            with pytest.raises(ModelCallError, match="valid base64 data"):
                client._build_contents(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Describe the image."},
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,a",
                                },
                            ],
                        }
                    ],
                    system_prompt=None,
                )
