"""Tests for AnthropicClient."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haindy.models.anthropic_client import AnthropicClient, _flatten_message_content
from haindy.models.errors import ModelCallError
from haindy.models.structured_output import build_json_schema_response_format


@pytest.fixture()
def patched_settings():
    settings = MagicMock()
    settings.anthropic_api_key = "test-key"
    # Simulate no anthropic_model field
    type(settings).anthropic_model = property(lambda self: None)
    with patch("haindy.models.anthropic_client.get_settings", return_value=settings):
        yield settings


def _make_response(
    text: str = "Hello",
    stop_reason: str = "end_turn",
    blocks: list[Any] | None = None,
) -> Any:
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    response_blocks = blocks or [SimpleNamespace(text=text)]
    return SimpleNamespace(
        content=response_blocks,
        stop_reason=stop_reason,
        usage=usage,
        model="claude-sonnet-4-6",
    )


class TestAnthropicClientInit:
    def test_default_model_falls_back_to_hardcoded(self, patched_settings: Any) -> None:
        client = AnthropicClient()
        assert client.model == "claude-sonnet-4-6"

    def test_custom_model_is_stored(self, patched_settings: Any) -> None:
        client = AnthropicClient(model="claude-haiku-3-5")
        assert client.model == "claude-haiku-3-5"

    def test_uses_api_key_from_settings(self, patched_settings: Any) -> None:
        client = AnthropicClient()
        assert client._api_key == "test-key"


class TestAnthropicClientCall:
    @pytest.mark.asyncio
    async def test_basic_call_returns_normalized_dict(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("world")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            result = await client.call(messages=[{"role": "user", "content": "Hello"}])

        assert result["content"] == "world"
        assert result["finish_reason"] == "end_turn"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_system_prompt_included(self, patched_settings: Any) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="Be concise.",
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "Be concise." in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_system_messages_in_list_extracted(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[
                    {"role": "system", "content": "System instruction."},
                    {"role": "user", "content": "Hello"},
                ]
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "System instruction." in call_kwargs["system"]
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_json_schema_parses_response_and_sets_output_config(
        self, patched_settings: Any
    ) -> None:
        response = _make_response('{"key": "value"}')
        response_format = build_json_schema_response_format(
            "haindy_test_schema_v1",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            result = await client.call(
                messages=[{"role": "user", "content": "output json"}],
                response_format=response_format,
            )

        assert result["content"] == {"key": "value"}
        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": response_format["json_schema"]["schema"],
            }
        }

    @pytest.mark.asyncio
    async def test_json_object_compatibility_uses_permissive_schema(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("{}")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[{"role": "user", "content": "go"}],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            }
        }

    @pytest.mark.asyncio
    async def test_multimodal_content_is_normalized_to_anthropic_blocks(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("ok")
        encoded = base64.b64encode(b"fake-image").decode("ascii")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this screenshot"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{encoded}",
                            },
                        ],
                    }
                ]
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this screenshot"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded,
                        },
                    },
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_anthropic_native_image_blocks_pass_through(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Inspect"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/image.png",
                                },
                            },
                        ],
                    }
                ]
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"][1] == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png",
            },
        }

    @pytest.mark.asyncio
    async def test_reasoning_level_maps_to_effort_on_supported_models(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_level="xhigh",
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "max"}

    @pytest.mark.asyncio
    async def test_unsupported_modalities_raise_request_validation_error(
        self, patched_settings: Any
    ) -> None:
        client = AnthropicClient()
        with pytest.raises(ModelCallError) as exc_info:
            await client.call(
                messages=[{"role": "user", "content": "hi"}],
                modalities={"text", "audio"},
            )

        assert exc_info.value.failure_kind == "request_validation_error"
        assert exc_info.value.response_payload == {"unsupported_modalities": ["audio"]}

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_are_flattened(
        self, patched_settings: Any
    ) -> None:
        response = _make_response(
            blocks=[SimpleNamespace(text="hello"), SimpleNamespace(text="world")]
        )
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            result = await client.call(messages=[{"role": "user", "content": "hi"}])

        assert result["content"] == "hello\nworld"

    @pytest.mark.asyncio
    async def test_structured_output_refusal_raises_typed_error(
        self, patched_settings: Any
    ) -> None:
        response = _make_response('{"key":"value"}', stop_reason="refusal")
        response_format = build_json_schema_response_format(
            "haindy_test_schema_v1",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            with pytest.raises(ModelCallError) as exc_info:
                await client.call(
                    messages=[{"role": "user", "content": "output json"}],
                    response_format=response_format,
                )

        assert exc_info.value.failure_kind == "response_refusal"

    @pytest.mark.asyncio
    async def test_structured_output_truncation_raises_typed_error(
        self, patched_settings: Any
    ) -> None:
        response = _make_response('{"key":"value"}', stop_reason="max_tokens")
        response_format = build_json_schema_response_format(
            "haindy_test_schema_v1",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            with pytest.raises(ModelCallError) as exc_info:
                await client.call(
                    messages=[{"role": "user", "content": "output json"}],
                    response_format=response_format,
                )

        assert exc_info.value.failure_kind == "response_truncated"

    @pytest.mark.asyncio
    async def test_structured_output_prefill_rejected(
        self, patched_settings: Any
    ) -> None:
        response_format = build_json_schema_response_format(
            "haindy_test_schema_v1",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )
        client = AnthropicClient()
        with pytest.raises(ModelCallError) as exc_info:
            await client.call(
                messages=[
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "{"},
                ],
                response_format=response_format,
            )

        assert exc_info.value.failure_kind == "request_validation_error"

    @pytest.mark.asyncio
    async def test_max_tokens_defaults_to_8192(self, patched_settings: Any) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(messages=[{"role": "user", "content": "hi"}])

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_custom_max_tokens_passed_through(
        self, patched_settings: Any
    ) -> None:
        response = _make_response("ok")
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            await client.call(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1024,
            )

        call_kwargs = mock_instance.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_usage_totals_computed_when_total_missing(
        self, patched_settings: Any
    ) -> None:
        block = SimpleNamespace(text="hi")
        usage = SimpleNamespace(input_tokens=3, output_tokens=7)
        response = SimpleNamespace(
            content=[block],
            stop_reason="end_turn",
            usage=usage,
            model="claude-sonnet-4-6",
        )
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            result = await client.call(messages=[{"role": "user", "content": "hi"}])

        assert result["usage"]["total_tokens"] == 10


class TestAnthropicClientStreaming:
    @pytest.mark.asyncio
    async def test_stream_observer_called(self, patched_settings: Any) -> None:
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

        final_msg = SimpleNamespace(
            content=[SimpleNamespace(text="hello world")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            model="claude-sonnet-4-6",
        )

        async def _text_stream():
            yield "hello "
            yield "world"

        fake_stream = MagicMock()
        fake_stream.text_stream = _text_stream()
        fake_stream.get_final_message = AsyncMock(return_value=final_msg)
        fake_stream.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream.__aexit__ = AsyncMock(return_value=None)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.messages.stream = MagicMock(return_value=fake_stream)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            obs = Obs()
            result = await client.call(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_observer=obs,
            )

        assert started == [True]
        assert ended == [True]
        assert "hello " in deltas
        assert "world" in deltas
        assert result["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_streaming_structured_output_parses_json(
        self, patched_settings: Any
    ) -> None:
        response_format = build_json_schema_response_format(
            "haindy_test_schema_v1",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )
        final_msg = SimpleNamespace(
            content=[SimpleNamespace(text='{"key":"value"}')],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            model="claude-sonnet-4-6",
        )

        async def _text_stream():
            yield '{"key":'
            yield '"value"}'

        fake_stream = MagicMock()
        fake_stream.text_stream = _text_stream()
        fake_stream.get_final_message = AsyncMock(return_value=final_msg)
        fake_stream.__aenter__ = AsyncMock(return_value=fake_stream)
        fake_stream.__aexit__ = AsyncMock(return_value=None)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.messages.stream = MagicMock(return_value=fake_stream)
            mock_cls.return_value = mock_instance

            client = AnthropicClient()
            result = await client.call(
                messages=[{"role": "user", "content": "hi"}],
                response_format=response_format,
                stream=True,
            )

        assert result["content"] == {"key": "value"}


class TestFlattenMessageContent:
    def test_string_content_returned_as_is(self) -> None:
        assert _flatten_message_content("hello") == "hello"

    def test_list_content_joined(self) -> None:
        content = [
            {"type": "text", "text": "foo"},
            {"type": "input_text", "text": "bar"},
        ]
        assert _flatten_message_content(content) == "foo\nbar"

    def test_non_string_content_stringified(self) -> None:
        assert _flatten_message_content(42) == "42"
