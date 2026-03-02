"""Tests for OpenAIClient streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.models.openai_client import OpenAIClient, ResponseStreamObserver


class RecordingObserver(ResponseStreamObserver):
    """Test observer that records streaming callbacks."""

    def __init__(self) -> None:
        self.started = False
        self.ended = False
        self.deltas: list[dict[str, int]] = []
        self.totals: dict[str, int] | None = None
        self.errors: list[Any] = []
        self.token_updates: list[dict[str, int]] = []

    def on_stream_start(self) -> None:  # pragma: no cover - simple flag
        self.started = True

    def on_text_delta(self, delta: str) -> None:  # pragma: no cover - ignored in tests
        return

    def on_usage_delta(self, delta: dict[str, int]) -> None:
        self.deltas.append(delta)

    def on_usage_total(self, totals: dict[str, int]) -> None:
        self.totals = totals

    def on_error(self, error: Any) -> None:  # pragma: no cover - defensive logging path
        self.errors.append(error)

    def on_stream_end(self) -> None:
        self.ended = True

    def on_token_progress(
        self, total_tokens: int, delta_tokens: int, delta_chars: int
    ) -> None:
        self.token_updates.append(
            {
                "total": total_tokens,
                "delta_tokens": delta_tokens,
                "delta_chars": delta_chars,
            }
        )


class FakeStream:
    """Async stream stub returning predetermined events and final response."""

    def __init__(self, events: list[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def __aiter__(self):
        async def iterator():
            for event in self._events:
                yield event

        return iterator()

    async def get_final_response(self) -> Any:
        return self._final_response


class FakeResponses:
    """Stub for AsyncOpenAI.responses that captures kwargs."""

    def __init__(self, events: list[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response
        self.last_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> FakeStream:
        self.last_kwargs = kwargs
        return FakeStream(self._events, self._final_response)


class FakeCreateResponses:
    """Stub for AsyncOpenAI.responses.create that captures kwargs."""

    def __init__(self, final_response: Any) -> None:
        self._final_response = final_response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return self._final_response


class FakeAsyncOpenAI:
    """Minimal AsyncOpenAI replacement for tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # events injected later by monkeypatching instance attribute
        self.responses: FakeResponses | None = None


@pytest.mark.asyncio
async def test_streaming_requests_usage_and_emits_final_delta(monkeypatch) -> None:
    usage = SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)
    final_response = SimpleNamespace(
        output_text="hello world",
        usage=usage,
        status="completed",
        model="gpt-5.2",
    )
    delta_event = SimpleNamespace(type="response.output_text.delta", delta="hello")
    events = [
        delta_event,
        SimpleNamespace(type="response.completed", response=final_response),
    ]

    fake_responses = FakeResponses(events=events, final_response=final_response)

    def fake_make_client(*args: Any, **kwargs: Any) -> FakeAsyncOpenAI:
        client = FakeAsyncOpenAI()
        client.responses = fake_responses
        return client

    dummy_settings = SimpleNamespace(
        openai_api_key="dummy",
        openai_request_timeout_seconds=30,
    )

    monkeypatch.setattr("src.models.openai_client.AsyncOpenAI", fake_make_client)
    monkeypatch.setattr("src.models.openai_client.get_settings", lambda: dummy_settings)
    monkeypatch.setattr(
        "src.models.openai_client.OpenAIClient._estimate_token_count",
        lambda self, text, encoder: len(text),
    )

    client = OpenAIClient(model="gpt-5.2", api_key="test-key")

    observer = RecordingObserver()
    result = await client._call_responses_api_streaming(
        final_messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=None,
        response_format=None,
        reasoning_level=None,
        system_prompt=None,
        observer=observer,
    )

    assert fake_responses.last_kwargs is not None
    assert "stream_options" not in fake_responses.last_kwargs.get("extra_body", {})

    assert observer.deltas == [
        {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    ]
    assert observer.totals == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert observer.started is True
    assert observer.ended is True
    assert not observer.errors
    assert observer.token_updates == [{"total": 5, "delta_tokens": 5, "delta_chars": 5}]

    assert result["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


def test_ensure_json_keyword_appends_input_message_when_missing(monkeypatch) -> None:
    dummy_settings = SimpleNamespace(
        openai_api_key="dummy",
        openai_request_timeout_seconds=30,
    )

    monkeypatch.setattr("src.models.openai_client.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("src.models.openai_client.get_settings", lambda: dummy_settings)

    client = OpenAIClient(model="gpt-5.2", api_key="test-key")
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": "Plan it"}]}
    ]
    instructions = client._ensure_json_keyword_for_response_format(
        instructions="You are a planner.",
        input_items=input_items,
        response_format={"type": "json_object"},
    )

    assert instructions == "You are a planner."
    assert input_items[-1]["role"] == "user"
    assert input_items[-1]["content"][0]["type"] == "input_text"
    assert input_items[-1]["content"][0]["text"] == "Respond with valid json."


def test_ensure_json_keyword_appends_input_even_if_only_instructions_mention_json(
    monkeypatch,
) -> None:
    dummy_settings = SimpleNamespace(
        openai_api_key="dummy",
        openai_request_timeout_seconds=30,
    )

    monkeypatch.setattr("src.models.openai_client.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("src.models.openai_client.get_settings", lambda: dummy_settings)

    client = OpenAIClient(model="gpt-5.2", api_key="test-key")
    input_items: list[dict[str, Any]] = []
    instructions = client._ensure_json_keyword_for_response_format(
        instructions="Return JSON only.",
        input_items=input_items,
        response_format={"type": "json_object"},
    )

    assert instructions == "Return JSON only."
    assert input_items[-1]["content"][0]["text"] == "Respond with valid json."


def test_ensure_json_keyword_keeps_existing_input_json_reference(monkeypatch) -> None:
    dummy_settings = SimpleNamespace(
        openai_api_key="dummy",
        openai_request_timeout_seconds=30,
    )

    monkeypatch.setattr("src.models.openai_client.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("src.models.openai_client.get_settings", lambda: dummy_settings)

    client = OpenAIClient(model="gpt-5.2", api_key="test-key")
    input_items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Return json object only"}],
        }
    ]

    instructions = client._ensure_json_keyword_for_response_format(
        instructions="You are a planner.",
        input_items=input_items,
        response_format={"type": "json_object"},
    )

    assert instructions == "You are a planner."
    assert len(input_items) == 1


@pytest.mark.asyncio
async def test_call_responses_api_injects_json_keyword_into_input(monkeypatch) -> None:
    usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
    final_response = SimpleNamespace(
        output_text="{}",
        usage=usage,
        status="completed",
        model="gpt-5.2",
    )
    fake_responses = FakeCreateResponses(final_response=final_response)

    def fake_make_client(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(responses=fake_responses)

    dummy_settings = SimpleNamespace(
        openai_api_key="dummy",
        openai_request_timeout_seconds=30,
    )
    monkeypatch.setattr("src.models.openai_client.AsyncOpenAI", fake_make_client)
    monkeypatch.setattr("src.models.openai_client.get_settings", lambda: dummy_settings)

    client = OpenAIClient(model="gpt-5.2", api_key="test-key")
    result = await client._call_responses_api(
        final_messages=[{"role": "user", "content": "Assess context."}],
        temperature=0.0,
        max_tokens=None,
        response_format={"type": "json_object"},
        reasoning_level="none",
        system_prompt="You are a strict evaluator.",
    )

    assert result["content"] == {}
    assert fake_responses.last_kwargs is not None
    input_items = fake_responses.last_kwargs["input"]
    input_texts = [
        content.get("text", "")
        for item in input_items
        for content in item.get("content", [])
        if isinstance(content, dict) and isinstance(content.get("text"), str)
    ]
    assert any("json" in text.lower() for text in input_texts)
