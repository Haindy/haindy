"""Tests for OpenAIClient streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.models.openai_client import OpenAIClient, ResponseStreamObserver


class RecordingObserver(ResponseStreamObserver):
    """Test observer that records streaming callbacks."""

    def __init__(self) -> None:
        self.started = False
        self.ended = False
        self.deltas: List[Dict[str, int]] = []
        self.totals: Dict[str, int] | None = None
        self.errors: List[Any] = []
        self.token_updates: List[Dict[str, int]] = []

    def on_stream_start(self) -> None:  # pragma: no cover - simple flag
        self.started = True

    def on_text_delta(self, delta: str) -> None:  # pragma: no cover - ignored in tests
        return

    def on_usage_delta(self, delta: Dict[str, int]) -> None:
        self.deltas.append(delta)

    def on_usage_total(self, totals: Dict[str, int]) -> None:
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

    def __init__(self, events: List[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response

    async def __aenter__(self) -> "FakeStream":
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

    def __init__(self, events: List[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response
        self.last_kwargs: Dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> FakeStream:
        self.last_kwargs = kwargs
        return FakeStream(self._events, self._final_response)


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
        model="gpt-5",
    )
    delta_event = SimpleNamespace(type="response.output_text.delta", delta="hello")
    events = [delta_event, SimpleNamespace(type="response.completed", response=final_response)]

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

    client = OpenAIClient(model="gpt-5", api_key="test-key")

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
    assert observer.totals == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert observer.started is True
    assert observer.ended is True
    assert not observer.errors
    assert observer.token_updates == [
        {"total": 5, "delta_tokens": 5, "delta_chars": 5}
    ]

    assert result["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
