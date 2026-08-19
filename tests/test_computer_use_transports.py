"""Tests for provider-specific Computer Use transports."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import haindy.agents.computer_use.transports as transport_module
from haindy.agents.computer_use.transports import (
    OpenAIResponsesHTTPTransport,
    OpenAIResponsesWebSocketTransport,
)
from tests.computer_use_session_support import make_session

pytest_plugins = ("tests.computer_use_session_support",)


class FakeSocket:
    """Minimal socket stub for connection timeout tests."""

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_timeout", "socket_timeout"),
    ((900.0, 10.0), (4.0, 4.0)),
)
async def test_websocket_transport_caps_initial_open_timeout(
    monkeypatch,
    request_timeout: float,
    socket_timeout: float,
) -> None:
    connect_kwargs: dict[str, Any] = {}
    wait_timeouts: list[float | None] = []
    original_wait_for = asyncio.wait_for

    async def connect(*args: Any, **kwargs: Any) -> FakeSocket:
        del args
        connect_kwargs.update(kwargs)
        return FakeSocket()

    async def recording_wait_for(awaitable: Any, *, timeout: float | None) -> Any:
        wait_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(transport_module.websockets, "connect", connect)
    monkeypatch.setattr(transport_module.asyncio, "wait_for", recording_wait_for)
    client = type(
        "Client",
        (),
        {"base_url": "https://api.openai.com/v1", "api_key": "test-key"},
    )()
    transport = OpenAIResponsesWebSocketTransport(
        client=client,
        timeout_seconds=request_timeout,
    )

    await transport._ensure_socket()

    assert connect_kwargs["open_timeout"] == socket_timeout
    assert connect_kwargs["close_timeout"] == socket_timeout
    assert wait_timeouts == [socket_timeout]


@pytest.mark.asyncio
async def test_stalled_initial_websocket_handshake_falls_back_to_http_once(
    mock_client,
    mock_browser,
    session_settings,
    monkeypatch,
) -> None:
    connect_calls = 0

    async def stalled_connect(*args: Any, **kwargs: Any) -> Any:
        nonlocal connect_calls
        del args, kwargs
        connect_calls += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(transport_module.websockets, "connect", stalled_connect)
    monkeypatch.setattr(
        OpenAIResponsesWebSocketTransport,
        "_MAX_OPEN_TIMEOUT_SECONDS",
        0.01,
    )
    mock_client.base_url = "https://api.openai.com/v1"
    mock_client.api_key = "test-key"
    mock_client.responses.create.return_value = {"id": "resp_http"}
    session_settings.openai_cu_transport = "responses_websocket"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="openai",
    )
    payload = {"model": "gpt-5.6-sol", "input": "Inspect the screen."}

    response = await session._create_response(payload)

    assert response == {"id": "resp_http"}
    assert connect_calls == 1
    mock_client.responses.create.assert_awaited_once_with(**payload)
    assert isinstance(session._openai_transport, OpenAIResponsesHTTPTransport)
