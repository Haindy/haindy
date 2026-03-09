"""Transport helpers for provider-specific Computer Use requests."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol
from urllib.parse import urljoin

import websockets
from openai import AsyncOpenAI

logger = logging.getLogger("src.agents.computer_use.transport")


class ComputerUseTransport(Protocol):
    """Minimal transport interface for Computer Use request/response cycles."""

    async def request(self, payload: dict[str, Any]) -> Any:
        """Send a provider request and return a response object/dict."""

    async def close(self) -> None:
        """Close transport state for the current session."""


class OpenAIResponsesHTTPTransport:
    """Compatibility transport using the standard Responses HTTP API."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def request(self, payload: dict[str, Any]) -> Any:
        return await self._client.responses.create(**payload)

    async def close(self) -> None:
        return None


class OpenAIResponsesWebSocketTransport:
    """Persistent WebSocket transport for Responses API WebSocket mode."""

    _RESPONSE_DONE_EVENTS = {"response.done", "response.completed"}
    _ERROR_EVENTS = {"error", "response.failed"}

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        timeout_seconds: float,
    ) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._socket: websockets.ClientConnection | None = None
        self._lock = asyncio.Lock()

    async def request(self, payload: dict[str, Any]) -> Any:
        async with self._lock:
            normalized_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"stream", "background"}
            }
            attempts = (
                {"type": "response.create", "response": normalized_payload},
                {"type": "response.create", **normalized_payload},
            )
            last_error: Exception | None = None
            for event in attempts:
                try:
                    socket = await self._ensure_socket()
                    await socket.send(json.dumps(event))
                    return await self._read_response(socket)
                except Exception as exc:
                    last_error = exc
                    await self.close()
            if last_error is not None:
                raise last_error
            raise RuntimeError("OpenAI WebSocket transport failed without an error")

    async def close(self) -> None:
        if self._socket is None:
            return
        try:
            await self._socket.close()
        except Exception:
            logger.debug("Failed to close OpenAI CU websocket transport", exc_info=True)
        finally:
            self._socket = None

    async def _ensure_socket(self) -> websockets.ClientConnection:
        if self._socket is not None:
            return self._socket

        base_url = str(self._client.base_url)
        websocket_url = self._to_websocket_url(base_url)
        headers = {"Authorization": f"Bearer {self._client.api_key}"}
        self._socket = await websockets.connect(
            websocket_url,
            additional_headers=headers,
            open_timeout=self._timeout_seconds,
            close_timeout=self._timeout_seconds,
            max_size=None,
        )
        return self._socket

    async def _read_response(self, socket: websockets.ClientConnection) -> Any:
        while True:
            raw_message = await asyncio.wait_for(
                socket.recv(),
                timeout=self._timeout_seconds,
            )
            event = json.loads(raw_message)
            event_type = str(event.get("type") or "").strip()
            if not event_type:
                continue

            if event_type in self._RESPONSE_DONE_EVENTS:
                if "response" in event:
                    return event["response"]
                return event

            if event_type in self._ERROR_EVENTS:
                error_payload = event.get("error") or event
                raise RuntimeError(
                    f"OpenAI Responses WebSocket error: {json.dumps(error_payload)}"
                )

    @staticmethod
    def _to_websocket_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.startswith("https://"):
            normalized = "wss://" + normalized[len("https://") :]
        elif normalized.startswith("http://"):
            normalized = "ws://" + normalized[len("http://") :]
        if not normalized.endswith("/v1"):
            normalized = normalized.rstrip("/")
        return urljoin(normalized + "/", "responses")


__all__ = [
    "ComputerUseTransport",
    "OpenAIResponsesHTTPTransport",
    "OpenAIResponsesWebSocketTransport",
]
