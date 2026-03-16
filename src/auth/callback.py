"""Temporary localhost callback capture for OAuth browser logins."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit


class OAuthCallbackCapture:
    """Listen for a single localhost OAuth callback and capture its full URL."""

    def __init__(self, redirect_uri: str) -> None:
        parsed = urlsplit(redirect_uri)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._path = parsed.path or "/"
        self._server: asyncio.Server | None = None
        self._event = asyncio.Event()
        self._redirect_url: str | None = None

    async def start(self) -> None:
        """Start the callback listener."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._host,
            port=self._port,
        )

    async def close(self) -> None:
        """Stop the callback listener."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def wait_for_redirect(self, timeout_seconds: float) -> str | None:
        """Wait for a captured redirect URL."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None
        return self._redirect_url

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < 2:
                return

            target = parts[1]
            host = f"{self._host}:{self._port}"
            while True:
                line = await reader.readline()
                if not line or line in {b"\r\n", b"\n"}:
                    break
                try:
                    header = line.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    continue
                if header.lower().startswith("host:"):
                    host = header.split(":", 1)[1].strip()

            if target.startswith(self._path):
                self._redirect_url = f"http://{host}{target}"
                self._event.set()

            response_body = b"Authorization received. You can return to HAINDY."
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain; charset=utf-8\r\n"
                + f"Content-Length: {len(response_body)}\r\n\r\n".encode()
                + response_body
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                return
