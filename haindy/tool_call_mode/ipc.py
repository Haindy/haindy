"""NDJSON helpers for tool-call CLI <-> daemon IPC."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from .models import ToolCallEnvelope, ToolCallRequest


async def send_request(socket_path: Path, request: ToolCallRequest) -> ToolCallEnvelope:
    """Send one request to the daemon and parse a single-line response.

    On Linux/macOS uses a Unix domain socket at ``socket_path``.
    On Windows reads the TCP port from the sibling ``daemon.port`` file and
    connects via TCP loopback.
    """

    if sys.platform == "win32":
        port_file = socket_path.parent / "daemon.port"
        try:
            port = int(port_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError) as exc:
            raise ConnectionError(
                f"Cannot read daemon TCP port from {port_file}: {exc}"
            ) from exc
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
    else:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))

    try:
        payload = json.dumps(request.model_dump(mode="json")) + "\n"
        writer.write(payload.encode("utf-8"))
        await writer.drain()

        raw = await reader.readline()
        if not raw:
            raise ConnectionError(
                "Tool-call daemon closed the connection without a response."
            )
        return ToolCallEnvelope.model_validate_json(raw.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()


async def read_request(reader: asyncio.StreamReader) -> ToolCallRequest:
    """Read exactly one request line from the socket."""

    raw = await reader.readline()
    if not raw:
        raise ConnectionError("Tool-call client disconnected before sending a request.")
    return ToolCallRequest.model_validate_json(raw.decode("utf-8"))


async def write_envelope(
    writer: asyncio.StreamWriter,
    envelope: ToolCallEnvelope,
) -> None:
    """Write one envelope line to the socket."""

    payload = envelope.model_dump_json() + "\n"
    writer.write(payload.encode("utf-8"))
    await writer.drain()
