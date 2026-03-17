"""Async daemon entrypoint for one tool-call session."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from typing import Any

from src.monitoring.logger import get_logger

from .ipc import read_request, write_envelope
from .logging import setup_tool_call_logging
from .models import CommandStatus, ExitReason, make_envelope, public_command_name
from .paths import (
    cleanup_session_artifacts,
    ensure_session_layout,
    get_daemon_log_path,
    get_socket_path,
    save_session_metadata,
    write_pid_file,
)
from .runtime import ToolCallSessionRuntime

logger = get_logger(__name__)


class ToolCallDaemon:
    """Single-session asyncio daemon serving one Unix socket."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: str,
        idle_timeout_seconds: int,
        android_serial: str | None = None,
        android_app: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.socket_path = get_socket_path(session_id)
        self.runtime = ToolCallSessionRuntime(
            session_id=session_id,
            backend=backend,
            idle_timeout_seconds=idle_timeout_seconds,
            android_serial=android_serial,
            android_app=android_app,
        )
        self._server: asyncio.AbstractServer | None = None
        self._command_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Run the session daemon until shutdown."""

        ensure_session_layout(self.session_id)
        write_pid_file(self.session_id, os.getpid())
        self.runtime.set_pid(os.getpid())

        await self.runtime.start()
        save_session_metadata(self.runtime.metadata)

        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )
        self._signal_ready()
        logger.info(
            "Tool-call daemon ready",
            extra={"session_id": self.session_id, "socket_path": str(self.socket_path)},
        )

        idle_task = asyncio.create_task(self._idle_watchdog())
        try:
            async with self._server:
                await self._shutdown_event.wait()
        finally:
            idle_task.cancel()
            with suppress(asyncio.CancelledError):
                await idle_task
            await self.runtime.stop()
            cleanup_session_artifacts(self.session_id)
            logger.info(
                "Tool-call daemon stopped", extra={"session_id": self.session_id}
            )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await read_request(reader)
            if self._command_lock.locked() and request.command != "session_close":
                envelope = make_envelope(
                    session_id=self.session_id,
                    command=public_command_name(request.command),
                    status=CommandStatus.ERROR,
                    response=(
                        "Session is busy executing a previous command. Retry when the current command completes."
                    ),
                    screenshot_path=None,
                    exit_reason=ExitReason.SESSION_BUSY,
                    duration_ms=0,
                    actions_taken=0,
                )
            else:
                async with self._command_lock:
                    timeout_seconds = int(request.options.get("timeout_seconds", 300))
                    try:
                        envelope = await asyncio.wait_for(
                            self.runtime.handle_request(request),
                            timeout=max(timeout_seconds, 1),
                        )
                    except asyncio.TimeoutError:
                        envelope = make_envelope(
                            session_id=self.session_id,
                            command=public_command_name(request.command),
                            status=CommandStatus.ERROR,
                            response=(
                                "Command timed out before completion. The session is still alive and can accept another command."
                            ),
                            screenshot_path=self.runtime.metadata.latest_screenshot_path,
                            exit_reason=ExitReason.COMMAND_TIMEOUT,
                            duration_ms=timeout_seconds * 1000,
                            actions_taken=0,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Tool-call command failed",
                            extra={
                                "session_id": self.session_id,
                                "command": request.command,
                            },
                        )
                        envelope = make_envelope(
                            session_id=self.session_id,
                            command=public_command_name(request.command),
                            status=CommandStatus.ERROR,
                            response=(
                                f"Haindy encountered an internal error. Details: {exc}."
                            ),
                            screenshot_path=self.runtime.metadata.latest_screenshot_path,
                            exit_reason=ExitReason.AGENT_ERROR,
                            duration_ms=0,
                            actions_taken=0,
                        )
            await write_envelope(writer, envelope)
        finally:
            writer.close()
            await writer.wait_closed()
            if self.runtime.is_close_requested():
                self._shutdown_event.set()

    async def _idle_watchdog(self) -> None:
        """Shutdown the daemon after the configured idle timeout."""

        timeout_seconds = self.runtime.idle_timeout_seconds
        while True:
            await asyncio.sleep(1.0)
            if self._command_lock.locked():
                continue
            idle_seconds = time.monotonic() - self.runtime.last_activity_monotonic
            if idle_seconds >= timeout_seconds:
                logger.info(
                    "Tool-call session reached idle timeout",
                    extra={
                        "session_id": self.session_id,
                        "idle_seconds": int(idle_seconds),
                    },
                )
                self._shutdown_event.set()
                return

    @staticmethod
    def _signal_ready() -> None:
        fd_text = os.environ.get("HAINDY_READINESS_FD", "").strip()
        if not fd_text:
            return
        try:
            fd = int(fd_text)
        except ValueError:
            return
        os.write(fd, b"1")
        os.close(fd)


async def run_daemon_from_args(args: Any) -> int:
    """Run the daemon entrypoint from parsed CLI args."""

    ensure_session_layout(args.session_id)
    setup_tool_call_logging(
        log_path=get_daemon_log_path(args.session_id),
        run_id=args.session_id,
        debug_to_stderr=bool(getattr(args, "debug", False)),
    )
    daemon = ToolCallDaemon(
        session_id=args.session_id,
        backend=args.backend,
        idle_timeout_seconds=args.idle_timeout,
        android_serial=args.android_serial,
        android_app=args.android_app,
    )
    try:
        await daemon.run()
        return 0
    except Exception as exc:
        daemon.runtime.metadata.status = "error"
        daemon.runtime.metadata.notes = str(exc)
        save_session_metadata(daemon.runtime.metadata)
        logger.exception(
            "Tool-call daemon startup failed",
            extra={"session_id": args.session_id},
        )
        raise
