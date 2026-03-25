"""Public CLI entrypoints for tool-call mode."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

from haindy.config.settings import get_settings
from haindy.runtime.environment import normalize_automation_backend

from .daemon import run_daemon_from_args
from .ipc import send_request
from .launcher import (
    ToolCallDaemonLaunchError,
    launch_tool_call_daemon,
    public_cli_program_name,
)
from .models import (
    CommandStatus,
    ExitReason,
    SessionListEntry,
    ToolCallEnvelope,
    ToolCallRequest,
    envelope_exit_code,
    make_envelope,
    public_command_name,
)
from .paths import (
    cleanup_session_artifacts,
    cleanup_stale_sessions,
    ensure_session_layout,
    get_daemon_log_path,
    get_sessions_root,
    get_socket_path,
    is_process_alive,
    load_session_metadata,
    read_pid,
    save_session_metadata,
    terminate_session_process,
)

TOOL_CALL_COMMANDS = {"session", "act", "test", "screenshot", "__tool_call_daemon"}


class ToolCallUsageError(ValueError):
    """Raised when tool-call CLI parsing should return a JSON usage envelope."""


class ToolCallArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that keeps usage failures in-band as JSON."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: Any = None,
    ) -> Any:
        parsed = super().parse_args(args=args, namespace=namespace)

        root_debug = bool(getattr(parsed, "_root_debug", False))
        root_json = bool(getattr(parsed, "_root_json", False))
        root_session = getattr(parsed, "_root_session", None)
        if hasattr(parsed, "_root_debug"):
            delattr(parsed, "_root_debug")
        if hasattr(parsed, "_root_json"):
            delattr(parsed, "_root_json")
        if hasattr(parsed, "_root_session"):
            delattr(parsed, "_root_session")

        parsed.debug = bool(getattr(parsed, "debug", False) or root_debug)
        parsed.json = bool(getattr(parsed, "json", True) or root_json)
        if not getattr(parsed, "session", None) and root_session:
            parsed.session = root_session
        return parsed

    def error(self, message: str) -> NoReturn:
        raise ToolCallUsageError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        if status == 0:
            raise SystemExit(0)
        raise ToolCallUsageError(message.strip() if message else "Invalid arguments.")


def is_tool_call_command(argv: list[str] | None) -> bool:
    """Return True when argv targets the tool-call command surface."""

    if not argv:
        return False
    return argv[0] in TOOL_CALL_COMMANDS


def create_tool_call_parser() -> argparse.ArgumentParser:
    """Create the public tool-call parser."""

    parser = ToolCallArgumentParser(
        prog=public_cli_program_name(),
        description="HAINDY tool-call mode",
    )
    parser.add_argument(
        "--debug",
        dest="_root_debug",
        action="store_true",
        help="Enable verbose daemon logging to stderr.",
    )
    parser.add_argument(
        "--json",
        dest="_root_json",
        action="store_true",
        help="Emit JSON to stdout (always enabled in tool-call mode).",
    )
    parser.add_argument(
        "--session",
        dest="_root_session",
        help="Explicit session id for commands that operate on an existing session.",
    )

    common_parser = ToolCallArgumentParser(add_help=False)
    common_parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose daemon logging to stderr.",
    )
    common_parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit JSON to stdout (always enabled in tool-call mode).",
    )
    session_parser_parent = ToolCallArgumentParser(add_help=False)
    session_parser_parent.add_argument(
        "--session",
        default=argparse.SUPPRESS,
        help="Explicit session id for commands that operate on an existing session.",
    )

    subparsers = parser.add_subparsers(
        dest="tool_command",
        required=True,
        parser_class=ToolCallArgumentParser,
    )

    session_parser = subparsers.add_parser(
        "session",
        help="Manage sessions",
        parents=[common_parser],
    )
    session_subparsers = session_parser.add_subparsers(
        dest="session_command",
        required=True,
        parser_class=ToolCallArgumentParser,
    )

    session_new = session_subparsers.add_parser(
        "new",
        help="Start a new session",
        parents=[common_parser],
    )
    target_group = session_new.add_mutually_exclusive_group()
    target_group.add_argument(
        "--android", action="store_true", help="Use Android ADB backend."
    )
    target_group.add_argument(
        "--desktop", action="store_true", help="Use desktop backend."
    )
    target_group.add_argument("--ios", action="store_true", help="Use iOS idb backend.")
    session_new.add_argument("--android-serial", help="Target Android serial.")
    session_new.add_argument("--android-app", help="Launch Android package on start.")
    session_new.add_argument("--ios-udid", help="Target iOS device or simulator UDID.")
    session_new.add_argument("--ios-app", help="Launch iOS bundle ID on start.")
    session_new.add_argument(
        "--url",
        help="Desktop URL startup support is intentionally deferred in this build.",
    )
    session_new.add_argument(
        "--idle-timeout",
        type=int,
        default=1800,
        help="Kill the daemon after this many idle seconds.",
    )

    session_list = session_subparsers.add_parser(
        "list",
        help="List live sessions",
        parents=[common_parser],
    )
    del session_list

    session_status = session_subparsers.add_parser(
        "status",
        help="Describe current session state",
        parents=[common_parser, session_parser_parent],
    )
    session_status.add_argument("--timeout", type=int, default=300)

    session_close = session_subparsers.add_parser(
        "close",
        help="Close a session",
        parents=[common_parser, session_parser_parent],
    )
    session_close.add_argument("--force", action="store_true")

    session_set = session_subparsers.add_parser(
        "set",
        help="Set a session variable",
        parents=[common_parser, session_parser_parent],
    )
    session_set.add_argument("name")
    session_set.add_argument("value", nargs="?")
    session_set.add_argument("--value-file", type=Path)
    session_set.add_argument("--secret", action="store_true")

    session_unset = session_subparsers.add_parser(
        "unset",
        help="Unset a session variable",
        parents=[common_parser, session_parser_parent],
    )
    session_unset.add_argument("name")

    session_vars = session_subparsers.add_parser(
        "vars",
        help="List session variables",
        parents=[common_parser, session_parser_parent],
    )
    del session_vars

    screenshot_parser = subparsers.add_parser(
        "screenshot",
        help="Take a screenshot and return its path (no AI model call)",
        parents=[common_parser, session_parser_parent],
    )
    screenshot_parser.add_argument("--timeout", type=int, default=30)

    act_parser = subparsers.add_parser(
        "act",
        help="Execute one direct action",
        parents=[common_parser, session_parser_parent],
    )
    act_parser.add_argument("instruction")
    act_parser.add_argument("--timeout", type=int, default=300)

    test_parser = subparsers.add_parser(
        "test",
        help="Run one planned scenario",
        parents=[common_parser, session_parser_parent],
    )
    test_parser.add_argument("scenario")
    test_parser.add_argument("--max-steps", type=int, default=20)
    test_parser.add_argument("--timeout", type=int, default=300)

    return parser


def create_tool_call_daemon_parser() -> argparse.ArgumentParser:
    """Create the hidden daemon parser."""

    parser = argparse.ArgumentParser(
        prog=f"{public_cli_program_name()} __tool_call_daemon"
    )
    parser.add_argument("__tool_call_daemon")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--idle-timeout", type=int, default=1800)
    parser.add_argument("--android-serial")
    parser.add_argument("--android-app")
    parser.add_argument("--ios-udid")
    parser.add_argument("--ios-app")
    parser.add_argument("--debug", action="store_true")
    return parser


async def run_tool_call_cli(argv: list[str]) -> int:
    """Run the public tool-call CLI and print exactly one JSON object."""

    parser = create_tool_call_parser()
    try:
        args = parser.parse_args(argv)
    except ToolCallUsageError as exc:
        envelope, exit_code = _usage_error(str(exc))
        print(envelope.model_dump_json())
        return exit_code

    if args.tool_command == "session":
        if args.session_command == "new":
            envelope, exit_code = await _handle_session_new(args)
        elif args.session_command == "list":
            envelope, exit_code = _handle_session_list()
        elif args.session_command == "close":
            envelope, exit_code = await _handle_session_close(args)
        elif args.session_command == "status":
            envelope, exit_code = await _send_session_request(
                args,
                ToolCallRequest(
                    command="session_status",
                    options={"timeout_seconds": args.timeout},
                ),
            )
        elif args.session_command == "set":
            envelope, exit_code = await _handle_session_set(args)
        elif args.session_command == "unset":
            envelope, exit_code = await _send_session_request(
                args,
                ToolCallRequest(
                    command="session_unset",
                    var_name=args.name,
                ),
            )
        elif args.session_command == "vars":
            envelope, exit_code = await _send_session_request(
                args,
                ToolCallRequest(command="session_vars"),
            )
        else:  # pragma: no cover - parser guarantees this branch is unreachable
            envelope, exit_code = _usage_error("Unknown session subcommand.")
    elif args.tool_command == "screenshot":
        envelope, exit_code = await _send_session_request(
            args,
            ToolCallRequest(
                command="screenshot",
                options={"timeout_seconds": args.timeout},
            ),
        )
    elif args.tool_command == "act":
        envelope, exit_code = await _send_session_request(
            args,
            ToolCallRequest(
                command="act",
                instruction=args.instruction,
                options={"timeout_seconds": args.timeout},
            ),
        )
    elif args.tool_command == "test":
        envelope, exit_code = await _send_session_request(
            args,
            ToolCallRequest(
                command="test",
                instruction=args.scenario,
                options={
                    "timeout_seconds": args.timeout,
                    "max_steps": args.max_steps,
                },
            ),
        )
    else:  # pragma: no cover - parser guarantees this branch is unreachable
        envelope, exit_code = _usage_error("Unknown tool-call command.")

    print(envelope.model_dump_json())
    return exit_code


async def run_tool_call_daemon_cli(argv: list[str]) -> int:
    """Run the hidden daemon command."""

    parser = create_tool_call_daemon_parser()
    args = parser.parse_args(argv)
    return await run_daemon_from_args(args)


async def _handle_session_new(args: argparse.Namespace) -> tuple[ToolCallEnvelope, int]:
    settings = get_settings()
    backend = _resolve_requested_backend(args, settings)

    if args.url:
        return _usage_error(
            "Desktop `--url` startup support is deferred in this build."
        )

    cleanup_stale_sessions()
    session_id = str(uuid4())
    ensure_session_layout(session_id)

    started = time.perf_counter()
    try:
        launch = launch_tool_call_daemon(
            session_id=session_id,
            backend=backend,
            idle_timeout=args.idle_timeout,
            android_serial=args.android_serial,
            android_app=args.android_app,
            ios_udid=getattr(args, "ios_udid", None),
            ios_app=getattr(args, "ios_app", None),
            debug=bool(args.debug),
        )
    except ToolCallDaemonLaunchError as exc:
        envelope = make_envelope(
            session_id=session_id,
            command="session",
            status=CommandStatus.ERROR,
            response=(
                "Session daemon could not be launched. "
                f"Check {get_daemon_log_path(session_id)}. Details: {exc}"
            ),
            screenshot_path=None,
            exit_reason=ExitReason.AGENT_ERROR,
            duration_ms=int((time.perf_counter() - started) * 1000),
            actions_taken=0,
        )
        return envelope, 1

    try:
        ready_byte = await asyncio.wait_for(
            asyncio.to_thread(os.read, launch.readiness_fd, 1),
            timeout=30.0,
        )
        if ready_byte != b"1":
            envelope = make_envelope(
                session_id=session_id,
                command="session",
                status=CommandStatus.ERROR,
                response=(
                    "Session daemon exited during startup. "
                    f"Check {get_daemon_log_path(session_id)}."
                ),
                screenshot_path=None,
                exit_reason=ExitReason.AGENT_ERROR,
                duration_ms=int((time.perf_counter() - started) * 1000),
                actions_taken=0,
            )
            return envelope, 1
    except asyncio.TimeoutError:
        envelope = make_envelope(
            session_id=session_id,
            command="session",
            status=CommandStatus.ERROR,
            response=(
                "Timed out waiting for the session daemon to become ready. "
                f"Check {get_daemon_log_path(session_id)}."
            ),
            screenshot_path=None,
            exit_reason=ExitReason.AGENT_ERROR,
            duration_ms=int((time.perf_counter() - started) * 1000),
            actions_taken=0,
        )
        return envelope, 1
    finally:
        os.close(launch.readiness_fd)

    metadata = load_session_metadata(session_id)
    if metadata is None or not is_process_alive(metadata.pid):
        envelope = make_envelope(
            session_id=session_id,
            command="session",
            status=CommandStatus.ERROR,
            response=(
                "Session daemon exited before startup completed. "
                f"Check {get_daemon_log_path(session_id)}."
            ),
            screenshot_path=None,
            exit_reason=ExitReason.AGENT_ERROR,
            duration_ms=int((time.perf_counter() - started) * 1000),
            actions_taken=0,
        )
        return envelope, 1

    response = _startup_response_from_metadata(metadata)
    envelope = make_envelope(
        session_id=session_id,
        command="session",
        status=CommandStatus.SUCCESS,
        response=response,
        screenshot_path=metadata.latest_screenshot_path,
        exit_reason=ExitReason.COMPLETED,
        duration_ms=int((time.perf_counter() - started) * 1000),
        actions_taken=0,
    )
    return envelope, 0


def _handle_session_list() -> tuple[ToolCallEnvelope, int]:
    sessions: list[SessionListEntry] = []
    started = time.perf_counter()
    sessions_root = get_sessions_root()
    if sessions_root.exists():
        for session_dir in sorted(sessions_root.iterdir()):
            if not session_dir.is_dir():
                continue
            metadata = load_session_metadata(session_dir.name)
            if metadata is None or not is_process_alive(metadata.pid):
                cleanup_session_artifacts(session_dir.name)
                continue
            last_command_text = metadata.last_command_at or metadata.created_at
            last_command_at = _parse_iso(last_command_text)
            idle_seconds = int(max((time.time() - last_command_at.timestamp()), 0))
            sessions.append(
                SessionListEntry(
                    session_id=metadata.session_id,
                    backend=(
                        "android"
                        if metadata.backend == "mobile_adb"
                        else "ios"
                        if metadata.backend == "mobile_ios"
                        else "desktop"
                    ),
                    created_at=metadata.created_at,
                    steps_executed=metadata.actions_executed,
                    idle_seconds=idle_seconds,
                )
            )

    envelope = make_envelope(
        session_id=None,
        command="session",
        status=CommandStatus.SUCCESS,
        response=f"{len(sessions)} active sessions found.",
        screenshot_path=None,
        exit_reason=ExitReason.COMPLETED,
        duration_ms=int((time.perf_counter() - started) * 1000),
        actions_taken=0,
        sessions=sessions,
    )
    return envelope, 0


async def _handle_session_set(
    args: argparse.Namespace,
) -> tuple[ToolCallEnvelope, int]:
    if args.value is not None and args.value_file is not None:
        return _usage_error("Pass either a value or --value-file, not both.")
    value = args.value
    if args.value_file is not None:
        value = args.value_file.read_text(encoding="utf-8")
    if value is None:
        return _usage_error("`session set` requires either a value or --value-file.")
    return await _send_session_request(
        args,
        ToolCallRequest(
            command="session_set",
            var_name=args.name,
            var_value=value,
            var_secret=bool(args.secret),
        ),
    )


async def _handle_session_close(
    args: argparse.Namespace,
) -> tuple[ToolCallEnvelope, int]:
    session_id = getattr(args, "session", None)
    if not session_id:
        return _usage_error("`--session` is required.")

    metadata = load_session_metadata(session_id)
    if metadata is None or not is_process_alive(metadata.pid):
        return _missing_session(session_id)

    if args.force:
        terminated = terminate_session_process(session_id, force=False)
        if terminated:
            await asyncio.sleep(0.5)
        if is_process_alive(read_pid(session_id)):
            terminate_session_process(session_id, force=True)
            await asyncio.sleep(0.1)
        if is_process_alive(read_pid(session_id)):
            envelope = make_envelope(
                session_id=session_id,
                command="session",
                status=CommandStatus.ERROR,
                response=f"Session {session_id} could not be force-closed.",
                screenshot_path=metadata.latest_screenshot_path,
                exit_reason=ExitReason.AGENT_ERROR,
                duration_ms=0,
                actions_taken=0,
            )
            return envelope, 1
        metadata.status = "closed"
        metadata.closed_at = datetime.now(timezone.utc).isoformat()
        metadata.notes = "Force-closed by CLI."
        save_session_metadata(metadata)
        cleanup_session_artifacts(session_id)
        envelope = make_envelope(
            session_id=session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=(
                "Session force-closed. "
                f"{metadata.actions_executed} device actions were executed during this session."
            ),
            screenshot_path=None,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )
        return envelope, 0

    return await _send_session_request(
        args,
        ToolCallRequest(command="session_close"),
    )


async def _send_session_request(
    args: argparse.Namespace,
    request: ToolCallRequest,
) -> tuple[ToolCallEnvelope, int]:
    session_id = getattr(args, "session", None)
    if not session_id:
        return _usage_error("`--session` is required.")

    metadata = load_session_metadata(session_id)
    if metadata is None or not is_process_alive(metadata.pid):
        return _missing_session(session_id)

    socket_path = get_socket_path(session_id)
    if not socket_path.exists():
        return _missing_session(session_id)

    try:
        envelope = await send_request(socket_path, request)
    except Exception:
        envelope = make_envelope(
            session_id=session_id,
            command=public_command_name(request.command),
            status=CommandStatus.ERROR,
            response=(
                "Haindy daemon connection lost mid-command. "
                f"Check {get_daemon_log_path(session_id)}."
            ),
            screenshot_path=metadata.latest_screenshot_path,
            exit_reason=ExitReason.AGENT_ERROR,
            duration_ms=0,
            actions_taken=0,
        )
        return envelope, 1

    return envelope, envelope_exit_code(envelope)


def _resolve_requested_backend(
    args: argparse.Namespace,
    settings: object,
) -> str:
    if args.android:
        return "mobile_adb"
    if args.desktop:
        return "desktop"
    if getattr(args, "ios", False):
        return "mobile_ios"
    return normalize_automation_backend(
        getattr(settings, "automation_backend", "desktop")
    )


def _startup_response_from_metadata(metadata: object) -> str:
    backend = getattr(metadata, "backend", "desktop")
    if backend == "mobile_adb":
        serial = getattr(metadata, "android_serial", None)
        serial_suffix = f" Device found: {serial}." if serial else ""
        return f"Session started with Android ADB backend.{serial_suffix}"
    if backend == "mobile_ios":
        udid = getattr(metadata, "ios_udid", None)
        udid_suffix = f" Device UDID: {udid}." if udid else ""
        return f"Session started with iOS idb backend.{udid_suffix}"
    return "Session started with desktop backend."


def _missing_session(session_id: str) -> tuple[ToolCallEnvelope, int]:
    envelope = make_envelope(
        session_id=session_id,
        command="session",
        status=CommandStatus.ERROR,
        response=f"No active session found for {session_id}.",
        screenshot_path=None,
        exit_reason=ExitReason.AGENT_ERROR,
        duration_ms=0,
        actions_taken=0,
    )
    return envelope, 3


def _usage_error(message: str) -> tuple[ToolCallEnvelope, int]:
    envelope = make_envelope(
        session_id=None,
        command="session",
        status=CommandStatus.ERROR,
        response=message,
        screenshot_path=None,
        exit_reason=ExitReason.AGENT_ERROR,
        duration_ms=0,
        actions_taken=0,
    )
    return envelope, 2


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
