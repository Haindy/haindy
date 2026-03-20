"""Helpers for launching detached tool-call daemons."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PUBLIC_CLI_NAME = "haindy"
_HIDDEN_DAEMON_COMMAND = "__tool_call_daemon"


class ToolCallDaemonLaunchError(RuntimeError):
    """Raised when the daemon launcher cannot start a detached child."""


@dataclass(frozen=True)
class ToolCallDaemonLaunch:
    """Startup handle returned to the caller after daemon launch."""

    command: tuple[str, ...]
    readiness_fd: int


def public_cli_program_name() -> str:
    """Return the canonical public CLI program name shown to users."""

    return PUBLIC_CLI_NAME


def resolve_cli_executable_argv() -> list[str]:
    """Return argv for re-executing the HAINDY CLI from the current env."""

    script_path = _resolve_installed_cli_script()
    if script_path is not None:
        return [script_path]
    return [sys.executable, "-m", "src.main"]


def build_tool_call_daemon_command(
    *,
    session_id: str,
    backend: str,
    idle_timeout: int,
    android_serial: str | None = None,
    android_app: str | None = None,
    debug: bool = False,
) -> list[str]:
    """Build the argv used to launch the hidden daemon entrypoint."""

    command = [
        *resolve_cli_executable_argv(),
        _HIDDEN_DAEMON_COMMAND,
        "--session-id",
        session_id,
        "--backend",
        backend,
        "--idle-timeout",
        str(idle_timeout),
    ]
    if android_serial:
        command.extend(["--android-serial", android_serial])
    if android_app:
        command.extend(["--android-app", android_app])
    if debug:
        command.append("--debug")
    return command


def launch_tool_call_daemon(
    *,
    session_id: str,
    backend: str,
    idle_timeout: int,
    android_serial: str | None = None,
    android_app: str | None = None,
    debug: bool = False,
    env: Mapping[str, str] | None = None,
) -> ToolCallDaemonLaunch:
    """Launch the hidden tool-call daemon as a detached grandchild process."""

    if not hasattr(os, "fork"):
        raise ToolCallDaemonLaunchError(
            "Tool-call daemonization requires POSIX fork support."
        )

    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    command = tuple(
        build_tool_call_daemon_command(
            session_id=session_id,
            backend=backend,
            idle_timeout=idle_timeout,
            android_serial=android_serial,
            android_app=android_app,
            debug=debug,
        )
    )
    child_env = dict(env or os.environ)
    child_env["HAINDY_READINESS_FD"] = str(write_fd)

    try:
        _spawn_detached_process(command, env=child_env)
    except Exception:
        os.close(read_fd)
        os.close(write_fd)
        raise

    os.close(write_fd)
    return ToolCallDaemonLaunch(command=command, readiness_fd=read_fd)


def _spawn_detached_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
) -> None:
    """Double-fork, detach, and exec the requested command."""

    try:
        first_pid = os.fork()
    except OSError as exc:
        raise ToolCallDaemonLaunchError(
            f"Unable to fork the tool-call daemon launcher: {exc}."
        ) from exc

    if first_pid == 0:
        try:
            os.setsid()
            try:
                second_pid = os.fork()
            except OSError:
                os._exit(1)
            if second_pid > 0:
                os._exit(0)
            _redirect_stdio_to_devnull()
            os.execvpe(command[0], list(command), dict(env))
        except BaseException:
            os._exit(1)

    status = _waitpid_blocking(first_pid)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return
    if os.WIFSIGNALED(status):
        signal_number = os.WTERMSIG(status)
        raise ToolCallDaemonLaunchError(
            "Tool-call daemon launcher terminated before detaching "
            f"(signal {signal_number})."
        )
    raise ToolCallDaemonLaunchError(
        "Tool-call daemon launcher exited before detaching."
    )


def _waitpid_blocking(pid: int) -> int:
    """Wait for a child process while tolerating EINTR."""

    while True:
        try:
            _, status = os.waitpid(pid, 0)
            return status
        except InterruptedError:
            continue


def _redirect_stdio_to_devnull() -> None:
    """Disconnect stdio from the parent terminal before exec."""

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        for destination in (0, 1, 2):
            os.dup2(devnull_fd, destination)
    finally:
        if devnull_fd > 2:
            os.close(devnull_fd)


def _resolve_installed_cli_script() -> str | None:
    """Return the best available installed console-script path, if present."""

    current = _normalize_executable_candidate(sys.argv[0], require_name=True)
    if current is not None:
        return current

    interpreter_sibling = _normalize_executable_candidate(
        Path(sys.executable).with_name(PUBLIC_CLI_NAME),
        require_name=False,
    )
    if interpreter_sibling is not None:
        return interpreter_sibling

    on_path = shutil.which(PUBLIC_CLI_NAME)
    return _normalize_executable_candidate(on_path, require_name=False)


def _normalize_executable_candidate(
    candidate: str | os.PathLike[str] | None,
    *,
    require_name: bool,
) -> str | None:
    """Validate a candidate console-script path and normalize it to a string."""

    if candidate is None:
        return None
    path = Path(candidate).expanduser()
    if require_name and path.name != PUBLIC_CLI_NAME:
        return None
    if not path.is_absolute():
        resolved = shutil.which(str(path))
        if resolved is None:
            return None
        path = Path(resolved)
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        return None
    if resolved_path.name != PUBLIC_CLI_NAME:
        return None
    if not os.access(resolved_path, os.X_OK):
        return None
    return str(resolved_path)
