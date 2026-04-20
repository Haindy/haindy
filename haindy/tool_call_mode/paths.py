"""Filesystem helpers for tool-call session state."""

from __future__ import annotations

import json
import os
import shutil
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

from haindy.config.settings import get_settings

from .models import SessionMetadata


def get_haindy_home() -> Path:
    """Return the base directory for tool-call session state."""

    env_override = os.environ.get("HAINDY_HOME")
    if env_override is not None:
        raw = env_override.strip() or "~/.haindy"
        return Path(raw).expanduser()
    return Path(getattr(get_settings(), "haindy_home", Path("~/.haindy"))).expanduser()


def get_sessions_root() -> Path:
    """Return the root directory for all session directories."""

    return get_haindy_home() / "sessions"


def get_session_dir(session_id: str) -> Path:
    """Return the directory for one session."""

    return get_sessions_root() / session_id


def get_socket_path(session_id: str) -> Path:
    """Return the daemon socket path for one session."""

    return get_session_dir(session_id) / "daemon.sock"


def get_pid_path(session_id: str) -> Path:
    """Return the daemon pid file path for one session."""

    return get_session_dir(session_id) / "daemon.pid"


def get_metadata_path(session_id: str) -> Path:
    """Return the session metadata file path."""

    return get_session_dir(session_id) / "session.json"


def get_screenshots_dir(session_id: str) -> Path:
    """Return the session screenshots directory."""

    return get_session_dir(session_id) / "screenshots"


def get_logs_dir(session_id: str) -> Path:
    """Return the session logs directory."""

    return get_session_dir(session_id) / "logs"


def get_action_artifacts_dir(session_id: str) -> Path:
    """Return the session-local action artifact directory."""

    return get_session_dir(session_id) / "action_artifacts"


def get_daemon_log_path(session_id: str) -> Path:
    """Return the daemon log path for one session."""

    return get_logs_dir(session_id) / "daemon.log"


def ensure_session_layout(session_id: str) -> Path:
    """Create the session directory structure and return the session dir."""

    session_dir = get_session_dir(session_id)
    get_sessions_root().mkdir(parents=True, exist_ok=True)
    get_screenshots_dir(session_id).mkdir(parents=True, exist_ok=True)
    get_logs_dir(session_id).mkdir(parents=True, exist_ok=True)
    get_action_artifacts_dir(session_id).mkdir(parents=True, exist_ok=True)
    return session_dir


def write_pid_file(session_id: str, pid: int) -> None:
    """Persist the daemon pid for one session."""

    get_pid_path(session_id).write_text(f"{int(pid)}\n", encoding="utf-8")


def read_pid(session_id: str) -> int | None:
    """Read the daemon pid if it exists."""

    path = get_pid_path(session_id)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (TypeError, ValueError):
        return None


def is_process_alive(pid: int | None) -> bool:
    """Return True when a pid currently exists."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_session_metadata(session_id: str) -> SessionMetadata | None:
    """Load persisted metadata for one session, if present."""

    path = get_metadata_path(session_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return SessionMetadata.model_validate(payload)
    except Exception:
        return None


def save_session_metadata(metadata: SessionMetadata) -> None:
    """Persist one session metadata payload."""

    path = get_metadata_path(metadata.session_id)
    path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def cleanup_session_artifacts(session_id: str, *, remove_dir: bool = False) -> None:
    """Best-effort cleanup for dead session artifacts."""

    socket_path = get_socket_path(session_id)
    pid_path = get_pid_path(session_id)
    with contextlib_suppress(FileNotFoundError):
        socket_path.unlink()
    with contextlib_suppress(FileNotFoundError):
        pid_path.unlink()
    if remove_dir:
        shutil.rmtree(get_session_dir(session_id), ignore_errors=True)


def cleanup_stale_sessions() -> None:
    """Remove stale socket files for dead daemons."""

    sessions_root = get_sessions_root()
    if not sessions_root.exists():
        return
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        pid = read_pid(session_id)
        if is_process_alive(pid):
            continue
        cleanup_session_artifacts(session_id)


def prune_dead_sessions(*, older_than_days: int) -> list[str]:
    """Delete dead session directories older than the requested age."""

    threshold = datetime.now(timezone.utc) - timedelta(
        days=max(int(older_than_days), 0)
    )
    pruned: list[str] = []
    sessions_root = get_sessions_root()
    if not sessions_root.exists():
        return pruned

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        metadata = load_session_metadata(session_id)
        pid = read_pid(session_id)
        if is_process_alive(pid):
            continue
        if metadata is None:
            continue
        try:
            created_at = datetime.fromisoformat(metadata.created_at)
        except ValueError:
            continue
        if created_at > threshold:
            continue
        cleanup_session_artifacts(session_id, remove_dir=True)
        pruned.append(session_id)
    return pruned


def terminate_session_process(session_id: str, *, force: bool = False) -> bool:
    """Terminate a session daemon by pid if it is still running."""

    pid = read_pid(session_id)
    if not is_process_alive(pid):
        return False
    assert pid is not None
    sig = getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
    os.kill(pid, sig)
    return True


class contextlib_suppress:
    """Tiny local suppress helper to keep this module dependency-free."""

    def __init__(self, *exceptions: type[BaseException]) -> None:
        self._exceptions = exceptions

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: object,
    ) -> bool:
        return bool(exc_type and issubclass(exc_type, self._exceptions))
