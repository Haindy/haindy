"""Tests for tool-call session path helpers."""

import os
import shutil
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from haindy.tool_call_mode.models import SessionMetadata
from haindy.tool_call_mode.paths import (
    ensure_session_layout,
    get_action_artifacts_dir,
    get_daemon_log_path,
    get_haindy_home,
    get_logs_dir,
    get_screenshots_dir,
    get_session_dir,
    get_sessions_root,
    get_socket_path,
    is_session_daemon_live,
    load_session_metadata,
    prune_dead_sessions,
    save_session_metadata,
    write_pid_file,
)


def test_get_haindy_home_prefers_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "custom-home"))

    assert get_haindy_home() == (tmp_path / "custom-home")


def test_session_layout_stays_under_home_sessions(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "haindy-home"
    session_id = "session-123"
    monkeypatch.setenv("HAINDY_HOME", str(home))

    session_dir = ensure_session_layout(session_id)

    assert get_sessions_root() == home / "sessions"
    assert session_dir == home / "sessions" / session_id
    assert get_screenshots_dir(session_id) == session_dir / "screenshots"
    assert get_logs_dir(session_id) == session_dir / "logs"
    assert get_daemon_log_path(session_id) == session_dir / "logs" / "daemon.log"
    assert get_action_artifacts_dir(session_id) == session_dir / "action_artifacts"
    assert get_screenshots_dir(session_id).is_dir()
    assert get_logs_dir(session_id).is_dir()
    assert get_action_artifacts_dir(session_id).is_dir()


def test_save_and_load_session_metadata_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    session_id = "session-123"
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="desktop",
        idle_timeout_seconds=1800,
    )
    metadata.pid = 1234
    metadata.latest_screenshot_path = "/tmp/shot.png"
    save_session_metadata(metadata)

    loaded = load_session_metadata(session_id)

    assert loaded is not None
    assert loaded.session_id == session_id
    assert loaded.backend == "desktop"
    assert loaded.pid == 1234
    assert loaded.latest_screenshot_path == "/tmp/shot.png"


def test_prune_dead_sessions_removes_only_old_dead_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))

    old_dead = "old-dead"
    old_created_recently_closed = "old-created-recently-closed"
    recent_dead = "recent-dead"
    live_session = "live-session"

    old_metadata = SessionMetadata.new(
        session_id=old_dead,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "created_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        }
    )
    recently_closed_metadata = SessionMetadata.new(
        session_id=old_created_recently_closed,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "created_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "closed_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "status": "closed",
        }
    )
    recent_metadata = SessionMetadata.new(
        session_id=recent_dead,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "created_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        }
    )
    live_metadata = SessionMetadata.new(
        session_id=live_session,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "created_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "pid": 4321,
        }
    )

    for metadata in (
        old_metadata,
        recently_closed_metadata,
        recent_metadata,
        live_metadata,
    ):
        get_session_dir(metadata.session_id).mkdir(parents=True, exist_ok=True)
        save_session_metadata(metadata)
    write_pid_file(live_session, 4321)
    get_socket_path(live_session).touch()

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == 4321,
    )
    monkeypatch.setattr(Path, "is_socket", lambda path: path.name == "daemon.sock")

    pruned = prune_dead_sessions(older_than_days=7)

    assert pruned == [old_dead]
    assert get_session_dir(old_dead).exists() is False
    assert get_session_dir(old_created_recently_closed).exists() is True
    assert get_session_dir(recent_dead).exists() is True
    assert get_session_dir(live_session).exists() is True


def test_closed_session_with_reused_pid_is_not_live(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    session_id = "closed-reused-pid"
    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "pid": os.getpid(),
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)
    write_pid_file(session_id, os.getpid())
    get_socket_path(session_id).touch()

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == os.getpid(),
    )

    assert is_session_daemon_live(metadata) is False


def test_session_requires_matching_pid_file_and_socket(monkeypatch) -> None:
    short_home = Path("/tmp") / f"hdy-{os.getpid()}-live"
    shutil.rmtree(short_home, ignore_errors=True)
    monkeypatch.setenv("HAINDY_HOME", str(short_home))
    session_id = "s"
    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(update={"pid": os.getpid(), "status": "ready"})
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == os.getpid(),
    )

    assert is_session_daemon_live(metadata) is False
    write_pid_file(session_id, os.getpid())
    assert is_session_daemon_live(metadata) is False
    get_socket_path(session_id).touch()
    assert is_session_daemon_live(metadata) is False
    get_socket_path(session_id).unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(get_socket_path(session_id)))
        server.listen(1)
        assert is_session_daemon_live(metadata) is True
    finally:
        server.close()
        shutil.rmtree(short_home, ignore_errors=True)


def test_ready_session_with_unreachable_reused_pid_is_not_live(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    session_id = "ready-reused-pid"
    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="desktop",
        idle_timeout_seconds=1800,
    ).model_copy(update={"pid": os.getpid(), "status": "ready"})
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)
    write_pid_file(session_id, os.getpid())
    get_socket_path(session_id).touch()

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == os.getpid(),
    )

    assert is_session_daemon_live(metadata) is False
