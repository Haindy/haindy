"""Tests for tool-call session path helpers."""

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

    for metadata in (old_metadata, recent_metadata, live_metadata):
        get_session_dir(metadata.session_id).mkdir(parents=True, exist_ok=True)
        save_session_metadata(metadata)
    write_pid_file(live_session, 4321)

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == 4321,
    )

    pruned = prune_dead_sessions(older_than_days=7)

    assert pruned == [old_dead]
    assert get_session_dir(old_dead).exists() is False
    assert get_session_dir(recent_dead).exists() is True
    assert get_session_dir(live_session).exists() is True
