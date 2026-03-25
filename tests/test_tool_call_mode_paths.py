"""Tests for tool-call session path helpers."""

from pathlib import Path

from haindy.tool_call_mode.models import SessionMetadata
from haindy.tool_call_mode.paths import (
    get_haindy_home,
    get_session_dir,
    load_session_metadata,
    save_session_metadata,
)


def test_get_haindy_home_prefers_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "custom-home"))

    assert get_haindy_home() == (tmp_path / "custom-home")


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
