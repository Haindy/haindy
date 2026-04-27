"""Tests for tool-call parser and CLI helpers."""

import json
import os
from pathlib import Path

import pytest

from haindy.tool_call_mode.cli import (
    _handle_session_list,
    _handle_session_new,
    _handle_session_prune,
    create_tool_call_parser,
    is_tool_call_command,
    run_tool_call_cli,
)
from haindy.tool_call_mode.launcher import ToolCallDaemonLaunch
from haindy.tool_call_mode.models import SessionMetadata
from haindy.tool_call_mode.paths import (
    get_session_dir,
    save_session_metadata,
    write_pid_file,
)


def test_tool_call_parser_accepts_session_after_subcommand() -> None:
    parser = create_tool_call_parser()

    parsed = parser.parse_args(["act", "tap the login button", "--session", "abc123"])

    assert parsed.tool_command == "act"
    assert parsed.session == "abc123"
    assert parsed.instruction == "tap the login button"


def test_tool_call_parser_accepts_global_flags_before_subcommand() -> None:
    parser = create_tool_call_parser()

    parsed = parser.parse_args(
        ["--debug", "--json", "act", "tap the login button", "--session", "abc123"]
    )

    assert parsed.debug is True
    assert parsed.json is True
    assert parsed.tool_command == "act"
    assert parsed.session == "abc123"


def test_tool_call_parser_accepts_session_before_subcommand() -> None:
    parser = create_tool_call_parser()

    parsed = parser.parse_args(["--session", "abc123", "act", "tap the login button"])

    assert parsed.tool_command == "act"
    assert parsed.session == "abc123"
    assert parsed.instruction == "tap the login button"


def test_tool_call_parser_accepts_session_set_value_file_after_subcommand(
    tmp_path: Path,
) -> None:
    parser = create_tool_call_parser()
    value_file = tmp_path / "secret.txt"
    value_file.write_text("hunter2", encoding="utf-8")

    parsed = parser.parse_args(
        [
            "session",
            "set",
            "PASSWORD",
            "--value-file",
            str(value_file),
            "--secret",
            "--session",
            "abc123",
        ]
    )

    assert parsed.tool_command == "session"
    assert parsed.session_command == "set"
    assert parsed.session == "abc123"
    assert parsed.value_file == value_file
    assert parsed.secret is True


def test_tool_call_parser_accepts_test_status_and_explore_commands() -> None:
    parser = create_tool_call_parser()

    test_status = parser.parse_args(["test-status", "--session", "abc123"])
    explore = parser.parse_args(
        [
            "explore",
            "map the settings screen",
            "--max-steps",
            "8",
            "--timeout",
            "120",
            "--session",
            "abc123",
        ]
    )
    explore_status = parser.parse_args(["explore-status", "--session", "abc123"])

    assert test_status.tool_command == "test-status"
    assert test_status.session == "abc123"
    assert explore.tool_command == "explore"
    assert explore.goal == "map the settings screen"
    assert explore.max_steps == 8
    assert explore.timeout == 120
    assert explore.session == "abc123"
    assert explore_status.tool_command == "explore-status"
    assert explore_status.session == "abc123"


def test_tool_call_parser_accepts_session_prune() -> None:
    parser = create_tool_call_parser()

    parsed = parser.parse_args(["session", "prune", "--older-than", "7"])

    assert parsed.tool_command == "session"
    assert parsed.session_command == "prune"
    assert parsed.older_than == 7


def test_is_tool_call_command_skips_global_flags() -> None:
    assert (
        is_tool_call_command(
            ["--debug", "--json", "--session", "abc123", "explore-status"]
        )
        is True
    )
    assert is_tool_call_command(["--session", "abc123", "run", "--plan", "req.md"]) is (
        False
    )


def test_handle_session_list_filters_dead_sessions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))

    live_session = "live-session"
    dead_session = "dead-session"

    get_session_dir(live_session).mkdir(parents=True, exist_ok=True)
    live_metadata = SessionMetadata.new(
        session_id=live_session,
        backend="desktop",
        idle_timeout_seconds=1800,
    )
    save_session_metadata(live_metadata)

    get_session_dir(dead_session).mkdir(parents=True, exist_ok=True)
    dead_metadata = SessionMetadata.new(
        session_id=dead_session,
        backend="mobile_adb",
        idle_timeout_seconds=1800,
    )
    dead_metadata.pid = 999999
    save_session_metadata(dead_metadata)
    write_pid_file(dead_session, 999999)

    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.is_session_daemon_live",
        lambda metadata: metadata is not None and metadata.session_id == live_session,
    )

    envelope, exit_code = _handle_session_list()

    assert exit_code == 0
    assert envelope.status.value == "success"
    assert envelope.sessions is not None
    assert [entry.session_id for entry in envelope.sessions] == [live_session]


def test_handle_session_list_ignores_closed_session_with_reused_pid(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    session_id = "closed-reused-pid"
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)
    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="mobile_ios",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "pid": os.getpid(),
            "status": "closed",
            "closed_at": "2026-04-21T10:36:34+00:00",
        }
    )
    save_session_metadata(metadata)
    write_pid_file(session_id, os.getpid())
    (get_session_dir(session_id) / "daemon.sock").touch()

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == os.getpid(),
    )

    envelope, exit_code = _handle_session_list()

    assert exit_code == 0
    assert envelope.sessions == []


@pytest.mark.asyncio
async def test_session_close_rejects_closed_session_with_reused_pid(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    session_id = "closed-reused-pid"
    get_session_dir(session_id).mkdir(parents=True, exist_ok=True)
    metadata = SessionMetadata.new(
        session_id=session_id,
        backend="mobile_ios",
        idle_timeout_seconds=1800,
    ).model_copy(
        update={
            "pid": os.getpid(),
            "status": "closed",
            "closed_at": "2026-04-21T10:36:34+00:00",
        }
    )
    save_session_metadata(metadata)
    write_pid_file(session_id, os.getpid())
    (get_session_dir(session_id) / "daemon.sock").touch()

    monkeypatch.setattr(
        "haindy.tool_call_mode.paths.is_process_alive",
        lambda pid: pid == os.getpid(),
    )

    exit_code = await run_tool_call_cli(["session", "close", "--session", session_id])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert payload["response"] == f"No active session found for {session_id}."


def test_handle_session_prune_reports_pruned_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.prune_dead_sessions",
        lambda *, older_than_days: ["old-a", "old-b"],
    )
    args = create_tool_call_parser().parse_args(
        ["session", "prune", "--older-than", "14"]
    )

    envelope, exit_code = _handle_session_prune(args)

    assert exit_code == 0
    assert envelope.status.value == "success"
    assert envelope.response == "Pruned 2 session directories older than 14 day(s)."


@pytest.mark.asyncio
async def test_run_tool_call_cli_returns_json_usage_envelope_on_bad_args(
    capsys,
) -> None:
    exit_code = await run_tool_call_cli(["act", "tap the login button"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["command"] == "act"
    assert (
        "`--session` is required." in payload["response"]
        or "required" in payload["response"]
    )


@pytest.mark.asyncio
async def test_handle_session_new_launches_daemon_with_expected_settings(
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    session_id = "session-detach-test"
    read_fd, write_fd = os.pipe()

    def fake_launch_tool_call_daemon(**kwargs):
        captured_kwargs.update(kwargs)
        os.write(write_fd, b"1")
        os.close(write_fd)
        return ToolCallDaemonLaunch(
            command=("haindy", "__tool_call_daemon"),
            readiness_fd=read_fd,
        )

    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.cleanup_stale_sessions", lambda: None
    )
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.ensure_session_layout", lambda value: None
    )
    monkeypatch.setattr("haindy.tool_call_mode.cli.uuid4", lambda: session_id)
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.get_settings",
        lambda: type(
            "Settings",
            (),
            {"automation_backend": "desktop", "haindy_home": Path("/tmp/haindy")},
        )(),
    )
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.launch_tool_call_daemon",
        fake_launch_tool_call_daemon,
    )
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.load_session_metadata",
        lambda _: SessionMetadata.new(
            session_id=session_id,
            backend="mobile_adb",
            idle_timeout_seconds=1800,
            android_serial="emulator-5554",
            android_app="co.playerup.flutterApp",
        ).model_copy(update={"pid": 1234}),
    )
    monkeypatch.setattr(
        "haindy.tool_call_mode.cli.is_session_daemon_live", lambda metadata: True
    )

    envelope, exit_code = await _handle_session_new(
        create_tool_call_parser().parse_args(
            [
                "session",
                "new",
                "--android",
                "--android-serial",
                "emulator-5554",
                "--android-app",
                "co.playerup.flutterApp",
            ]
        )
    )

    assert exit_code == 0
    assert envelope.status.value == "success"
    assert captured_kwargs["session_id"] == session_id
    assert captured_kwargs["backend"] == "mobile_adb"
    assert captured_kwargs["idle_timeout"] == 1800
    assert captured_kwargs["android_serial"] == "emulator-5554"
    assert captured_kwargs["android_app"] == "co.playerup.flutterApp"
