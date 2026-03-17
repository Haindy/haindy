"""Tests for tool-call parser and CLI helpers."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.tool_call_mode.cli import (
    _handle_session_list,
    _handle_session_new,
    create_tool_call_parser,
    run_tool_call_cli,
)
from src.tool_call_mode.models import SessionMetadata
from src.tool_call_mode.paths import (
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
    live_metadata.pid = 4321
    save_session_metadata(live_metadata)
    write_pid_file(live_session, 4321)

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
        "src.tool_call_mode.cli.is_process_alive", lambda pid: pid == 4321
    )

    envelope, exit_code = _handle_session_list()

    assert exit_code == 0
    assert envelope.status.value == "success"
    assert envelope.sessions is not None
    assert [entry.session_id for entry in envelope.sessions] == [live_session]


@pytest.mark.asyncio
async def test_run_tool_call_cli_returns_json_usage_envelope_on_bad_args(
    capsys,
) -> None:
    exit_code = await run_tool_call_cli(["act", "tap the login button"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["command"] == "session"
    assert (
        "`--session` is required." in payload["response"]
        or "required" in payload["response"]
    )


@pytest.mark.asyncio
async def test_handle_session_new_detaches_daemon_stdio(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None

        async def wait(self) -> int:
            await asyncio.sleep(60)
            return 0

    captured_kwargs: dict[str, object] = {}
    session_id = "session-detach-test"

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured_kwargs.update(kwargs)
        readiness_fd = int(kwargs["env"]["HAINDY_READINESS_FD"])
        os.write(readiness_fd, b"1")
        return FakeProcess()

    monkeypatch.setattr("src.tool_call_mode.cli.cleanup_stale_sessions", lambda: None)
    monkeypatch.setattr(
        "src.tool_call_mode.cli.ensure_session_layout", lambda value: None
    )
    monkeypatch.setattr("src.tool_call_mode.cli.uuid4", lambda: session_id)
    monkeypatch.setattr(
        "src.tool_call_mode.cli.get_settings",
        lambda: type(
            "Settings",
            (),
            {"automation_backend": "desktop", "haindy_home": Path("/tmp/haindy")},
        )(),
    )
    monkeypatch.setattr(
        "src.tool_call_mode.cli.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "src.tool_call_mode.cli.load_session_metadata",
        lambda _: SessionMetadata.new(
            session_id=session_id,
            backend="mobile_adb",
            idle_timeout_seconds=1800,
            android_serial="emulator-5554",
            android_app="co.playerup.flutterApp",
        ).model_copy(update={"pid": 1234}),
    )
    monkeypatch.setattr("src.tool_call_mode.cli.is_process_alive", lambda pid: True)

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
    assert captured_kwargs["start_new_session"] is True
    assert captured_kwargs["stdin"] == subprocess.DEVNULL
    assert captured_kwargs["stdout"] == subprocess.DEVNULL
    assert captured_kwargs["stderr"] == subprocess.DEVNULL
