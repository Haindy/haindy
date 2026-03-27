"""Tests for tool-call daemon launch helpers and survival semantics."""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from haindy.tool_call_mode.daemon import ToolCallDaemon
from haindy.tool_call_mode.launcher import (
    public_cli_program_name,
    resolve_cli_executable_argv,
)
from haindy.tool_call_mode.paths import (
    cleanup_session_artifacts,
    is_process_alive,
    load_session_metadata,
)


def test_resolve_cli_executable_argv_prefers_current_haindy_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "haindy"
    script_path.write_text("#!/bin/sh\n", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr("haindy.tool_call_mode.launcher.sys.argv", [str(script_path)])
    monkeypatch.setattr(
        "haindy.tool_call_mode.launcher.sys.executable",
        str(tmp_path / "python"),
    )
    monkeypatch.setattr("haindy.tool_call_mode.launcher.shutil.which", lambda _: None)

    assert resolve_cli_executable_argv() == [str(script_path.resolve())]


def test_resolve_cli_executable_argv_falls_back_to_python_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")
    python_path.chmod(python_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr("haindy.tool_call_mode.launcher.sys.argv", ["pytest"])
    monkeypatch.setattr(
        "haindy.tool_call_mode.launcher.sys.executable",
        str(python_path),
    )
    monkeypatch.setattr("haindy.tool_call_mode.launcher.shutil.which", lambda _: None)

    assert resolve_cli_executable_argv() == [str(python_path), "-m", "haindy.main"]


def test_tool_call_daemon_records_external_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = ToolCallDaemon(
        session_id="signal-test",
        backend="desktop",
        idle_timeout_seconds=1800,
    )
    saved_notes: list[str | None] = []

    def _save_metadata(metadata: object) -> None:
        saved_notes.append(getattr(metadata, "notes", None))

    monkeypatch.setattr(
        "haindy.tool_call_mode.daemon.save_session_metadata", _save_metadata
    )

    daemon._handle_shutdown_signal("SIGTERM")

    assert daemon.runtime.metadata.status == "closing"
    assert (
        daemon.runtime.metadata.notes == "External shutdown signal received: SIGTERM."
    )
    assert daemon._shutdown_event.is_set() is True
    assert daemon._shutdown_reason == "signal:SIGTERM"
    assert saved_notes[-1] == "External shutdown signal received: SIGTERM."


@pytest.mark.skipif(
    os.name != "posix", reason="Tool-call daemon uses POSIX process semantics."
)
def test_tool_call_daemon_survives_session_new_process_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_path = _haindy_cli_path()
    base_dir = Path(tempfile.mkdtemp(prefix="haindy-tool-call-"))
    home_dir = base_dir / "home"
    sitecustomize_dir = base_dir / "stub-runtime"
    sitecustomize_dir.mkdir()
    (sitecustomize_dir / "sitecustomize.py").write_text(
        dedent("""
            from __future__ import annotations

            import time
            from datetime import datetime, timezone

            from haindy.tool_call_mode.models import CommandStatus, ExitReason, make_envelope, public_command_name
            from haindy.tool_call_mode.paths import save_session_metadata
            from haindy.tool_call_mode.runtime import ToolCallSessionRuntime


            def _now() -> str:
                return datetime.now(timezone.utc).isoformat()


            async def _fake_start(self):
                self.metadata.status = "ready"
                self.metadata.last_command_at = _now()
                save_session_metadata(self.metadata)
                return make_envelope(
                    session_id=self.session_id,
                    command="session",
                    status=CommandStatus.SUCCESS,
                    response="Stub runtime started.",
                    screenshot_path=None,
                    exit_reason=ExitReason.COMPLETED,
                    duration_ms=0,
                    actions_taken=0,
                )


            async def _fake_stop(self):
                self.metadata.status = "closed"
                self.metadata.closed_at = _now()
                save_session_metadata(self.metadata)


            async def _fake_handle_request(self, request):
                self._last_activity_monotonic = time.monotonic()
                self.metadata.last_command_at = _now()
                if request.command == "session_close":
                    self._close_requested = True
                    response = "Session closed. 0 device actions were executed during this session."
                elif request.command == "session_status":
                    response = "Session is active. Stub runtime ready."
                else:
                    response = f"Stub handled {request.command}."
                envelope = make_envelope(
                    session_id=self.session_id,
                    command=public_command_name(request.command),
                    status=CommandStatus.SUCCESS,
                    response=response,
                    screenshot_path=self.metadata.latest_screenshot_path,
                    exit_reason=ExitReason.COMPLETED,
                    duration_ms=0,
                    actions_taken=0,
                )
                self.metadata.last_command_name = envelope.command
                self.metadata.commands_executed += 1
                save_session_metadata(self.metadata)
                return envelope


            ToolCallSessionRuntime.start = _fake_start
            ToolCallSessionRuntime.stop = _fake_stop
            ToolCallSessionRuntime.handle_request = _fake_handle_request
            """),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HAINDY_HOME"] = str(home_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(sitecustomize_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    monkeypatch.setenv("HAINDY_HOME", str(home_dir))

    session_id: str | None = None
    metadata_pid: int | None = None
    try:
        new_result = subprocess.run(
            [str(cli_path), "session", "new", "--desktop"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=30,
        )
        assert new_result.returncode == 0, new_result.stderr
        new_payload = json.loads(new_result.stdout)
        session_id = str(new_payload["session_id"])
        assert new_payload["status"] == "success"

        metadata = load_session_metadata(session_id)
        assert metadata is not None
        metadata_pid = metadata.pid
        assert metadata.status == "ready"
        assert is_process_alive(metadata_pid) is True

        status_result = subprocess.run(
            [str(cli_path), "session", "status", "--session", session_id],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=30,
        )
        assert status_result.returncode == 0, status_result.stderr
        status_payload = json.loads(status_result.stdout)
        assert status_payload["status"] == "success"
        assert "Stub runtime ready." in status_payload["response"]

        close_result = subprocess.run(
            [str(cli_path), "session", "close", "--session", session_id],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=30,
        )
        assert close_result.returncode == 0, close_result.stderr
    finally:
        if session_id is not None:
            force_close = subprocess.run(
                [str(cli_path), "session", "close", "--session", session_id, "--force"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=30,
            )
            if force_close.returncode not in {0, 3}:
                pytest.fail(force_close.stderr or force_close.stdout)
            cleanup_session_artifacts(session_id)
        if metadata_pid and is_process_alive(metadata_pid):
            os.kill(metadata_pid, signal.SIGKILL)
        shutil.rmtree(base_dir, ignore_errors=True)


def test_installed_haindy_help_smoke() -> None:
    cli_path = _haindy_cli_path()

    result = subprocess.run(
        [str(cli_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert f"{public_cli_program_name()} run --plan requirements.md" in result.stdout
    assert "python -m haindy.main run --plan requirements.md" in result.stdout
    assert result.stdout.index(
        f"{public_cli_program_name()} run --plan"
    ) < result.stdout.index("python -m haindy.main run --plan")


def _haindy_cli_path() -> Path:
    cli_path = Path(sys.executable).with_name(public_cli_program_name())
    assert cli_path.exists(), (
        f"Expected installed console script at {cli_path}. "
        'Run `.venv/bin/pip install -e ".[dev]"` before tests.'
    )
    return cli_path.resolve()
