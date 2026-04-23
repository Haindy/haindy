"""Tests for the HAINDY test-forensics artifact locator helper."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from haindy.config.settings import SETTINGS_ENV_VARS, build_project_data_dir

_HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "haindy-test-forensics"
    / "scripts"
    / "locate_run_artifacts.py"
)


@pytest.fixture
def helper_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "haindy_test_forensics_locator",
        _HELPER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _isolate_haindy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_name in SETTINGS_ENV_VARS.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trace(
    data_root: Path,
    run_id: str,
    *,
    success: bool = False,
    model_log_path: Path | None = None,
) -> Path:
    trace_path = data_root / "traces" / f"{run_id}.json"
    metadata: dict[str, object] = {
        "automation_backend": "desktop",
        "test_plan_name": "Forensics fixture",
    }
    if model_log_path is not None:
        metadata["model_log_path"] = str(model_log_path)
    _write_json(
        trace_path,
        {
            "run_id": run_id,
            "success": success,
            "started_at": "2026-04-10T00:00:00Z",
            "ended_at": "2026-04-10T00:00:10Z",
            "run_metadata": metadata,
            "steps": [
                {
                    "scenario": "Login",
                    "step_number": 1,
                    "step_action": "Tap login",
                    "expected_result": "Dashboard appears",
                    "step_result": {
                        "status": "failed",
                        "actual_result": "Login screen stayed visible",
                        "error_message": None,
                        "screenshot_before": "before.png",
                        "screenshot_after": "after.png",
                    },
                }
            ],
        },
    )
    return trace_path


def _write_session(
    haindy_home: Path,
    session_id: str,
    *,
    run_id: str | None = None,
    created_at: str = "2026-04-10T00:00:00+00:00",
) -> Path:
    session_dir = haindy_home / "sessions" / session_id
    latest_artifact = (
        session_dir / "action_artifacts" / "case_step_001.json" if run_id else None
    )
    _write_json(
        session_dir / "session.json",
        {
            "session_id": session_id,
            "backend": "desktop",
            "created_at": created_at,
            "status": "ready",
            "last_command_at": created_at,
            "last_command_name": "test" if run_id else "explore",
            "latest_screenshot_path": str(session_dir / "screenshots" / "step_001.png"),
            "latest_background_run_id": run_id,
            "latest_test_phase": "completed" if run_id else None,
            "latest_test_progress_at": created_at if run_id else None,
            "latest_test_action_artifact_path": (
                str(latest_artifact) if latest_artifact else None
            ),
        },
    )
    (session_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    (session_dir / "screenshots" / "step_001.png").write_bytes(b"png")
    (session_dir / "logs").mkdir(parents=True, exist_ok=True)
    (session_dir / "logs" / "daemon.log").write_text("daemon\n", encoding="utf-8")
    (session_dir / "logs" / "ai_interactions.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    if latest_artifact:
        _write_json(
            latest_artifact,
            {
                "run_id": run_id,
                "session_id": session_id,
                "phase": "completed",
            },
        )
    return session_dir


def _run_helper(
    helper_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, dict[str, object], str]:
    exit_code = helper_module.main(list(args))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else {}
    return exit_code, payload, captured.err


def test_run_id_uses_default_project_data_root(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    haindy_home = tmp_path / "haindy-home"
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HAINDY_HOME", str(haindy_home))
    data_root = build_project_data_dir(haindy_home, project_dir)
    trace_path = _write_trace(data_root, "run_default")

    exit_code, payload, stderr = _run_helper(
        helper_module,
        capsys,
        "--run-id",
        "run_default",
    )

    assert exit_code == 0
    assert stderr == ""
    assert payload["data_root"] == str(data_root)
    assert payload["trace_path"] == str(trace_path)


def test_latest_failed_uses_exact_data_dir_override(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root = tmp_path / "configured-data"
    monkeypatch.setenv("HAINDY_DATA_DIR", str(data_root))
    _write_trace(data_root, "passed", success=True)
    _write_trace(data_root, "failed", success=False)

    exit_code, payload, _stderr = _run_helper(helper_module, capsys)

    assert exit_code == 0
    assert payload["run_id"] == "failed"
    assert payload["data_root"] == str(data_root)


def test_report_dir_uses_reports_dir_override(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "run_reports"
    data_root = tmp_path / "data"
    reports_dir = tmp_path / "custom-reports"
    monkeypatch.setenv("HAINDY_DATA_DIR", str(data_root))
    monkeypatch.setenv("HAINDY_REPORTS_DIR", str(reports_dir))
    _write_trace(data_root, run_id)
    action_json = reports_dir / run_id / "run-actions.json"
    html_report = reports_dir / run_id / "index.html"
    action_json.parent.mkdir(parents=True)
    action_json.write_text("{}", encoding="utf-8")
    html_report.write_text("<html></html>", encoding="utf-8")

    exit_code, payload, _stderr = _run_helper(
        helper_module,
        capsys,
        "--run-id",
        run_id,
    )

    assert exit_code == 0
    assert payload["reports_dir"] == str(reports_dir)
    assert payload["report_dir"] == str(reports_dir / run_id)
    assert payload["actions_json_paths"] == [str(action_json)]
    assert payload["html_report_paths"] == [str(html_report)]


def test_session_id_links_session_artifacts_to_trace(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "session-run"
    data_root = tmp_path / "data"
    haindy_home = tmp_path / "haindy-home"
    monkeypatch.setenv("HAINDY_DATA_DIR", str(data_root))
    monkeypatch.setenv("HAINDY_HOME", str(haindy_home))
    _write_trace(data_root, run_id)
    session_dir = _write_session(haindy_home, "session-a", run_id=run_id)

    exit_code, payload, _stderr = _run_helper(
        helper_module,
        capsys,
        "--session-id",
        "session-a",
    )

    assert exit_code == 0
    assert payload["mode"] == "tool_call_session"
    assert payload["session_id"] == "session-a"
    assert payload["session_dir"] == str(session_dir)
    assert payload["action_artifact_paths"] == [
        str(session_dir / "action_artifacts" / "case_step_001.json")
    ]
    linked_trace = payload["linked_trace"]
    assert isinstance(linked_trace, dict)
    assert linked_trace["run_id"] == run_id


def test_run_id_summary_correlates_matching_tool_call_sessions(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "correlated-run"
    data_root = tmp_path / "data"
    haindy_home = tmp_path / "haindy-home"
    monkeypatch.setenv("HAINDY_DATA_DIR", str(data_root))
    monkeypatch.setenv("HAINDY_HOME", str(haindy_home))
    _write_trace(data_root, run_id)
    _write_session(haindy_home, "matching-session", run_id=run_id)
    _write_session(haindy_home, "other-session", run_id="other-run")

    exit_code, payload, _stderr = _run_helper(
        helper_module,
        capsys,
        "--run-id",
        run_id,
    )

    assert exit_code == 0
    sessions = payload["matching_tool_call_sessions"]
    assert isinstance(sessions, list)
    assert [session["session_id"] for session in sessions] == ["matching-session"]


def test_latest_session_uses_newest_session_metadata(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    haindy_home = tmp_path / "haindy-home"
    monkeypatch.setenv("HAINDY_HOME", str(haindy_home))
    older = _write_session(haindy_home, "older")
    newer = _write_session(haindy_home, "newer")
    os.utime(older / "session.json", (100, 100))
    os.utime(newer / "session.json", (200, 200))

    exit_code, payload, _stderr = _run_helper(
        helper_module,
        capsys,
        "--latest-session",
    )

    assert exit_code == 0
    assert payload["session_id"] == "newer"


def test_session_without_trace_still_returns_session_summary(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    haindy_home = tmp_path / "haindy-home"
    data_root = tmp_path / "data"
    monkeypatch.setenv("HAINDY_HOME", str(haindy_home))
    monkeypatch.setenv("HAINDY_DATA_DIR", str(data_root))
    _write_session(haindy_home, "explore-session")

    exit_code, payload, _stderr = _run_helper(
        helper_module,
        capsys,
        "--session-id",
        "explore-session",
    )

    assert exit_code == 0
    assert payload["run_id"] is None
    assert payload["linked_trace"] is None
    assert payload["action_artifact_paths"] == []


def test_run_lookup_does_not_fall_back_to_legacy_data_dir(
    helper_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    configured_data = tmp_path / "configured-data"
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HAINDY_DATA_DIR", str(configured_data))
    _write_trace(project_dir / "data", "legacy-run")

    exit_code = helper_module.main(["--run-id", "legacy-run"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert f"under {configured_data / 'traces'}" in captured.err
