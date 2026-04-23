#!/usr/bin/env python3
"""Locate HAINDY run and tool-call session artifacts for forensics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SETTINGS_NOTE = (
    "Artifact roots come from the effective HAINDY settings. Legacy ./data paths "
    "are not scanned unless they are configured through HAINDY_DATA_DIR or "
    "storage.data_dir."
)


def _repo_root() -> Path:
    return _REPO_ROOT


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runtime_settings() -> Any:
    from haindy.config.settings import load_settings

    return load_settings()


def _expand_path(value: Path | str) -> Path:
    return Path(value).expanduser()


def _display_path(path: Path, repo_root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _trace_files(data_root: Path) -> list[Path]:
    trace_dir = data_root / "traces"
    return sorted(trace_dir.glob("*.json"))


def _select_trace(
    data_root: Path,
    run_id: str | None,
    latest: bool,
    latest_failed: bool,
) -> Path:
    trace_dir = data_root / "traces"
    if run_id:
        candidate = trace_dir / f"{run_id}.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Trace for run_id {run_id!r} not found under {trace_dir}"
            )
        return candidate

    traces = _trace_files(data_root)
    if not traces:
        raise FileNotFoundError(f"No trace files found under {trace_dir}")

    if latest:
        return traces[-1]

    for trace_path in reversed(traces):
        trace = _load_json(trace_path)
        if latest_failed and not bool(trace.get("success")):
            return trace_path

    raise FileNotFoundError(f"No failed trace files found under {trace_dir}")


def _session_root(haindy_home: Path) -> Path:
    return _expand_path(haindy_home) / "sessions"


def _session_dirs(haindy_home: Path) -> list[Path]:
    root = _session_root(haindy_home)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _metadata_path(session_dir: Path) -> Path:
    return session_dir / "session.json"


def _load_session_metadata(session_dir: Path) -> dict[str, Any] | None:
    path = _metadata_path(session_dir)
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _select_session(haindy_home: Path, session_id: str | None) -> Path:
    if session_id:
        candidate = _session_root(haindy_home) / session_id
        if not candidate.exists():
            raise FileNotFoundError(f"Tool-call session {session_id!r} not found")
        return candidate

    sessions = _session_dirs(haindy_home)
    if not sessions:
        raise FileNotFoundError(
            f"No tool-call sessions found under {_session_root(haindy_home)}"
        )

    def sort_key(session_dir: Path) -> float:
        metadata = _metadata_path(session_dir)
        try:
            if metadata.exists():
                return metadata.stat().st_mtime
            return session_dir.stat().st_mtime
        except OSError:
            return 0.0

    return max(sessions, key=sort_key)


def _session_id_from_dir(session_dir: Path, metadata: dict[str, Any] | None) -> str:
    if metadata and metadata.get("session_id"):
        return str(metadata["session_id"])
    return session_dir.name


def _action_artifact_files(session_dir: Path) -> list[Path]:
    return sorted((session_dir / "action_artifacts").glob("*.json"))


def _action_artifact_run_id(path: Path) -> str | None:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("run_id")
    return str(value) if value else None


def _matching_action_artifacts(session_dir: Path, run_id: str | None) -> list[Path]:
    artifacts = _action_artifact_files(session_dir)
    if not run_id:
        return artifacts
    return [path for path in artifacts if _action_artifact_run_id(path) == run_id]


def _session_matches_run(session_dir: Path, run_id: str) -> bool:
    metadata = _load_session_metadata(session_dir) or {}
    if metadata.get("latest_background_run_id") == run_id:
        return True
    return bool(_matching_action_artifacts(session_dir, run_id))


def _sessions_for_run(haindy_home: Path, run_id: str) -> list[Path]:
    return [
        session_dir
        for session_dir in _session_dirs(haindy_home)
        if _session_matches_run(session_dir, run_id)
    ]


def _failed_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        step_result = step.get("step_result") or {}
        status = step_result.get("status")
        if status in {"failed", "error"}:
            failed.append(
                {
                    "scenario": step.get("scenario"),
                    "step_number": step.get("step_number"),
                    "action": step.get("step_action"),
                    "expected_result": step.get("expected_result"),
                    "actual_result": step_result.get("actual_result"),
                    "error_message": step_result.get("error_message"),
                    "screenshot_before": step_result.get("screenshot_before"),
                    "screenshot_after": step_result.get("screenshot_after"),
                }
            )
    return failed


def _path_if_exists(path: Path, repo_root: Path) -> str | None:
    return _display_path(path, repo_root) if path.exists() else None


def _display_paths(paths: Sequence[Path], repo_root: Path) -> list[str]:
    return [_display_path(path, repo_root) for path in paths]


def _model_log_path(trace: dict[str, Any], configured_model_log_path: Path) -> str:
    metadata_path = (trace.get("run_metadata") or {}).get("model_log_path")
    return str(metadata_path or configured_model_log_path)


def _summarize_trace(
    trace_path: Path,
    *,
    repo_root: Path,
    data_root: Path,
    reports_dir: Path,
    configured_model_log_path: Path,
) -> dict[str, Any]:
    trace = _load_json(trace_path)
    run_id = str(trace.get("run_id") or trace_path.stem)
    report_dir = reports_dir / run_id
    actions_json = sorted(report_dir.glob("*-actions.json"))
    html_reports = sorted(report_dir.glob("*.html"))
    debug_dir = repo_root / "debug_screenshots" / run_id

    steps = trace.get("steps", [])
    statuses = [
        (step.get("step_result") or {}).get("status")
        for step in steps
        if isinstance(step, dict)
    ]
    passed_steps = sum(1 for status in statuses if status == "passed")
    failed_steps = _failed_steps(trace)

    model_log_path = _model_log_path(trace, configured_model_log_path)

    return {
        "mode": "trace",
        "run_id": run_id,
        "data_root": str(data_root),
        "reports_dir": str(reports_dir),
        "trace_path": _display_path(trace_path, repo_root),
        "success": bool(trace.get("success")),
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "automation_backend": (trace.get("run_metadata") or {}).get(
            "automation_backend"
        ),
        "test_plan_name": (trace.get("run_metadata") or {}).get("test_plan_name"),
        "model_log_path": model_log_path,
        "model_screenshot_dir": _path_if_exists(
            data_root / "model_logs" / "screenshots",
            repo_root,
        ),
        "report_dir": (
            _display_path(report_dir, repo_root) if report_dir.exists() else None
        ),
        "actions_json_paths": _display_paths(actions_json, repo_root),
        "html_report_paths": _display_paths(html_reports, repo_root),
        "debug_screenshot_dir": (
            _display_path(debug_dir, repo_root) if debug_dir.exists() else None
        ),
        "total_steps": len(steps),
        "passed_steps": passed_steps,
        "failed_steps": failed_steps,
        "first_failed_step": failed_steps[0] if failed_steps else None,
        "model_log_rg_hint": f'rg -n \'"run_id": "{run_id}"\' {model_log_path}',
        "settings_note": _SETTINGS_NOTE,
    }


def _summarize_session(
    session_dir: Path,
    *,
    repo_root: Path,
    data_root: Path,
    reports_dir: Path,
    configured_model_log_path: Path,
    run_id_filter: str | None = None,
    include_linked_trace: bool = True,
) -> dict[str, Any]:
    metadata = _load_session_metadata(session_dir) or {}
    session_id = _session_id_from_dir(session_dir, metadata)
    latest_run_id = metadata.get("latest_background_run_id")
    run_id = run_id_filter or (str(latest_run_id) if latest_run_id else None)
    action_artifacts = _matching_action_artifacts(session_dir, run_id_filter)
    screenshot_dir = session_dir / "screenshots"
    screenshots = sorted(screenshot_dir.glob("*"))
    daemon_log_path = session_dir / "logs" / "daemon.log"
    ai_log_path = session_dir / "logs" / "ai_interactions.jsonl"
    trace_path = data_root / "traces" / f"{run_id}.json" if run_id else None
    linked_trace = None
    if include_linked_trace and trace_path and trace_path.exists():
        linked_trace = _summarize_trace(
            trace_path,
            repo_root=repo_root,
            data_root=data_root,
            reports_dir=reports_dir,
            configured_model_log_path=configured_model_log_path,
        )

    return {
        "mode": "tool_call_session",
        "session_id": session_id,
        "run_id": run_id,
        "data_root": str(data_root),
        "reports_dir": str(reports_dir),
        "session_dir": _display_path(session_dir, repo_root),
        "session_metadata_path": _path_if_exists(
            _metadata_path(session_dir), repo_root
        ),
        "session_status": metadata.get("status"),
        "backend": metadata.get("backend"),
        "created_at": metadata.get("created_at"),
        "last_command_at": metadata.get("last_command_at"),
        "last_command_name": metadata.get("last_command_name"),
        "latest_background_run_id": latest_run_id,
        "latest_test_phase": metadata.get("latest_test_phase"),
        "latest_test_progress_at": metadata.get("latest_test_progress_at"),
        "latest_test_action_artifact_path": metadata.get(
            "latest_test_action_artifact_path"
        ),
        "latest_screenshot_path": metadata.get("latest_screenshot_path"),
        "action_artifact_paths": _display_paths(action_artifacts, repo_root),
        "session_screenshot_dir": _path_if_exists(screenshot_dir, repo_root),
        "session_screenshot_count": len(screenshots),
        "latest_session_screenshot_paths": _display_paths(screenshots[-5:], repo_root),
        "daemon_log_path": _path_if_exists(daemon_log_path, repo_root),
        "ai_interactions_log_path": _path_if_exists(ai_log_path, repo_root),
        "linked_trace": linked_trace,
        "model_log_path": str(configured_model_log_path),
        "model_log_rg_hint": (
            f'rg -n \'"run_id": "{run_id}"\' {configured_model_log_path}'
            if run_id
            else None
        ),
        "settings_note": _SETTINGS_NOTE,
    }


def _summarize_run_id(
    run_id: str,
    *,
    repo_root: Path,
    data_root: Path,
    reports_dir: Path,
    haindy_home: Path,
    configured_model_log_path: Path,
) -> dict[str, Any]:
    trace_path = data_root / "traces" / f"{run_id}.json"
    matching_sessions = [
        _summarize_session(
            session_dir,
            repo_root=repo_root,
            data_root=data_root,
            reports_dir=reports_dir,
            configured_model_log_path=configured_model_log_path,
            run_id_filter=run_id,
            include_linked_trace=False,
        )
        for session_dir in _sessions_for_run(haindy_home, run_id)
    ]

    if not trace_path.exists():
        if matching_sessions:
            return {
                "mode": "run_id_session_only",
                "run_id": run_id,
                "data_root": str(data_root),
                "reports_dir": str(reports_dir),
                "trace_path": None,
                "matching_tool_call_sessions": matching_sessions,
                "warning": (
                    f"No trace for run_id {run_id!r} was found under "
                    f"{data_root / 'traces'}, but matching tool-call session "
                    "artifacts were found."
                ),
                "settings_note": _SETTINGS_NOTE,
            }
        raise FileNotFoundError(
            f"Trace for run_id {run_id!r} not found under {data_root / 'traces'}"
        )

    summary = _summarize_trace(
        trace_path,
        repo_root=repo_root,
        data_root=data_root,
        reports_dir=reports_dir,
        configured_model_log_path=configured_model_log_path,
    )
    summary["matching_tool_call_sessions"] = matching_sessions
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate HAINDY run artifacts for failed-run forensics."
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--run-id", help="Specific HAINDY run_id to inspect.")
    selector.add_argument("--session-id", help="Specific tool-call session to inspect.")
    selector.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest trace whether it passed or failed.",
    )
    selector.add_argument(
        "--latest-failed",
        action="store_true",
        help="Use the latest failed trace. This is the default.",
    )
    selector.add_argument(
        "--latest-session",
        action="store_true",
        help="Use the newest tool-call session directory.",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    settings = _load_runtime_settings()
    data_root = _expand_path(settings.data_dir)
    reports_dir = _expand_path(settings.reports_dir)
    haindy_home = _expand_path(settings.haindy_home)
    configured_model_log_path = _expand_path(settings.model_log_path)
    latest_failed = (
        not any(
            [
                args.run_id,
                args.session_id,
                args.latest,
                args.latest_failed,
                args.latest_session,
            ]
        )
        or args.latest_failed
    )

    try:
        if args.session_id or args.latest_session:
            session_dir = _select_session(
                haindy_home,
                session_id=args.session_id,
            )
            summary = _summarize_session(
                session_dir,
                repo_root=repo_root,
                data_root=data_root,
                reports_dir=reports_dir,
                configured_model_log_path=configured_model_log_path,
            )
        elif args.run_id:
            summary = _summarize_run_id(
                args.run_id,
                repo_root=repo_root,
                data_root=data_root,
                reports_dir=reports_dir,
                haindy_home=haindy_home,
                configured_model_log_path=configured_model_log_path,
            )
        else:
            trace_path = _select_trace(
                data_root=data_root,
                run_id=None,
                latest=args.latest,
                latest_failed=latest_failed,
            )
            summary = _summarize_trace(
                trace_path,
                repo_root=repo_root,
                data_root=data_root,
                reports_dir=reports_dir,
                configured_model_log_path=configured_model_log_path,
            )
            run_id = str(summary.get("run_id") or "")
            summary["matching_tool_call_sessions"] = (
                [
                    _summarize_session(
                        session_dir,
                        repo_root=repo_root,
                        data_root=data_root,
                        reports_dir=reports_dir,
                        configured_model_log_path=configured_model_log_path,
                        run_id_filter=run_id,
                        include_linked_trace=False,
                    )
                    for session_dir in _sessions_for_run(haindy_home, run_id)
                ]
                if run_id
                else []
            )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
