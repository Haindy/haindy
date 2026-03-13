#!/usr/bin/env python3
"""Locate HAINDY run artifacts for failed-run forensics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_files(repo_root: Path) -> list[Path]:
    trace_dir = repo_root / "data" / "traces"
    return sorted(trace_dir.glob("*.json"))


def _select_trace(
    repo_root: Path,
    run_id: str | None,
    latest: bool,
    latest_failed: bool,
) -> Path:
    traces = _trace_files(repo_root)
    if not traces:
        raise FileNotFoundError("No trace files found under data/traces")

    if run_id:
        candidate = repo_root / "data" / "traces" / f"{run_id}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Trace for run_id {run_id!r} not found")
        return candidate

    if latest:
        return traces[-1]

    for trace_path in reversed(traces):
        trace = _load_json(trace_path)
        if latest_failed and not bool(trace.get("success")):
            return trace_path

    raise FileNotFoundError("No failed trace files found under data/traces")


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


def _summarize(trace_path: Path, repo_root: Path) -> dict[str, Any]:
    trace = _load_json(trace_path)
    run_id = str(trace.get("run_id") or trace_path.stem)
    report_dir = repo_root / "reports" / run_id
    actions_json = sorted(
        str(path.relative_to(repo_root)) for path in report_dir.glob("*-actions.json")
    )
    html_reports = sorted(
        str(path.relative_to(repo_root)) for path in report_dir.glob("*.html")
    )
    debug_dir = repo_root / "debug_screenshots" / run_id

    steps = trace.get("steps", [])
    statuses = [
        (step.get("step_result") or {}).get("status")
        for step in steps
        if isinstance(step, dict)
    ]
    passed_steps = sum(1 for status in statuses if status == "passed")
    failed_steps = _failed_steps(trace)

    model_log_path = (
        (trace.get("run_metadata") or {}).get("model_log_path")
    ) or "data/model_logs/model_calls.jsonl"

    return {
        "run_id": run_id,
        "trace_path": str(trace_path.relative_to(repo_root)),
        "success": bool(trace.get("success")),
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "automation_backend": (trace.get("run_metadata") or {}).get(
            "automation_backend"
        ),
        "test_plan_name": (trace.get("run_metadata") or {}).get("test_plan_name"),
        "model_log_path": model_log_path,
        "report_dir": (
            str(report_dir.relative_to(repo_root)) if report_dir.exists() else None
        ),
        "actions_json_paths": actions_json,
        "html_report_paths": html_reports,
        "debug_screenshot_dir": (
            str(debug_dir.relative_to(repo_root)) if debug_dir.exists() else None
        ),
        "total_steps": len(steps),
        "passed_steps": passed_steps,
        "failed_steps": failed_steps,
        "first_failed_step": failed_steps[0] if failed_steps else None,
        "model_log_rg_hint": f'rg -n \'"run_id": "{run_id}"\' {model_log_path}',
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate HAINDY run artifacts for failed-run forensics."
    )
    parser.add_argument("--run-id", help="Specific HAINDY run_id to inspect.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest trace whether it passed or failed.",
    )
    parser.add_argument(
        "--latest-failed",
        action="store_true",
        help="Use the latest failed trace. This is the default.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    latest_failed = args.latest_failed or (not args.run_id and not args.latest)

    try:
        trace_path = _select_trace(
            repo_root=repo_root,
            run_id=args.run_id,
            latest=args.latest,
            latest_failed=latest_failed,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = _summarize(trace_path, repo_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
