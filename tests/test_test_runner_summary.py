"""Summary calculation tests for TestRunner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from haindy.core.types import TestReport as RunnerTestReport
from haindy.core.types import TestStatus
from tests.support_test_runner import (
    _build_multi_case_plan,
    make_case_result,
    runner_factory,
)


def test_calculate_summary_accounts_for_skipped_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan = _build_multi_case_plan()
    now = datetime.now(timezone.utc)

    runner._current_test_plan = plan
    runner._test_report = RunnerTestReport(
        test_plan_id=plan.plan_id,
        test_plan_name=plan.name,
        started_at=now,
        completed_at=now,
        status=TestStatus.FAILED,
        created_by=runner.name,
        test_cases=[
            make_case_result(
                plan.test_cases[0],
                status=TestStatus.PASSED,
                started_at=now,
                completed_at=now,
                steps_total=6,
                steps_completed=6,
            ),
            make_case_result(
                plan.test_cases[1],
                status=TestStatus.FAILED,
                started_at=now,
                completed_at=now,
                steps_total=6,
                steps_failed=1,
                error_message="First assertion failed",
            ),
            make_case_result(
                plan.test_cases[2],
                status=TestStatus.SKIPPED,
                started_at=now,
                completed_at=now,
                steps_total=6,
                error_message="Blocked by earlier failure",
            ),
            make_case_result(
                plan.test_cases[3],
                status=TestStatus.SKIPPED,
                started_at=now,
                completed_at=now,
                steps_total=6,
                error_message="Blocked by earlier failure",
            ),
        ],
    )

    summary = runner._calculate_summary()

    assert summary.total_test_cases == 4
    assert summary.completed_test_cases == 4
    assert summary.passed_test_cases == 1
    assert summary.failed_test_cases == 1
    assert summary.skipped_test_cases == 2
    assert summary.completed_steps == 6
    assert summary.failed_steps == 1


def test_print_summary_reports_skipped_case_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan = _build_multi_case_plan()
    now = datetime.now(timezone.utc)

    runner._current_test_plan = plan
    runner._test_report = RunnerTestReport(
        test_plan_id=plan.plan_id,
        test_plan_name=plan.name,
        started_at=now,
        completed_at=now,
        status=TestStatus.FAILED,
        created_by=runner.name,
        test_cases=[
            make_case_result(
                plan.test_cases[0],
                status=TestStatus.PASSED,
                started_at=now,
                completed_at=now,
                steps_total=1,
                steps_completed=1,
            ),
            make_case_result(
                plan.test_cases[1],
                status=TestStatus.FAILED,
                started_at=now,
                completed_at=now,
                steps_total=1,
                steps_failed=1,
                error_message="Failed",
            ),
            make_case_result(
                plan.test_cases[2],
                status=TestStatus.SKIPPED,
                started_at=now,
                completed_at=now,
                steps_total=1,
                error_message="Skipped",
            ),
            make_case_result(
                plan.test_cases[3],
                status=TestStatus.SKIPPED,
                started_at=now,
                completed_at=now,
                steps_total=1,
                error_message="Skipped",
            ),
        ],
    )
    runner._test_report.summary = runner._calculate_summary()

    runner._print_summary()
    output = capsys.readouterr().out

    assert "Test Cases run: 4/4" in output
    assert "Test Cases passed: 1/4" in output
    assert "Test Cases failed: 1/4" in output
    assert "Test Cases skipped: 2/4" in output
