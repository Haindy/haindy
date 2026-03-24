"""Tests for the TestRunner bug-report collaborator."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haindy.agents.test_runner_bug_reports import (
    BugReportRequest,
    TestRunnerBugReportBuilder,
)
from haindy.core.types import (
    StepIntent,
    StepResult,
    TestCase,
    TestCasePriority,
    TestCaseResult,
    TestPlan,
    TestStatus,
    TestStep,
)


def _build_bug_report_request() -> BugReportRequest:
    step = TestStep(
        step_number=2,
        description="Open the dashboard",
        action="Open the dashboard",
        expected_result="Dashboard is visible",
        intent=StepIntent.VALIDATION,
    )
    test_case = TestCase(
        test_id="TC001",
        name="Dashboard access",
        description="Verify the dashboard opens for valid users.",
        priority=TestCasePriority.HIGH,
        steps=[step],
    )
    test_plan = TestPlan(
        name="Smoke plan",
        description="Basic regression coverage",
        requirements_source="unit-test",
        test_cases=[test_case],
    )
    now = datetime.now(timezone.utc)
    case_result = TestCaseResult(
        case_id=test_case.case_id,
        test_id=test_case.test_id,
        name=test_case.name,
        status=TestStatus.FAILED,
        started_at=now,
        completed_at=now,
        steps_total=1,
        steps_completed=0,
        steps_failed=1,
        step_results=[],
    )
    step_result = StepResult(
        step_id=step.step_id,
        step_number=step.step_number,
        status=TestStatus.FAILED,
        started_at=now,
        completed_at=now,
        action=step.action,
        expected_result=step.expected_result,
        actual_result="Spinner never cleared",
        error_message="Navigation timed out",
        screenshot_after="reports/after.png",
    )
    return BugReportRequest(
        test_plan=test_plan,
        step_result=step_result,
        step=step,
        test_case=test_case,
        case_result=case_result,
        verification_result={
            "verdict": "FAIL",
            "reasoning": "Dashboard never appeared",
            "actual_result": "Spinner never cleared",
            "confidence": 0.71,
            "is_blocker": False,
            "blocker_reasoning": "",
        },
    )


@pytest.mark.asyncio
async def test_build_bug_report_applies_plan_level_blocker_override() -> None:
    request = _build_bug_report_request()
    call_openai = AsyncMock(
        side_effect=[
            {
                "content": {
                    "error_type": "timeout",
                    "severity": "medium",
                    "bug_description": "Dashboard load timed out",
                    "reasoning": "Navigation never completed",
                }
            },
            {
                "content": {
                    "severity": "critical",
                    "should_block": True,
                    "blocker_reason": "All later cases require the dashboard.",
                    "notes": "Root cause likely upstream auth or API latency.",
                    "recommended_actions": [
                        "Inspect backend latency",
                        "Review auth logs",
                    ],
                }
            },
        ]
    )
    model_logger = SimpleNamespace(log_call=AsyncMock())
    builder = TestRunnerBugReportBuilder(
        model="gpt-5.4",
        call_openai=call_openai,
        model_logger=model_logger,
    )

    result = await builder.build_bug_report(request)

    assert result is not None
    assert result.bug_report.error_type == "timeout"
    assert result.bug_report.description == "Dashboard load timed out"
    assert result.bug_report.severity.value == "critical"
    assert result.bug_report.plan_blocker is True
    assert result.is_blocker is True
    assert result.blocker_reasoning == "All later cases require the dashboard."
    assert result.bug_report.plan_recommendations == [
        "Inspect backend latency",
        "Review auth logs",
    ]
    assert "Plan-level notes: Root cause likely upstream auth or API latency." in (
        result.bug_report.error_details
    )


@pytest.mark.asyncio
async def test_build_bug_report_returns_none_for_non_failed_steps() -> None:
    request = _build_bug_report_request()
    request = BugReportRequest(
        test_plan=request.test_plan,
        step_result=request.step_result.model_copy(
            update={"status": TestStatus.PASSED}
        ),
        step=request.step,
        test_case=request.test_case,
        case_result=request.case_result,
        verification_result=request.verification_result,
    )
    builder = TestRunnerBugReportBuilder(
        model="gpt-5.4",
        call_openai=AsyncMock(),
        model_logger=SimpleNamespace(log_call=AsyncMock()),
    )

    result = await builder.build_bug_report(request)

    assert result is None
