"""Focused tests for the enhanced HTML reporter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.types import (
    BugReport,
    BugSeverity,
    StepIntent,
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    TestReport,
    TestState,
    TestStatus,
    TestStep,
)
from src.monitoring.enhanced_reporter import EnhancedReporter
from src.monitoring.enhanced_reporter_data import (
    clean_ai_conversation,
    extract_template_data,
    screenshot_to_data_uri,
)


def _build_test_state(tmp_path: Path) -> tuple[TestState, dict[str, object]]:
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    before_path.write_bytes(b"before_png")
    after_path.write_bytes(b"after_png")

    step = TestStep(
        step_number=1,
        description="Click submit",
        action="Click submit",
        expected_result="Form submits",
        intent=StepIntent.VALIDATION,
    )
    case = TestCase(
        test_id="TC001",
        name="Submit form",
        description="Submission flow",
        steps=[step],
    )
    plan = TestPlan(
        name="Submission plan",
        description="Verify submission flow.",
        requirements_source="unit-test",
        test_cases=[case],
    )
    step_result = StepResult(
        step_id=step.step_id,
        step_number=1,
        status=TestStatus.FAILED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        action=step.action,
        expected_result=step.expected_result,
        actual_result="Button stayed disabled",
        screenshot_before=str(before_path),
        screenshot_after=str(after_path),
        error_message="Submit failed",
        confidence=0.25,
    )
    case_result = TestCaseResult(
        case_id=case.case_id,
        test_id=case.test_id,
        name=case.name,
        status=TestStatus.FAILED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        steps_total=1,
        steps_completed=0,
        steps_failed=1,
        step_results=[step_result],
        error_message="1 step failed",
    )
    bug_report = BugReport(
        step_id=step.step_id,
        test_case_id=case.case_id,
        test_plan_id=plan.plan_id,
        step_number=1,
        description="Submit button did not trigger form submission",
        severity=BugSeverity.HIGH,
        error_type="assertion_failed",
        expected_result="Form submits",
        actual_result="Button stayed disabled",
        error_details="The button remained disabled after input.",
        reproduction_steps=["Open form", "Populate fields", "Click submit"],
    )
    report = TestReport(
        test_plan_id=plan.plan_id,
        test_plan_name=plan.name,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status=TestStatus.FAILED,
        test_cases=[case_result],
        bugs=[bug_report],
        environment={"runtime": "desktop"},
        artifacts={"trace_path": "trace.json"},
    )
    state = TestState(test_plan=plan, test_report=report)
    action_storage = {
        "test_cases": [
            {
                "test_case_id": str(case.case_id),
                "steps": [
                    {
                        "step_id": str(step.step_id),
                        "test_runner_interpretation": {
                            "prompt": "Plan this step",
                            "response": {"actions": ["click submit"]},
                        },
                        "actions": [
                            {
                                "action_type": "click",
                                "target": "Submit",
                                "description": "Click submit",
                                "timestamp_start": "2026-03-06T12:00:00+00:00",
                                "timestamp_end": "2026-03-06T12:00:01+00:00",
                                "result": {"success": False},
                                "automation_calls": [{"type": "click"}],
                                "screenshots": {
                                    "before": str(before_path),
                                    "after": str(after_path),
                                },
                                "ai_conversation": {
                                    "action_agent_execution": {
                                        "messages": [
                                            {
                                                "role": "user",
                                                "content": [
                                                    {
                                                        "type": "text",
                                                        "text": "Click it",
                                                    },
                                                    {
                                                        "type": "image_url",
                                                        "image_url": "data:image/png;base64,AAA",
                                                    },
                                                ],
                                            }
                                        ]
                                    }
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    return state, action_storage


def test_screenshot_to_data_uri_handles_missing_path(tmp_path: Path) -> None:
    assert screenshot_to_data_uri(tmp_path / "missing.png") is None


def test_clean_ai_conversation_replaces_image_payloads() -> None:
    cleaned = clean_ai_conversation(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": "data:image/png;base64,AAA"},
                    ],
                }
            ]
        }
    )
    assert cleaned["messages"][0]["content"][1] == {
        "type": "text",
        "text": "[IMAGE: Screenshot provided to AI]",
    }


def test_extract_template_data_embeds_screenshots_and_actions(tmp_path: Path) -> None:
    test_state, action_storage = _build_test_state(tmp_path)

    data = extract_template_data(test_state, action_storage)

    assert data["overall_status"] == "failed"
    assert data["test_cases"][0]["steps"][0]["screenshots"]["before"].startswith(
        "data:image/png;base64,"
    )
    assert data["test_cases"][0]["steps"][0]["actions"][0]["screenshots"] is not None
    assert (
        data["test_cases"][0]["steps"][0]["actions"][0]["ai_conversation"]["messages"][
            0
        ]["content"][1]["text"]
        == "[IMAGE: Screenshot provided to AI]"
    )


def test_generate_report_writes_html_and_actions(tmp_path: Path) -> None:
    reporter = EnhancedReporter()
    test_state, action_storage = _build_test_state(tmp_path)

    report_path, actions_path = reporter.generate_report(
        test_state, tmp_path, action_storage
    )

    assert report_path.exists()
    assert actions_path is not None and actions_path.exists()
    assert "Submission plan" in report_path.read_text(encoding="utf-8")
    stored_actions = json.loads(actions_path.read_text(encoding="utf-8"))
    assert stored_actions["test_cases"][0]["steps"][0]["actions"][0]["action_type"] == (
        "click"
    )


def test_generate_report_requires_test_report(tmp_path: Path) -> None:
    plan = TestPlan(
        name="Plan",
        description="desc",
        requirements_source="unit-test",
        test_cases=[],
    )
    reporter = EnhancedReporter()
    test_state = TestState(test_plan=plan)

    with pytest.raises(ValueError, match="without a test report"):
        reporter.generate_report(test_state, tmp_path)
