"""Template data extraction helpers for the enhanced HTML reporter."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.types import BugReport, TestCase, TestState


def screenshot_to_data_uri(path: str | Path | None) -> str | None:
    """Read a screenshot file and return a base64 data URI, or None."""
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except (OSError, ValueError):
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def clean_ai_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
    """Remove embedded image payloads from stored AI conversations."""
    if not conversation or "messages" not in conversation:
        return conversation

    cleaned_conversation: dict[str, Any] = {"messages": []}
    for message in conversation.get("messages", []):
        cleaned_message = {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
        }
        content = cleaned_message["content"]
        if isinstance(content, list):
            cleaned_content: list[Any] = []
            for item in content:
                if not isinstance(item, dict):
                    cleaned_content.append(item)
                    continue
                if item.get("type") == "text":
                    cleaned_content.append(item)
                elif item.get("type") == "image_url":
                    cleaned_content.append(
                        {
                            "type": "text",
                            "text": "[IMAGE: Screenshot provided to AI]",
                        }
                    )
            cleaned_message["content"] = cleaned_content
        cleaned_conversation["messages"].append(cleaned_message)
    return cleaned_conversation


def _find_bug_report(
    bugs: list[BugReport],
    *,
    step_id: Any,
) -> BugReport | None:
    for bug in bugs:
        if bug.step_id == step_id:
            return bug
    return None


def _find_test_case_actions(
    action_storage: dict[str, Any] | None,
    *,
    case_id: Any,
) -> dict[str, Any] | None:
    if not action_storage:
        return None
    for stored_case in action_storage.get("test_cases", []):
        if stored_case.get("test_case_id") == str(case_id):
            if isinstance(stored_case, dict):
                return stored_case
            return None
    return None


def _find_test_case_meta(test_state: TestState, *, case_id: Any) -> TestCase | None:
    for plan_case in test_state.test_plan.test_cases:
        if plan_case.case_id == case_id:
            return plan_case
    return None


def _build_step_actions_data(
    step_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions_data: list[dict[str, Any]] = []
    for action in step_actions:
        action_duration = 0
        if action.get("timestamp_start") and action.get("timestamp_end"):
            start = datetime.fromisoformat(action["timestamp_start"])
            end = datetime.fromisoformat(action["timestamp_end"])
            action_duration = int((end - start).total_seconds() * 1000)

        ai_conversation = action.get("ai_conversation", {}).get(
            "action_agent_execution"
        )
        if ai_conversation:
            ai_conversation = clean_ai_conversation(ai_conversation)

        raw_screenshots = action.get("screenshots", {}) or {}
        screenshots = {
            key: screenshot_to_data_uri(value)
            for key, value in raw_screenshots.items()
            if key in {"before", "after"} and value
        }

        actions_data.append(
            {
                "action_type": action.get("action_type", "unknown"),
                "target": action.get("target", ""),
                "value": action.get("value"),
                "description": action.get("description", ""),
                "duration": action_duration,
                "result": action.get("result"),
                "ai_conversation": ai_conversation,
                "automation_calls": action.get("automation_calls", []),
                "screenshots": screenshots or None,
            }
        )
    return actions_data


def _build_step_data(
    test_state: TestState,
    *,
    case_result: Any,
    action_storage: dict[str, Any] | None,
    bugs: list[BugReport],
) -> list[dict[str, Any]]:
    steps_data: list[dict[str, Any]] = []
    case_actions = _find_test_case_actions(action_storage, case_id=case_result.case_id)
    for step in case_result.step_results:
        bug_report = _find_bug_report(bugs, step_id=step.step_id)
        step_actions: list[dict[str, Any]] = []
        test_runner_conversation = None
        if case_actions:
            for stored_step in case_actions.get("steps", []):
                if stored_step.get("step_id") != str(step.step_id):
                    continue
                step_actions = stored_step.get("actions", [])
                test_runner_conversation = stored_step.get("test_runner_interpretation")
                break

        screenshots: dict[str, str] | None = None
        if step.screenshot_before or step.screenshot_after:
            screenshots = {}
            if step.screenshot_before:
                before_uri = screenshot_to_data_uri(step.screenshot_before)
                if before_uri:
                    screenshots["before"] = before_uri
            if step.screenshot_after:
                after_uri = screenshot_to_data_uri(step.screenshot_after)
                if after_uri:
                    screenshots["after"] = after_uri
            if not screenshots:
                screenshots = None

        step_duration = (step.completed_at - step.started_at).total_seconds()
        steps_data.append(
            {
                "id": str(step.step_id),
                "number": step.step_number,
                "status": step.status.value,
                "action": step.action,
                "expected_result": step.expected_result,
                "actual_result": step.actual_result,
                "error_message": step.error_message,
                "confidence": step.confidence,
                "duration": round(step_duration, 2),
                "screenshots": screenshots,
                "bug_report": {
                    "description": bug_report.description,
                    "severity": bug_report.severity.value,
                    "error_type": bug_report.error_type,
                    "expected_result": bug_report.expected_result,
                    "actual_result": bug_report.actual_result,
                    "error_details": bug_report.error_details,
                    "reproduction_steps": bug_report.reproduction_steps,
                }
                if bug_report
                else None,
                "test_runner_conversation": test_runner_conversation,
                "actions": _build_step_actions_data(step_actions),
            }
        )
    return steps_data


def extract_template_data(
    test_state: TestState,
    action_storage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract data from test state for template rendering."""
    test_report = test_state.test_report
    if test_report is None:
        raise ValueError("Cannot extract report template data without a test report")

    total_steps = sum(tc.steps_total for tc in test_report.test_cases)
    completed_steps = sum(tc.steps_completed for tc in test_report.test_cases)
    success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0

    if test_report.completed_at and test_report.started_at:
        duration = (test_report.completed_at - test_report.started_at).total_seconds()
    else:
        duration = 0

    test_cases_data: list[dict[str, Any]] = []
    for case_result in test_report.test_cases:
        case_duration = 0.0
        if case_result.completed_at and case_result.started_at:
            case_duration = (
                case_result.completed_at - case_result.started_at
            ).total_seconds()

        test_case_meta = _find_test_case_meta(test_state, case_id=case_result.case_id)
        test_cases_data.append(
            {
                "id": str(case_result.case_id),
                "name": case_result.name,
                "status": case_result.status.value,
                "priority": test_case_meta.priority.value
                if test_case_meta
                else "medium",
                "prerequisites": test_case_meta.prerequisites if test_case_meta else [],
                "steps_total": case_result.steps_total,
                "steps_completed": case_result.steps_completed,
                "steps_failed": case_result.steps_failed,
                "error_message": case_result.error_message,
                "duration": round(case_duration, 2),
                "steps": _build_step_data(
                    test_state,
                    case_result=case_result,
                    action_storage=action_storage,
                    bugs=test_report.bugs,
                ),
            }
        )

    return {
        "test_plan_id": str(test_report.test_plan_id),
        "test_plan_name": test_report.test_plan_name,
        "test_plan_description": test_state.test_plan.description,
        "test_plan_created": test_state.test_plan.created_at.isoformat(),
        "test_started": test_report.started_at.isoformat(),
        "test_completed": test_report.completed_at.isoformat()
        if test_report.completed_at
        else "In Progress",
        "environment": json.dumps(test_report.environment, indent=2)
        if test_report.environment
        else "{}",
        "overall_status": test_report.status.value,
        "duration": round(duration, 2),
        "success_rate": round(success_rate, 1),
        "test_cases": test_cases_data,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": test_report.artifacts or {},
    }
