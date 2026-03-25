"""Bug-report planning and assembly helpers for TestRunner."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from haindy.core.types import (
    BugReport,
    BugSeverity,
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    TestStatus,
    TestStep,
)
from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)

CallOpenAIFunc = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class BugReportRequest:
    """Inputs required to build a bug report for a failed step."""

    test_plan: TestPlan
    step_result: StepResult
    step: TestStep
    test_case: TestCase
    case_result: TestCaseResult
    verification_result: dict[str, Any]


@dataclass(frozen=True)
class BugReportBuildResult:
    """Built bug report plus runner-facing blocker metadata."""

    bug_report: BugReport
    plan_level_assessment: dict[str, Any] | None
    is_blocker: bool
    blocker_reasoning: str


class TestRunnerBugReportBuilder:
    """Owns bug classification and plan-level impact assessment."""

    def __init__(
        self,
        *,
        model: str,
        call_openai: CallOpenAIFunc,
        model_logger: Any,
    ) -> None:
        self._model = model
        self._call_openai = call_openai
        self._model_logger = model_logger

    async def build_bug_report(
        self,
        request: BugReportRequest,
    ) -> BugReportBuildResult | None:
        """Create a detailed bug report for a failed step."""
        if request.step_result.status != TestStatus.FAILED:
            return None

        classification = await self._classify_bug(request)
        bug_report = self._build_bug_report(request, classification)

        plan_assessment: dict[str, Any] | None = None
        try:
            plan_assessment = await self._evaluate_bug_plan_context(
                bug_report=bug_report,
                request=request,
                initial_severity=classification.severity,
            )
        except Exception as exc:
            logger.error(
                "Plan-level bug assessment failed",
                extra={
                    "error": str(exc),
                    "bug_id": str(bug_report.bug_id),
                },
            )

        is_blocker = bool(request.verification_result.get("is_blocker", False))
        blocker_reasoning = str(
            request.verification_result.get("blocker_reasoning") or ""
        )

        if plan_assessment:
            plan_severity = plan_assessment.get("severity")
            if isinstance(plan_severity, str):
                plan_severity_enum = _severity_from_string(plan_severity)
                if plan_severity_enum:
                    bug_report.plan_recommended_severity = plan_severity_enum
                    if _severity_rank(plan_severity_enum) < _severity_rank(
                        bug_report.severity
                    ):
                        bug_report.severity = plan_severity_enum
                else:
                    logger.warning(
                        "Plan-level assessment returned unknown severity",
                        extra={"severity": plan_severity},
                    )

            blocker_flag = plan_assessment.get("should_block")
            if blocker_flag is not None:
                bug_report.plan_blocker = bool(blocker_flag)
                is_blocker = bool(blocker_flag)
                blocker_reasoning = str(plan_assessment.get("blocker_reason") or "")
                if bug_report.plan_blocker:
                    reasoning = (
                        blocker_reasoning
                        or "Plan-level assessment marked this failure as blocking."
                    )
                    bug_report.plan_blocker_reason = reasoning
                    blocker_reasoning = reasoning
                    bug_report.error_details = _append_error_detail(
                        bug_report.error_details,
                        f"Plan-level blocker reasoning: {reasoning}",
                    )
                elif blocker_reasoning:
                    bug_report.plan_blocker_reason = blocker_reasoning
                    bug_report.error_details = _append_error_detail(
                        bug_report.error_details,
                        f"Plan-level blocker reasoning: {blocker_reasoning}",
                    )

            notes = plan_assessment.get("notes")
            if isinstance(notes, str):
                bug_report.plan_assessment_notes = notes
                bug_report.error_details = _append_error_detail(
                    bug_report.error_details,
                    f"Plan-level notes: {notes}",
                )

            recommendations = plan_assessment.get("recommended_actions")
            if isinstance(recommendations, list):
                bug_report.plan_recommendations = [
                    str(item) for item in recommendations
                ]

        logger.info(
            "Bug report created",
            extra={
                "bug_id": str(bug_report.bug_id),
                "severity": bug_report.severity.value,
                "error_type": classification.error_type,
                "plan_blocker": bug_report.plan_blocker,
            },
        )

        return BugReportBuildResult(
            bug_report=bug_report,
            plan_level_assessment=plan_assessment,
            is_blocker=is_blocker,
            blocker_reasoning=blocker_reasoning,
        )

    async def _classify_bug(
        self,
        request: BugReportRequest,
    ) -> _BugClassification:
        verification_result = request.verification_result
        prompt = f"""Analyze this test failure and create a bug report:

Test Case: {request.test_case.name}
Failed Step: {request.step.action}
Expected Result: {request.step.expected_result}

Verification Results:
- Verdict: {verification_result.get("verdict", "FAIL")}
- Reasoning: {verification_result.get("reasoning", request.step_result.error_message)}
- Actual Result: {verification_result.get("actual_result", request.step_result.actual_result)}
- Is Blocker: {verification_result.get("is_blocker", False)}
- Blocker Reasoning: {verification_result.get("blocker_reasoning", "N/A")}

Error Details: {request.step_result.error_message}
Step is Optional: {request.step.optional}

Determine:
1. error_type: One of (element_not_found, assertion_failed, timeout, navigation_error, api_error, validation_error, unknown_error)
2. severity: One of (critical, high, medium, low)
   - If is_blocker=true, severity should be at least "high"
   - critical: Blocks all testing, core functionality broken
   - high: Major feature broken, blocks test case
   - medium: Feature partially working, workaround possible
   - low: Minor issue, cosmetic or edge case
3. bug_description: A clear, concise description for developers

Respond in JSON format with keys: error_type, severity, bug_description, reasoning"""

        response = await self._call_openai(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        await self._model_logger.log_call(
            agent="test_runner.bug_report",
            model=self._model,
            prompt=prompt,
            request_payload={
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            response=response,
            metadata={
                "step_number": request.step.step_number,
                "test_case": request.test_case.name,
            },
        )

        raw_result = response.get("content", {})
        if isinstance(raw_result, str):
            parsed_result = json.loads(raw_result)
            result: dict[str, Any] = (
                parsed_result if isinstance(parsed_result, dict) else {}
            )
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            result = {}

        error_type = str(result.get("error_type") or "unknown_error")
        severity = _severity_from_string(result.get("severity")) or BugSeverity.MEDIUM

        logger.debug(
            "AI bug classification",
            extra={
                "error_type": error_type,
                "severity": severity.value,
                "reasoning": result.get("reasoning", ""),
            },
        )

        return _BugClassification(
            error_type=error_type,
            severity=severity,
            bug_description=str(
                result.get(
                    "bug_description",
                    f"Step {request.step.step_number} failed: {request.step.action}",
                )
            ),
        )

    def _build_bug_report(
        self,
        request: BugReportRequest,
        classification: _BugClassification,
    ) -> BugReport:
        verification_result = request.verification_result

        reproduction_steps = [
            f"1. Execute test case: {request.test_case.name}",
            f"2. Navigate to step {request.step.step_number}: {request.step.action}",
        ]
        for index, step_result in enumerate(request.case_result.step_results[-3:]):
            if step_result.status == TestStatus.PASSED:
                reproduction_steps.append(
                    f"{index + 3}. Previous step completed: Step {step_result.step_number}"
                )
        reproduction_steps.append(
            f"{len(reproduction_steps) + 1}. Execute failing step: {request.step.action}"
        )

        error_details_parts: list[str] = []
        if verification_result.get("reasoning"):
            error_details_parts.append(
                f"Verification reasoning: {verification_result['reasoning']}"
            )
        if verification_result.get("is_blocker"):
            error_details_parts.append(
                f"Blocker: Yes - {verification_result.get('blocker_reasoning', 'N/A')}"
            )
        else:
            error_details_parts.append("Blocker: No")
        if request.step_result.error_message:
            error_details_parts.append(
                f"Error message: {request.step_result.error_message}"
            )

        return BugReport(
            step_id=request.step.step_id,
            test_case_id=request.test_case.case_id,
            test_plan_id=request.test_plan.plan_id,
            step_number=request.step.step_number,
            description=classification.bug_description,
            severity=classification.severity,
            error_type=classification.error_type,
            expected_result=request.step.expected_result,
            actual_result=request.step_result.actual_result,
            screenshot_path=request.step_result.screenshot_after,
            error_details="\n".join(error_details_parts),
            reproduction_steps=reproduction_steps,
        )

    async def _evaluate_bug_plan_context(
        self,
        *,
        bug_report: BugReport,
        request: BugReportRequest,
        initial_severity: BugSeverity,
    ) -> dict[str, Any] | None:
        """Ask the model to evaluate bug impact using full test plan context."""
        plan_payload = request.test_plan.model_dump(mode="json")
        plan_json = json.dumps(plan_payload, indent=2)

        bug_payload = bug_report.model_dump(mode="json")
        bug_payload["severity"] = bug_report.severity.value
        bug_payload["initial_severity"] = initial_severity.value

        test_case_context = {
            "test_case_id": request.test_case.test_id,
            "name": request.test_case.name,
            "description": request.test_case.description,
            "priority": request.test_case.priority.value,
            "prerequisites": request.test_case.prerequisites,
            "postconditions": request.test_case.postconditions,
        }

        step_context = {
            "step_number": request.step.step_number,
            "action": request.step.action,
            "expected_result": request.step.expected_result,
            "optional": request.step.optional,
        }

        verification_context = {
            "verdict": request.verification_result.get("verdict"),
            "reasoning": request.verification_result.get("reasoning"),
            "actual_result": request.verification_result.get("actual_result"),
            "confidence": request.verification_result.get("confidence"),
            "is_blocker": request.verification_result.get("is_blocker"),
            "blocker_reasoning": request.verification_result.get("blocker_reasoning"),
        }

        prompt = (
            "You are a senior QA lead reviewing an automated test failure. "
            "Use the complete test plan to reason about downstream impact. "
            "If the failure prevents any remaining test cases from achieving their purpose, "
            "treat it as blocking.\n\n"
            "Test Plan (JSON):\n"
            f"{plan_json}\n\n"
            "Failed Test Case Context:\n"
            f"{json.dumps(test_case_context, indent=2)}\n\n"
            "Failed Step Context:\n"
            f"{json.dumps(step_context, indent=2)}\n\n"
            "Existing Bug Report:\n"
            f"{json.dumps(bug_payload, indent=2)}\n\n"
            "Verification Summary:\n"
            f"{json.dumps(verification_context, indent=2)}\n\n"
            "Respond with JSON using this schema:\n"
            "{\n"
            '  "severity": "critical|high|medium|low",\n'
            '  "should_block": true|false,\n'
            '  "blocker_reason": "Why later cases cannot proceed (or empty)",\n'
            '  "notes": "Additional context for the report",\n'
            '  "recommended_actions": ["Optional suggestions"]\n'
            "}"
        )

        response = await self._call_openai(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        await self._model_logger.log_call(
            agent="test_runner.bug_plan_assessment",
            model=self._model,
            prompt=prompt,
            request_payload={
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            response=response,
            metadata={
                "step_number": request.step.step_number,
                "test_case": request.test_case.name,
            },
        )

        content = response.get("content", "{}")
        if isinstance(content, str):
            parsed_content = json.loads(content)
            return parsed_content if isinstance(parsed_content, dict) else None
        if isinstance(content, dict):
            return content
        return None


@dataclass(frozen=True)
class _BugClassification:
    error_type: str
    severity: BugSeverity
    bug_description: str


def _severity_rank(severity: BugSeverity) -> int:
    severity_order = {
        BugSeverity.CRITICAL: 0,
        BugSeverity.HIGH: 1,
        BugSeverity.MEDIUM: 2,
        BugSeverity.LOW: 3,
    }
    return severity_order.get(severity, 99)


def _severity_from_string(value: Any) -> BugSeverity | None:
    severity_map = {
        "critical": BugSeverity.CRITICAL,
        "high": BugSeverity.HIGH,
        "medium": BugSeverity.MEDIUM,
        "low": BugSeverity.LOW,
    }
    return severity_map.get(str(value or "").strip().lower())


def _append_error_detail(existing: str | None, line: str) -> str:
    if not existing:
        return line
    return f"{existing}\n{line}"
