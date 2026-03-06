"""Shared support code for TestRunner regression tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.test_runner import TestRunner
from src.core.enhanced_types import (
    EnhancedActionResult,
    ExecutionResult,
    ValidationResult,
)
from src.core.types import (
    StepIntent,
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    TestStatus,
    TestStep,
)
from src.runtime.execution_replay_cache import ExecutionReplayCacheKey


class _StubAutomationDriver:
    def __init__(self) -> None:
        self._capturing = False

    def start_capture(self) -> None:
        self._capturing = True

    def stop_capture(self) -> list[dict[str, object]]:
        self._capturing = False
        return []

    async def screenshot(self) -> bytes:
        return b"stub-screenshot"

    async def get_viewport_size(self) -> tuple[int, int]:
        return (1080, 2400)


class _StubActionAgent:
    def __init__(self) -> None:
        self.conversation_history: list[dict[str, object]] = []
        self.last_test_step: TestStep | None = None
        self.last_test_context: dict[str, object] | None = None

    async def execute_action(
        self,
        test_step: TestStep,
        test_context: dict[str, object],
        screenshot: bytes | None = None,
        record_driver_actions: bool = False,
    ) -> EnhancedActionResult:
        del screenshot, record_driver_actions
        self.last_test_step = test_step
        self.last_test_context = test_context
        return EnhancedActionResult(
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=test_context,
            validation=ValidationResult(valid=True, confidence=1.0, reasoning="ok"),
            execution=ExecutionResult(success=True, execution_time_ms=1.0),
            overall_success=True,
        )


class _StubTraceWriter:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.path = Path("data/traces/test_trace.json")

    def set_run_metadata(self, metadata: dict[str, object]) -> None:
        del metadata

    def record_cache_event(self, event: dict[str, object]) -> None:
        del event

    def record_step(self, **kwargs: object) -> None:
        del kwargs


def _patch_runner_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        task_plan_cache_path=tmp_path / "task_plan_cache.json",
        execution_replay_cache_path=tmp_path / "execution_replay_cache.json",
        enable_execution_replay_cache=True,
        model_log_path=tmp_path / "model_calls.jsonl",
        actions_computer_tool_stabilization_wait_ms=500,
        actions_computer_tool_action_timeout_ms=7000,
        desktop_coordinate_cache_path=tmp_path / "desktop_coordinates.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coordinates.json",
        desktop_keyboard_layout="us",
        mobile_keyboard_layout="us",
        max_screenshots=None,
    )
    monkeypatch.setattr("src.agents.test_runner.get_settings", lambda: settings)
    monkeypatch.setattr(
        "src.agents.test_runner.get_model_logger",
        lambda *args, **kwargs: SimpleNamespace(log_call=AsyncMock()),
    )
    monkeypatch.setattr("src.agents.test_runner.get_run_id", lambda: "test-run")
    monkeypatch.setattr("src.agents.test_runner.RunTraceWriter", _StubTraceWriter)


def runner_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    action_agent: _StubActionAgent | None = None,
    automation_driver: _StubAutomationDriver | None = None,
) -> TestRunner:
    _patch_runner_dependencies(monkeypatch, tmp_path)
    return TestRunner(
        automation_driver=automation_driver or _StubAutomationDriver(),
        action_agent=action_agent,
    )


def _build_test_case() -> tuple[TestPlan, TestCase, TestStep]:
    step = TestStep(
        step_number=1,
        description="Attempt protected navigation",
        action="Attempt protected navigation",
        expected_result="Signed-out view remains visible",
        intent=StepIntent.VALIDATION,
    )
    case = TestCase(
        test_id="TC001",
        name="Sign-out protections",
        description="Ensure signed-out users cannot access authenticated content.",
        steps=[step],
    )
    plan = TestPlan(
        name="Mobile auth checks",
        description="Regression checks for sign-out enforcement.",
        requirements_source="unit-test",
        test_cases=[case],
    )
    return plan, case, step


def _build_multi_case_plan(case_count: int = 4) -> TestPlan:
    test_cases: list[TestCase] = []
    for index in range(case_count):
        step = TestStep(
            step_number=1,
            description=f"Execute case {index + 1}",
            action=f"Execute case {index + 1}",
            expected_result=f"Case {index + 1} completes",
            intent=StepIntent.VALIDATION,
        )
        test_cases.append(
            TestCase(
                test_id=f"TC{index + 1:03d}",
                name=f"Case {index + 1}",
                description=f"Case {index + 1} description",
                steps=[step],
            )
        )
    return TestPlan(
        name="Summary regression plan",
        description="Exercise summary accounting.",
        requirements_source="unit-test",
        test_cases=test_cases,
    )


def make_case_result(
    case: TestCase,
    *,
    status: TestStatus = TestStatus.IN_PROGRESS,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    steps_total: int | None = None,
    steps_completed: int = 0,
    steps_failed: int = 0,
    step_results: list[StepResult] | None = None,
    error_message: str | None = None,
) -> TestCaseResult:
    now = datetime.now(timezone.utc)
    return TestCaseResult(
        case_id=case.case_id,
        test_id=case.test_id,
        name=case.name,
        status=status,
        started_at=started_at or now,
        completed_at=completed_at,
        steps_total=steps_total if steps_total is not None else len(case.steps),
        steps_completed=steps_completed,
        steps_failed=steps_failed,
        step_results=step_results or [],
        error_message=error_message,
    )


def make_step_result(
    step: TestStep,
    *,
    status: TestStatus = TestStatus.IN_PROGRESS,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    action: str | None = None,
    expected_result: str | None = None,
    actual_result: str = "",
) -> StepResult:
    now = datetime.now(timezone.utc)
    return StepResult(
        step_id=step.step_id,
        step_number=step.step_number,
        status=status,
        started_at=started_at or now,
        completed_at=completed_at or now,
        action=action or step.action,
        expected_result=expected_result or step.expected_result,
        actual_result=actual_result,
    )


def make_capture_test_step_screenshot(tmp_path: Path):
    async def _fake_capture_test_step_screenshot(
        *,
        test_case: TestCase,
        step: TestStep,
        suffix: str,
        origin: str | None = None,
        update_latest: bool = False,
    ) -> SimpleNamespace:
        del origin, update_latest
        return SimpleNamespace(
            screenshot_bytes=b"captured-screenshot",
            screenshot_path=str(
                tmp_path / f"tc{test_case.test_id}_step{step.step_number}_{suffix}.png"
            ),
        )

    return _fake_capture_test_step_screenshot


def make_replay_key(
    runner: TestRunner, plan: TestPlan, case: TestCase, step: TestStep
) -> ExecutionReplayCacheKey:
    return ExecutionReplayCacheKey(
        scenario=plan.name,
        step=runner._plan_cache_key(step, case),
        environment="desktop",
        resolution=(1080, 2400),
        keyboard_layout="us",
        plan_fingerprint=runner._plan_fingerprint(),
    )
