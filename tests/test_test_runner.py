"""Focused regression tests for TestRunner environment propagation."""

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
    TestCase,
    TestCaseResult,
    TestPlan,
    TestState,
    TestStatus,
    TestStep,
)


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


def _patch_runner_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        task_plan_cache_path=tmp_path / "task_plan_cache.json",
        execution_replay_cache_path=tmp_path / "execution_replay_cache.json",
        model_log_path=tmp_path / "model_calls.jsonl",
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


@pytest.mark.asyncio
async def test_execute_action_propagates_mobile_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runner_dependencies(monkeypatch, tmp_path)
    driver = _StubAutomationDriver()
    action_agent = _StubActionAgent()
    runner = TestRunner(automation_driver=driver, action_agent=action_agent)

    plan, case, step = _build_test_case()
    runner._environment = "mobile_adb"
    runner._current_test_plan = plan
    runner._current_test_case = case
    runner._current_step_actions = []
    runner._test_state = TestState(
        test_plan=plan,
        context={
            "automation_backend": "mobile_adb",
            "target_type": "mobile_adb",
        },
    )

    action = {
        "type": "click",
        "target": "Back control",
        "description": "Tap back control",
        "critical": True,
    }
    result = await runner._execute_action(action, step)

    assert result["success"] is True
    assert action_agent.last_test_step is not None
    assert action_agent.last_test_step.environment == "mobile_adb"
    assert action_agent.last_test_context is not None
    assert action_agent.last_test_context["automation_backend"] == "mobile_adb"
    assert action_agent.last_test_context["target_type"] == "mobile_adb"


@pytest.mark.asyncio
async def test_interpret_step_adds_mobile_specific_guidance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runner_dependencies(monkeypatch, tmp_path)
    runner = TestRunner(automation_driver=_StubAutomationDriver())
    plan, case, step = _build_test_case()
    case_result = TestCaseResult(
        case_id=case.case_id,
        test_id=case.test_id,
        name=case.name,
        status=TestStatus.IN_PROGRESS,
        started_at=datetime.now(timezone.utc),
        steps_total=len(case.steps),
        steps_completed=0,
        steps_failed=0,
        step_results=[],
    )

    runner._environment = "mobile_adb"
    runner._current_test_plan = plan
    runner._current_test_case = case

    async def _fake_get_interpretation_screenshot(
        test_step: TestStep,
        test_case: TestCase,
    ) -> tuple[bytes, str, str]:
        del test_step, test_case
        return b"interpretation-screenshot", "debug_screenshots/mock.png", "unit_test"

    captured_prompt: dict[str, str] = {}

    async def _fake_call_openai(
        messages: list[dict[str, object]],
        response_format: dict[str, str] | None = None,
    ) -> dict[str, object]:
        del response_format
        content = messages[0]["content"]
        assert isinstance(content, list)
        captured_prompt["text"] = str(content[0]["text"])
        return {
            "content": {
                "actions": [
                    {
                        "type": "skip_navigation",
                        "target": "already on signed-out screen",
                        "description": "No navigation needed",
                        "critical": True,
                        "computer_use_prompt": "",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        runner,
        "_get_interpretation_screenshot",
        _fake_get_interpretation_screenshot,
    )
    monkeypatch.setattr(runner, "call_openai", _fake_call_openai)

    actions, _ = await runner._interpret_step(
        step,
        case,
        case_result,
        use_cache=False,
    )

    prompt = captured_prompt["text"]
    assert len(actions) == 1
    assert "Runtime backend: mobile_adb" in prompt
    assert "Do NOT propose keyboard/browser shortcuts like Alt+Left" in prompt
    assert 'use `key_press` with value "back"' in prompt
