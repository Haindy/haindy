"""Focused step-processing tests for TestRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.core.types import ActionType, StepIntent, TestState, TestStatus
from tests.support_test_runner import (
    _build_test_case,
    _StubActionAgent,
    make_capture_test_step_screenshot,
    make_case_result,
    runner_factory,
)


@pytest.mark.asyncio
async def test_execute_action_propagates_mobile_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    action_agent = _StubActionAgent()
    runner = runner_factory(
        monkeypatch,
        tmp_path,
        action_agent=action_agent,
    )
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

    result = await runner._execute_action(
        {
            "type": "click",
            "target": "Back control",
            "description": "Tap back control",
            "critical": True,
        },
        step,
    )

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
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    case_result = make_case_result(case)

    runner._environment = "mobile_adb"
    runner._current_test_plan = plan
    runner._current_test_case = case

    async def _fake_get_interpretation_screenshot(*args, **kwargs):
        del args, kwargs
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

    actions, _ = await runner._interpret_step(step, case, case_result, use_cache=False)

    prompt = captured_prompt["text"]
    assert len(actions) == 1
    assert "Runtime backend: mobile_adb" in prompt
    assert "Do NOT propose keyboard/browser shortcuts like Alt+Left" in prompt
    assert 'use `key_press` with value "back"' in prompt
    assert "type: One of [navigate, click, type, wait, assert" in prompt
    assert (
        "Use `wait` when the step is explicitly about allowing loading or startup "
        "to finish"
    ) in prompt


@pytest.mark.asyncio
async def test_execute_action_accepts_wait_action_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    action_agent = _StubActionAgent()
    runner = runner_factory(
        monkeypatch,
        tmp_path,
        action_agent=action_agent,
    )
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

    result = await runner._execute_action(
        {
            "type": "wait",
            "target": "signed-out welcome entry UI",
            "description": "Wait until the signed-out welcome entry UI is visible",
            "critical": True,
            "computer_use_prompt": (
                "Wait until the signed-out welcome entry UI is visible. "
                "If it is already visible, stop."
            ),
        },
        step,
    )

    assert result["success"] is True
    assert action_agent.last_test_step is not None
    assert action_agent.last_test_step.action_instruction is not None
    assert action_agent.last_test_step.action_instruction.action_type == ActionType.WAIT
    assert (
        action_agent.last_test_step.action_instruction.computer_use_prompt
        == "Wait until the signed-out welcome entry UI is visible. If it is already visible, stop."
    )


@pytest.mark.asyncio
async def test_interpret_step_includes_setup_steps_in_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    setup_step = step.model_copy(
        update={
            "step_number": 1,
            "description": "Reset the app to a clean state with no active user session.",
            "action": "Reset the app to a clean state with no active user session.",
            "expected_result": "App launches to the initial signed-out screen.",
            "intent": StepIntent.SETUP,
        }
    )
    case = case.model_copy(update={"setup_steps": [setup_step], "steps": [step]})
    plan = plan.model_copy(update={"test_cases": [case]})
    case_result = make_case_result(case)

    runner._environment = "mobile_adb"
    runner._current_test_plan = plan
    runner._current_test_case = case

    async def _fake_get_interpretation_screenshot(*args, **kwargs):
        del args, kwargs
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
                        "type": "reset_app",
                        "target": "PlayerUp app",
                        "description": "Reset the app to a clean signed-out state",
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
        setup_step,
        case,
        case_result,
        use_cache=False,
    )

    prompt = captured_prompt["text"]
    assert len(actions) == 1
    assert "Step position: 1 of 2" in prompt
    assert (
        "Setup Step 1: Reset the app to a clean state with no active user session."
        in prompt
    )
    assert "Step 1: Attempt protected navigation" in prompt
    assert (
        "- Setup Step 1: Reset the app to a clean state with no active user session. "
        "(intent: setup, expected: App launches to the initial signed-out screen.) "
        "[CURRENT STEP]"
    ) in prompt
    assert (
        "- Step 1: Attempt protected navigation (intent: validation, expected: "
        "Signed-out view remains visible)"
    ) in prompt


@pytest.mark.asyncio
async def test_execute_setup_step_uses_ai_verification_in_normal_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    setup_step = step.model_copy(update={"intent": StepIntent.SETUP})
    case = case.model_copy(update={"steps": [setup_step]})
    plan = plan.model_copy(update={"test_cases": [case]})

    runner._current_test_plan = plan
    runner._current_test_case = case
    runner._current_test_case_actions = {"steps": []}
    runner._execution_history = []

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    monkeypatch.setattr(runner, "_try_execution_replay", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runner,
        "_interpret_step",
        AsyncMock(
            return_value=(
                [
                    {
                        "type": "click",
                        "target": "Login button",
                        "description": "Tap Login",
                        "critical": True,
                    }
                ],
                False,
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "_execute_action",
        AsyncMock(
            return_value={
                "success": True,
                "action_type": "click",
                "target": "Login button",
                "outcome": "Tapped Login",
                "confidence": 0.9,
                "error": None,
                "driver_actions": [{"type": "click", "x": 10, "y": 10}],
            }
        ),
    )
    verify_mock = AsyncMock(
        return_value={
            "verdict": "PASS",
            "reasoning": "Setup reached expected screen",
            "actual_result": "Sign-in screen is visible.",
            "confidence": 0.93,
            "is_blocker": False,
            "blocker_reasoning": "",
        }
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)
    store_mock = AsyncMock()
    persist_mock = AsyncMock()
    invalidate_coord_mock = AsyncMock()
    monkeypatch.setattr(runner, "_store_execution_replay", store_mock)
    monkeypatch.setattr(runner, "_persist_coordinate_cache", persist_mock)
    monkeypatch.setattr(runner, "_invalidate_coordinate_cache", invalidate_coord_mock)

    result = await runner._execute_test_step(
        setup_step,
        case,
        make_case_result(case, steps_total=len(case.steps)),
    )

    assert result.status == TestStatus.PASSED
    assert verify_mock.await_count == 1
    assert runner._current_step_data["verification_mode"] == "ai"
    store_mock.assert_awaited_once()
    persist_mock.assert_awaited_once()
    invalidate_coord_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_setup_step_failed_ai_verification_does_not_store_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    setup_step = step.model_copy(update={"intent": StepIntent.SETUP})
    case = case.model_copy(update={"steps": [setup_step]})
    plan = plan.model_copy(update={"test_cases": [case]})

    runner._current_test_plan = plan
    runner._current_test_case = case
    runner._current_test_case_actions = {"steps": []}
    runner._execution_history = []

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    monkeypatch.setattr(runner, "_try_execution_replay", AsyncMock(return_value=None))
    monkeypatch.setattr(
        runner,
        "_interpret_step",
        AsyncMock(
            return_value=(
                [
                    {
                        "type": "click",
                        "target": "Login button",
                        "description": "Tap Login",
                        "critical": True,
                    }
                ],
                False,
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "_execute_action",
        AsyncMock(
            return_value={
                "success": True,
                "action_type": "click",
                "target": "Login button",
                "outcome": "Tapped Login",
                "confidence": 0.9,
                "error": None,
                "driver_actions": [{"type": "click", "x": 10, "y": 10}],
            }
        ),
    )
    verify_mock = AsyncMock(
        return_value={
            "verdict": "FAIL",
            "reasoning": "Login form is not visible after action",
            "actual_result": "Still on welcome screen.",
            "confidence": 0.86,
            "is_blocker": False,
            "blocker_reasoning": "",
        }
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)
    store_mock = AsyncMock()
    persist_mock = AsyncMock()
    invalidate_coord_mock = AsyncMock()
    monkeypatch.setattr(runner, "_store_execution_replay", store_mock)
    monkeypatch.setattr(runner, "_persist_coordinate_cache", persist_mock)
    monkeypatch.setattr(runner, "_invalidate_coordinate_cache", invalidate_coord_mock)

    result = await runner._execute_test_step(
        setup_step,
        case,
        make_case_result(case, steps_total=len(case.steps)),
    )

    assert result.status == TestStatus.FAILED
    assert result.error_message == "Login form is not visible after action"
    assert verify_mock.await_count == 1
    assert runner._current_step_data["verification_mode"] == "ai"
    store_mock.assert_not_awaited()
    persist_mock.assert_not_awaited()
    invalidate_coord_mock.assert_awaited_once()
