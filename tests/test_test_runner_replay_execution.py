"""Replay execution tests for TestRunner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.types import StepIntent, TestStatus
from tests.support_test_runner import (
    _build_test_case,
    make_capture_test_step_screenshot,
    make_case_result,
    make_replay_key,
    make_step_result,
    runner_factory,
)


@pytest.mark.asyncio
async def test_replay_verification_waits_when_model_requests_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()

    runner._current_test_plan = plan
    runner._current_step_actions = []
    runner._current_step_data = {}

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    replay_key = make_replay_key(runner, plan, case, step)
    monkeypatch.setattr(
        runner._replay_service,
        "execution_replay_key",
        AsyncMock(return_value=replay_key),
    )
    monkeypatch.setattr(
        runner._execution_replay_cache,
        "lookup",
        Mock(return_value=SimpleNamespace(actions=[{"type": "click", "x": 1, "y": 2}])),
    )
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.replay_driver_actions",
        AsyncMock(),
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.asyncio.sleep", sleep_mock
    )

    verify_mock = AsyncMock(
        side_effect=[
            {
                "verdict": "FAIL",
                "reasoning": "Still loading",
                "actual_result": "Loading indicator visible",
                "confidence": 0.62,
                "is_blocker": False,
                "blocker_reasoning": "",
                "request_additional_wait": True,
                "recommended_wait_ms": 2500,
                "wait_reasoning": "Spinner still present",
            },
            {
                "verdict": "PASS",
                "reasoning": "Navigation completed",
                "actual_result": "Home screen is visible",
                "confidence": 0.88,
                "is_blocker": False,
                "blocker_reasoning": "",
                "request_additional_wait": False,
                "recommended_wait_ms": 0,
                "wait_reasoning": "",
            },
        ]
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)

    result = await runner._try_execution_replay(
        step=step,
        test_case=case,
        step_result=make_step_result(step),
        screenshot_before=b"before",
        execution_history=[],
        next_test_case=None,
    )

    assert result is not None
    assert result.status == TestStatus.PASSED
    assert verify_mock.await_count == 2
    assert verify_mock.await_args_list[0].kwargs["replay_wait_budget_ms"] == 30000
    assert verify_mock.await_args_list[1].kwargs["replay_wait_budget_ms"] == 27500
    sleep_mock.assert_awaited_once_with(2.5)
    assert runner._current_step_data["replay_validation_wait_spent_ms"] == 2500
    assert runner._current_step_data["replay_validation_wait_cycles"] == 1
    assert (
        runner._current_step_data["replay_validation_wait_budget_remaining_ms"] == 27500
    )
    assert result.screenshot_after is not None
    assert "_replay_wait_1_after" in result.screenshot_after


@pytest.mark.asyncio
async def test_replay_verification_uses_budget_cap_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()

    runner._current_test_plan = plan
    runner._current_step_actions = []
    runner._current_step_data = {}

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    replay_key = make_replay_key(runner, plan, case, step)
    monkeypatch.setattr(
        runner._replay_service,
        "execution_replay_key",
        AsyncMock(return_value=replay_key),
    )
    monkeypatch.setattr(
        runner._execution_replay_cache,
        "lookup",
        Mock(return_value=SimpleNamespace(actions=[{"type": "click", "x": 1, "y": 2}])),
    )
    invalidate_mock = Mock()
    monkeypatch.setattr(runner._execution_replay_cache, "invalidate", invalidate_mock)
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.replay_driver_actions",
        AsyncMock(),
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.asyncio.sleep", sleep_mock
    )

    verify_mock = AsyncMock(
        side_effect=[
            {
                "verdict": "FAIL",
                "reasoning": "Potential in-flight transition",
                "actual_result": "Ambiguous",
                "confidence": 0.45,
                "is_blocker": False,
                "blocker_reasoning": "",
                "request_additional_wait": True,
                "recommended_wait_ms": 60000,
                "wait_reasoning": "May still be settling",
            },
            {
                "verdict": "FAIL",
                "reasoning": "Still not complete",
                "actual_result": "No successful navigation observed",
                "confidence": 0.55,
                "is_blocker": True,
                "blocker_reasoning": "Cannot proceed",
                "request_additional_wait": True,
                "recommended_wait_ms": 5000,
                "wait_reasoning": "Requested but no budget remains",
            },
        ]
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)

    step_result = make_step_result(step)
    result = await runner._try_execution_replay(
        step=step,
        test_case=case,
        step_result=step_result,
        screenshot_before=b"before",
        execution_history=[],
        next_test_case=None,
    )

    assert result is None
    assert step_result.status == TestStatus.FAILED
    assert verify_mock.await_count == 2
    assert verify_mock.await_args_list[0].kwargs["replay_wait_budget_ms"] == 30000
    assert verify_mock.await_args_list[1].kwargs["replay_wait_budget_ms"] == 0
    sleep_mock.assert_awaited_once_with(30.0)
    invalidate_mock.assert_called_once_with(replay_key)
    assert runner._current_step_data["replay_validation_wait_spent_ms"] == 30000
    assert runner._current_step_data["replay_validation_wait_cycles"] == 1
    assert runner._current_step_data["replay_validation_wait_budget_remaining_ms"] == 0


@pytest.mark.asyncio
async def test_replay_setup_step_uses_ai_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    setup_step = step.model_copy(update={"intent": StepIntent.SETUP})

    runner._current_test_plan = plan
    runner._current_step_actions = []
    runner._current_step_data = {}

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    monkeypatch.setattr(
        runner._replay_service,
        "execution_replay_key",
        AsyncMock(return_value=make_replay_key(runner, plan, case, setup_step)),
    )
    monkeypatch.setattr(
        runner._execution_replay_cache,
        "lookup",
        Mock(return_value=SimpleNamespace(actions=[{"type": "click", "x": 1, "y": 2}])),
    )
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.replay_driver_actions",
        AsyncMock(),
    )
    verify_mock = AsyncMock(
        return_value={
            "verdict": "PASS",
            "reasoning": "Setup completed with visible expected UI",
            "actual_result": "Expected setup UI is visible",
            "confidence": 0.9,
            "is_blocker": False,
            "blocker_reasoning": "",
            "request_additional_wait": False,
            "recommended_wait_ms": 0,
            "wait_reasoning": "",
        }
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)

    result = await runner._try_execution_replay(
        step=setup_step,
        test_case=case,
        step_result=make_step_result(setup_step),
        screenshot_before=b"before",
        execution_history=[],
        next_test_case=None,
    )

    assert result is not None
    assert result.status == TestStatus.PASSED
    assert verify_mock.await_count == 1
    assert verify_mock.await_args.kwargs["replay_wait_budget_ms"] == 30000
    assert runner._current_step_data["verification_mode"] == "ai"


@pytest.mark.asyncio
async def test_replay_setup_step_fail_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    setup_step = step.model_copy(update={"intent": StepIntent.SETUP})

    runner._current_test_plan = plan
    runner._current_step_actions = []
    runner._current_step_data = {}

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        make_capture_test_step_screenshot(tmp_path),
    )
    replay_key = make_replay_key(runner, plan, case, setup_step)
    monkeypatch.setattr(
        runner._replay_service,
        "execution_replay_key",
        AsyncMock(return_value=replay_key),
    )
    monkeypatch.setattr(
        runner._execution_replay_cache,
        "lookup",
        Mock(return_value=SimpleNamespace(actions=[{"type": "click", "x": 1, "y": 2}])),
    )
    invalidate_mock = Mock()
    monkeypatch.setattr(runner._execution_replay_cache, "invalidate", invalidate_mock)
    monkeypatch.setattr(
        "src.runtime.execution_replay_service.replay_driver_actions",
        AsyncMock(),
    )
    verify_mock = AsyncMock(
        return_value={
            "verdict": "FAIL",
            "reasoning": "Setup target screen not reached",
            "actual_result": "Still on previous screen",
            "confidence": 0.8,
            "is_blocker": True,
            "blocker_reasoning": "Cannot proceed",
            "request_additional_wait": False,
            "recommended_wait_ms": 0,
            "wait_reasoning": "",
        }
    )
    monkeypatch.setattr(runner, "_verify_expected_outcome", verify_mock)

    step_result = make_step_result(setup_step)
    result = await runner._try_execution_replay(
        step=setup_step,
        test_case=case,
        step_result=step_result,
        screenshot_before=b"before",
        execution_history=[],
        next_test_case=None,
    )

    assert result is None
    assert step_result.status == TestStatus.FAILED
    assert verify_mock.await_count == 1
    invalidate_mock.assert_called_once_with(replay_key)


@pytest.mark.asyncio
async def test_replay_step_preserves_post_replay_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()

    runner._current_test_plan = plan
    runner._current_test_case = case
    runner._current_test_case_actions = {"steps": []}

    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"

    async def _fake_capture_test_step_screenshot(*, suffix: str, **kwargs):
        test_case = kwargs["test_case"]
        step_obj = kwargs["step"]
        del kwargs
        return SimpleNamespace(
            screenshot_bytes=b"captured-screenshot",
            screenshot_path=str(
                before_path
                if suffix == "before"
                else tmp_path
                / f"tc{test_case.test_id}_step{step_obj.step_number}_{suffix}.png"
            ),
        )

    async def _fake_try_execution_replay(
        *,
        step,
        test_case,
        step_result,
        screenshot_before,
        execution_history,
        next_test_case,
    ):
        del step, test_case, screenshot_before, execution_history, next_test_case
        step_result.status = TestStatus.PASSED
        step_result.actual_result = "Replay succeeded"
        step_result.screenshot_after = str(after_path)
        runner._artifacts.update_latest_snapshot(
            b"after-screenshot",
            str(after_path),
            "step_1_after",
        )
        return step_result

    monkeypatch.setattr(
        runner._artifacts,
        "capture_test_step_screenshot",
        _fake_capture_test_step_screenshot,
    )
    monkeypatch.setattr(runner, "_try_execution_replay", _fake_try_execution_replay)

    result = await runner._execute_test_step(
        step,
        case,
        make_case_result(case, steps_total=len(case.steps)),
    )

    assert result.status == TestStatus.PASSED
    assert runner._artifacts.latest_screenshot_path == str(after_path)
    assert runner._artifacts.latest_screenshot_origin == "step_1_after"
