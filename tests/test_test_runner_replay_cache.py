"""Replay cache accounting tests for TestRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.runtime.execution_replay_cache import ExecutionReplayCacheKey
from tests.support_test_runner import (
    _build_test_case,
    runner_factory,
)


def test_replay_enabled_ignores_can_be_replayed_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    _, _, step = _build_test_case()
    replay_disabled_hint_step = step.model_copy(update={"can_be_replayed": False})

    assert runner._replay_enabled(replay_disabled_hint_step) is True


def test_replay_stabilization_wait_has_two_second_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)

    runner._settings.actions_computer_tool_stabilization_wait_ms = 500
    assert runner._replay_stabilization_wait_ms() == 2000

    runner._settings.actions_computer_tool_stabilization_wait_ms = 2500
    assert runner._replay_stabilization_wait_ms() == 2500


@pytest.mark.asyncio
async def test_execution_replay_key_includes_plan_fingerprint_and_changes_with_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)

    plan_a, case_a, step_a = _build_test_case()
    runner._current_test_plan = plan_a
    key_a = await runner._execution_replay_key(step_a, case_a)

    assert key_a is not None
    assert key_a.plan_fingerprint == runner._plan_fingerprint()
    assert key_a.plan_fingerprint != ""

    plan_b = plan_a.model_copy(deep=True)
    plan_b.test_cases[0].steps[0].expected_result = "Different expected outcome text"
    case_b = plan_b.test_cases[0]
    step_b = case_b.steps[0]
    runner._current_test_plan = plan_b
    key_b = await runner._execution_replay_key(step_b, case_b)

    assert key_b is not None
    assert key_b.plan_fingerprint == runner._plan_fingerprint()
    assert key_b.scenario == key_a.scenario
    assert key_b.step == key_a.step
    assert key_b.environment == key_a.environment
    assert key_b.resolution == key_a.resolution
    assert key_b.keyboard_layout == key_a.keyboard_layout
    assert key_b.plan_fingerprint != key_a.plan_fingerprint


@pytest.mark.asyncio
async def test_store_execution_replay_skips_validation_only_and_stores_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = runner_factory(monkeypatch, tmp_path)
    plan, case, step = _build_test_case()
    runner._current_test_plan = plan

    replay_key = ExecutionReplayCacheKey(
        scenario=plan.name,
        step=runner._plan_cache_key(step, case),
        environment="desktop",
        resolution=(1080, 2400),
        keyboard_layout="us",
        plan_fingerprint="fingerprint",
    )
    key_mock = AsyncMock(return_value=replay_key)
    monkeypatch.setattr(runner._replay_service, "execution_replay_key", key_mock)
    store_mock = Mock()
    monkeypatch.setattr(runner._execution_replay_cache, "store", store_mock)

    validation_only_results = [
        {
            "action": {"type": "assert"},
            "result": {"driver_actions": [{"type": "click", "x": 1, "y": 2}]},
        },
        {
            "action": {"type": "skip_navigation"},
            "result": {"driver_actions": [{"type": "click", "x": 3, "y": 4}]},
        },
        {
            "action": {"type": "wait"},
            "result": {"driver_actions": [{"type": "wait", "duration_ms": 100}]},
        },
        {
            "action": {"type": "screenshot"},
            "result": {"driver_actions": [{"type": "move", "x": 5, "y": 6}]},
        },
    ]

    await runner._store_execution_replay(step, case, validation_only_results)
    key_mock.assert_not_awaited()
    store_mock.assert_not_called()

    actionable_results = [
        {
            "action": {"type": "click"},
            "result": {
                "driver_actions": [
                    {"type": "move", "x": 20, "y": 30},
                    {
                        "type": "click",
                        "x": 20,
                        "y": 30,
                        "button": "left",
                        "click_count": 1,
                    },
                ]
            },
        }
    ]

    await runner._store_execution_replay(step, case, actionable_results)
    key_mock.assert_awaited_once()
    store_mock.assert_called_once_with(
        replay_key,
        [
            {"type": "move", "x": 20, "y": 30},
            {
                "type": "click",
                "x": 20,
                "y": 30,
                "button": "left",
                "click_count": 1,
            },
        ],
    )
