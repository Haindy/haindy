"""ActionAgent behavior tests for computer-use-only execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.action_agent import ActionAgent
from src.core.enhanced_types import (
    EnhancedActionResult,
    ExecutionResult,
    ValidationResult,
)
from src.core.types import ActionInstruction, ActionType, TestStep


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.agents.action_agent.get_settings",
        lambda: SimpleNamespace(
            desktop_coordinate_cache_path=Path("data/desktop_cache/test_coordinates.json"),
            computer_use_model="computer-use-preview",
            cu_provider="openai",
        ),
    )


def test_build_computer_use_goal_prefers_explicit_prompt() -> None:
    step = TestStep(
        step_number=1,
        description="Do not use this fallback",
        action="Click",
        expected_result="Done",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click",
            expected_outcome="Done",
            computer_use_prompt="Use this exact prepared prompt",
        ),
    )

    goal = ActionAgent._build_computer_use_goal(step, step.action_instruction)
    assert goal == "Use this exact prepared prompt"


def test_build_computer_use_goal_generates_structured_prompt_when_needed() -> None:
    step = TestStep(
        step_number=2,
        description="Type into search field",
        action="Type",
        expected_result="Text appears",
        action_instruction=ActionInstruction(
            action_type=ActionType.TYPE,
            description="Type query",
            target="Search field",
            value="openai",
            expected_outcome="Query visible",
        ),
    )

    goal = ActionAgent._build_computer_use_goal(step, step.action_instruction)
    assert "Action type: type" in goal
    assert "Target: Search field" in goal
    assert "Value: openai" in goal


@pytest.mark.asyncio
async def test_execute_action_routes_skip_navigation_without_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    agent = ActionAgent()
    step = TestStep(
        step_number=1,
        description="Navigation already satisfied",
        action="skip_navigation",
        expected_result="No-op",
        action_instruction=ActionInstruction(
            action_type=ActionType.SKIP_NAVIGATION,
            description="Skip navigation",
            expected_outcome="No navigation needed",
        ),
    )

    result = await agent.execute_action(step, {})
    assert result.overall_success is True
    assert result.execution.success is True


@pytest.mark.asyncio
async def test_execute_action_delegates_to_computer_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)
    agent = ActionAgent()
    step = TestStep(
        step_number=3,
        description="Click submit",
        action="click submit",
        expected_result="Submitted",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click submit",
            expected_outcome="Submitted",
        ),
    )

    expected = EnhancedActionResult(
        test_step_id=step.step_id,
        test_step=step,
        test_context={},
        validation=ValidationResult(valid=True, confidence=0.8, reasoning="ok"),
        execution=ExecutionResult(success=True, execution_time_ms=5.0),
        overall_success=True,
    )

    async def _fake(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(agent, "_execute_computer_tool_workflow", _fake)

    result = await agent.execute_action(step, {})
    assert result is expected
