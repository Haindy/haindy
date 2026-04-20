"""ActionAgent behavior tests for computer-use-only execution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haindy.agents.action_agent import ActionAgent
from haindy.agents.computer_use.types import ComputerUseSessionResult
from haindy.agents.computer_use.visual_state import VisualBounds, VisualFrame
from haindy.core.enhanced_types import (
    EnhancedActionResult,
    ExecutionResult,
    ValidationResult,
)
from haindy.core.types import ActionInstruction, ActionType, TestCase, TestStep


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "haindy.agents.action_agent.get_settings",
        lambda: SimpleNamespace(
            linux_coordinate_cache_path=Path("data/linux_cache/test_coordinates.json"),
            computer_use_model="gpt-5.4",
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

    goal = ActionAgent._build_computer_use_goal(step, step.action_instruction)  # type: ignore[arg-type]
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

    goal = ActionAgent._build_computer_use_goal(step, step.action_instruction)  # type: ignore[arg-type]
    assert "Action type: type" in goal
    assert "Target: Search field" in goal
    assert "Value: openai" in goal


def test_build_action_session_metadata_preserves_mobile_app_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    agent = ActionAgent()
    step = TestStep(
        step_number=4,
        description="Open the configured app.",
        action="open_app",
        expected_result="App is foregrounded",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Open the configured app",
            expected_outcome="App is foregrounded",
        ),
    )

    metadata, environment, safety_identifier = agent._build_action_session_metadata(
        step_session=None,
        test_step=step,
        instruction=step.action_instruction,  # type: ignore[arg-type]
        interaction_mode="execute",
        current_url="android://screen",
        context_lookup={
            "environment": "mobile_adb",
            "app_package": "co.playerup.app",
            "app_activity": ".MainActivity",
        },
    )

    assert environment == "mobile_adb"
    assert safety_identifier
    assert metadata["app_package"] == "co.playerup.app"
    assert metadata["app_activity"] == ".MainActivity"
    assert metadata["current_url"] == "android://screen"
    assert metadata["response_reporting_scope"] == "state_only"


def test_build_step_validation_prompt_scopes_verdict_to_current_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    agent = ActionAgent()
    test_case = TestCase(  # type: ignore[call-arg]
        test_id="TC002",
        name="Case-insensitive sign-in",
        description=(
            "Verify sign-in succeeds and later confirm the displayed account email "
            "is normalized."
        ),
        steps=[],
    )
    step = TestStep(
        step_number=7,
        description='Tap "Account Settings".',
        action='Tap "Account Settings".',
        expected_result=(
            "The Account Settings screen is displayed and shows the account email "
            "address."
        ),
    )

    prompt = agent._build_step_validation_prompt(
        test_case=test_case,
        step=step,
        action_results=[],
        execution_history=[],
        next_test_case=None,
    )

    assert "Test case description:" not in prompt
    assert (
        "Evaluate ONLY whether this current step achieved its stated expected result."
        in prompt
    )
    assert "Do NOT fail this step based on broader test-case goals" in prompt


def test_new_computer_use_session_skips_openai_client_for_google(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "haindy.agents.action_agent.get_settings",
        lambda: SimpleNamespace(
            linux_coordinate_cache_path=Path("data/linux_cache/test_coordinates.json"),
            macos_coordinate_cache_path=Path("data/macos_cache/test_coordinates.json"),
            mobile_coordinate_cache_path=Path(
                "data/mobile_cache/test_coordinates.json"
            ),
            ios_coordinate_cache_path=Path("data/ios_cache/test_coordinates.json"),
            computer_use_model="gpt-5.4",
            cu_provider="google",
        ),
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.provider = "google"

    monkeypatch.setattr("haindy.agents.action_agent.ComputerUseSession", _FakeSession)

    agent = ActionAgent(automation_driver=object())  # type: ignore[arg-type]

    agent._new_computer_use_session(debug_logger=None, environment="desktop")

    assert captured["client"] is None


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
    assert result.execution.success is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_execute_action_delegates_to_computer_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    async def _fake(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return expected

    monkeypatch.setattr(agent, "_execute_computer_tool_workflow", _fake)

    result = await agent.execute_action(step, {})
    assert result is expected


@pytest.mark.asyncio
async def test_execute_action_persists_artifact_frame_instead_of_model_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr("haindy.agents.action_agent.get_debug_logger", lambda: None)

    visual_patch = VisualFrame(
        frame_id="patch_1",
        kind="patch",
        image_bytes=b"model_patch_png",
        screen_size=(1280, 720),
        bounds=VisualBounds(x=80, y=120, width=220, height=160),
        parent_keyframe_id="keyframe_0",
        diff_bounds=VisualBounds(x=90, y=130, width=40, height=20),
    )
    artifact_frame = VisualFrame(
        frame_id="keyframe_1",
        kind="keyframe",
        image_bytes=b"artifact_full_png",
        screen_size=(1280, 720),
        bounds=VisualBounds(x=0, y=0, width=1280, height=720),
    )

    class _FakeSession:
        async def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ComputerUseSessionResult(
                final_output="Done",
                final_visual_frame=visual_patch,
                final_artifact_frame=artifact_frame,
            )

    agent = ActionAgent(
        automation_driver=SimpleNamespace(  # type: ignore[arg-type]
            get_page_url=AsyncMock(return_value="https://example.com"),
            get_page_title=AsyncMock(return_value="Example"),
            get_viewport_size=AsyncMock(return_value=(1280, 720)),
        )
    )
    monkeypatch.setattr(
        agent, "_new_computer_use_session", lambda *_args, **_kwargs: _FakeSession()
    )

    step = TestStep(
        step_number=5,
        description="Type into the email field",
        action="type into email field",
        expected_result="The text appears in the email field",
        action_instruction=ActionInstruction(
            action_type=ActionType.TYPE,
            description="Type email",
            target="Email",
            value="user@example.com",
            expected_outcome="The text appears in the email field",
        ),
    )

    result = await agent.execute_action(
        step,
        {"environment": "desktop"},
        screenshot=b"initial_png",
    )

    assert result.environment_state_after is not None
    assert result.environment_state_after.screenshot == b"artifact_full_png"
    assert result.environment_state_after.frame_kind == "keyframe"
    assert result.environment_state_after.patch_bounds is None
