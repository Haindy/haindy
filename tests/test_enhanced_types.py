"""Enhanced action result model tests."""

from src.core.enhanced_types import (
    AIAnalysis,
    ComputerToolTurn,
    CoordinateResult,
    EnhancedActionResult,
    EnvironmentState,
    ExecutionResult,
    SafetyEvent,
    ValidationResult,
)
from src.core.types import TestStep


def _step() -> TestStep:
    return TestStep(
        step_number=1,
        description="Click submit",
        action="Click submit button",
        expected_result="Submission succeeds",
    )


def test_dict_for_compatibility_exposes_provider_neutral_coordinates() -> None:
    step = _step()
    result = EnhancedActionResult(
        test_step_id=step.step_id,
        test_step=step,
        test_context={"run": "unit"},
        validation=ValidationResult(valid=True, confidence=0.9, reasoning="ok"),
        coordinates=CoordinateResult(
            target_reference="submit_button",
            pixel_coordinates=(900, 650),
            relative_x=0.5,
            relative_y=0.5,
            confidence=0.86,
            reasoning="Clear target",
        ),
        execution=ExecutionResult(success=True, execution_time_ms=120.0),
        environment_state_before=EnvironmentState(
            url="https://example.com/form",
            title="Form",
            viewport_size=(1280, 720),
        ),
        environment_state_after=EnvironmentState(
            url="https://example.com/success",
            title="Success",
            viewport_size=(1280, 720),
        ),
        ai_analysis=AIAnalysis(
            success=True,
            confidence=0.8,
            actual_outcome="Form submitted",
            matches_expected=True,
        ),
        overall_success=True,
    )

    compat = result.dict_for_compatibility()
    assert compat["target_reference"] == "submit_button"
    assert compat["pixel_coordinates"] == (900, 650)
    assert compat["execution_success"] is True
    assert compat["url_after"] == "https://example.com/success"


def test_computer_actions_and_safety_events_are_serialized_in_compat_dict() -> None:
    step = _step()
    result = EnhancedActionResult(
        test_step_id=step.step_id,
        test_step=step,
        test_context={},
        validation=ValidationResult(valid=True, confidence=0.7, reasoning="ok"),
        execution=ExecutionResult(success=True, execution_time_ms=10.0),
        computer_actions=[
            ComputerToolTurn(call_id="call_1", action_type="click", parameters={"x": 1, "y": 2}, status="executed")
        ],
        safety_events=[
            SafetyEvent(call_id="call_1", code="none", message="No safety issues")
        ],
        overall_success=True,
    )

    compat = result.dict_for_compatibility()
    assert len(compat["computer_actions"]) == 1
    assert len(compat["safety_events"]) == 1
