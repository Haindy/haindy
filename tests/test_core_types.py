"""Core type tests for the computer-use data model."""

from src.core.types import (
    ActionInstruction,
    ActionType,
    CoordinateReference,
    ResolvedAction,
    TestCase,
    TestCasePriority,
    TestPlan,
    TestStep,
)


def test_resolved_action_holds_coordinate_reference() -> None:
    instruction = ActionInstruction(
        action_type=ActionType.CLICK,
        description="Click submit",
        target="Submit button",
        expected_outcome="Form submits",
    )
    action = ResolvedAction(
        instruction=instruction,
        coordinates=CoordinateReference(
            target_reference="submit_button",
            pixel_coordinates=(800, 500),
            confidence=0.88,
        ),
    )

    assert action.instruction.action_type == ActionType.CLICK
    assert action.coordinates is not None
    assert action.coordinates.target_reference == "submit_button"


def test_test_plan_synchronizes_flattened_steps_from_test_cases() -> None:
    step = TestStep(
        step_number=1,
        description="Open homepage",
        action="Navigate to homepage",
        expected_result="Homepage loads",
    )
    case = TestCase(
        test_id="TC001",
        name="Homepage smoke",
        description="Open homepage and verify load",
        priority=TestCasePriority.MEDIUM,
        steps=[step],
    )

    plan = TestPlan(
        name="Smoke Plan",
        description="Basic smoke plan",
        requirements_source="REQ-1",
        test_cases=[case],
    )

    assert len(plan.test_cases) == 1
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Open homepage"


def test_test_plan_builds_default_case_from_legacy_steps() -> None:
    step = TestStep(
        step_number=1,
        description="Open settings",
        action="Click Settings",
        expected_result="Settings view is visible",
    )

    plan = TestPlan(
        name="Legacy Plan",
        description="Legacy format plan",
        requirements_source="REQ-2",
        test_cases=[],
        steps=[step],
    )

    assert len(plan.test_cases) == 1
    assert plan.test_cases[0].steps[0].action == "Click Settings"
