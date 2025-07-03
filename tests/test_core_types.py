"""
Unit tests for core data types.
"""

from datetime import datetime
from uuid import UUID

import pytest

from src.core.types import (
    ActionInstruction,
    ActionResult,
    ActionType,
    ConfidenceLevel,
    GridAction,
    GridCoordinate,
    TestPlan,
    TestState,
    TestStatus,
    TestStep,
)


class TestGridCoordinate:
    """Tests for GridCoordinate model."""

    def test_valid_coordinate(self):
        """Test creating a valid grid coordinate."""
        coord = GridCoordinate(
            cell="M23",
            offset_x=0.7,
            offset_y=0.4,
            confidence=0.95,
            refined=True,
        )
        assert coord.cell == "M23"
        assert coord.offset_x == 0.7
        assert coord.offset_y == 0.4
        assert coord.confidence == 0.95
        assert coord.refined is True

    def test_default_offsets(self):
        """Test default offset values."""
        coord = GridCoordinate(cell="A1", confidence=0.8)
        assert coord.offset_x == 0.5
        assert coord.offset_y == 0.5
        assert coord.refined is False

    def test_invalid_offsets(self):
        """Test validation of offset values."""
        with pytest.raises(ValueError):
            GridCoordinate(cell="A1", offset_x=1.5, confidence=0.8)

        with pytest.raises(ValueError):
            GridCoordinate(cell="A1", offset_y=-0.1, confidence=0.8)

    def test_invalid_confidence(self):
        """Test validation of confidence score."""
        with pytest.raises(ValueError):
            GridCoordinate(cell="A1", confidence=1.5)

        with pytest.raises(ValueError):
            GridCoordinate(cell="A1", confidence=-0.1)


class TestActionInstruction:
    """Tests for ActionInstruction model."""

    def test_click_instruction(self):
        """Test creating a click instruction."""
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the login button",
            target="Login button",
            expected_outcome="Login form appears",
        )
        assert instruction.action_type == ActionType.CLICK
        assert instruction.target == "Login button"
        assert instruction.value is None
        assert instruction.timeout == 5000

    def test_type_instruction(self):
        """Test creating a type instruction."""
        instruction = ActionInstruction(
            action_type=ActionType.TYPE,
            description="Enter username",
            target="Username field",
            value="test@example.com",
            expected_outcome="Username field is filled",
            timeout=3000,
        )
        assert instruction.action_type == ActionType.TYPE
        assert instruction.value == "test@example.com"
        assert instruction.timeout == 3000


class TestTestStep:
    """Tests for TestStep model."""

    def test_create_step(self):
        """Test creating a test step."""
        instruction = ActionInstruction(
            action_type=ActionType.NAVIGATE,
            description="Go to login page",
            value="https://example.com/login",
            expected_outcome="Login page loads",
        )
        step = TestStep(
            step_number=1,
            description="Navigate to login page",
            action_instruction=instruction,
        )
        assert step.step_number == 1
        assert isinstance(step.step_id, UUID)
        assert step.dependencies == []
        assert step.optional is False
        assert step.max_retries == 3

    def test_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        dep_id = UUID("12345678-1234-5678-1234-567812345678")
        instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Submit form",
            expected_outcome="Form submitted",
        )
        step = TestStep(
            step_number=5,
            description="Submit the form",
            action_instruction=instruction,
            dependencies=[dep_id],
            optional=True,
            max_retries=1,
        )
        assert dep_id in step.dependencies
        assert step.optional is True
        assert step.max_retries == 1


class TestTestPlan:
    """Tests for TestPlan model."""

    def test_create_test_plan(self):
        """Test creating a test plan."""
        steps = [
            TestStep(
                step_number=1,
                description="Navigate to site",
                action_instruction=ActionInstruction(
                    action_type=ActionType.NAVIGATE,
                    description="Go to homepage",
                    value="https://example.com",
                    expected_outcome="Homepage loads",
                ),
            ),
            TestStep(
                step_number=2,
                description="Click login",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click login button",
                    target="Login button",
                    expected_outcome="Login form appears",
                ),
            ),
        ]

        plan = TestPlan(
            name="Login Flow Test",
            description="Test the login functionality",
            requirements="User should be able to log in with valid credentials",
            steps=steps,
            tags=["login", "authentication"],
        )

        assert plan.name == "Login Flow Test"
        assert len(plan.steps) == 2
        assert isinstance(plan.plan_id, UUID)
        assert isinstance(plan.created_at, datetime)
        assert plan.tags == ["login", "authentication"]


class TestTestState:
    """Tests for TestState model."""

    def test_initial_state(self):
        """Test creating initial test state."""
        plan = TestPlan(
            name="Test Plan",
            description="Test",
            requirements="Requirements",
            steps=[],
        )
        state = TestState(test_plan=plan)

        assert state.test_plan == plan
        assert state.current_step is None
        assert state.completed_steps == []
        assert state.failed_steps == []
        assert state.skipped_steps == []
        assert state.status == TestStatus.PENDING
        assert state.error_count == 0
        assert state.warning_count == 0

    def test_state_with_progress(self):
        """Test state with execution progress."""
        plan = TestPlan(
            name="Test Plan",
            description="Test",
            requirements="Requirements",
            steps=[],
        )
        step_id = UUID("12345678-1234-5678-1234-567812345678")
        
        state = TestState(
            test_plan=plan,
            status=TestStatus.IN_PROGRESS,
            completed_steps=[step_id],
            error_count=1,
            start_time=datetime.utcnow(),
        )

        assert state.status == TestStatus.IN_PROGRESS
        assert step_id in state.completed_steps
        assert state.error_count == 1
        assert isinstance(state.start_time, datetime)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_levels(self):
        """Test confidence level values."""
        assert ConfidenceLevel.VERY_HIGH == "very_high"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.VERY_LOW == "very_low"


class TestActionResult:
    """Tests for ActionResult model."""

    def test_successful_result(self):
        """Test creating a successful action result."""
        action = GridAction(
            instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click button",
                expected_outcome="Button clicked",
            ),
            coordinate=GridCoordinate(
                cell="B5",
                confidence=0.9,
            ),
        )
        
        result = ActionResult(
            success=True,
            action=action,
            execution_time_ms=150,
            confidence=0.95,
        )
        
        assert result.success is True
        assert result.action == action
        assert result.execution_time_ms == 150
        assert result.confidence == 0.95
        assert result.error_message is None
        assert isinstance(result.action_id, UUID)
        assert isinstance(result.timestamp, datetime)

    def test_failed_result(self):
        """Test creating a failed action result."""
        action = GridAction(
            instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click button",
                expected_outcome="Button clicked",
            ),
            coordinate=GridCoordinate(
                cell="B5",
                confidence=0.9,
            ),
        )
        
        result = ActionResult(
            success=False,
            action=action,
            execution_time_ms=5000,
            confidence=0.3,
            error_message="Element not found at specified coordinates",
        )
        
        assert result.success is False
        assert result.error_message == "Element not found at specified coordinates"
        assert result.confidence == 0.3