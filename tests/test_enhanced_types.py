"""
Tests for enhanced types models.
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from src.core.enhanced_types import (
    ValidationResult, CoordinateResult, ExecutionResult,
    AIAnalysis, BrowserState, EnhancedActionResult,
    ActionPattern
)
from src.core.types import TestStep, ActionInstruction, ActionType


@pytest.fixture
def sample_test_step():
    """Create a sample test step."""
    return TestStep(
        step_number=1,
        description="Click the login button",
        action_instruction=ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click the login button",
            target="Login button",
            expected_outcome="Login form is submitted"
        )
    )


@pytest.fixture
def sample_test_context():
    """Create sample test context."""
    return {
        "test_plan_name": "Login Test",
        "current_step_description": "Click login button",
        "previous_steps_summary": "Navigated to login page",
        "total_steps": 3,
        "completed_steps": 1
    }


class TestValidationResult:
    """Tests for ValidationResult model."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            valid=True,
            confidence=0.95,
            reasoning="Login button is clearly visible",
            concerns=[],
            suggestions=[]
        )
        
        assert result.valid is True
        assert result.confidence == 0.95
        assert result.reasoning == "Login button is clearly visible"
        assert result.concerns == []
        assert result.suggestions == []
        assert isinstance(result.timestamp, datetime)
    
    def test_validation_result_with_failures(self):
        """Test validation result with failures."""
        result = ValidationResult(
            valid=False,
            confidence=0.3,
            reasoning="Cannot find login button",
            concerns=["Button not visible", "Page may not have loaded"],
            suggestions=["Wait for page load", "Check if on correct page"]
        )
        
        assert result.valid is False
        assert result.confidence == 0.3
        assert len(result.concerns) == 2
        assert len(result.suggestions) == 2


class TestCoordinateResult:
    """Tests for CoordinateResult model."""
    
    def test_coordinate_result_creation(self):
        """Test creating a coordinate result."""
        result = CoordinateResult(
            grid_cell="M23",
            grid_coordinates=(960, 540),
            offset_x=0.5,
            offset_y=0.5,
            confidence=0.92,
            reasoning="Button found in center",
            refined=False
        )
        
        assert result.grid_cell == "M23"
        assert result.grid_coordinates == (960, 540)
        assert result.offset_x == 0.5
        assert result.offset_y == 0.5
        assert result.confidence == 0.92
        assert result.refined is False
    
    def test_coordinate_result_with_refinement(self):
        """Test coordinate result with refinement."""
        result = CoordinateResult(
            grid_cell="B7",
            grid_coordinates=(150, 200),
            offset_x=0.7,
            offset_y=0.3,
            confidence=0.95,
            reasoning="Refined position found",
            refined=True,
            refinement_details={"method": "zoom", "iterations": 1}
        )
        
        assert result.refined is True
        assert result.refinement_details == {"method": "zoom", "iterations": 1}


class TestExecutionResult:
    """Tests for ExecutionResult model."""
    
    def test_execution_success(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            execution_time_ms=523.4
        )
        
        assert result.success is True
        assert result.execution_time_ms == 523.4
        assert result.error_message is None
    
    def test_execution_failure(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            execution_time_ms=100.5,
            error_message="Element not clickable",
            error_traceback="Traceback..."
        )
        
        assert result.success is False
        assert result.error_message == "Element not clickable"
        assert result.error_traceback == "Traceback..."


class TestAIAnalysis:
    """Tests for AIAnalysis model."""
    
    def test_ai_analysis_creation(self):
        """Test creating AI analysis."""
        analysis = AIAnalysis(
            success=True,
            confidence=0.9,
            actual_outcome="Login form submitted",
            matches_expected=True,
            ui_changes=["Form disappeared", "Loading spinner appeared"],
            recommendations=[],
            anomalies=[]
        )
        
        assert analysis.success is True
        assert analysis.confidence == 0.9
        assert analysis.matches_expected is True
        assert len(analysis.ui_changes) == 2


class TestBrowserState:
    """Tests for BrowserState model."""
    
    def test_browser_state_creation(self):
        """Test creating browser state."""
        state = BrowserState(
            url="https://example.com/login",
            title="Login Page",
            viewport_size=(1920, 1080),
            screenshot=b"fake_screenshot_data"
        )
        
        assert state.url == "https://example.com/login"
        assert state.title == "Login Page"
        assert state.viewport_size == (1920, 1080)
        assert state.screenshot == b"fake_screenshot_data"


class TestEnhancedActionResult:
    """Tests for EnhancedActionResult model."""
    
    def test_enhanced_result_creation(self, sample_test_step, sample_test_context):
        """Test creating enhanced action result."""
        result = EnhancedActionResult(
            test_step_id=sample_test_step.step_id,
            test_step=sample_test_step,
            test_context=sample_test_context,
            validation=ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Action is valid"
            )
        )
        
        assert isinstance(result.action_id, UUID)
        assert result.test_step_id == sample_test_step.step_id
        assert result.test_step == sample_test_step
        assert result.validation.valid is True
        assert result.overall_success is False  # No execution yet
    
    def test_enhanced_result_full_success(self, sample_test_step, sample_test_context):
        """Test enhanced result with full success."""
        result = EnhancedActionResult(
            test_step_id=sample_test_step.step_id,
            test_step=sample_test_step,
            test_context=sample_test_context,
            validation=ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Valid"
            ),
            coordinates=CoordinateResult(
                grid_cell="M23",
                grid_coordinates=(960, 540),
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.9,
                reasoning="Found button"
            ),
            browser_state_before=BrowserState(
                url="https://example.com",
                title="Example",
                viewport_size=(1920, 1080)
            ),
            browser_state_after=BrowserState(
                url="https://example.com/dashboard",
                title="Dashboard",
                viewport_size=(1920, 1080)
            ),
            execution=ExecutionResult(
                success=True,
                execution_time_ms=500.0
            ),
            overall_success=True
        )
        
        assert result.overall_success is True
        assert result.failure_phase is None
    
    def test_enhanced_result_dict_compatibility(self, sample_test_step, sample_test_context):
        """Test backward compatibility dictionary conversion."""
        result = EnhancedActionResult(
            test_step_id=sample_test_step.step_id,
            test_step=sample_test_step,
            test_context=sample_test_context,
            validation=ValidationResult(
                valid=True,
                confidence=0.95,
                reasoning="Valid"
            ),
            coordinates=CoordinateResult(
                grid_cell="M23",
                grid_coordinates=(960, 540),
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.9,
                reasoning="Found"
            )
        )
        
        # Convert to dict
        dict_result = result.dict_for_compatibility()
        
        # Check key fields
        assert dict_result["action_type"] == "click"
        assert dict_result["validation_passed"] is True
        assert dict_result["grid_cell"] == "M23"
        assert dict_result["grid_coordinates"] == (960, 540)
        assert dict_result["coordinate_confidence"] == 0.9
        assert dict_result["test_context"] == sample_test_context


class TestActionPattern:
    """Tests for ActionPattern model."""
    
    def test_action_pattern_creation(self):
        """Test creating action pattern."""
        pattern = ActionPattern(
            action_type="click",
            target_description="Login button",
            grid_coordinates=CoordinateResult(
                grid_cell="M23",
                grid_coordinates=(960, 540),
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.95,
                reasoning="Button center"
            ),
            playwright_command="page.click('button#login')",
            confidence=0.9,
            success_count=5,
            failure_count=1
        )
        
        assert pattern.action_type == "click"
        assert pattern.success_count == 5
        assert pattern.failure_count == 1
        assert pattern.confidence == 0.9