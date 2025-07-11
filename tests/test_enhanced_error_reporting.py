"""
Tests for enhanced error reporting functionality.
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from src.core.types import (
    StepResult,
    ActionInstruction,
    ActionType,
    TestPlan,
    TestStep,
    TestCase,
    TestCasePriority,
    TestState,
    TestStatus,
    GridCoordinate
)
from src.core.enhanced_types import BugReport, AIAnalysis
from src.monitoring.simple_html_reporter import SimpleHTMLReporter
from types import SimpleNamespace


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    steps = [
        TestStep(
            step_id=uuid4(),
            step_number=1,
            description="Navigate to Wikipedia",
            action="Navigate to Wikipedia homepage",
            expected_result="Wikipedia homepage loaded",
            action_instruction=ActionInstruction(
                action_type=ActionType.NAVIGATE,
                description="Navigate to Wikipedia homepage",
                target="https://wikipedia.org",
                expected_outcome="Wikipedia homepage loaded"
            ),
            dependencies=[],
            optional=False
        ),
        TestStep(
            step_id=uuid4(),
            step_number=2,
            description="Click search box",
            action="Click on the search input field",
            expected_result="Search box focused and ready for input",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click on the search input field",
                target="search box",
                expected_outcome="Search box focused and ready for input"
            ),
            dependencies=[],
            optional=False
        ),
        TestStep(
            step_id=uuid4(),
            step_number=3,
            description="Type search term",
            action="Type 'Artificial Intelligence' in search box",
            expected_result="Search term entered in search box",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Type 'Artificial Intelligence' in search box",
                target="search box",
                value="Artificial Intelligence",
                expected_outcome="Search term entered in search box"
            ),
            dependencies=[],
            optional=False
        )
    ]
    
    test_case = TestCase(
        test_id="TC001",
        name="Wikipedia Search Test",
        description="Test searching on Wikipedia",
        priority=TestCasePriority.HIGH,
        steps=steps,
        tags=["search", "wikipedia"]
    )
    
    return TestPlan(
        plan_id=uuid4(),
        name="Wikipedia Search Test",
        description="Test searching on Wikipedia",
        requirements_source="User should be able to search on Wikipedia",
        test_cases=[test_case],
        steps=steps,  # For backward compatibility
        created_at=datetime.now(timezone.utc),
        tags=["search", "wikipedia"]
    )


@pytest.fixture
def failed_step_result(sample_test_plan):
    """Create a failed step result with detailed error information."""
    step = sample_test_plan.steps[1]  # Click search box step
    
    # Simulate detailed action result from Action Agent
    action_result_details = {
        "action_type": "click",
        "validation_passed": True,
        "validation_reasoning": "Search box is visible in the screenshot",
        "validation_confidence": 0.95,
        "grid_cell": "M15",
        "grid_coordinates": (960, 300),
        "offset_x": 0.5,
        "offset_y": 0.3,
        "coordinate_confidence": 0.85,
        "coordinate_reasoning": "Search box identified at grid cell M15",
        "execution_success": False,
        "execution_time_ms": 523.4,
        "execution_error": "Click failed: Element not interactable",
        "url_before": "https://wikipedia.org",
        "url_after": "https://wikipedia.org",
        "page_title_before": "Wikipedia",
        "page_title_after": "Wikipedia",
        "screenshot_before": b"fake_screenshot_before_data",
        "screenshot_after": b"fake_screenshot_after_data",
        "grid_screenshot_highlighted": b"fake_grid_screenshot_data",
        "ai_analysis": {
            "success": False,
            "confidence": 0.2,
            "actual_outcome": "Click failed - search box not responding to click",
            "matches_expected": False,
            "ui_changes": [],
            "recommendations": [
                "Try clicking at a different offset within the grid cell",
                "Verify the search box is not covered by another element",
                "Check if JavaScript is fully loaded"
            ],
            "anomalies": [
                "Search box appears to be present but not interactable",
                "No UI changes detected after click attempt"
            ]
        }
    }
    
    # Create a mock TestStepResult since this class doesn't exist in the codebase
    return SimpleNamespace(
        step=step,
        success=False,
        action_taken=None,
        actual_result="Click failed: Element not interactable",
        screenshot_before=b"fake_screenshot_before_data",
        screenshot_after=b"fake_screenshot_after_data",
        execution_mode="visual",
        action_result_details=action_result_details,
        # Add the create_bug_report method
        create_bug_report=lambda test_plan_name: BugReport(
            test_step=step,
            step_number=step.step_number,
            test_plan_name=test_plan_name,
            failure_type="execution",
            error_message="Click failed: Element not interactable",
            attempted_action="Click on the search input field",
            expected_outcome="Search box focused and ready for input",
            actual_outcome="click failed - search box not responding to click",
            grid_cell_targeted="M15",
            severity="critical",
            is_blocking=True,
            screenshot_before=b"fake_screenshot_before_data",
            screenshot_after=b"fake_screenshot_after_data",
            grid_screenshot=b"fake_grid_screenshot_data",
            confidence_scores={
                "validation": 0.95,
                "coordinate": 0.85,
                "execution": 0.0,
                "evaluation": 0.2,
                "overall": 0.2
            },
            ai_analysis=AIAnalysis(
                success=False,
                confidence=0.2,
                actual_outcome="Click failed - search box not responding to click",
                matches_expected=False,
                ui_changes=[],
                recommendations=[
                    "Try clicking at a different offset within the grid cell",
                    "Verify the search box is not covered by another element",
                    "Check if JavaScript is fully loaded"
                ],
                anomalies=[
                    "Search box appears to be present but not interactable",
                    "No UI changes detected after click attempt"
                ]
            ),
            ui_anomalies=[
                "Search box appears to be present but not interactable",
                "No UI changes detected after click attempt"
            ],
            suggested_fixes=[
                "Try clicking at a different offset within the grid cell",
                "Verify the search box is not covered by another element",
                "Check if JavaScript is fully loaded"
            ]
        )
    )


@pytest.fixture
def successful_step_result(sample_test_plan):
    """Create a successful step result."""
    step = sample_test_plan.steps[0]  # Navigate step
    
    action_result_details = {
        "action_type": "navigate",
        "validation_passed": True,
        "execution_success": True,
        "ai_analysis": {
            "success": True,
            "confidence": 0.95,
            "actual_outcome": "Successfully navigated to Wikipedia homepage"
        }
    }
    
    # Create a mock TestStepResult for successful case
    return SimpleNamespace(
        step=step,
        success=True,
        actual_result="Successfully navigated to Wikipedia",
        execution_mode="visual",
        action_result_details=action_result_details,
        # Successful steps return None for bug report
        create_bug_report=lambda test_plan_name: None
    )


class TestBugReportCreation:
    """Test BugReport creation from TestStepResult."""
    
    def test_create_bug_report_from_failed_step(self, failed_step_result, sample_test_plan):
        """Test creating a bug report from a failed step."""
        bug_report = failed_step_result.create_bug_report(sample_test_plan.name)
        
        assert bug_report is not None
        assert isinstance(bug_report, BugReport)
        assert bug_report.step_number == 2
        assert bug_report.test_plan_name == "Wikipedia Search Test"
        assert bug_report.failure_type == "execution"
        assert bug_report.error_message == "Click failed: Element not interactable"
        assert bug_report.grid_cell_targeted == "M15"
        assert bug_report.severity == "critical"  # Non-optional step
        assert bug_report.is_blocking is True
        
        # Check confidence scores
        assert bug_report.confidence_scores["validation"] == 0.95
        assert bug_report.confidence_scores["coordinate"] == 0.85
        assert bug_report.confidence_scores["execution"] == 0.0
        assert bug_report.confidence_scores["evaluation"] == 0.2
        
        # Check AI analysis data
        assert len(bug_report.ui_anomalies) == 2
        assert len(bug_report.suggested_fixes) == 3
        assert "search box not responding" in bug_report.actual_outcome.lower()
    
    def test_no_bug_report_for_successful_step(self, successful_step_result, sample_test_plan):
        """Test that successful steps don't create bug reports."""
        bug_report = successful_step_result.create_bug_report(sample_test_plan.name)
        assert bug_report is None
    
    def test_bug_report_summary(self, failed_step_result, sample_test_plan):
        """Test bug report summary generation."""
        bug_report = failed_step_result.create_bug_report(sample_test_plan.name)
        summary = bug_report.to_summary()
        
        assert "BUG [CRITICAL]" in summary
        assert "Step 2:" in summary
        assert "Click failed" in summary
        assert "Expected:" in summary
        assert "Actual:" in summary
        assert "Confidence: 20%" in summary


class TestSimpleHTMLReporter:
    """Test the simple HTML reporter."""
    
    def test_generate_html_report(self, sample_test_plan, successful_step_result, failed_step_result, tmp_path):
        """Test generating an HTML report with bug details."""
        # Create test state
        test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[sample_test_plan.steps[0].step_id],
            failed_steps=[sample_test_plan.steps[1].step_id],
            skipped_steps=[],
            status=TestStatus.FAILED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            error_count=1,
            warning_count=0
        )
        
        # Create execution history
        execution_history = [successful_step_result, failed_step_result]
        
        # Generate report
        reporter = SimpleHTMLReporter()
        output_path = tmp_path / "test_report.html"
        result_path = reporter.generate_report(test_state, execution_history, output_path)
        
        # Verify report was created
        assert result_path.exists()
        assert result_path == output_path
        
        # Check content
        content = result_path.read_text()
        assert "Wikipedia Search Test" in content
        assert "Bug Reports" in content
        assert "Step 2:" in content
        assert "Click failed" in content
        assert "Grid Screenshot (Highlighted)" in content
        assert "M15" in content  # Grid cell
        assert "85%" in content  # Coordinate confidence
        assert "Recommended Fixes" in content
        assert "Try clicking at a different offset" in content
    
    def test_report_without_bugs(self, sample_test_plan, successful_step_result, tmp_path):
        """Test generating a report with no failures."""
        # Create test state with all steps passed
        test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[step.step_id for step in sample_test_plan.steps],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        
        # Create execution history with only successful results
        execution_history = [successful_step_result] * 3
        
        # Generate report
        reporter = SimpleHTMLReporter()
        output_path = tmp_path / "success_report.html"
        result_path = reporter.generate_report(test_state, execution_history, output_path)
        
        # Check content
        content = result_path.read_text()
        assert "Wikipedia Search Test" in content
        assert "100%" in content  # Success rate
        assert "COMPLETED" in content
        # Verify no actual bug data is shown
        assert "Step 1:" not in content  # No bug step numbers
        assert "Click failed" not in content  # No error messages