"""
End-to-end integration tests for the complete HAINDY workflow.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.controller import BrowserController
from src.core.types import ScopeTriageResult, TestState, TestStatus
from src.main import async_main, load_scenario
from src.monitoring.reporter import TestReporter
from src.orchestration.communication import MessageBus
from src.orchestration.coordinator import WorkflowCoordinator
from src.orchestration.state_manager import StateManager


class TestEndToEndIntegration:
    """Test complete workflow integration."""
    
    @pytest.fixture
    def mock_browser_controller(self):
        """Mock browser controller."""
        controller = AsyncMock(spec=BrowserController)
        controller.start = AsyncMock()
        controller.navigate = AsyncMock()
        controller.stop = AsyncMock()
        controller.driver = MagicMock()
        return controller
    
    @pytest.fixture
    def mock_coordinator(self):
        """Mock workflow coordinator."""
        coordinator = AsyncMock(spec=WorkflowCoordinator)
        coordinator.initialize = AsyncMock()
        coordinator.generate_test_plan = AsyncMock()
        coordinator.execute_test_from_requirements = AsyncMock()
        coordinator.cleanup = AsyncMock()
        return coordinator
    
    @pytest.fixture
    def sample_test_state(self):
        """Create a sample test state."""
        from uuid import uuid4
        from datetime import datetime, timezone
        from src.core.types import (
            TestPlan,
            TestStep,
            ActionInstruction,
            ActionResult,
            GridAction,
            GridCoordinate,
            TestCaseResult,
            TestReport,
            TestSummary,
        )
        
        # Create a test plan
        from src.core.types import TestCase, TestCasePriority
        test_case = TestCase(
            test_id="TC001",
            name="Login Test",
            description="Test login functionality",
            priority=TestCasePriority.HIGH,
            steps=[
                TestStep(
                    step_number=1,
                    description="Navigate to login page",
                    action="Navigate to the login page",
                    expected_result="Login form is visible",
                    action_instruction=ActionInstruction(
                        action_type="navigate",
                        description="Navigate to the login page",
                        target="login page",
                        expected_outcome="Login form is visible",
                    ),
                ),
                TestStep(
                    step_number=2,
                    description="Enter credentials",
                    action="Enter username and password",
                    expected_result="Credentials entered",
                    action_instruction=ActionInstruction(
                        action_type="type",
                        description="Enter username and password",
                        target="form fields",
                        expected_outcome="Credentials entered",
                    ),
                ),
                TestStep(
                    step_number=3,
                    description="Submit form",
                    action="Click login button",
                    expected_result="Login successful",
                    action_instruction=ActionInstruction(
                        action_type="click",
                        description="Click login button",
                        target="login button",
                        expected_outcome="Login successful",
                    ),
                ),
            ],
        )
        
        test_plan = TestPlan(
            name="Test Login Flow",
            description="Testing login functionality",
            requirements_source="Test the login flow",
            test_cases=[test_case],
            steps=test_case.steps  # For backward compatibility
        )
        
        # Create test state
        all_steps = test_case.steps
        test_state = TestState(
            test_plan=test_plan,
            status=TestStatus.COMPLETED,
            completed_steps=[step.step_id for step in all_steps],
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        test_case_result = TestCaseResult(
            case_id=uuid4(),
            test_id=test_case.test_id,
            name=test_case.name,
            status=TestStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            steps_total=len(all_steps),
            steps_completed=len(all_steps),
            steps_failed=0,
            step_results=[],
            bugs=[],
        )

        test_summary = TestSummary(
            total_test_cases=1,
            completed_test_cases=1,
            failed_test_cases=0,
            total_steps=len(all_steps),
            completed_steps=len(all_steps),
            failed_steps=0,
            critical_bugs=0,
            high_bugs=0,
            medium_bugs=0,
            low_bugs=0,
            success_rate=1.0,
            execution_time_seconds=1.0,
        )

        test_report = TestReport(
            test_plan_id=test_plan.plan_id,
            test_plan_name=test_plan.name,
            started_at=test_state.start_time or datetime.now(timezone.utc),
            completed_at=test_state.end_time,
            status=TestStatus.COMPLETED,
            test_cases=[test_case_result],
            summary=test_summary,
        )
        
        # For compatibility with reporter that expects different structure
        # Create a mock object that has the expected attributes
        from types import SimpleNamespace
        # Create a more complete mock that works around reporter bugs
        mock_state = SimpleNamespace(
            test_id=test_plan.plan_id,
            plan_id=test_plan.plan_id,
            status=TestStatus.COMPLETED,
            test_plan=test_state.test_plan,
            completed_steps=test_state.completed_steps,
            failed_steps=test_state.failed_steps,
            skipped_steps=test_state.skipped_steps,
            start_time=test_state.start_time,
            end_time=test_state.end_time,
            error_count=0,
            warning_count=0,
            errors=[],
            step_results={
                "step_1": ActionResult(
                    success=True,
                    action=GridAction(
                        instruction=all_steps[0].action_instruction,
                        coordinate=GridCoordinate(cell="M15", offset_x=0.5, offset_y=0.5, confidence=0.95),
                    ),
                    screenshot_after="/tmp/screenshot1.png",
                    execution_time_ms=250,
                    confidence=0.95,
                ),
                "step_2": ActionResult(
                    success=True,
                    action=GridAction(
                        instruction=all_steps[1].action_instruction,
                        coordinate=GridCoordinate(cell="M15", offset_x=0.5, offset_y=0.5, confidence=0.95),
                    ),
                    screenshot_after="/tmp/screenshot2.png",
                    execution_time_ms=300,
                    confidence=0.92,
                ),
                "step_3": ActionResult(
                    success=True,
                    action=GridAction(
                        instruction=all_steps[2].action_instruction,
                        coordinate=GridCoordinate(cell="M15", offset_x=0.5, offset_y=0.5, confidence=0.95),
                    ),
                    screenshot_after="/tmp/screenshot3.png",
                    execution_time_ms=280,
                    confidence=0.98,
                ),
            })
        mock_state.test_report = test_report
        
        return mock_state
    
    def test_load_scenario(self, tmp_path):
        """Test loading test scenario from JSON."""
        # Create test scenario file
        scenario_data = {
            "name": "Test Scenario",
            "requirements": "Test requirements",
            "url": "https://example.com",
        }
        scenario_file = tmp_path / "test_scenario.json"
        scenario_file.write_text(json.dumps(scenario_data))
        
        # Load scenario
        scenario = load_scenario(scenario_file)
        
        assert scenario["name"] == "Test Scenario"
        assert scenario["requirements"] == "Test requirements"
        assert scenario["url"] == "https://example.com"
    
    def test_load_scenario_missing_fields(self, tmp_path):
        """Test loading scenario with missing required fields."""
        # Create invalid scenario file
        scenario_data = {
            "name": "Test Scenario",
            # Missing 'requirements' and 'url'
        }
        scenario_file = tmp_path / "invalid_scenario.json"
        scenario_file.write_text(json.dumps(scenario_data))
        
        # Should exit with error
        with pytest.raises(SystemExit):
            load_scenario(scenario_file)
    
    @pytest.mark.asyncio
    async def test_main_with_requirements(
        self,
        mock_browser_controller,
        mock_coordinator,
        sample_test_state,
        tmp_path,
    ):
        """Test main execution with requirements argument."""
        # Mock the test execution
        mock_coordinator.execute_test_from_requirements.return_value = sample_test_state
        
        # Mock interactive input
        with patch("src.main.get_interactive_requirements") as mock_get_req:
            mock_get_req.return_value = ("Test the login flow", "https://example.com")
            
            with patch("src.main.BrowserController", return_value=mock_browser_controller):
                with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                    # Run main with requirements
                    exit_code = await async_main([
                        "--requirements",
                        "--output", str(tmp_path),
                        "--format", "json",
                    ])
        
        # Verify success
        assert exit_code == 0
        
        # Verify interactive mode was called
        mock_get_req.assert_called_once()
        
        # Verify components were initialized
        mock_browser_controller.start.assert_called_once()
        mock_coordinator.initialize.assert_called_once()
        
        # Verify test was executed
        mock_coordinator.execute_test_from_requirements.assert_called_once()
        
        # Verify report was generated
        reports = list(tmp_path.glob("test_report_*.json"))
        assert len(reports) == 1
    
    @pytest.mark.asyncio
    async def test_main_with_scenario_file(
        self,
        mock_browser_controller,
        mock_coordinator,
        sample_test_state,
        tmp_path,
    ):
        """Test main execution with scenario file."""
        # Create scenario file
        scenario_data = {
            "name": "Test Scenario",
            "requirements": "Test the application",
            "url": "https://example.com",
        }
        scenario_file = tmp_path / "test_scenario.json"
        scenario_file.write_text(json.dumps(scenario_data))
        
        # Mock the test execution
        mock_coordinator.execute_test_from_requirements.return_value = sample_test_state
        
        with patch("src.main.BrowserController", return_value=mock_browser_controller):
            with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                # Run main with scenario file
                exit_code = await async_main([
                    "--json-test-plan", str(scenario_file),
                    "--output", str(tmp_path),
                ])
        
        # Verify success
        assert exit_code == 0
        
        # Verify test was executed with scenario data
        mock_coordinator.execute_test_from_requirements.assert_called_once_with(
            requirements="Test the application",
            initial_url="https://example.com",
        )
    
    @pytest.mark.asyncio
    async def test_main_plan_only_mode(
        self,
        mock_browser_controller,
        mock_coordinator,
        tmp_path,
    ):
        """Test main execution in plan-only mode."""
        from src.core.types import TestPlan, TestStep, ActionInstruction
        
        # Mock test plan
        from src.core.types import TestCase, TestCasePriority
        test_case = TestCase(
            test_id="TC001",
            name="Login Test",
            description="Test login functionality",
            priority=TestCasePriority.HIGH,
            steps=[
                TestStep(
                    step_number=1,
                    description="Navigate to login page",
                    action="Navigate to the login page",
                    expected_result="Login form is visible",
                    action_instruction=ActionInstruction(
                        action_type="navigate",
                        description="Navigate to the login page",
                        target="login page",
                        expected_outcome="Login form is visible",
                    ),
                ),
                TestStep(
                    step_number=2,
                    description="Enter username",
                    action="Enter username in the field",
                    expected_result="Username entered",
                    action_instruction=ActionInstruction(
                        action_type="type",
                        description="Enter username in the field",
                        target="username field",
                        value="testuser",
                        expected_outcome="Username entered",
                    ),
                ),
            ],
        )
        
        mock_plan = TestPlan(
            name="Test Login Flow",
            description="Testing login functionality",
            requirements_source="Test login",
            test_cases=[test_case],
            steps=test_case.steps  # For backward compatibility
        )
        planner_instance = MagicMock()
        triage_result = ScopeTriageResult(
            in_scope="Scope triage summary",
            explicit_exclusions=["Do not test FMC."],
            ambiguous_points=[],
            blocking_questions=[],
        )
        pipeline_mock = AsyncMock(return_value=(mock_plan, triage_result))
        
        # Mock interactive input
        with patch("src.main.get_interactive_requirements") as mock_get_req:
            mock_get_req.return_value = ("Test login", "https://example.com")
            
            with patch("src.main.BrowserController", return_value=mock_browser_controller):
                with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                    with patch("src.main.ScopeTriageAgent", return_value=MagicMock()), \
                         patch("src.main.TestPlannerAgent", return_value=planner_instance), \
                         patch("src.main.run_scope_triage_and_plan", pipeline_mock):
                        # Run main in plan-only mode
                        exit_code = await async_main([
                            "--requirements",
                            "--plan-only",
                        ])
        
        # Verify success
        assert exit_code == 0
        
        # Verify only plan was generated, not executed
        pipeline_mock.assert_awaited_once()
        mock_coordinator.generate_test_plan.assert_not_called()
        mock_coordinator.execute_test_from_requirements.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_main_with_timeout(
        self,
        mock_browser_controller,
        mock_coordinator,
        tmp_path,
    ):
        """Test main execution with timeout."""
        # Mock timeout during execution
        mock_coordinator.execute_test_from_requirements.side_effect = asyncio.TimeoutError()
        
        # Mock interactive input
        with patch("src.main.get_interactive_requirements") as mock_get_req:
            mock_get_req.return_value = ("Test login", "https://example.com")
            
            with patch("src.main.BrowserController", return_value=mock_browser_controller):
                with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                    # Run main with short timeout
                    exit_code = await async_main([
                        "--requirements",
                        "--timeout", "1",
                    ])
        
        # Verify timeout exit code
        assert exit_code == 2
    
    @pytest.mark.asyncio
    async def test_test_reporter_integration(self, sample_test_state, tmp_path):
        """Test TestReporter integration with TestState."""
        reporter = TestReporter()
        
        # Generate HTML report
        html_path, _ = await reporter.generate_report(
            test_state=sample_test_state,
            output_dir=tmp_path,
            format="html",
        )
        
        assert html_path.exists()
        assert html_path.suffix == ".html"
        
        # Verify HTML content
        html_content = html_path.read_text()
        assert "HAINDY Test Report" in html_content
        assert "Test Execution Report" in html_content
        assert "passed" in html_content.lower()
        
        # Generate JSON report
        json_path, actions_path = await reporter.generate_report(
            test_state=sample_test_state,
            output_dir=tmp_path,
            format="json",
        )
        
        assert json_path.exists()
        assert json_path.suffix == ".json"
        
        # Verify JSON content
        json_data = json.loads(json_path.read_text())
        assert json_data["summary"]["outcome"] == "passed"
        assert json_data["steps"]["total"] == 3
        assert json_data["steps"]["passed"] == 3
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, tmp_path):
        """Test the complete workflow with real components (mocked browser)."""
        # Create test scenario
        scenario_data = {
            "name": "Integration Test",
            "requirements": "Navigate to test page and verify title",
            "url": "https://example.com",
        }
        scenario_file = tmp_path / "integration_test.json"
        scenario_file.write_text(json.dumps(scenario_data))
        
        # Mock browser driver
        mock_driver = MagicMock()
        mock_driver.navigate = AsyncMock()
        mock_driver.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
        
        # Create real components
        message_bus = MessageBus()
        state_manager = StateManager()
        
        with patch("src.browser.controller.PlaywrightDriver", return_value=mock_driver):
            # This would run the actual workflow if agents were fully implemented
            # For now, we verify the components can be created and initialized
            coordinator = WorkflowCoordinator(
                message_bus=message_bus,
                state_manager=state_manager,
                browser_driver=mock_driver,
            )
            
            # Verify components are properly connected
            assert coordinator.message_bus == message_bus
            assert coordinator.state_manager == state_manager
            assert coordinator.browser_driver == mock_driver
            
            # Cleanup
            await message_bus.shutdown()
            await state_manager.shutdown()
    
    def test_cli_help(self, capsys):
        """Test CLI help output."""
        with pytest.raises(SystemExit) as exc_info:
            from src.main import create_parser
            parser = create_parser()
            parser.parse_args(["--help"])
        
        assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        assert "HAINDY - Autonomous AI Testing Agent" in captured.out
        assert "--requirements" in captured.out
        assert "--json-test-plan" in captured.out
        assert "Examples:" in captured.out
