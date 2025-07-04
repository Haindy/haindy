"""
End-to-end integration tests for the complete HAINDY workflow.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.controller import BrowserController
from src.core.types import TestState, TestStatus
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
        controller.initialize = AsyncMock()
        controller.navigate = AsyncMock()
        controller.cleanup = AsyncMock()
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
        from src.core.types import ActionResult, GridAction, GridCoordinate, ActionInstruction
        
        # Create sample grid actions
        action1 = GridAction(
            instruction=ActionInstruction(
                action_type="click",
                description="Click the login button",
                target="Login button",
                expected_outcome="Login form submitted",
            ),
            coordinates=[
                GridCoordinate(
                    cell="M15",
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.95,
                )
            ],
        )
        
        test_state = TestState(
            test_id=uuid4(),
            status=TestStatus.COMPLETED,
            current_step=3,
            step_results={
                "step_1": ActionResult(
                    success=True,
                    action=action1,
                    screenshot_after="/tmp/screenshot1.png",
                    execution_time_ms=250,
                ),
                "step_2": ActionResult(
                    success=True,
                    action=action1,  # Reuse for simplicity
                    screenshot_after="/tmp/screenshot2.png",
                    execution_time_ms=300,
                ),
                "step_3": ActionResult(
                    success=True,
                    action=action1,  # Reuse for simplicity
                    screenshot_after="/tmp/screenshot3.png",
                    execution_time_ms=280,
                ),
            },
            errors=[],
        )
        return test_state
    
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
        
        with patch("src.main.BrowserController", return_value=mock_browser_controller):
            with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                # Run main with requirements
                exit_code = await async_main([
                    "--requirements", "Test the login flow",
                    "--url", "https://example.com",
                    "--output", str(tmp_path),
                    "--format", "json",
                ])
        
        # Verify success
        assert exit_code == 0
        
        # Verify components were initialized
        mock_browser_controller.initialize.assert_called_once()
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
                    "--scenario", str(scenario_file),
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
        mock_plan = TestPlan(
            name="Test Login Flow",
            description="Testing login functionality",
            requirements="Test login",
            steps=[
                TestStep(
                    step_number=1,
                    description="Navigate to login page",
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
        mock_coordinator.generate_test_plan.return_value = mock_plan
        
        with patch("src.main.BrowserController", return_value=mock_browser_controller):
            with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                # Run main in plan-only mode
                exit_code = await async_main([
                    "--requirements", "Test login",
                    "--url", "https://example.com",
                    "--plan-only",
                ])
        
        # Verify success
        assert exit_code == 0
        
        # Verify only plan was generated, not executed
        mock_coordinator.generate_test_plan.assert_called_once()
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
        
        with patch("src.main.BrowserController", return_value=mock_browser_controller):
            with patch("src.main.WorkflowCoordinator", return_value=mock_coordinator):
                # Run main with short timeout
                exit_code = await async_main([
                    "--requirements", "Test login",
                    "--url", "https://example.com",
                    "--timeout", "1",
                ])
        
        # Verify timeout exit code
        assert exit_code == 2
    
    @pytest.mark.asyncio
    async def test_test_reporter_integration(self, sample_test_state, tmp_path):
        """Test TestReporter integration with TestState."""
        reporter = TestReporter()
        
        # Generate HTML report
        html_path = await reporter.generate_report(
            test_state=sample_test_state,
            output_dir=tmp_path,
            format="html",
        )
        
        assert html_path.exists()
        assert html_path.suffix == ".html"
        
        # Verify HTML content
        html_content = html_path.read_text()
        assert "HAINDY Test Report" in html_content
        assert "Test Execution Summary" in html_content
        assert "passed" in html_content.lower()
        
        # Generate JSON report
        json_path = await reporter.generate_report(
            test_state=sample_test_state,
            output_dir=tmp_path,
            format="json",
        )
        
        assert json_path.exists()
        assert json_path.suffix == ".json"
        
        # Verify JSON content
        json_data = json.loads(json_path.read_text())
        assert json_data["outcome"] == "passed"
        assert json_data["total_steps"] == 3
        assert json_data["passed_steps"] == 3
    
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
        assert "--scenario" in captured.out
        assert "Examples:" in captured.out