"""
Tests for the Test Runner Agent.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from datetime import datetime, timezone
from src.agents.test_runner import (
    ExecutionMode,
    TestRunnerAgent,
    ActionResult,
    TestStepResult,
)
from src.core.types import (
    ActionInstruction,
    ActionType,
    EvaluationResult,
    GridAction,
    GridCoordinate,
    TestPlan,
    TestState,
    TestStep,
    TestStatus,
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.openai_temperature = 0.7
    settings.openai_max_retries = 3
    return settings


@pytest.fixture
def mock_browser_driver():
    """Mock browser driver."""
    driver = AsyncMock()
    driver.navigate = AsyncMock()
    driver.screenshot = AsyncMock(return_value=b"mock_screenshot")
    driver.click = AsyncMock()
    driver.type = AsyncMock()
    driver.type_text = AsyncMock()
    driver.wait = AsyncMock()
    driver.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    return driver


@pytest.fixture
def mock_action_agent():
    """Mock action agent."""
    agent = AsyncMock()
    # Create a mock GridAction with proper structure
    mock_coordinate = GridCoordinate(
        cell="M23",
        offset_x=0.5,
        offset_y=0.5,
        confidence=0.95,
        refined=False
    )
    mock_instruction = ActionInstruction(
        action_type=ActionType.CLICK,
        description="Click button",
        target="button",
        expected_outcome="Button clicked"
    )
    mock_grid_action = GridAction(
        instruction=mock_instruction,
        coordinate=mock_coordinate
    )
    agent.determine_action = AsyncMock(return_value=mock_grid_action)
    
    # Mock the new execute_action method
    async def mock_execute_action(test_step, test_context, screenshot=None):
        return {
            "action_type": test_step.action_instruction.action_type.value,
            "validation_passed": True,
            "validation_reasoning": "Action is valid",
            "validation_confidence": 0.95,
            "grid_cell": "M23",
            "grid_coordinates": (960, 540),
            "offset_x": 0.5,
            "offset_y": 0.5,
            "coordinate_confidence": 0.95,
            "execution_success": True,
            "execution_time_ms": 523.4,
            "screenshot_after": b"mock_screenshot_after",
            "ai_analysis": {
                "success": True,
                "confidence": 0.9,
                "actual_outcome": "Action completed successfully"
            }
        }
    
    agent.execute_action = AsyncMock(side_effect=mock_execute_action)
    return agent


# Evaluator agent fixture removed - evaluation now handled by Action Agent's AI analysis


@pytest.fixture
def test_runner_agent(mock_browser_driver, mock_action_agent):
    """Create a TestRunnerAgent instance for testing."""
    agent = TestRunnerAgent(
        browser_driver=mock_browser_driver,
        action_agent=mock_action_agent
    )
    agent._client = AsyncMock()
    # Ensure the action agent's call_openai is properly mocked if needed
    if hasattr(mock_action_agent, 'call_openai'):
        mock_action_agent.call_openai = AsyncMock()
    return agent


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    plan_id = uuid4()
    step1_id = uuid4()
    step2_id = uuid4()
    step3_id = uuid4()
    step4_id = uuid4()
    
    return TestPlan(
        plan_id=plan_id,
        name="Login Test",
        description="Test user login functionality",
        requirements="Test that users can log in with valid credentials",
        steps=[
            TestStep(
                step_id=step1_id,
                step_number=1,
                description="Navigate to login page",
                action_instruction=ActionInstruction(
                    action_type=ActionType.NAVIGATE,
                    description="Navigate to the login page",
                    target="login page URL",
                    expected_outcome="Login page displayed"
                ),
                dependencies=[],
                optional=False
            ),
            TestStep(
                step_id=step2_id,
                step_number=2,
                description="Enter username",
                action_instruction=ActionInstruction(
                    action_type=ActionType.TYPE,
                    description="Enter username in the username field",
                    target="username field",
                    value="testuser",
                    expected_outcome="Username entered"
                ),
                dependencies=[step1_id],
                optional=False
            ),
            TestStep(
                step_id=step3_id,
                step_number=3,
                description="Enter password",
                action_instruction=ActionInstruction(
                    action_type=ActionType.TYPE,
                    description="Enter password in the password field",
                    target="password field",
                    value="testpass",
                    expected_outcome="Password entered"
                ),
                dependencies=[step1_id],
                optional=False
            ),
            TestStep(
                step_id=step4_id,
                step_number=4,
                description="Click login button",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click the login button",
                    target="login button",
                    expected_outcome="User logged in successfully"
                ),
                dependencies=[step2_id, step3_id],
                optional=False
            ),
        ],
        tags=["login", "authentication"]
    )


class TestTestRunnerAgent:
    """Test cases for TestRunnerAgent."""
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_success(
        self, test_runner_agent, sample_test_plan
    ):
        """Test successful test plan execution."""
        # Mock AI response for progress analysis
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "assessment": "Test progressing well",
                "concerns": [],
                "recommendations": ["Continue with remaining steps"]
            })
        })
        
        # Execute test plan
        result = await test_runner_agent.execute_test_plan(
            sample_test_plan,
            initial_url="https://example.com/login"
        )
        
        # Verify
        assert isinstance(result, TestState)
        # Debug: print execution history if test failed
        if result.status != TestStatus.COMPLETED:
            for i, step_result in enumerate(test_runner_agent._execution_history):
                print(f"Step {i+1}: success={step_result.success}, result={step_result.actual_result}")
        assert result.status == TestStatus.COMPLETED
        assert len(result.completed_steps) == 4
        assert len(result.failed_steps) == 0
        assert len(result.skipped_steps) == 0
        
        # Verify browser interactions
        test_runner_agent.browser_driver.navigate.assert_called_once_with(
            "https://example.com/login"
        )
        # Action Agent now handles screenshots internally, so browser driver screenshot calls are reduced
        assert test_runner_agent.browser_driver.screenshot.call_count >= 4  # Initial screenshots for each step
        assert test_runner_agent.action_agent.execute_action.call_count == 4
        # Evaluation is now handled by Action Agent's AI analysis
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_with_failure(
        self, test_runner_agent, sample_test_plan, mock_action_agent
    ):
        """Test test plan execution with a critical step failure."""
        # Make step 2 fail via Action Agent's AI analysis
        async def mock_execute_action_with_failure(test_step, test_context, screenshot=None):
            if test_step.step_number == 2:
                return {
                    "action_type": test_step.action_instruction.action_type.value,
                    "validation_passed": True,
                    "grid_cell": "M23",
                    "grid_coordinates": (960, 540),
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "coordinate_confidence": 0.95,
                    "execution_success": False,
                    "screenshot_after": b"mock_screenshot_after",
                    "ai_analysis": {
                        "success": False,  # Step 2 fails
                        "confidence": 0.90,
                        "actual_outcome": "Error: Username field not found",
                        "anomalies": ["Username field missing"]
                    }
                }
            else:
                return {
                    "action_type": test_step.action_instruction.action_type.value,
                    "validation_passed": True,
                    "grid_cell": "M23",
                    "grid_coordinates": (960, 540),
                    "offset_x": 0.5,
                    "offset_y": 0.5,
                    "coordinate_confidence": 0.95,
                    "execution_success": True,
                    "screenshot_after": b"mock_screenshot_after",
                    "ai_analysis": {
                        "success": True,
                        "confidence": 0.95,
                        "actual_outcome": "Action completed successfully"
                    }
                }
        
        mock_action_agent.execute_action.side_effect = mock_execute_action_with_failure
        
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "assessment": "Test failed early",
                "concerns": ["Critical step failed"],
                "recommendations": ["Investigate page load issue"]
            })
        })
        
        # Execute
        result = await test_runner_agent.execute_test_plan(sample_test_plan)
        
        # Debug output
        print(f"Test execution history:")
        for i, step_result in enumerate(test_runner_agent._execution_history):
            print(f"  Step {i+1}: success={step_result.success}, result={step_result.actual_result}")
        print(f"Completed steps: {len(result.completed_steps)}")
        print(f"Failed steps: {len(result.failed_steps)}")
        
        # Verify
        assert result.status == TestStatus.FAILED
        assert len(result.completed_steps) == 1  # Only first step completed
        assert len(result.failed_steps) == 1  # Second step failed
    
    @pytest.mark.asyncio
    async def test_execute_step_visual_mode(
        self, test_runner_agent, sample_test_plan
    ):
        """Test executing a step in visual mode."""
        step = sample_test_plan.steps[0]
        # Set current test plan
        test_runner_agent._current_test_plan = sample_test_plan
        
        # Execute
        result = await test_runner_agent._execute_step(step, ExecutionMode.VISUAL)
        
        # Verify
        assert isinstance(result, TestStepResult)
        assert result.success is True
        assert result.execution_mode == "visual"
        assert result.action_taken is not None
        assert result.action_taken.action_type == "navigate"
        
        # Verify agent calls
        test_runner_agent.action_agent.execute_action.assert_called_once()
        # Evaluation is now handled by Action Agent's AI analysis
    
    @pytest.mark.asyncio
    async def test_execute_step_scripted_mode(
        self, test_runner_agent, sample_test_plan
    ):
        """Test executing a step in scripted mode."""
        step = sample_test_plan.steps[0]
        # Set current test plan
        test_runner_agent._current_test_plan = sample_test_plan
        
        # Add a scripted action
        step_key = f"{sample_test_plan.plan_id}:{step.step_number}"
        test_runner_agent._scripted_actions[step_key] = {
            "action_type": "click",
            "x": 960,
            "y": 540,
            "grid_cell": "M23",
            "offset_x": 0.5,
            "offset_y": 0.5
        }
        
        # Execute
        result = await test_runner_agent._execute_step(step, ExecutionMode.SCRIPTED)
        
        # Verify
        assert result.success is True
        assert result.execution_mode == "scripted"
        assert result.action_taken.confidence == 1.0
        
        # Verify browser click was called
        test_runner_agent.browser_driver.click.assert_called_once()
        # Coordinates should be calculated from grid, not hardcoded
        call_args = test_runner_agent.browser_driver.click.call_args[0]
        assert len(call_args) == 2  # x, y coordinates
    
    @pytest.mark.asyncio
    async def test_execute_step_hybrid_mode_fallback(
        self, test_runner_agent, sample_test_plan
    ):
        """Test hybrid mode falling back to visual when scripted fails."""
        step = sample_test_plan.steps[0]
        # Set current test plan
        test_runner_agent._current_test_plan = sample_test_plan
        
        # Add a scripted action that will fail
        step_key = f"{sample_test_plan.plan_id}:{step.step_number}"
        test_runner_agent._scripted_actions[step_key] = {
            "action_type": "click",
            "grid_cell": "M23",
            "offset_x": 0.5,
            "offset_y": 0.5
        }
        
        # Make scripted execution fail on first call, succeed on second
        test_runner_agent.browser_driver.click.side_effect = [
            Exception("Element not found"),  # First call fails (scripted)
            None  # Second call succeeds (visual fallback)
        ]
        
        # Execute
        result = await test_runner_agent._execute_step(step, ExecutionMode.HYBRID)
        
        # Verify fallback to visual
        assert result.success is True
        assert result.action_taken is not None
        assert result.action_taken.action_type == "navigate"
        test_runner_agent.action_agent.execute_action.assert_called_once()
        # Browser click should be called once (scripted failed, visual succeeded through Action Agent)
        assert test_runner_agent.browser_driver.click.call_count == 1
    
    def test_check_dependencies_met(self, test_runner_agent, sample_test_plan):
        """Test dependency checking when dependencies are met."""
        # Initialize test state with completed steps
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[sample_test_plan.steps[0].step_id, sample_test_plan.steps[1].step_id],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        
        # Check step 4 which depends on steps 2 and 3
        step4 = sample_test_plan.steps[3]
        assert test_runner_agent._check_dependencies(step4) is False  # Step 3 not completed
        
        # Add step 3
        test_runner_agent._test_state.completed_steps.append(sample_test_plan.steps[2].step_id)
        
        # Now dependencies should be met
        assert test_runner_agent._check_dependencies(step4) is True
    
    def test_check_dependencies_failed(self, test_runner_agent, sample_test_plan):
        """Test dependency checking when a dependency failed."""
        # Initialize test state with failed step
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[],  # Step 1 not in completed steps
            failed_steps=[sample_test_plan.steps[0].step_id],  # Step 1 failed
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=1,
            warning_count=0
        )
        
        # Check step 2 which depends on step 1
        step2 = sample_test_plan.steps[1]
        assert test_runner_agent._check_dependencies(step2) is False
    
    def test_record_action(self, test_runner_agent, sample_test_plan):
        """Test recording successful actions."""
        test_runner_agent._current_test_plan = sample_test_plan
        step = sample_test_plan.steps[0]
        
        action = ActionResult(
            action_type="click",
            grid_cell="B7",
            offset_x=0.3,
            offset_y=0.7,
            confidence=0.95,
            requires_refinement=False
        )
        
        # Record action
        test_runner_agent._record_action(step, action)
        
        # Verify
        step_key = f"{sample_test_plan.plan_id}:{step.step_number}"
        assert step_key in test_runner_agent._scripted_actions
        recorded = test_runner_agent._scripted_actions[step_key]
        assert recorded["action_type"] == "click"
        assert recorded["grid_cell"] == "B7"
        assert recorded["offset_x"] == 0.3
        assert recorded["offset_y"] == 0.7
        # No pixel coordinates should be stored anymore
    
    @pytest.mark.asyncio
    async def test_determine_execution_mode(
        self, test_runner_agent, sample_test_plan
    ):
        """Test execution mode determination."""
        test_runner_agent._current_test_plan = sample_test_plan
        step = sample_test_plan.steps[0]
        
        # No scripted action - should use visual
        mode = await test_runner_agent._determine_execution_mode(step)
        assert mode == ExecutionMode.VISUAL
        
        # Add scripted action
        step_key = f"{sample_test_plan.plan_id}:{step.step_number}"
        test_runner_agent._scripted_actions[step_key] = {"action_type": "click"}
        
        # Should now use hybrid
        mode = await test_runner_agent._determine_execution_mode(step)
        assert mode == ExecutionMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_get_next_action(self, test_runner_agent, sample_test_plan):
        """Test getting next action recommendation."""
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": "Proceed with the current step as planned"
        })
        
        test_state = TestState(
            test_plan=sample_test_plan,
            current_step=sample_test_plan.steps[0],
            completed_steps=[],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        
        # Get next action
        recommendation = await test_runner_agent.get_next_action(
            sample_test_plan, test_state
        )
        
        assert recommendation == "Proceed with the current step as planned"
        test_runner_agent.call_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_progress(self, test_runner_agent, sample_test_plan):
        """Test AI-based progress analysis."""
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=sample_test_plan.steps[2],
            completed_steps=[sample_test_plan.steps[0].step_id, sample_test_plan.steps[1].step_id],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        
        test_runner_agent._execution_history = [
            TestStepResult(
                step=sample_test_plan.steps[0],
                success=True,
                action_taken=None,
                actual_result="Success",
                execution_mode="visual"
            ),
            TestStepResult(
                step=sample_test_plan.steps[1],
                success=True,
                action_taken=None,
                actual_result="Success",  
                execution_mode="visual"
            ),
        ]
        
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "assessment": "Test progressing smoothly",
                "concerns": [],
                "recommendations": ["Continue with login flow"]
            })
        })
        
        # Analyze progress
        await test_runner_agent._analyze_progress()
        
        # Verify AI was called and analysis stored
        test_runner_agent.call_openai.assert_called_once()
        assert "latest_analysis" in test_runner_agent._test_state.context
        analysis = test_runner_agent._test_state.context["latest_analysis"]
        assert analysis["assessment"] == "Test progressing smoothly"
    
    @pytest.mark.asyncio
    async def test_mark_step_skipped(self, test_runner_agent, sample_test_plan):
        """Test marking a step as skipped."""
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=sample_test_plan.steps[3],
            completed_steps=[sample_test_plan.steps[0].step_id, sample_test_plan.steps[1].step_id, sample_test_plan.steps[2].step_id],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        test_runner_agent._execution_history = []
        test_runner_agent._step_index = 3  # Current step index
        
        # Mark step as skipped
        step = sample_test_plan.steps[3]
        test_runner_agent._mark_step_skipped(step)
        
        # Verify
        assert len(test_runner_agent._execution_history) == 1
        result = test_runner_agent._execution_history[0]
        assert result.success is False
        assert "Skipped due to unmet dependencies" in result.actual_result
        assert result.execution_mode == "skipped"
        assert step.step_id in test_runner_agent._test_state.skipped_steps
    
    @pytest.mark.asyncio
    async def test_judge_final_test_result_success(self, test_runner_agent, sample_test_plan):
        """Test final judgment on successful test execution."""
        # Set up completed test state
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[s.step_id for s in sample_test_plan.steps],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        
        # Add successful execution history
        test_runner_agent._execution_history = [
            TestStepResult(
                step=step,
                success=True,
                action_taken=ActionResult(
                    action_type=step.action_instruction.action_type.value,
                    grid_cell="M23",
                    confidence=0.95
                ),
                actual_result="Step completed successfully",
                execution_mode="visual",
                action_result_details={
                    "validation_passed": True,
                    "execution_success": True,
                    "ai_analysis": {"success": True}
                }
            )
            for step in sample_test_plan.steps
        ]
        
        # Mock AI judgment
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "overall_success": True,
                "confidence": 0.95,
                "reasoning": "All test steps completed successfully",
                "key_issues": [],
                "recommendations": []
            })
        })
        
        # Get final judgment
        judgment = await test_runner_agent.judge_final_test_result()
        
        # Verify
        assert judgment["overall_success"] is True
        assert judgment["confidence"] == 0.95
        assert judgment["execution_stats"]["success_rate"] == 1.0
        assert len(judgment["key_issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_judge_final_test_result_with_failures(self, test_runner_agent, sample_test_plan):
        """Test final judgment on test with failures."""
        # Set up test state with failures
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_plan=sample_test_plan,
            current_step=None,
            completed_steps=[sample_test_plan.steps[0].step_id],
            failed_steps=[sample_test_plan.steps[1].step_id],
            skipped_steps=[sample_test_plan.steps[2].step_id, sample_test_plan.steps[3].step_id],
            status=TestStatus.FAILED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            error_count=1,
            warning_count=0
        )
        
        # Add mixed execution history
        test_runner_agent._execution_history = [
            TestStepResult(
                step=sample_test_plan.steps[0],
                success=True,
                action_taken=None,
                actual_result="Success",
                execution_mode="visual"
            ),
            TestStepResult(
                step=sample_test_plan.steps[1],
                success=False,
                action_taken=None,
                actual_result="Failed to find element",
                execution_mode="visual",
                action_result_details={
                    "validation_passed": False,
                    "validation_reasoning": "Element not visible",
                    "execution_error": "Element not found"
                }
            )
        ]
        
        # Mock AI judgment
        test_runner_agent.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "overall_success": False,
                "confidence": 0.85,
                "reasoning": "Critical step failed, preventing test completion",
                "key_issues": ["Username field not found", "Page may not have loaded"],
                "recommendations": ["Verify page load", "Check element selectors"]
            })
        })
        
        # Get final judgment
        judgment = await test_runner_agent.judge_final_test_result()
        
        # Verify
        assert judgment["overall_success"] is False
        assert judgment["confidence"] == 0.85
        assert judgment["execution_stats"]["success_rate"] == 0.25  # 1/4 steps
        assert len(judgment["key_issues"]) == 2
        assert len(judgment["recommendations"]) == 2