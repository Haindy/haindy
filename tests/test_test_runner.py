"""
Tests for the Test Runner Agent.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.agents.test_runner import (
    ExecutionMode,
    TestRunnerAgent,
    ActionResult,
    TestPlan,
    TestState,
    TestStep,
    TestStepResult,
)
from src.core.types import EvaluationResult


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
    driver.wait_for_load = AsyncMock()
    driver.wait_for_idle = AsyncMock()
    driver.take_screenshot = AsyncMock(return_value=b"mock_screenshot")
    driver.click = AsyncMock()
    driver.type_text = AsyncMock()
    return driver


@pytest.fixture
def mock_action_agent():
    """Mock action agent."""
    agent = AsyncMock()
    agent.determine_action = AsyncMock(return_value=ActionResult(
        action_type="click",
        grid_cell="M23",
        offset_x=0.5,
        offset_y=0.5,
        confidence=0.95,
        requires_refinement=False
    ))
    return agent


@pytest.fixture
def mock_evaluator_agent():
    """Mock evaluator agent."""
    agent = AsyncMock()
    agent.evaluate_result = AsyncMock(return_value=EvaluationResult(
        step_id=uuid4(),
        success=True,
        confidence=0.95,
        expected_outcome="Expected outcome",
        actual_outcome="Actual outcome matches expected",
        deviations=[],
        suggestions=[]
    ))
    return agent


@pytest.fixture
def test_runner_agent(mock_browser_driver, mock_action_agent, mock_evaluator_agent):
    """Create a TestRunnerAgent instance for testing."""
    agent = TestRunnerAgent(
        browser_driver=mock_browser_driver,
        action_agent=mock_action_agent,
        evaluator_agent=mock_evaluator_agent
    )
    agent._client = AsyncMock()
    return agent


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    test_id = uuid4()
    return TestPlan(
        test_id=test_id,
        name="Login Test",
        description="Test user login functionality",
        prerequisites=["User account exists"],
        steps=[
            TestStep(
                id=uuid4(),
                step_number=1,
                action="Navigate to login page",
                expected_result="Login page displayed",
                depends_on=[],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=2,
                action="Enter username",
                expected_result="Username entered",
                depends_on=[1],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=3,
                action="Enter password",
                expected_result="Password entered",
                depends_on=[1],
                is_critical=True
            ),
            TestStep(
                id=uuid4(),
                step_number=4,
                action="Click login button",
                expected_result="User logged in successfully",
                depends_on=[2, 3],
                is_critical=True
            ),
        ],
        success_criteria=["User can log in"],
        edge_cases=[]
    )


class TestTestRunnerAgent:
    """Test cases for TestRunnerAgent."""
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_success(
        self, test_runner_agent, sample_test_plan
    ):
        """Test successful test plan execution."""
        # Mock AI response for progress analysis
        test_runner_agent.call_ai = AsyncMock(return_value={
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
        assert result.test_status == "completed"
        assert len(result.completed_steps) == 4
        assert len(result.remaining_steps) == 0
        
        # Verify browser interactions
        test_runner_agent.browser_driver.navigate.assert_called_once_with(
            "https://example.com/login"
        )
        assert test_runner_agent.browser_driver.take_screenshot.call_count >= 8  # Before/after each step
        assert test_runner_agent.action_agent.determine_action.call_count == 4
        assert test_runner_agent.evaluator_agent.evaluate_result.call_count == 4
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_with_failure(
        self, test_runner_agent, sample_test_plan, mock_evaluator_agent
    ):
        """Test test plan execution with a critical step failure."""
        # Make step 2 fail
        mock_evaluator_agent.evaluate_result.side_effect = [
            EvaluationResult(
                step_id=uuid4(),
                success=True,
                confidence=0.95,
                expected_outcome="Expected",
                actual_outcome="Actual",
                deviations=[],
                suggestions=[]
            ),
            EvaluationResult(
                step_id=uuid4(),
                success=False,  # Step 2 fails
                confidence=0.90,
                expected_outcome="Username entered",
                actual_outcome="Error: Username field not found",
                deviations=["Username field missing"],
                suggestions=["Check if page loaded correctly"]
            ),
            # Remaining steps won't be executed
        ]
        
        test_runner_agent.call_ai = AsyncMock(return_value={
            "content": json.dumps({
                "assessment": "Test failed early",
                "concerns": ["Critical step failed"],
                "recommendations": ["Investigate page load issue"]
            })
        })
        
        # Execute
        result = await test_runner_agent.execute_test_plan(sample_test_plan)
        
        # Verify
        assert result.test_status == "failed"
        assert len(result.completed_steps) == 2  # Only first 2 steps attempted
        assert len(result.remaining_steps) == 2  # Steps 3 and 4 not attempted
    
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
        assert result.action_taken.action_type == "click"
        
        # Verify agent calls
        test_runner_agent.action_agent.determine_action.assert_called_once()
        test_runner_agent.evaluator_agent.evaluate_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_step_scripted_mode(
        self, test_runner_agent, sample_test_plan
    ):
        """Test executing a step in scripted mode."""
        step = sample_test_plan.steps[0]
        # Set current test plan
        test_runner_agent._current_test_plan = sample_test_plan
        
        # Add a scripted action
        step_key = f"{sample_test_plan.test_id}:{step.step_number}"
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
        test_runner_agent.browser_driver.click.assert_called_once_with(960, 540)
    
    @pytest.mark.asyncio
    async def test_execute_step_hybrid_mode_fallback(
        self, test_runner_agent, sample_test_plan
    ):
        """Test hybrid mode falling back to visual when scripted fails."""
        step = sample_test_plan.steps[0]
        # Set current test plan
        test_runner_agent._current_test_plan = sample_test_plan
        
        # Add a scripted action that will fail
        step_key = f"{sample_test_plan.test_id}:{step.step_number}"
        test_runner_agent._scripted_actions[step_key] = {
            "action_type": "click",
            "x": 960,
            "y": 540
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
        assert result.action_taken.action_type == "click"
        test_runner_agent.action_agent.determine_action.assert_called_once()
        # Browser click should be called twice (once failed, once succeeded)
        assert test_runner_agent.browser_driver.click.call_count == 2
    
    def test_check_dependencies_met(self, test_runner_agent, sample_test_plan):
        """Test dependency checking when dependencies are met."""
        # Add some completed steps to history
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
        
        # Check step 4 which depends on steps 2 and 3
        step4 = sample_test_plan.steps[3]
        assert test_runner_agent._check_dependencies(step4) is False  # Step 3 not completed
        
        # Add step 3
        test_runner_agent._execution_history.append(
            TestStepResult(
                step=sample_test_plan.steps[2],
                success=True,
                action_taken=None,
                actual_result="Success",
                execution_mode="visual"
            )
        )
        
        # Now dependencies should be met
        assert test_runner_agent._check_dependencies(step4) is True
    
    def test_check_dependencies_failed(self, test_runner_agent, sample_test_plan):
        """Test dependency checking when a dependency failed."""
        # Add failed step to history
        test_runner_agent._execution_history = [
            TestStepResult(
                step=sample_test_plan.steps[0],
                success=False,  # Failed
                action_taken=None,
                actual_result="Failed",
                execution_mode="visual"
            )
        ]
        
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
        step_key = f"{sample_test_plan.test_id}:{step.step_number}"
        assert step_key in test_runner_agent._scripted_actions
        recorded = test_runner_agent._scripted_actions[step_key]
        assert recorded["action_type"] == "click"
        assert recorded["grid_cell"] == "B7"
        assert recorded["x"] == 0.3 * 1920
        assert recorded["y"] == 0.7 * 1080
    
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
        step_key = f"{sample_test_plan.test_id}:{step.step_number}"
        test_runner_agent._scripted_actions[step_key] = {"action_type": "click"}
        
        # Should now use hybrid
        mode = await test_runner_agent._determine_execution_mode(step)
        assert mode == ExecutionMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_get_next_action(self, test_runner_agent, sample_test_plan):
        """Test getting next action recommendation."""
        test_runner_agent.call_ai = AsyncMock(return_value={
            "content": "Proceed with the current step as planned"
        })
        
        test_state = TestState(
            test_id=sample_test_plan.test_id,
            current_step=0,
            completed_steps=[],
            remaining_steps=[0, 1, 2, 3],
            test_status="in_progress"
        )
        
        # Get next action
        recommendation = await test_runner_agent.get_next_action(
            sample_test_plan, test_state
        )
        
        assert recommendation == "Proceed with the current step as planned"
        test_runner_agent.call_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_progress(self, test_runner_agent, sample_test_plan):
        """Test AI-based progress analysis."""
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_id=sample_test_plan.test_id,
            current_step=2,
            completed_steps=[0, 1],
            remaining_steps=[2, 3],
            test_status="in_progress"
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
        
        test_runner_agent.call_ai = AsyncMock(return_value={
            "content": json.dumps({
                "assessment": "Test progressing smoothly",
                "concerns": [],
                "recommendations": ["Continue with login flow"]
            })
        })
        
        # Analyze progress
        await test_runner_agent._analyze_progress()
        
        # Verify AI was called and analysis stored
        test_runner_agent.call_ai.assert_called_once()
        assert "latest_analysis" in test_runner_agent._test_state.context
        analysis = test_runner_agent._test_state.context["latest_analysis"]
        assert analysis["assessment"] == "Test progressing smoothly"
    
    @pytest.mark.asyncio
    async def test_mark_step_skipped(self, test_runner_agent, sample_test_plan):
        """Test marking a step as skipped."""
        test_runner_agent._current_test_plan = sample_test_plan
        test_runner_agent._test_state = TestState(
            test_id=sample_test_plan.test_id,
            current_step=3,
            completed_steps=[0, 1, 2],
            remaining_steps=[3],
            test_status="in_progress"
        )
        test_runner_agent._execution_history = []
        
        # Mark step as skipped
        step = sample_test_plan.steps[3]
        test_runner_agent._mark_step_skipped(step)
        
        # Verify
        assert len(test_runner_agent._execution_history) == 1
        result = test_runner_agent._execution_history[0]
        assert result.success is False
        assert "Skipped due to unmet dependencies" in result.actual_result
        assert result.execution_mode == "skipped"
        assert 3 not in test_runner_agent._test_state.remaining_steps
        assert 3 in test_runner_agent._test_state.completed_steps