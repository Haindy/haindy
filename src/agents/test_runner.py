"""
Test Runner Agent implementation.

Orchestrates test execution by coordinating other agents and managing test state.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from src.agents.action_agent import ActionAgent
from src.agents.base_agent import BaseAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.test_planner import TestPlannerAgent
from src.browser.driver import BrowserDriver
from src.config.agent_prompts import TEST_RUNNER_SYSTEM_PROMPT
from src.core.types import (
    EvaluationResult,
    GridAction,
    GridCoordinate,
    ActionInstruction,
)
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


# Custom types for Test Runner
class TestStep(BaseModel):
    """Simplified test step for test runner."""
    id: UUID = Field(default_factory=uuid4)
    step_number: int
    action: str
    expected_result: str
    depends_on: List[int] = Field(default_factory=list)
    is_critical: bool = True


class TestPlan(BaseModel):
    """Simplified test plan for test runner."""
    test_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    prerequisites: List[str] = Field(default_factory=list)
    steps: List[TestStep]
    success_criteria: List[str] = Field(default_factory=list)
    edge_cases: List[str] = Field(default_factory=list)


class TestState(BaseModel):
    """Test execution state."""
    test_id: UUID
    current_step: int
    completed_steps: List[int]
    remaining_steps: List[int]
    test_status: str = "pending"
    context: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Simplified action result for test runner."""
    action_type: str
    grid_cell: str
    offset_x: float = 0.5
    offset_y: float = 0.5
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    requires_refinement: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestStepResult(BaseModel):
    """Result of executing a single test step."""
    step: TestStep
    success: bool
    action_taken: Optional[ActionResult] = None
    actual_result: str
    screenshot_before: Optional[bytes] = None
    screenshot_after: Optional[bytes] = None
    evaluation: Optional[EvaluationResult] = None
    execution_mode: str = "visual"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionMode(Enum):
    """Execution mode for test steps."""
    
    VISUAL = "visual"  # Use AI visual interaction
    SCRIPTED = "scripted"  # Use recorded script
    HYBRID = "hybrid"  # Try scripted, fallback to visual


class TestRunnerAgent(BaseAgent):
    """
    AI agent that orchestrates test execution.
    
    This agent coordinates between Test Planner, Action, and Evaluator agents
    to execute test plans step by step. It maintains test state, handles
    branching logic, and decides on execution strategies.
    """
    
    def __init__(
        self,
        name: str = "TestRunnerAgent",
        browser_driver: Optional[BrowserDriver] = None,
        action_agent: Optional[ActionAgent] = None,
        evaluator_agent: Optional[EvaluatorAgent] = None,
        **kwargs
    ):
        """
        Initialize the Test Runner Agent.
        
        Args:
            name: Agent name
            browser_driver: Browser driver instance
            action_agent: Action agent instance for visual interactions
            evaluator_agent: Evaluator agent instance for result validation
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_RUNNER_SYSTEM_PROMPT
        self.browser_driver = browser_driver
        self.action_agent = action_agent
        self.evaluator_agent = evaluator_agent
        
        # Test execution state
        self._current_test_plan: Optional[TestPlan] = None
        self._test_state: Optional[TestState] = None
        self._execution_history: List[TestStepResult] = []
        self._scripted_actions: Dict[str, Dict[str, Any]] = {}
    
    async def execute_test_plan(
        self, test_plan: TestPlan, initial_url: Optional[str] = None
    ) -> TestState:
        """
        Execute a complete test plan.
        
        Args:
            test_plan: The test plan to execute
            initial_url: Optional starting URL for the test
            
        Returns:
            Final test state after execution
        """
        logger.info("Starting test plan execution", extra={
            "test_plan_id": str(test_plan.test_id),
            "test_plan_name": test_plan.name,
            "total_steps": len(test_plan.steps)
        })
        
        # Initialize test state
        self._current_test_plan = test_plan
        self._test_state = TestState(
            test_id=test_plan.test_id,
            current_step=0,
            completed_steps=[],
            remaining_steps=list(range(len(test_plan.steps))),
            test_status="in_progress",
            context={"test_plan_name": test_plan.name}
        )
        self._execution_history = []
        
        # Navigate to initial URL if provided
        if initial_url and self.browser_driver:
            logger.info("Navigating to initial URL", extra={"url": initial_url})
            await self.browser_driver.navigate(initial_url)
            await self.browser_driver.wait_for_load()
        
        # Execute steps
        try:
            while self._test_state.test_status == "in_progress":
                await self._execute_next_step()
        except Exception as e:
            logger.error("Test execution failed", extra={
                "error": str(e),
                "current_step": self._test_state.current_step
            })
            self._test_state.test_status = "failed"
            self._test_state.context["error"] = str(e)
        
        logger.info("Test execution completed", extra={
            "test_status": self._test_state.test_status,
            "completed_steps": len(self._test_state.completed_steps),
            "total_steps": len(test_plan.steps)
        })
        
        return self._test_state
    
    async def _execute_next_step(self) -> None:
        """Execute the next step in the test plan."""
        if not self._test_state.remaining_steps:
            # All steps completed
            self._test_state.test_status = "completed"
            return
        
        # Get next step
        step_index = self._test_state.remaining_steps[0]
        current_step = self._current_test_plan.steps[step_index]
        
        logger.info("Executing test step", extra={
            "step_number": current_step.step_number,
            "action": current_step.action,
            "is_critical": current_step.is_critical
        })
        
        # Check dependencies
        if not self._check_dependencies(current_step):
            logger.warning("Step dependencies not met, skipping", extra={
                "step_number": current_step.step_number,
                "depends_on": current_step.depends_on
            })
            self._mark_step_skipped(current_step)
            return
        
        # Determine execution mode
        execution_mode = await self._determine_execution_mode(current_step)
        
        # Execute the step
        step_result = await self._execute_step(current_step, execution_mode)
        
        # Update state based on result
        await self._update_state(current_step, step_result)
    
    def _check_dependencies(self, step: TestStep) -> bool:
        """Check if all step dependencies are satisfied."""
        if not step.depends_on:
            return True
        
        for dep_step_num in step.depends_on:
            # Check if dependency was completed successfully
            dep_result = next(
                (r for r in self._execution_history if r.step.step_number == dep_step_num),
                None
            )
            if not dep_result or not dep_result.success:
                return False
        
        return True
    
    async def _determine_execution_mode(self, step: TestStep) -> ExecutionMode:
        """Determine the best execution mode for a step."""
        # Check if we have a scripted action for this step
        step_key = f"{self._current_test_plan.test_id}:{step.step_number}"
        
        if step_key in self._scripted_actions:
            # We have a recorded action, try scripted mode
            return ExecutionMode.HYBRID
        
        # Default to visual mode for new steps
        return ExecutionMode.VISUAL
    
    async def _execute_step(
        self, step: TestStep, mode: ExecutionMode
    ) -> TestStepResult:
        """Execute a single test step."""
        screenshot_before = None
        screenshot_after = None
        action_result = None
        evaluation_result = None
        
        try:
            # Capture pre-action screenshot
            if self.browser_driver:
                screenshot_before = await self.browser_driver.take_screenshot()
            
            # Execute action based on mode
            if mode == ExecutionMode.VISUAL:
                action_result = await self._execute_visual_action(step)
            elif mode == ExecutionMode.SCRIPTED:
                action_result = await self._execute_scripted_action(step)
            else:  # HYBRID
                action_result = await self._execute_hybrid_action(step)
            
            # Wait for action to complete
            if self.browser_driver:
                await self.browser_driver.wait_for_idle()
                screenshot_after = await self.browser_driver.take_screenshot()
            
            # Evaluate result
            if self.evaluator_agent and screenshot_after:
                evaluation_result = await self.evaluator_agent.evaluate_result(
                    screenshot_after,
                    step.expected_result,
                    step_id=step.id
                )
            
            # Create step result
            success = evaluation_result.success if evaluation_result else True
            
            return TestStepResult(
                step=step,
                success=success,
                action_taken=action_result,
                actual_result=evaluation_result.actual_outcome if evaluation_result else "Action completed",
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                evaluation=evaluation_result,
                execution_mode=mode.value
            )
            
        except Exception as e:
            logger.error("Step execution failed", extra={
                "step_number": step.step_number,
                "error": str(e)
            })
            
            return TestStepResult(
                step=step,
                success=False,
                action_taken=None,
                actual_result=f"Error: {str(e)}",
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                evaluation=None,
                execution_mode=mode.value
            )
    
    async def _execute_visual_action(self, step: TestStep) -> Optional[ActionResult]:
        """Execute action using visual AI interaction."""
        if not self.action_agent or not self.browser_driver:
            logger.warning("Visual execution not available - missing dependencies")
            return None
        
        # Get current screenshot
        screenshot = await self.browser_driver.take_screenshot()
        
        # Get action coordinates from Action Agent
        action_result = await self.action_agent.determine_action(
            screenshot, step.action
        )
        
        if action_result and action_result.confidence >= 0.8:
            # Execute the action
            await self._perform_browser_action(action_result)
            
            # Record successful action for future use
            self._record_action(step, action_result)
        
        return action_result
    
    async def _execute_scripted_action(self, step: TestStep) -> Optional[ActionResult]:
        """Execute action using recorded script."""
        if not self._current_test_plan:
            return None
        step_key = f"{self._current_test_plan.test_id}:{step.step_number}"
        scripted_action = self._scripted_actions.get(step_key)
        
        if not scripted_action or not self.browser_driver:
            return None
        
        try:
            # Execute the scripted action
            action_type = scripted_action["action_type"]
            
            if action_type == "click":
                await self.browser_driver.click(
                    scripted_action["x"],
                    scripted_action["y"]
                )
            elif action_type == "type":
                await self.browser_driver.type_text(
                    scripted_action["text"]
                )
            
            # Return a synthetic action result
            return ActionResult(
                action_type=action_type,
                grid_cell=scripted_action.get("grid_cell", "N/A"),
                offset_x=scripted_action.get("offset_x", 0),
                offset_y=scripted_action.get("offset_y", 0),
                confidence=1.0,
                requires_refinement=False
            )
            
        except Exception as e:
            logger.warning("Scripted action failed", extra={
                "step": step_key,
                "error": str(e)
            })
            return None
    
    async def _execute_hybrid_action(self, step: TestStep) -> Optional[ActionResult]:
        """Try scripted execution, fall back to visual if needed."""
        # Try scripted first
        result = await self._execute_scripted_action(step)
        
        if result:
            return result
        
        # Fall back to visual
        logger.info("Falling back to visual execution", extra={
            "step_number": step.step_number
        })
        return await self._execute_visual_action(step)
    
    async def _perform_browser_action(self, action: ActionResult) -> None:
        """Perform the actual browser action."""
        if not self.browser_driver:
            return
        
        # Calculate absolute coordinates
        # This is simplified - real implementation would use grid system
        x = action.offset_x * 1920  # Assuming 1920x1080 for now
        y = action.offset_y * 1080
        
        if action.action_type == "click":
            await self.browser_driver.click(x, y)
        elif action.action_type == "type":
            # For typing, we'd need additional context
            pass
    
    def _record_action(self, step: TestStep, action: ActionResult) -> None:
        """Record successful action for future scripted execution."""
        if not self._current_test_plan:
            return
        step_key = f"{self._current_test_plan.test_id}:{step.step_number}"
        
        self._scripted_actions[step_key] = {
            "action_type": action.action_type,
            "grid_cell": action.grid_cell,
            "offset_x": action.offset_x,
            "offset_y": action.offset_y,
            "x": action.offset_x * 1920,  # Simplified
            "y": action.offset_y * 1080,
            "timestamp": action.timestamp.isoformat()
        }
    
    async def _update_state(self, step: TestStep, result: TestStepResult) -> None:
        """Update test state after step execution."""
        # Add to history
        self._execution_history.append(result)
        
        # Update state
        self._test_state.remaining_steps.remove(step.step_number - 1)
        self._test_state.completed_steps.append(step.step_number - 1)
        
        if result.success:
            # Move to next step
            if self._test_state.remaining_steps:
                self._test_state.current_step = self._test_state.remaining_steps[0]
        else:
            # Handle failure
            if step.is_critical:
                logger.error("Critical step failed, ending test", extra={
                    "step_number": step.step_number
                })
                self._test_state.test_status = "failed"
            else:
                logger.warning("Non-critical step failed, continuing", extra={
                    "step_number": step.step_number
                })
        
        # Check for completion
        if not self._test_state.remaining_steps and self._test_state.test_status == "in_progress":
            self._test_state.test_status = "completed"
        
        # Use AI to analyze overall progress and determine next action
        await self._analyze_progress()
    
    def _mark_step_skipped(self, step: TestStep) -> None:
        """Mark a step as skipped due to unmet dependencies."""
        result = TestStepResult(
            step=step,
            success=False,
            action_taken=None,
            actual_result="Skipped due to unmet dependencies",
            screenshot_before=None,
            screenshot_after=None,
            evaluation=None,
            execution_mode="skipped"
        )
        
        self._execution_history.append(result)
        self._test_state.remaining_steps.remove(step.step_number - 1)
        self._test_state.completed_steps.append(step.step_number - 1)
    
    async def _analyze_progress(self) -> None:
        """Use AI to analyze test progress and make decisions."""
        # Build context for AI
        context = {
            "test_plan": self._current_test_plan.name,
            "completed_steps": len(self._test_state.completed_steps),
            "total_steps": len(self._current_test_plan.steps),
            "recent_results": [
                {
                    "step": r.step.action,
                    "success": r.success,
                    "result": r.actual_result
                }
                for r in self._execution_history[-3:]  # Last 3 results
            ]
        }
        
        prompt = f"""Analyze the current test execution progress:

Test Plan: {context['test_plan']}
Progress: {context['completed_steps']}/{context['total_steps']} steps completed

Recent Results:
{json.dumps(context['recent_results'], indent=2)}

Based on this information, provide:
1. Overall assessment of test progress
2. Any concerns or patterns noticed
3. Recommendations for next steps

Respond in JSON format with keys: assessment, concerns, recommendations"""
        
        try:
            response = await self.call_ai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.get("content", "{}"))
            logger.info("Test progress analysis", extra=analysis)
            
            # Store analysis in context
            self._test_state.context["latest_analysis"] = analysis
            
        except Exception as e:
            logger.warning("Failed to analyze progress", extra={"error": str(e)})
    
    async def get_next_action(
        self, test_plan: TestPlan, current_state: TestState
    ) -> Optional[str]:
        """
        Determine the next action based on current state.
        
        This method can be used for more dynamic test execution where
        the Test Runner makes decisions about what to do next.
        """
        # Find current step
        if current_state.remaining_steps:
            next_step_index = current_state.remaining_steps[0]
            next_step = test_plan.steps[next_step_index]
            
            # Use AI to determine if we should proceed or adapt
            prompt = f"""Current test state:
- Test: {test_plan.name}
- Current step: {next_step.action}
- Expected result: {next_step.expected_result}
- Recent failures: {sum(1 for r in self._execution_history[-3:] if not r.success)}

Should we:
1. Proceed with the current step as planned
2. Skip this step
3. Retry a previous step
4. Abort the test

Provide your recommendation with reasoning."""
            
            response = await self.call_ai(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.get("content", "Proceed with current step")
        
        return None