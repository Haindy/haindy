"""
Test Runner Agent implementation.

Orchestrates test execution by coordinating other agents and managing test state.
"""

import asyncio
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
    TestPlan,
    TestStep,
    TestState,
    TestStatus,
)
from src.monitoring.logger import get_logger

logger = get_logger(__name__)




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
    # Enhanced debugging information from Action Agent
    action_result_details: Optional[Dict[str, Any]] = None


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
            "test_plan_id": str(test_plan.plan_id),
            "test_plan_name": test_plan.name,
            "total_steps": len(test_plan.steps)
        })
        
        # Initialize test state
        self._current_test_plan = test_plan
        self._test_state = TestState(
            test_plan=test_plan,
            current_step=None,
            completed_steps=[],
            failed_steps=[],
            skipped_steps=[],
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc),
            error_count=0,
            warning_count=0
        )
        self._execution_history = []
        self._step_index = 0  # Track current step index
        
        # Navigate to initial URL if provided
        if initial_url and self.browser_driver:
            logger.info("Navigating to initial URL", extra={"url": initial_url})
            await self.browser_driver.navigate(initial_url)
        
        # Execute steps
        try:
            while self._test_state.status == TestStatus.IN_PROGRESS:
                await self._execute_next_step()
        except Exception as e:
            logger.error("Test execution failed", extra={
                "error": str(e),
                "current_step": self._test_state.current_step
            })
            self._test_state.status = TestStatus.FAILED
            self._test_state.error_count += 1
        
        # Set end time
        self._test_state.end_time = datetime.now(timezone.utc)
        
        logger.info("Test execution completed", extra={
            "test_status": self._test_state.status.value,
            "completed_steps": len(self._test_state.completed_steps),
            "total_steps": len(test_plan.steps)
        })
        
        return self._test_state
    
    async def _execute_next_step(self) -> None:
        """Execute the next step in the test plan."""
        # Check if all steps are completed
        if self._step_index >= len(self._test_state.test_plan.steps):
            # All steps completed
            self._test_state.status = TestStatus.COMPLETED
            return
        
        # Get next step
        current_step = self._test_state.test_plan.steps[self._step_index]
        self._test_state.current_step = current_step
        
        logger.info("Executing test step", extra={
            "step_number": current_step.step_number,
            "description": current_step.description,
            "action_type": current_step.action_instruction.action_type.value,
            "optional": current_step.optional
        })
        
        # Check dependencies
        if not self._check_dependencies(current_step):
            logger.warning("Step dependencies not met, skipping", extra={
                "step_number": current_step.step_number,
                "dependencies": [str(d) for d in current_step.dependencies]
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
        if not step.dependencies:
            return True
        
        # Check if all dependencies are in completed_steps
        for dep_step_id in step.dependencies:
            if dep_step_id not in self._test_state.completed_steps:
                return False
        
        return True
    
    async def _determine_execution_mode(self, step: TestStep) -> ExecutionMode:
        """Determine the best execution mode for a step."""
        # Check if we have a scripted action for this step
        step_key = f"{self._current_test_plan.plan_id}:{step.step_number}"
        
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
                screenshot_before = await self.browser_driver.screenshot()
            
            # Initialize step result details
            self._current_step_result_details = None
            
            # Execute action based on mode
            if mode == ExecutionMode.VISUAL:
                action_result = await self._execute_visual_action(step)
            elif mode == ExecutionMode.SCRIPTED:
                action_result = await self._execute_scripted_action(step)
            else:  # HYBRID
                action_result = await self._execute_hybrid_action(step)
            
            # Action Agent now handles waiting and screenshot capture
            # Get screenshot after from Action Agent results if available
            if self._current_step_result_details and self._current_step_result_details.get("screenshot_after"):
                screenshot_after = self._current_step_result_details["screenshot_after"]
            elif self.browser_driver:
                # Fallback for non-visual modes
                await asyncio.sleep(1.0)
                screenshot_after = await self.browser_driver.screenshot()
                
                # Debug: Save screenshots for inspection
                import os
                os.makedirs("debug_screenshots", exist_ok=True)
                with open(f"debug_screenshots/step_{step.step_number}_after.png", "wb") as f:
                    f.write(screenshot_after)
            
            # Evaluate result
            if self.evaluator_agent and screenshot_after:
                evaluation_result = await self.evaluator_agent.evaluate_result(
                    screenshot_after,
                    step.action_instruction.expected_outcome,
                    step_id=step.step_id
                )
            
            # Create step result
            success = evaluation_result.success if evaluation_result else True
            
            # Debug logging
            if evaluation_result and not evaluation_result.success:
                logger.warning("Step evaluation failed", extra={
                    "step_number": step.step_number,
                    "expected": evaluation_result.expected_outcome,
                    "actual": evaluation_result.actual_outcome,
                    "confidence": evaluation_result.confidence,
                    "deviations": evaluation_result.deviations
                })
            
            return TestStepResult(
                step=step,
                success=success,
                action_taken=action_result,
                actual_result=evaluation_result.actual_outcome if evaluation_result else "Action completed",
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                evaluation=evaluation_result,
                execution_mode=mode.value,
                action_result_details=self._current_step_result_details
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("Step execution failed", extra={
                "step_number": step.step_number,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": tb
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
    
    def _build_test_context(self, step: TestStep) -> Dict[str, Any]:
        """Build comprehensive context for Action Agent."""
        # Get recent step summaries
        recent_steps = []
        for result in self._execution_history[-3:]:  # Last 3 steps
            recent_steps.append({
                "step_number": result.step.step_number,
                "description": result.step.description,
                "success": result.success,
                "result": result.actual_result
            })
        
        # Build previous steps summary
        previous_steps_summary = ""
        if recent_steps:
            previous_steps_summary = "Recent steps: " + "; ".join([
                f"Step {s['step_number']} ({s['description']}): {'Success' if s['success'] else 'Failed'}"
                for s in recent_steps
            ])
        
        return {
            "test_plan_name": self._current_test_plan.name if self._current_test_plan else "Unknown",
            "test_plan_description": self._current_test_plan.description if self._current_test_plan else "Unknown",
            "current_step_description": step.description,
            "current_step_number": step.step_number,
            "total_steps": len(self._current_test_plan.steps) if self._current_test_plan else 0,
            "completed_steps": len(self._test_state.completed_steps) if self._test_state else 0,
            "previous_steps_summary": previous_steps_summary,
            "recent_failures": sum(1 for r in self._execution_history[-3:] if not r.success),
            "expected_outcome": step.action_instruction.expected_outcome,
            "step_dependencies": [str(d) for d in step.dependencies] if step.dependencies else [],
            "optional_step": step.optional
        }
    
    async def _execute_visual_action(self, step: TestStep) -> Optional[ActionResult]:
        """Execute action using visual AI interaction."""
        if not self.action_agent or not self.browser_driver:
            logger.warning("Visual execution not available - missing dependencies")
            return None
        
        # Build comprehensive context for Action Agent
        test_context = self._build_test_context(step)
        
        # Get current screenshot
        screenshot = await self.browser_driver.screenshot()
        
        # Use new Action Agent method that owns full action lifecycle
        action_result_dict = await self.action_agent.execute_action(
            test_step=step,
            test_context=test_context,
            screenshot=screenshot
        )
        
        # Store detailed result for debugging
        self._current_step_result_details = action_result_dict
        
        # Convert to ActionResult for backward compatibility
        action_result = None
        if action_result_dict.get("validation_passed") and action_result_dict.get("grid_cell"):
            action_result = ActionResult(
                action_type=action_result_dict["action_type"],
                grid_cell=action_result_dict["grid_cell"],
                offset_x=action_result_dict["offset_x"],
                offset_y=action_result_dict["offset_y"],
                confidence=action_result_dict["coordinate_confidence"],
                requires_refinement=False
            )
            
            # Record successful action for future use if execution succeeded
            if action_result_dict.get("execution_success"):
                self._record_action(step, action_result)
        
        return action_result
    
    async def _execute_scripted_action(self, step: TestStep) -> Optional[ActionResult]:
        """Execute action using recorded script."""
        if not self._current_test_plan:
            return None
        step_key = f"{self._current_test_plan.plan_id}:{step.step_number}"
        scripted_action = self._scripted_actions.get(step_key)
        
        if not scripted_action or not self.browser_driver:
            return None
        
        try:
            # Get viewport size to calculate pixels from grid coordinates
            viewport_width, viewport_height = await self.browser_driver.get_viewport_size()
            
            # Create grid overlay to convert coordinates
            from src.grid.overlay import GridOverlay
            grid = GridOverlay()
            grid.initialize(viewport_width, viewport_height)
            
            from src.core.types import GridCoordinate
            coord = GridCoordinate(
                cell=scripted_action["grid_cell"],
                offset_x=scripted_action["offset_x"],
                offset_y=scripted_action["offset_y"],
                confidence=1.0
            )
            
            # Convert to pixel coordinates
            x, y = grid.coordinate_to_pixels(coord)
            
            # Execute the scripted action
            action_type = scripted_action["action_type"]
            
            if action_type == "click":
                await self.browser_driver.click(x, y)
            elif action_type == "type":
                # Click to focus first
                await self.browser_driver.click(x, y)
                await self.browser_driver.wait(200)
                # Type the text from instruction
                if step.action_instruction.value:
                    await self.browser_driver.type_text(step.action_instruction.value)
                    await self.browser_driver.wait(500)
            
            # Return a synthetic action result
            return ActionResult(
                action_type=action_type,
                grid_cell=scripted_action.get("grid_cell", "N/A"),
                offset_x=scripted_action.get("offset_x", 0.5),
                offset_y=scripted_action.get("offset_y", 0.5),
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
    
    
    def _record_action(self, step: TestStep, action: ActionResult) -> None:
        """Record successful action for future scripted execution."""
        if not self._current_test_plan:
            return
        step_key = f"{self._current_test_plan.plan_id}:{step.step_number}"
        
        # Don't record pixel coordinates here - they should be calculated
        # dynamically based on viewport size when replaying
        self._scripted_actions[step_key] = {
            "action_type": action.action_type,
            "grid_cell": action.grid_cell,
            "offset_x": action.offset_x,
            "offset_y": action.offset_y,
            "timestamp": action.timestamp.isoformat()
        }
    
    async def _update_state(self, step: TestStep, result: TestStepResult) -> None:
        """Update test state after step execution."""
        # Add to history
        self._execution_history.append(result)
        
        if result.success:
            # Add to completed steps
            self._test_state.completed_steps.append(step.step_id)
            # Move to next step
            self._step_index += 1
        else:
            # Add to failed steps
            self._test_state.failed_steps.append(step.step_id)
            self._test_state.error_count += 1
            
            # Handle failure
            if not step.optional:
                logger.error("Required step failed, ending test", extra={
                    "step_number": step.step_number
                })
                self._test_state.status = TestStatus.FAILED
            else:
                logger.warning("Optional step failed, continuing", extra={
                    "step_number": step.step_number
                })
                # Move to next step even if optional step failed
                self._step_index += 1
        
        # Check for completion
        if self._step_index >= len(self._test_state.test_plan.steps) and self._test_state.status == TestStatus.IN_PROGRESS:
            self._test_state.status = TestStatus.COMPLETED
        
        # Use AI to analyze overall progress and determine next action
        # TODO: Fix this after unifying TestState
        # await self._analyze_progress()
    
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
        self._test_state.skipped_steps.append(step.step_id)
        # Move to next step
        self._step_index += 1
    
    async def _analyze_progress(self) -> None:
        """Use AI to analyze test progress and make decisions."""
        # Build context for AI
        context = {
            "test_plan": self._current_test_plan.name,
            "completed_steps": len(self._test_state.completed_steps),
            "total_steps": len(self._current_test_plan.steps),
            "recent_results": [
                {
                    "step": r.step.description,
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
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.get("content", "{}"))
            logger.info("Test progress analysis", extra=analysis)
            
            # Store analysis in context
            self._test_state.context["latest_analysis"] = analysis
            
        except Exception as e:
            logger.warning("Failed to analyze progress", extra={"error": str(e)})
    
    async def judge_final_test_result(self) -> Dict[str, Any]:
        """
        Make final judgment on test execution with full context.
        
        Returns comprehensive analysis of test execution success/failure.
        """
        if not self._test_state or not self._current_test_plan:
            return {
                "overall_success": False,
                "confidence": 0.0,
                "reasoning": "No test state available"
            }
        
        # Build comprehensive summary
        total_steps = len(self._current_test_plan.steps)
        completed_steps = len(self._test_state.completed_steps)
        failed_steps = len(self._test_state.failed_steps)
        skipped_steps = len(self._test_state.skipped_steps)
        
        # Get detailed failure information
        failure_details = []
        for result in self._execution_history:
            if not result.success:
                detail = {
                    "step_number": result.step.step_number,
                    "description": result.step.description,
                    "expected": result.step.action_instruction.expected_outcome,
                    "actual": result.actual_result,
                    "optional": result.step.optional
                }
                # Add enhanced debugging info if available
                if result.action_result_details:
                    detail["validation_passed"] = result.action_result_details.get("validation_passed", False)
                    detail["validation_reasoning"] = result.action_result_details.get("validation_reasoning", "")
                    detail["execution_error"] = result.action_result_details.get("execution_error", "")
                    detail["ai_analysis"] = result.action_result_details.get("ai_analysis", {})
                failure_details.append(detail)
        
        # Use AI to make final judgment
        prompt = f"""Analyze the complete test execution and provide final judgment:

Test Plan: {self._current_test_plan.name}
Description: {self._current_test_plan.description}

Execution Summary:
- Total Steps: {total_steps}
- Completed Successfully: {completed_steps}
- Failed: {failed_steps}
- Skipped: {skipped_steps}

Failure Details:
{json.dumps(failure_details, indent=2)}

Based on this information, provide:
1. Overall success (true/false) - consider if the test objective was achieved
2. Confidence in judgment (0.0-1.0)
3. Detailed reasoning
4. Key issues identified
5. Recommendations for improvement

Respond in JSON format with keys: overall_success, confidence, reasoning, key_issues, recommendations"""
        
        try:
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            judgment = json.loads(response.get("content", "{}"))
            
            # Add execution statistics
            judgment["execution_stats"] = {
                "total_steps": total_steps,
                "completed": completed_steps,
                "failed": failed_steps,
                "skipped": skipped_steps,
                "success_rate": completed_steps / total_steps if total_steps > 0 else 0
            }
            
            logger.info("Final test judgment", extra=judgment)
            return judgment
            
        except Exception as e:
            logger.error("Failed to make final judgment", extra={"error": str(e)})
            return {
                "overall_success": self._test_state.status == TestStatus.COMPLETED,
                "confidence": 0.5,
                "reasoning": f"Automated judgment: {self._test_state.status.value}",
                "key_issues": [str(e)],
                "recommendations": ["Review test execution logs"],
                "execution_stats": {
                    "total_steps": total_steps,
                    "completed": completed_steps,
                    "failed": failed_steps,
                    "skipped": skipped_steps,
                    "success_rate": completed_steps / total_steps if total_steps > 0 else 0
                }
            }
    
    async def get_next_action(
        self, test_plan: TestPlan, current_state: TestState
    ) -> Optional[str]:
        """
        Determine the next action based on current state.
        
        This method can be used for more dynamic test execution where
        the Test Runner makes decisions about what to do next.
        """
        # Find current step
        if current_state.current_step:
            next_step = current_state.current_step
            
            # Use AI to determine if we should proceed or adapt
            prompt = f"""Current test state:
- Test: {test_plan.name}
- Current step: {next_step.description}
- Expected result: {next_step.action_instruction.expected_outcome}
- Recent failures: {sum(1 for r in self._execution_history[-3:] if not r.success)}

Should we:
1. Proceed with the current step as planned
2. Skip this step
3. Retry a previous step
4. Abort the test

Provide your recommendation with reasoning."""
            
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.get("content", "Proceed with current step")
        
        return None