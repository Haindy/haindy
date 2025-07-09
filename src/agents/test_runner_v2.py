"""
Refactored Test Runner Agent with reduced responsibilities.

This prototype demonstrates the new architecture where Test Runner focuses on
high-level orchestration while Action Agent handles all action execution details.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from src.agents.action_agent_v2 import ActionAgentV2
from src.agents.base_agent import BaseAgent
from src.browser.driver import BrowserDriver
from src.config.agent_prompts import TEST_RUNNER_SYSTEM_PROMPT
from src.core.enhanced_types import (
    BugReport,
    EnhancedActionResult,
    EnhancedTestState,
    EnhancedTestStepResult
)
from src.core.types import TestPlan, TestState, TestStatus
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class TestRunnerV2(BaseAgent):
    """
    Refactored Test Runner Agent focused on high-level orchestration.
    
    This agent:
    1. Manages test flow and state
    2. Provides context to Action Agent
    3. Makes high-level decisions about test progression
    4. Generates bug reports for failures
    """
    
    def __init__(
        self,
        name: str = "TestRunnerV2",
        browser_driver: Optional[BrowserDriver] = None,
        action_agent: Optional[ActionAgentV2] = None,
        **kwargs
    ):
        """Initialize the refactored Test Runner Agent."""
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_RUNNER_SYSTEM_PROMPT
        self.browser_driver = browser_driver
        self.action_agent = action_agent or ActionAgentV2(browser_driver=browser_driver)
        
        # Enhanced state tracking
        self._enhanced_state: Optional[EnhancedTestState] = None
        self._step_index: int = 0
        
    async def execute_test_plan(
        self,
        test_plan: TestPlan,
        initial_url: Optional[str] = None
    ) -> EnhancedTestState:
        """
        Execute a complete test plan with enhanced error tracking.
        
        Args:
            test_plan: The test plan to execute
            initial_url: Optional starting URL
            
        Returns:
            Enhanced test state with comprehensive execution details
        """
        logger.info("Starting enhanced test plan execution", extra={
            "test_plan_id": str(test_plan.plan_id),
            "test_plan_name": test_plan.name,
            "total_steps": len(test_plan.steps)
        })
        
        # Initialize enhanced state
        self._enhanced_state = EnhancedTestState(
            test_plan=test_plan,
            status=TestStatus.IN_PROGRESS,
            start_time=datetime.now(timezone.utc)
        )
        self._step_index = 0
        
        # Navigate to initial URL
        if initial_url and self.browser_driver:
            logger.info("Navigating to initial URL", extra={"url": initial_url})
            await self.browser_driver.navigate(initial_url)
            self._enhanced_state.total_browser_actions += 1
        
        # Execute test steps
        try:
            while (self._enhanced_state.status == TestStatus.IN_PROGRESS and 
                   self._step_index < len(test_plan.steps)):
                await self._execute_next_step()
                
        except Exception as e:
            logger.error("Test execution failed with unexpected error", extra={
                "error": str(e),
                "current_step": self._step_index
            })
            self._enhanced_state.status = TestStatus.FAILED
            self._enhanced_state.error_count += 1
            self._enhanced_state.errors.append({
                "type": "execution_error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Finalize execution
        self._enhanced_state.end_time = datetime.now(timezone.utc)
        
        # Generate final summary
        await self._generate_execution_summary()
        
        logger.info("Test execution completed", extra={
            "test_status": self._enhanced_state.status.value,
            "completed_steps": len(self._enhanced_state.completed_steps),
            "failed_steps": len(self._enhanced_state.failed_steps),
            "bug_reports": len(self._enhanced_state.bug_reports)
        })
        
        return self._enhanced_state
    
    async def _execute_next_step(self) -> None:
        """Execute the next step in the test plan."""
        current_step = self._enhanced_state.test_plan.steps[self._step_index]
        self._enhanced_state.current_step = current_step
        
        logger.info("Preparing to execute test step", extra={
            "step_number": current_step.step_number,
            "description": current_step.description,
            "action_type": current_step.action_instruction.action_type.value
        })
        
        # Build context for Action Agent
        test_context = self._build_test_context()
        
        # Delegate action execution to Action Agent
        action_result = await self.action_agent.execute_action(
            test_step=current_step,
            test_context=test_context
        )
        
        # Track API calls
        self._enhanced_state.total_api_calls += 1
        if action_result.screenshot_before:
            self._enhanced_state.total_screenshots += 1
        if action_result.screenshot_after:
            self._enhanced_state.total_screenshots += 1
        if action_result.execution_success:
            self._enhanced_state.total_browser_actions += 1
        
        # Store enhanced result
        self._enhanced_state.execution_history.append(action_result)
        
        # Create step result
        step_result = EnhancedTestStepResult(
            step=current_step,
            success=action_result.execution_success and action_result.validation_passed,
            action_result=action_result,
            execution_mode="visual"
        )
        
        # Judge overall success based on AI analysis and expected outcome
        success = await self._judge_step_success(current_step, action_result)
        
        # Update state based on result
        if success:
            self._enhanced_state.completed_steps.append(current_step.step_id)
            self._step_index += 1
        else:
            self._enhanced_state.failed_steps.append(current_step.step_id)
            self._enhanced_state.error_count += 1
            
            # Generate bug report for failure
            bug_report = await self._generate_bug_report(current_step, action_result)
            self._enhanced_state.bug_reports.append(bug_report)
            
            # Decide whether to continue or fail
            if not current_step.optional:
                logger.error("Required step failed, ending test", extra={
                    "step_number": current_step.step_number
                })
                self._enhanced_state.status = TestStatus.FAILED
            else:
                logger.warning("Optional step failed, continuing", extra={
                    "step_number": current_step.step_number
                })
                self._step_index += 1
        
        # Check if all steps completed
        if self._step_index >= len(self._enhanced_state.test_plan.steps):
            self._enhanced_state.status = TestStatus.COMPLETED
    
    def _build_test_context(self) -> Dict:
        """Build comprehensive context for the Action Agent."""
        # Get recent execution history
        recent_history = []
        for result in self._enhanced_state.execution_history[-3:]:
            recent_history.append({
                "action": result.action_type,
                "success": result.execution_success,
                "grid_cell": result.grid_cell,
                "url_change": f"{result.url_before} → {result.url_after}"
            })
        
        # Build context
        return {
            "test_plan_name": self._enhanced_state.test_plan.name,
            "test_plan_description": self._enhanced_state.test_plan.description,
            "current_step_number": self._step_index + 1,
            "current_step_description": self._enhanced_state.current_step.description if self._enhanced_state.current_step else "",
            "total_steps": len(self._enhanced_state.test_plan.steps),
            "completed_steps": len(self._enhanced_state.completed_steps),
            "failed_steps": len(self._enhanced_state.failed_steps),
            "recent_history": recent_history,
            "previous_steps_summary": self._get_previous_steps_summary()
        }
    
    def _get_previous_steps_summary(self) -> str:
        """Generate a summary of previous steps."""
        if not self._enhanced_state.execution_history:
            return "No previous steps"
        
        summary_parts = []
        for i, result in enumerate(self._enhanced_state.execution_history[-5:]):
            status = "✓" if result.execution_success else "✗"
            summary_parts.append(
                f"{status} Step {i+1}: {result.action_type} on {result.grid_cell}"
            )
        
        return "; ".join(summary_parts)
    
    async def _judge_step_success(
        self,
        step,
        action_result: EnhancedActionResult
    ) -> bool:
        """
        Judge if a step was successful based on all available information.
        
        This replaces the Evaluator Agent's functionality.
        """
        # First check: Did the action execute successfully?
        if not action_result.execution_success:
            return False
        
        # Second check: Did validation pass?
        if not action_result.validation_passed:
            return False
        
        # Third check: Does AI analysis indicate success?
        if action_result.ai_analysis:
            return action_result.ai_analysis.get("success", False)
        
        # Default: Consider it successful if no errors
        return True
    
    async def _generate_bug_report(
        self,
        step,
        action_result: EnhancedActionResult
    ) -> BugReport:
        """Generate a comprehensive bug report for a failed step."""
        # Determine failure category
        if not action_result.validation_passed:
            category = "validation"
            severity = "high"
        elif action_result.execution_error:
            category = "execution"
            severity = "critical"
        elif action_result.coordinate_confidence < 0.5:
            category = "ui_recognition"
            severity = "medium"
        else:
            category = "unexpected"
            severity = "medium"
        
        # Save screenshots
        import os
        from pathlib import Path
        
        screenshots_dir = Path("data/bug_reports/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        screenshot_paths = {}
        
        if action_result.grid_screenshot_highlighted:
            path = screenshots_dir / f"{timestamp}_step{step.step_number}_grid_highlighted.png"
            path.write_bytes(action_result.grid_screenshot_highlighted)
            screenshot_paths["grid_highlighted"] = str(path)
        
        if action_result.screenshot_before:
            path = screenshots_dir / f"{timestamp}_step{step.step_number}_before.png"
            path.write_bytes(action_result.screenshot_before)
            screenshot_paths["before"] = str(path)
        
        if action_result.screenshot_after:
            path = screenshots_dir / f"{timestamp}_step{step.step_number}_after.png"
            path.write_bytes(action_result.screenshot_after)
            screenshot_paths["after"] = str(path)
        
        # Build failure analysis
        failure_analysis = f"""
Step {step.step_number} failed during {action_result.action_type} action.

Validation: {action_result.validation_status.value}
- Reasoning: {action_result.validation_reasoning}
- Confidence: {action_result.validation_confidence}

Grid Selection: {action_result.grid_cell}
- Confidence: {action_result.coordinate_confidence}
- Reasoning: {action_result.coordinate_reasoning}

Execution: {'Success' if action_result.execution_success else 'Failed'}
- Time: {action_result.execution_time_ms}ms
- Error: {action_result.execution_error or 'None'}

AI Analysis: {action_result.ai_analysis.get('actual_outcome', 'Not available')}
"""
        
        # Generate recommendations
        recommendations = []
        if action_result.validation_confidence < 0.7:
            recommendations.append("Element may not be visible or ready. Consider adding wait conditions.")
        if action_result.coordinate_confidence < 0.7:
            recommendations.append("UI element detection is uncertain. Check if UI has changed.")
        if "timeout" in str(action_result.execution_error).lower():
            recommendations.append("Action timed out. Page may be slow or unresponsive.")
        
        return BugReport(
            test_plan_id=self._enhanced_state.test_plan.plan_id,
            test_plan_name=self._enhanced_state.test_plan.name,
            step_number=step.step_number,
            step_description=step.description,
            action_attempted=f"{action_result.action_type} on {action_result.grid_cell}",
            expected_outcome=step.action_instruction.expected_outcome,
            actual_outcome=action_result.ai_analysis.get("actual_outcome", "Action failed"),
            screenshots=screenshot_paths,
            ai_reasoning={
                "validation": action_result.validation_reasoning,
                "coordinates": action_result.coordinate_reasoning,
                "analysis": str(action_result.ai_analysis)
            },
            error_type=action_result.execution_error_type or category,
            error_message=action_result.execution_error or action_result.validation_reasoning,
            error_traceback=action_result.execution_traceback,
            browser_state={
                "url_before": action_result.url_before,
                "url_after": action_result.url_after,
                "title_before": action_result.page_title_before,
                "title_after": action_result.page_title_after
            },
            failure_analysis=failure_analysis,
            recommended_fixes=recommendations,
            severity=severity,
            category=category
        )
    
    async def _generate_execution_summary(self) -> None:
        """Generate a final execution summary with AI insights."""
        if not self._enhanced_state.bug_reports:
            return
        
        # Use AI to analyze patterns in failures
        prompt = f"""Analyze these test failures and provide insights:

Test: {self._enhanced_state.test_plan.name}
Total Steps: {len(self._enhanced_state.test_plan.steps)}
Failed Steps: {len(self._enhanced_state.failed_steps)}

Failures:
"""
        for report in self._enhanced_state.bug_reports:
            prompt += f"\n- Step {report.step_number}: {report.error_type} - {report.error_message}"
        
        prompt += "\n\nProvide a brief analysis of failure patterns and recommendations."
        
        try:
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}]
            )
            
            self._enhanced_state.metadata["failure_analysis"] = response.get("content", "")
        except Exception as e:
            logger.warning("Failed to generate AI summary", extra={"error": str(e)})