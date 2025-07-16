"""
Enhanced Test Runner Agent implementation for Phase 15.

This agent orchestrates test execution with intelligent step interpretation,
living document reporting, and comprehensive failure handling.
"""

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from src.agents.action_agent import ActionAgent
from src.agents.base_agent import BaseAgent
from src.agents.formatters import TestPlanFormatter
from src.browser.driver import BrowserDriver
from src.config.agent_prompts import TEST_RUNNER_SYSTEM_PROMPT
from src.core.types import (
    ActionInstruction,
    ActionType,
    BugReport,
    BugSeverity,
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    TestReport,
    TestState,
    TestStatus,
    TestStep,
    TestSummary,
)
from src.monitoring.logger import get_logger
from src.browser.instrumented_driver import InstrumentedBrowserDriver

logger = get_logger(__name__)


class TestRunner(BaseAgent):
    """
    Enhanced Test Runner Agent that orchestrates test execution with intelligence.
    
    Key improvements:
    - Hierarchical test plan execution (TestPlan → TestCase → TestStep)
    - Living document report generation
    - Intelligent step interpretation and decomposition
    - Comprehensive bug reporting
    - Smart failure handling and recovery
    """
    
    def __init__(
        self,
        name: str = "TestRunner",
        browser_driver: Optional[BrowserDriver] = None,
        action_agent: Optional[ActionAgent] = None,
        **kwargs
    ):
        """
        Initialize the Enhanced Test Runner.
        
        Args:
            name: Agent name
            browser_driver: Browser driver instance
            action_agent: Action agent for executing browser actions
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_RUNNER_SYSTEM_PROMPT
        self.browser_driver = browser_driver
        self.action_agent = action_agent
        
        # Current execution state
        self._current_test_plan: Optional[TestPlan] = None
        self._current_test_case: Optional[TestCase] = None
        self._current_test_step: Optional[TestStep] = None
        self._test_state: Optional[TestState] = None
        self._test_report: Optional[TestReport] = None
        
        # Execution context
        self._initial_url: Optional[str] = None
        self._execution_history: List[Dict[str, Any]] = []
        
        # Action storage for Phase 17
        self._action_storage: Dict[str, Any] = {
            "test_plan_id": None,
            "test_run_timestamp": None,
            "test_cases": []
        }
        self._current_test_case_actions: Optional[Dict[str, Any]] = None
        self._current_step_actions: Optional[List[Dict[str, Any]]] = None
    
    async def execute_test_plan(
        self,
        test_state: TestState,
        initial_url: Optional[str] = None
    ) -> TestState:
        """
        Execute a complete test plan with all test cases.
        
        Args:
            test_state: The test state containing the test plan
            initial_url: Optional starting URL
            
        Returns:
            Updated test state with comprehensive test report
        """
        test_plan = test_state.test_plan
        logger.info("Starting enhanced test plan execution", extra={
            "test_plan_id": str(test_plan.plan_id),
            "test_plan_name": test_plan.name,
            "total_test_cases": len(test_plan.test_cases)
        })
        
        # Initialize execution
        self._current_test_plan = test_plan
        self._initial_url = initial_url
        self._test_state = test_state
        
        # Initialize action storage for this test run
        self._action_storage = {
            "test_plan_id": str(test_plan.plan_id),
            "test_run_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_cases": []
        }
        
        # Initialize test report within the test state
        self._test_report = TestReport(
            test_plan_id=test_plan.plan_id,
            test_plan_name=test_plan.name,
            started_at=datetime.now(timezone.utc),
            status=TestStatus.IN_PROGRESS,
            environment={
                "initial_url": initial_url,
                "browser": "Chromium",
                "execution_mode": "enhanced"
            }
        )
        
        # Attach the test report to the test state
        test_state.test_report = self._test_report
        
        # Update test state timing and status
        test_state.status = TestStatus.IN_PROGRESS
        test_state.start_time = self._test_report.started_at
        
        # Initial report will be saved by the caller using TestReporter
        
        try:
            # Navigate to initial URL if provided
            if initial_url and self.browser_driver:
                logger.info("Navigating to initial URL", extra={"url": initial_url})
                await self.browser_driver.navigate(initial_url)
            
            # Execute each test case sequentially
            for i, test_case in enumerate(test_plan.test_cases):
                case_result = await self._execute_test_case(test_case)
                
                # Save report after each test case
                # Report updates will be handled by the caller
                
                # Check if failure should cascade
                if case_result.status == TestStatus.FAILED:
                    should_continue = await self._should_continue_after_failure(
                        test_case, case_result, test_plan, i
                    )
                    
                    if not should_continue:
                        logger.error("Test case failure blocks further execution", extra={
                            "failed_test_case": test_case.name,
                            "remaining_test_cases": len(test_plan.test_cases) - i - 1
                        })
                        
                        # Mark remaining test cases as blocked
                        for j in range(i + 1, len(test_plan.test_cases)):
                            blocked_case = test_plan.test_cases[j]
                            blocked_result = TestCaseResult(
                                case_id=blocked_case.case_id,
                                test_id=blocked_case.test_id,
                                name=blocked_case.name,
                                status=TestStatus.SKIPPED,
                                started_at=datetime.now(timezone.utc),
                                completed_at=datetime.now(timezone.utc),
                                steps_total=len(blocked_case.steps),
                                steps_completed=0,
                                steps_failed=0,
                                error_message=f"Blocked due to failure of test case: {test_case.name}"
                            )
                            self._test_report.test_cases.append(blocked_result)
                        
                        # Save report with blocked test cases
                        # Report updates will be handled by the caller
                        break
            
            # Finalize report
            self._test_report.completed_at = datetime.now(timezone.utc)
            self._test_report.status = self._determine_overall_status()
            self._test_report.summary = self._calculate_summary()
            
            # Update test state
            self._test_state.end_time = self._test_report.completed_at
            self._test_state.status = self._test_report.status
            
        except Exception as e:
            logger.error("Test execution failed with error", extra={
                "error": str(e),
                "test_plan": test_plan.name
            })
            self._test_report.status = TestStatus.FAILED
            self._test_state.status = TestStatus.FAILED
            raise
        
        finally:
            # Final report will be saved by the caller using TestReporter
            
            # Print summary to console
            self._print_summary()
        
        return self._test_state
    
    async def _execute_test_case(self, test_case: TestCase) -> TestCaseResult:
        """Execute a single test case with all its steps."""
        logger.info("Starting test case execution", extra={
            "test_case_id": test_case.test_id,
            "test_case_name": test_case.name,
            "priority": test_case.priority.value,
            "total_steps": len(test_case.steps)
        })
        
        self._current_test_case = test_case
        
        # Initialize action tracking for this test case
        self._current_test_case_actions = {
            "test_case_id": str(test_case.case_id),
            "test_case_name": test_case.name,
            "steps": []
        }
        self._action_storage["test_cases"].append(self._current_test_case_actions)
        
        # Initialize test case result
        case_result = TestCaseResult(
            case_id=test_case.case_id,
            test_id=test_case.test_id,
            name=test_case.name,
            status=TestStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            steps_total=len(test_case.steps),
            steps_completed=0,
            steps_failed=0
        )
        
        # Add to report
        self._test_report.test_cases.append(case_result)
        
        try:
            # Check prerequisites
            if not await self._verify_prerequisites(test_case.prerequisites):
                case_result.status = TestStatus.SKIPPED
                case_result.error_message = "Prerequisites not met"
                return case_result
            
            # Track if any step failed
            has_failed_steps = False
            
            # Execute each step
            for step in test_case.steps:
                step_result = await self._execute_test_step(step, test_case, case_result)
                case_result.step_results.append(step_result)
                
                # Update counters
                if step_result.status == TestStatus.PASSED:
                    case_result.steps_completed += 1
                elif step_result.status == TestStatus.FAILED:
                    case_result.steps_failed += 1
                    has_failed_steps = True
                    
                    # Create bug report for failed step
                    bug_report = await self._create_bug_report(
                        step_result, step, test_case, case_result
                    )
                    if bug_report:
                        case_result.bugs.append(bug_report)
                        self._test_report.bugs.append(bug_report)
                    
                    # Determine if we should continue
                    if not step.optional and await self._is_blocker_failure(step_result):
                        logger.error("Blocker failure detected, stopping test case", extra={
                            "step_number": step.step_number,
                            "test_case": test_case.name
                        })
                        case_result.status = TestStatus.FAILED
                        break
                
                # Save report after each step
                # Report updates will be handled by the caller
            
            # Determine final test case status
            if case_result.status != TestStatus.FAILED:  # Not already marked as failed by blocker
                if has_failed_steps:
                    case_result.status = TestStatus.FAILED
                    if not case_result.error_message:
                        case_result.error_message = f"{case_result.steps_failed} step(s) failed"
                elif await self._verify_postconditions(test_case.postconditions):
                    case_result.status = TestStatus.PASSED
                else:
                    case_result.status = TestStatus.FAILED
                    case_result.error_message = "Postconditions not met"
            
        except Exception as e:
            logger.error("Test case execution failed", extra={
                "error": str(e),
                "test_case": test_case.name
            })
            case_result.status = TestStatus.FAILED
            case_result.error_message = str(e)
        
        finally:
            case_result.completed_at = datetime.now(timezone.utc)
        
        return case_result
    
    async def _execute_test_step(
        self,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult
    ) -> StepResult:
        """Execute a single test step with intelligent interpretation."""
        logger.info("Executing test step", extra={
            "step_number": step.step_number,
            "action": step.action,
            "test_case": test_case.name
        })
        
        self._current_test_step = step
        
        # Initialize action tracking for this step
        self._current_step_actions = []
        self._current_step_data = {
            "step_number": step.step_number,
            "step_id": str(step.step_id),
            "step_description": step.action,
            "actions": self._current_step_actions
        }
        self._current_test_case_actions["steps"].append(self._current_step_data)
        
        # Initialize step result
        step_result = StepResult(
            step_id=step.step_id,
            step_number=step.step_number,
            status=TestStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),  # Will update later
            action=step.action,
            expected_result=step.expected_result,
            actual_result=""
        )
        
        try:
            # Check dependencies
            if not self._check_dependencies(step, case_result):
                step_result.status = TestStatus.SKIPPED
                step_result.actual_result = "Skipped due to unmet dependencies"
                return step_result
            
            # Capture before screenshot
            if self.browser_driver:
                screenshot_before = await self.browser_driver.screenshot()
                screenshot_path = self._save_screenshot(
                    screenshot_before,
                    f"tc{test_case.test_id}_step{step.step_number}_before"
                )
                step_result.screenshot_before = str(screenshot_path)
            
            # Interpret step and decompose into actions
            actions = await self._interpret_step(step, test_case, case_result)
            
            # Execute each action
            success = True
            actual_outcomes = []
            
            for action in actions:
                logger.debug("Executing sub-action", extra={
                    "action_type": action["type"],
                    "description": action.get("description", "")
                })
                
                action_result = await self._execute_action(action, step)
                step_result.actions_performed.append(action_result)
                
                if not action_result.get("success", False):
                    success = False
                    actual_outcomes.append(f"Failed: {action_result.get('error', 'Unknown error')}")
                    
                    # Determine if we should continue with remaining actions
                    if action.get("critical", True):
                        break
                else:
                    actual_outcomes.append(action_result.get("outcome", "Action completed"))
            
            # Capture after screenshot
            if self.browser_driver:
                await asyncio.sleep(1)  # Brief wait for UI to stabilize
                screenshot_after = await self.browser_driver.screenshot()
                screenshot_path = self._save_screenshot(
                    screenshot_after,
                    f"tc{test_case.test_id}_step{step.step_number}_after"
                )
                step_result.screenshot_after = str(screenshot_path)
            
            # Evaluate overall step result
            if success:
                # Verify expected outcome is achieved
                try:
                    verification = await self._verify_expected_outcome(
                        step.expected_result,
                        actual_outcomes
                    )
                    
                    if verification["success"]:
                        step_result.status = TestStatus.PASSED
                        step_result.actual_result = verification["actual_outcome"]
                    else:
                        step_result.status = TestStatus.FAILED
                        step_result.actual_result = verification["actual_outcome"]
                        step_result.error_message = verification.get("reason", "Verification failed")
                    
                    step_result.confidence = verification.get("confidence", 0.0)
                except Exception as e:
                    # AI verification failed - this is a fatal error
                    logger.error("AI verification failed - marking test as failed", extra={
                        "error": str(e),
                        "step_number": step.step_number,
                        "expected": step.expected_result,
                        "actual_outcomes": actual_outcomes
                    })
                    step_result.status = TestStatus.FAILED
                    step_result.actual_result = "; ".join(actual_outcomes) if actual_outcomes else "Unknown"
                    step_result.error_message = f"AI verification failed: {str(e)}"
                    step_result.confidence = 0.0
                    # Re-raise to trigger the outer exception handler
                    raise
            else:
                step_result.status = TestStatus.FAILED
                step_result.actual_result = "; ".join(actual_outcomes)
                step_result.error_message = "One or more actions failed"
            
        except Exception as e:
            logger.error("Step execution failed", extra={
                "error": str(e),
                "step_number": step.step_number
            })
            step_result.status = TestStatus.FAILED
            step_result.actual_result = f"Error: {str(e)}"
            step_result.error_message = str(e)
        
        finally:
            step_result.completed_at = datetime.now(timezone.utc)
            
            # Add to execution history
            self._execution_history.append({
                "test_case": test_case.name,
                "step": step.step_number,
                "action": step.action,
                "result": step_result.status.value,
                "timestamp": step_result.completed_at
            })
        
        return step_result
    
    async def _interpret_step(
        self,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult
    ) -> List[Dict[str, Any]]:
        """
        Interpret a test step and decompose it into executable actions.
        
        Returns a list of actions to be executed sequentially.
        """
        # Build context for interpretation
        # Convert recent history to JSON-serializable format
        recent_history = []
        for item in self._execution_history[-3:]:
            recent_history.append({
                "test_case": item.get("test_case", ""),
                "step": item.get("step", 0),
                "action": item.get("action", ""),
                "result": item.get("result", ""),
                "timestamp": item.get("timestamp").isoformat() if isinstance(item.get("timestamp"), datetime) else str(item.get("timestamp", ""))
            })
        
        context = {
            "test_case": test_case.name,
            "step_number": step.step_number,
            "action": step.action,
            "expected_result": step.expected_result,
            "recent_history": recent_history,
            "current_url": self._initial_url
        }
        
        # Use AI to interpret the step
        prompt = f"""Analyze this test step and break it down into specific browser actions:

Test Case: {context['test_case']}
Step {context['step_number']}: {context['action']}
Expected Result: {context['expected_result']}

Recent actions:
{json.dumps(context['recent_history'], indent=2)}

Break this down into a sequence of specific actions. For each action provide:
1. type: The action type - MUST be one of these exact values:
   - navigate: Go to a URL
   - click: Click on an element
   - type: Type text into a field
   - assert: Verify something on the page
   - key_press: Press a specific key (Enter, Tab, etc.)
   - scroll_to_element: Scroll until an element is visible
   - scroll_by_pixels: Scroll by a specific number of pixels
   - scroll_to_top: Scroll to top of page
   - scroll_to_bottom: Scroll to bottom of page
   - scroll_horizontal: Scroll horizontally
2. target: Describe the element in human terms (e.g., "the search input field", "the blue Login button", "the main navigation menu") - DO NOT use CSS selectors, IDs, or any DOM references
3. value: Required for certain action types:
   - For 'type': The text to type
   - For 'navigate': The URL to navigate to
   - For 'key_press': The key to press (e.g., "Enter", "Tab")
   - For 'scroll_by_pixels': Number of pixels (positive for down/right, negative for up/left)
4. description: Brief description of what this action does
5. critical: Whether failure should stop remaining actions (true/false)

IMPORTANT: Choose the most appropriate action type from the list above. For dropdown/select elements, use 'click' type.

Consider:
- Do we need to scroll to make elements visible?
- Are there multiple UI interactions needed?
- Should we verify intermediate states?

ACTION RULES:
1. Text Input Fields (search bars, text boxes, input fields):
   - DO NOT use 'click' followed by 'type' for text input fields
   - Use 'type' action directly - it will automatically focus the field
   - Text fields don't provide visual feedback when clicked, so click validation will fail
   - Example: For "Enter 'Python' in the search box", use only type action, not click+type

2. Navigation:
   - Always use 'navigate' for going to URLs, not click on address bar + type

3. Scrolling:
   - ONLY add scroll actions when truly necessary
   - Use common sense: elements like search bars, main navigation, headers are visible on page load
   - DO NOT scroll unless:
     a) The test explicitly mentions scrolling
     b) The element is logically below the fold (footer, comments, "load more" content)
     c) Previous actions suggest the element might not be visible
   - Example: Google's search bar is visible immediately - no scroll needed
   - Example: "Privacy Policy" link in footer - scroll needed

Respond with a JSON object containing an "actions" array."""

        # Log what we're sending to AI
        logger.info("Interpreting step with AI", extra={
            "step_number": step.step_number,
            "action": step.action,
            "expected_result": step.expected_result,
            "prompt_length": len(prompt)
        })
        
        try:
            
            try:
                response = await self.call_openai(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                logger.debug("OpenAI API call successful", extra={
                    "response_type": type(response).__name__,
                    "response_keys": list(response.keys()) if isinstance(response, dict) else None
                })
                
                # Store the Test Runner interpretation conversation
                if hasattr(self, '_current_step_data'):
                    self._current_step_data["test_runner_interpretation"] = {
                        "prompt": prompt,
                        "response": response.get("content", {})
                    }
                
            except Exception as api_error:
                logger.error("OpenAI API call failed", extra={
                    "api_error": str(api_error),
                    "api_error_type": type(api_error).__name__,
                    "traceback": traceback.format_exc()
                })
                raise
            
            # Parse AI response
            content = response.get("content", {})
            
            # Content should already be a dict when using json_object format
            if not isinstance(content, dict):
                error_msg = f"Expected dict content but got {type(content)}"
                raise TypeError(error_msg)
            
            actions = content.get("actions", [])
            
            # Ensure AI provided actions
            if not actions:
                raise ValueError(f"AI failed to provide actions for step {step.step_number}: {step.action}")
            
            logger.info("Step interpretation successful", extra={
                "step": step.step_number,
                "original_action": step.action,
                "decomposed_actions": len(actions)
            })
            
            return actions
            
        except Exception as e:
            logger.error("Failed to interpret step with AI", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "step": step.step_number,
                "action": step.action,
                "traceback": traceback.format_exc()
            })
            # Re-raise - no fallback, AI failure is fatal
            raise
    
    async def _execute_action(
        self,
        action: Dict[str, Any],
        step: TestStep
    ) -> Dict[str, Any]:
        """Execute a single decomposed action with comprehensive tracking."""
        action_type = action.get("type", "assert")
        action_id = str(uuid4())
        timestamp_start = datetime.now(timezone.utc)
        
        # Initialize action storage entry
        action_data = {
            "action_id": action_id,
            "action_type": action_type,
            "target": action.get("target", ""),
            "value": action.get("value"),
            "description": action.get("description", ""),
            "timestamp_start": timestamp_start.isoformat(),
            "timestamp_end": None,
            "ai_conversation": {
                "test_runner_interpretation": action.copy() if isinstance(action, dict) else None,
                "action_agent_execution": None
            },
            "browser_calls": [],
            "result": None,
            "screenshots": {}
        }
        
        try:
            if not self.action_agent or not self.browser_driver:
                error_result = {
                    "success": False,
                    "error": "Action agent or browser driver not available"
                }
                action_data["result"] = error_result
                action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                self._current_step_actions.append(action_data)
                return error_result
            
            # If browser driver is instrumented, start capturing calls
            if isinstance(self.browser_driver, InstrumentedBrowserDriver):
                self.browser_driver.start_capture()
            
            # Create ActionInstruction for the action
            instruction = ActionInstruction(
                action_type=ActionType(action_type),
                description=action.get("description", ""),
                target=action.get("target", ""),
                value=action.get("value"),
                expected_outcome=action.get("expected_outcome", step.expected_result)
            )
            
            # Create a temporary TestStep for the action
            action_step = TestStep(
                step_number=step.step_number,
                description=action.get("description", step.description),
                action=action.get("description", step.action),
                expected_result=action.get("expected_outcome", step.expected_result),
                action_instruction=instruction,
                optional=not action.get("critical", True)
            )
            
            # Build context
            test_context = {
                "test_plan_name": self._current_test_plan.name,
                "test_case_name": self._current_test_case.name,
                "step_number": step.step_number,
                "action_description": action.get("description", ""),
                "recent_actions": self._execution_history[-3:]
            }
            
            # Get screenshot before action
            screenshot = await self.browser_driver.screenshot()
            
            # Execute via Action Agent
            result = await self.action_agent.execute_action(
                test_step=action_step,
                test_context=test_context,
                screenshot=screenshot
            )
            
            # Stop capturing browser calls
            if isinstance(self.browser_driver, InstrumentedBrowserDriver):
                action_data["browser_calls"] = self.browser_driver.stop_capture()
            
            # Extract AI conversation from Action Agent
            # The action agent stores conversation history
            if hasattr(self.action_agent, 'conversation_history'):
                action_data["ai_conversation"]["action_agent_execution"] = {
                    "messages": self.action_agent.conversation_history.copy(),
                    "screenshot_path": None  # Will be filled by debug logger
                }
                # Clear conversation history for next action
                self.action_agent.conversation_history = []
            
            # Store comprehensive result
            action_data["result"] = {
                "success": (result.validation.valid if result.validation else False) and 
                          (result.execution.success if result.execution else False),
                "validation": result.validation.model_dump() if result.validation else None,
                "coordinates": result.coordinates.model_dump() if result.coordinates else None,
                "execution": result.execution.model_dump() if result.execution else None,
                "ai_analysis": result.ai_analysis.model_dump() if result.ai_analysis else None
            }
            
            # Store screenshot paths
            if result.browser_state_before and result.browser_state_before.screenshot_path:
                action_data["screenshots"]["before"] = result.browser_state_before.screenshot_path
            if result.browser_state_after and result.browser_state_after.screenshot_path:
                action_data["screenshots"]["after"] = result.browser_state_after.screenshot_path
            if result.grid_screenshot_highlighted:
                # Get path from debug logger if available
                from src.monitoring.debug_logger import get_debug_logger
                debug_logger = get_debug_logger()
                if debug_logger:
                    # The highlighted screenshot was likely saved by action agent
                    action_data["screenshots"]["grid_overlay"] = f"{debug_logger.debug_dir}/grid_highlighted_{action_id}.png"
            
            # Process result for compatibility
            success = (
                result.validation.valid if result.validation else False
            ) and (
                result.execution.success if result.execution else False
            )
            
            outcome = "Action completed"
            if result.ai_analysis:
                outcome = result.ai_analysis.actual_outcome
            
            compatibility_result = {
                "success": success,
                "action_type": action_type,
                "target": action.get("target", ""),
                "outcome": outcome,
                "confidence": result.ai_analysis.confidence if result.ai_analysis else 0.0,
                "error": result.execution.error_message if (result.execution and not success) else None
            }
            
            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            self._current_step_actions.append(action_data)
            
            return compatibility_result
            
        except Exception as e:
            logger.error("Action execution failed", extra={
                "error": str(e),
                "action_type": action_type
            })
            
            # Stop capturing if needed
            if isinstance(self.browser_driver, InstrumentedBrowserDriver):
                action_data["browser_calls"] = self.browser_driver.stop_capture()
            
            error_result = {
                "success": False,
                "action_type": action_type,
                "error": str(e)
            }
            
            action_data["result"] = error_result
            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            self._current_step_actions.append(action_data)
            
            return error_result
    
    async def _verify_expected_outcome(
        self,
        expected: str,
        actual_outcomes: List[str]
    ) -> Dict[str, Any]:
        """Use AI to verify if expected outcome was achieved."""
        prompt = f"""Compare the expected outcome with what actually happened:

Expected: {expected}

Actual outcomes:
{chr(10).join(f"- {outcome}" for outcome in actual_outcomes)}

Determine:
1. Was the expected outcome achieved? (true/false)
2. What was the actual outcome? (concise summary)
3. Confidence in this assessment (0.0-1.0)
4. If not achieved, what was the reason?

Respond in JSON format with keys: success, actual_outcome, confidence, reason"""

        try:
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            content = response.get("content", "{}")
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content
            
            # Debug logging
            logger.debug("Verification result", extra={
                "expected": expected,
                "actual_outcomes": actual_outcomes,
                "result": result
            })
            
            # Ensure required keys exist with proper types
            if "success" not in result:
                result["success"] = False
            else:
                # Ensure success is a boolean, not a string
                result["success"] = bool(result["success"])
            if "actual_outcome" not in result:
                result["actual_outcome"] = "; ".join(actual_outcomes)
            if "confidence" not in result:
                result["confidence"] = 0.5
                
            return result
            
        except Exception as e:
            logger.error("Failed to verify outcome with AI", extra={
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "expected": expected,
                "actual_outcomes": actual_outcomes
            })
            # Raise the exception - don't fallback
            raise
    
    async def _create_bug_report(
        self,
        step_result: StepResult,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult
    ) -> Optional[BugReport]:
        """Create a detailed bug report for a failed step."""
        if step_result.status != TestStatus.FAILED:
            return None
        
        # Use AI to determine error type and severity
        prompt = f"""Analyze this test failure and determine the bug severity and type:

Failed Step: {step.action}
Expected Result: {step.expected_result}
Actual Result: {step_result.actual_result}
Error Message: {step_result.error_message}
Step is Optional: {step.optional}

Determine:
1. error_type: One of (element_not_found, assertion_failed, timeout, navigation_error, api_error, validation_error, unknown_error)
2. severity: One of (critical, high, medium, low)
   - critical: Blocks all testing, core functionality broken
   - high: Major feature broken, blocks test case
   - medium: Feature partially working, workaround possible
   - low: Minor issue, cosmetic or edge case

Respond in JSON format with keys: error_type, severity, reasoning"""

        try:
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Content is already a dict when using json_object response format
            result = response.get("content", {})
            if isinstance(result, str):
                result = json.loads(result)
            error_type = result.get("error_type", "unknown_error")
            
            # Map severity string to enum
            severity_map = {
                "critical": BugSeverity.CRITICAL,
                "high": BugSeverity.HIGH,
                "medium": BugSeverity.MEDIUM,
                "low": BugSeverity.LOW
            }
            severity = severity_map.get(
                result.get("severity", "medium").lower(),
                BugSeverity.MEDIUM
            )
            
            logger.debug("AI bug classification", extra={
                "error_type": error_type,
                "severity": severity.value,
                "reasoning": result.get("reasoning", "")
            })
            
        except Exception as e:
            logger.error("Failed to classify bug with AI", extra={
                "error": str(e),
                "step": step.step_number
            })
            # Re-raise - AI failure is fatal
            raise
        
        # Build reproduction steps
        reproduction_steps = [
            f"1. Execute test case: {test_case.name}",
            f"2. Navigate to step {step.step_number}: {step.action}"
        ]
        
        # Add recent successful steps for context
        for i, result in enumerate(case_result.step_results[-3:]):
            if result.status == TestStatus.PASSED:
                reproduction_steps.append(
                    f"{i+3}. Previous step completed: Step {result.step_number}"
                )
        
        reproduction_steps.append(
            f"{len(reproduction_steps)+1}. Execute failing step: {step.action}"
        )
        
        bug_report = BugReport(
            step_id=step.step_id,
            test_case_id=test_case.case_id,
            test_plan_id=self._current_test_plan.plan_id,
            step_number=step.step_number,
            description=f"Step {step.step_number} failed: {step.action}",
            severity=severity,
            error_type=error_type,
            expected_result=step.expected_result,
            actual_result=step_result.actual_result,
            screenshot_path=step_result.screenshot_after,
            error_details=step_result.error_message,
            reproduction_steps=reproduction_steps
        )
        
        logger.info("Bug report created", extra={
            "bug_id": str(bug_report.bug_id),
            "severity": severity.value,
            "error_type": error_type
        })
        
        return bug_report
    
    async def _is_blocker_failure(self, step_result: StepResult) -> bool:
        """Determine if a failure should block test case execution."""
        # TODO: Once navigation actions and validation are fully implemented,
        # replace this hardcoded logic with AI-based determination
        # Navigation errors are always blockers
        if "navigation" in (step_result.error_message or "").lower():
            return True
        
        # Critical element not found
        if "not found" in (step_result.error_message or "").lower():
            # Use AI to determine criticality
            prompt = f"""Analyze if this failure should stop the test case:

Failed action: {step_result.action}
Error: {step_result.error_message}
Expected: {step_result.expected_result}
Actual: {step_result.actual_result}

Is this a critical failure that prevents continuing the test? Consider:
- Can subsequent steps still be meaningful?
- Is this a core functionality failure?
- Would continuing provide useful information?

Respond with JSON: {{"is_blocker": true/false, "reasoning": "explanation"}}"""

            try:
                response = await self.call_openai(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                # Content is already a dict when using json_object response format
                result = response.get("content", {})
                if isinstance(result, str):
                    result = json.loads(result)
                return result.get("is_blocker", True)
                
            except Exception:
                # Default to blocking on error
                return True
        
        return False
    
    async def _should_continue_after_failure(
        self,
        failed_test_case: TestCase,
        case_result: TestCaseResult,
        test_plan: TestPlan,
        current_index: int
    ) -> bool:
        """
        Determine if test execution should continue after a test case failure.
        
        Uses AI to analyze the failure impact on remaining test cases.
        """
        # Get remaining test cases
        remaining_cases = test_plan.test_cases[current_index + 1:]
        if not remaining_cases:
            return True  # No more test cases to run
        
        # Build context for AI decision
        remaining_names = [tc.name for tc in remaining_cases]
        
        prompt = f"""Analyze this test case failure and determine if testing should continue:

Failed Test Case: {failed_test_case.name}
Failure Summary:
- Steps completed: {case_result.steps_completed}/{case_result.steps_total}
- Steps failed: {case_result.steps_failed}
- Error: {case_result.error_message or 'Multiple step failures'}

Failed Test Case Description: {failed_test_case.description}

Remaining Test Cases:
{chr(10).join(f"- {name}" for name in remaining_names)}

Overall Test Plan: {test_plan.name}
Test Plan Description: {test_plan.description}

Question: Should we continue executing the remaining test cases, or does this failure block them?

Consider:
1. Is the failed functionality a prerequisite for other tests?
2. Can the remaining tests provide value despite this failure?
3. Would the remaining tests likely fail due to this failure?

Respond with JSON: {{"continue": true/false, "reasoning": "explanation"}}"""

        try:
            response = await self.call_openai(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Content is already a dict when using json_object response format
            result = response.get("content", {})
            if isinstance(result, str):
                result = json.loads(result)
            should_continue = result.get("continue", False)
            
            logger.info("Cascade failure decision", extra={
                "failed_test_case": failed_test_case.name,
                "decision": "continue" if should_continue else "stop",
                "reasoning": result.get("reasoning", "")
            })
            
            return should_continue
            
        except Exception as e:
            logger.error("Failed to determine cascade impact", extra={
                "error": str(e),
                "failed_test_case": failed_test_case.name
            })
            # On AI failure, default to stopping execution (conservative approach)
            return False
    
    async def _verify_prerequisites(self, prerequisites: List[str]) -> bool:
        """Verify test case prerequisites are met."""
        if not prerequisites:
            return True
        
        # For now, log prerequisites and assume they're met
        # In future, could implement actual verification
        logger.info("Checking prerequisites", extra={
            "prerequisites": prerequisites
        })
        
        return True
    
    async def _verify_postconditions(self, postconditions: List[str]) -> bool:
        """Verify test case postconditions are met."""
        if not postconditions:
            return True
        
        # Take screenshot for verification
        if self.browser_driver:
            screenshot = await self.browser_driver.screenshot()
            
            # Use AI to verify postconditions
            prompt = f"""Verify these postconditions are met based on the current state:

Postconditions:
{chr(10).join(f"- {pc}" for pc in postconditions)}

Analyze the screenshot and determine if all postconditions are satisfied.

Respond with JSON: {{"all_met": true/false, "details": ["condition: status", ...]}}"""

            # For now, assume postconditions are met
            # Full implementation would analyze screenshot with AI
            logger.info("Checking postconditions", extra={
                "postconditions": postconditions
            })
        
        return True
    
    def _check_dependencies(self, step: TestStep, case_result: TestCaseResult) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        
        # Check if all dependent steps completed successfully
        for dep_num in step.dependencies:
            # Find the step result for the dependency
            dep_result = next(
                (r for r in case_result.step_results if r.step_number == dep_num),
                None
            )
            
            if not dep_result or dep_result.status != TestStatus.PASSED:
                logger.warning("Step dependency not met", extra={
                    "step": step.step_number,
                    "dependency": dep_num,
                    "dependency_status": dep_result.status.value if dep_result else "not_found"
                })
                return False
        
        return True
    
    def _determine_overall_status(self) -> TestStatus:
        """Determine overall test execution status."""
        if not self._test_report.test_cases:
            return TestStatus.FAILED
        
        # If any test case failed, overall status is failed
        failed_cases = [tc for tc in self._test_report.test_cases 
                       if tc.status == TestStatus.FAILED]
        if failed_cases:
            return TestStatus.FAILED
        
        # If all completed, overall is completed
        all_completed = all(tc.status == TestStatus.PASSED 
                           for tc in self._test_report.test_cases)
        if all_completed:
            return TestStatus.PASSED
        
        # Otherwise, partial completion
        return TestStatus.SKIPPED
    
    def _calculate_summary(self) -> TestSummary:
        """Calculate test execution summary statistics."""
        total_cases = len(self._test_report.test_cases)
        completed_cases = sum(1 for tc in self._test_report.test_cases 
                             if tc.status == TestStatus.PASSED)
        failed_cases = sum(1 for tc in self._test_report.test_cases 
                          if tc.status == TestStatus.FAILED)
        
        total_steps = sum(tc.steps_total for tc in self._test_report.test_cases)
        completed_steps = sum(tc.steps_completed for tc in self._test_report.test_cases)
        failed_steps = sum(tc.steps_failed for tc in self._test_report.test_cases)
        
        # Count bugs by severity
        critical_bugs = sum(1 for bug in self._test_report.bugs 
                           if bug.severity == BugSeverity.CRITICAL)
        high_bugs = sum(1 for bug in self._test_report.bugs 
                       if bug.severity == BugSeverity.HIGH)
        medium_bugs = sum(1 for bug in self._test_report.bugs 
                         if bug.severity == BugSeverity.MEDIUM)
        low_bugs = sum(1 for bug in self._test_report.bugs 
                      if bug.severity == BugSeverity.LOW)
        
        # Calculate execution time
        if self._test_report.completed_at and self._test_report.started_at:
            execution_time = (
                self._test_report.completed_at - self._test_report.started_at
            ).total_seconds()
        else:
            execution_time = 0.0
        
        # Calculate success rate
        success_rate = completed_steps / total_steps if total_steps > 0 else 0.0
        
        return TestSummary(
            total_test_cases=total_cases,
            completed_test_cases=completed_cases,
            failed_test_cases=failed_cases,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            critical_bugs=critical_bugs,
            high_bugs=high_bugs,
            medium_bugs=medium_bugs,
            low_bugs=low_bugs,
            success_rate=success_rate,
            execution_time_seconds=execution_time
        )
    
    def _save_screenshot(self, screenshot: bytes, name: str) -> Path:
        """Save screenshot to disk and return path."""
        # Screenshots are now handled by the debug logger
        from src.monitoring.debug_logger import get_debug_logger
        debug_logger = get_debug_logger()
        if debug_logger:
            return debug_logger.save_screenshot(screenshot, name)
        # Fallback to temp directory
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            f.write(screenshot)
            return Path(f.name)
    
    
    def _print_summary(self) -> None:
        """Print test execution summary to console."""
        if not self._test_report or not self._test_report.summary:
            return
        
        s = self._test_report.summary
        
        print("\n" + "="*80)
        print(f"TEST EXECUTION SUMMARY: {self._test_report.test_plan_name}")
        print("="*80)
        
        print(f"\nStatus: {self._test_report.status.value.upper()}")
        print(f"Test Cases: {s.completed_test_cases}/{s.total_test_cases} completed")
        print(f"Steps: {s.completed_steps}/{s.total_steps} completed")
        print(f"Success Rate: {s.success_rate*100:.1f}%")
        print(f"Execution Time: {s.execution_time_seconds:.1f}s")
        
        if self._test_report.bugs:
            print(f"\nBugs Found: {len(self._test_report.bugs)}")
            print(f"  Critical: {s.critical_bugs}")
            print(f"  High: {s.high_bugs}")
            print(f"  Medium: {s.medium_bugs}")
            print(f"  Low: {s.low_bugs}")
            
            # Show critical bugs
            critical = [b for b in self._test_report.bugs 
                       if b.severity == BugSeverity.CRITICAL]
            if critical:
                print("\nCRITICAL BUGS:")
                for bug in critical:
                    print(f"  - {bug.description}")
        
        print("\n" + "="*80 + "\n")
    
    def get_action_storage(self) -> Dict[str, Any]:
        """Return the captured action storage data."""
        return self._action_storage