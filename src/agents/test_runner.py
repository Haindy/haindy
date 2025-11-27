"""
Enhanced Test Runner Agent implementation for Phase 15.

This agent orchestrates test execution with intelligent step interpretation,
living document reporting, and comprehensive failure handling.
"""

import asyncio
import base64
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
from src.config.settings import get_settings
from src.core.types import (
    ActionInstruction,
    ActionType,
    BugReport,
    BugSeverity,
    StepResult,
    TestCase,
    TestCaseResult,
    TestPlan,
    StepIntent,
    TestReport,
    TestState,
    TestStatus,
    TestStep,
    TestSummary,
)
from src.core.run_cache import PersistentRunCache
from src.monitoring.logger import get_logger
from src.browser.instrumented_driver import InstrumentedBrowserDriver

logger = get_logger(__name__)
MAX_TURN_ERROR_PREFIX = "Computer Use max turns exceeded"
LOOP_ERROR_PREFIX = "Computer Use loop detected"

COMPUTER_USE_PROMPT_MANUAL_TEMPLATE = """Computer Use execution context:
- Executor: OpenAI Computer Use tool powered by GPT-5 with medium reasoning effort.
- Environment: {environment_description}
- Inputs: Each prompt is delivered with the latest screenshot and scenario metadata; do not capture screenshots yourself.

Prompt construction rules:
1. Begin with a concise imperative goal that states the desired outcome.
2. Identify the UI target(s) using the exact labels a human sees (no CSS/XPath speculation).
3. Provide any text to enter or keys to press when relevant.
4. Restate the expected outcome so the executor can verify completion on its own.
5. Instruct the executor to act directly without seeking confirmation from the user.
6. Require a strategy shift after three identical failures and ask for an explanation if blocked.
7. Tell it to rely on the provided screenshot for context and to scroll or refocus if elements are off-screen.
8. For observation-only (`assert`) actions, explicitly forbid interactions and request a visual verification summary instead.
9. Avoid backend assumptions, hidden DOM references, or multi-step checklists—each prompt should cover one cohesive action.

If no interaction is required (`skip_navigation`), leave the computer_use_prompt empty.""".strip()


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
        run_cache: Optional[PersistentRunCache] = None,
        run_signature: Optional[str] = None,
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
        self._run_cache = run_cache
        self._run_signature = run_signature
        
        # Current execution state
        self._current_test_plan: Optional[TestPlan] = None
        self._current_test_case: Optional[TestCase] = None
        self._current_test_step: Optional[TestStep] = None
        self._test_state: Optional[TestState] = None
        self._test_report: Optional[TestReport] = None
        
        # Execution context
        self._initial_url: Optional[str] = None
        self._execution_history: List[Dict[str, Any]] = []
        self._settings = get_settings()
        self._disable_cache_for_case: Dict[str, bool] = {}
        self._plan_cache_source: str = "live"
        self._initial_screenshot_bytes: Optional[bytes] = None
        self._initial_screenshot_path: Optional[str] = None
        self._latest_screenshot_bytes: Optional[bytes] = None
        self._latest_screenshot_path: Optional[str] = None
        self._latest_screenshot_origin: Optional[str] = None
        self._computer_use_prompt_manual: str = self._build_computer_use_manual()
        
        # Action storage for Phase 17
        self._action_storage: Dict[str, Any] = {
            "test_plan_id": None,
            "test_run_timestamp": None,
            "test_cases": []
        }
        self._current_test_case_actions: Optional[Dict[str, Any]] = None
        self._current_step_actions: Optional[List[Dict[str, Any]]] = None

    def set_run_cache(
        self, run_cache: Optional[PersistentRunCache], run_signature: Optional[str]
    ) -> None:
        """Update the run cache configuration at runtime."""
        self._run_cache = run_cache
        self._run_signature = run_signature

    def _should_use_cache_for_case(self, test_case: TestCase) -> bool:
        """Determine if caching should be used for this case."""
        if not self._settings.run_cache_enabled:
            return False
        if not self._run_cache or not self._run_signature:
            return False
        return not self._disable_cache_for_case.get(str(test_case.case_id), False)

    def _invalidate_step_cache(self, test_case: TestCase, step: TestStep, reason: str) -> None:
        """Invalidate a cached step interpretation."""
        if not (self._run_cache and self._run_signature and self._settings.run_cache_enabled):
            return
        try:
            self._run_cache.invalidate_step(
                signature=self._run_signature,
                case_id=str(test_case.case_id),
                step_number=step.step_number,
                reason=reason,
            )
        except Exception:
            logger.debug("Failed to invalidate step cache", exc_info=True)

    def _invalidate_case_cache(self, test_case: TestCase, reason: str) -> None:
        """Invalidate cached data for the given case."""
        if not (self._run_cache and self._run_signature and self._settings.run_cache_enabled):
            return
        try:
            self._run_cache.mark_case_invalid(
                signature=self._run_signature,
                case_id=str(test_case.case_id),
                reason=reason,
            )
        except Exception:
            logger.debug("Failed to invalidate case cache", exc_info=True)

    def _build_computer_use_manual(self) -> str:
        """Construct the Computer Use manual with environment-aware context."""
        environment_description = self._computer_use_environment_description()
        return COMPUTER_USE_PROMPT_MANUAL_TEMPLATE.format(
            environment_description=environment_description
        )

    def _computer_use_environment_description(self) -> str:
        """Describe the execution surface for Computer Use prompts."""
        if self._settings.desktop_mode_enabled:
            return (
                "Linux desktop environment with OS-level mouse, keyboard, scrolling, "
                "and screenshot control. Interact with windows and applications directly; "
                "do not assume a sandboxed browser-only context."
            )
        return (
            "Chromium browser environment with full mouse, keyboard, scrolling, and "
            "screenshot control."
        )

    @staticmethod
    def _severity_rank(severity: BugSeverity) -> int:
        """Return an integer rank for comparing bug severities."""
        severity_order = {
            BugSeverity.CRITICAL: 0,
            BugSeverity.HIGH: 1,
            BugSeverity.MEDIUM: 2,
            BugSeverity.LOW: 3,
        }
        return severity_order.get(severity, 99)
    
    async def _ensure_initial_screenshot(self) -> None:
        """Capture and cache the initial browser screenshot."""
        if self._initial_screenshot_bytes is not None:
            return
        if not self.browser_driver:
            return
        
        wait_seconds = max(
            float(self._settings.actions_computer_tool_stabilization_wait_ms) / 1000.0,
            0.0,
        )
        if wait_seconds:
            await asyncio.sleep(wait_seconds)
        
        try:
            screenshot = await self.browser_driver.screenshot()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "Failed to capture initial screenshot",
                extra={"error": str(exc)},
            )
            return
        
        screenshot_path = self._save_screenshot(screenshot, "initial_state")
        self._initial_screenshot_bytes = screenshot
        self._initial_screenshot_path = str(screenshot_path)
        self._latest_screenshot_bytes = screenshot
        self._latest_screenshot_path = self._initial_screenshot_path
        self._latest_screenshot_origin = "initial_state"
        
        logger.info(
            "Captured initial browser screenshot",
            extra={"screenshot_path": self._initial_screenshot_path},
        )

    async def _get_interpretation_screenshot(
        self,
        step: TestStep,
        test_case: TestCase,
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """
        Resolve the screenshot to send alongside step interpretation.

        Returns a tuple of (screenshot_bytes, screenshot_path, source_label).
        """
        await self._ensure_initial_screenshot()

        if self._latest_screenshot_bytes and self._latest_screenshot_path:
            source = self._latest_screenshot_origin or "cached_snapshot"
            if (
                source == "initial_state"
                and step.step_number > 1
            ):
                source = "initial_state_cached"
            return (
                self._latest_screenshot_bytes,
                self._latest_screenshot_path,
                source,
            )

        if not self.browser_driver:
            return None, None, None

        try:
            screenshot = await self.browser_driver.screenshot()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "Failed to capture screenshot for interpretation",
                extra={"error": str(exc)},
            )
            return None, None, None

        screenshot_path = self._save_screenshot(
            screenshot,
            f"tc{test_case.test_id}_step{step.step_number}_context",
        )

        self._latest_screenshot_bytes = screenshot
        self._latest_screenshot_path = str(screenshot_path)
        if self._initial_screenshot_bytes is None:
            self._initial_screenshot_bytes = screenshot
            self._initial_screenshot_path = str(screenshot_path)
        self._latest_screenshot_origin = "fresh_capture"

        return screenshot, str(screenshot_path), "fresh_capture"
    
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
        self._initial_screenshot_bytes = None
        self._initial_screenshot_path = None
        self._latest_screenshot_bytes = None
        self._latest_screenshot_path = None
        self._latest_screenshot_origin = None
        self._disable_cache_for_case = {}
        try:
            self._plan_cache_source = getattr(test_state, "context", {}).get("plan_source", "live")
        except Exception:
            self._plan_cache_source = "live"
        
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
                "browser": "Desktop Firefox" if self._settings.desktop_mode_enabled else "Chromium",
                "execution_mode": "desktop" if self._settings.desktop_mode_enabled else "enhanced",
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
            if initial_url and self.browser_driver and not self._settings.desktop_mode_enabled:
                logger.info("Navigating to initial URL", extra={"url": initial_url})
                await self.browser_driver.navigate(initial_url)
            elif self._settings.desktop_mode_enabled:
                logger.info(
                    "Desktop mode active; skipping initial navigation and assuming target window is already open."
                )
            
            await self._ensure_initial_screenshot()
            
            # Execute each test case sequentially
            for i, test_case in enumerate(test_plan.test_cases):
                case_result = await self._execute_test_case(test_case)
                
                # Save report after each test case
                # Report updates will be handled by the caller
                
                # Check if failure should cascade
                if case_result.status == TestStatus.FAILED:
                    # Check if the last failed step was a blocker
                    is_blocker = False
                    for step_data in reversed(self._current_test_case_actions.get("steps", [])):
                        if step_data.get("is_blocker", False):
                            is_blocker = True
                            logger.info("Found blocker failure in test case", extra={
                                "test_case": test_case.name,
                                "step": step_data.get("step_number"),
                                "reasoning": step_data.get("blocker_reasoning", "")
                            })
                            break
                    
                    if is_blocker:
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

        if (
            self._settings.run_cache_enabled
            and self._run_cache
            and self._run_signature
            and self._plan_cache_source == "cache"
            and getattr(self._test_report, "status", None) == TestStatus.FAILED
        ):
            try:
                self._run_cache.invalidate_plan(self._run_signature, "test_run_failed")
            except Exception:
                logger.debug("Failed to invalidate plan cache", exc_info=True)
        
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
                    
                    # Determine if we should continue based on verification
                    is_blocker = self._current_step_data.get("is_blocker", False)
                    if not step.optional and is_blocker:
                        logger.error("Blocker failure detected, stopping test case", extra={
                            "step_number": step.step_number,
                            "test_case": test_case.name,
                            "blocker_reasoning": self._current_step_data.get("blocker_reasoning", "")
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

        if case_result.status == TestStatus.FAILED:
            self._disable_cache_for_case[str(test_case.case_id)] = True
            self._invalidate_case_cache(test_case, "case_failed")
        
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
        attempt = 0
        use_cache = self._should_use_cache_for_case(test_case)
        last_step_result: Optional[StepResult] = None

        while True:
            self._current_step_actions = []
            if attempt == 0:
                self._current_step_data = {
                    "step_number": step.step_number,
                    "step_id": str(step.step_id),
                    "step_description": step.action,
                    "actions": self._current_step_actions,
                    "step_intent": step.intent.value,
                }
                self._current_test_case_actions["steps"].append(self._current_step_data)
            else:
                self._current_step_data["actions"] = self._current_step_actions
                self._current_step_data["retry_attempt"] = attempt
            self._current_step_data["cache_policy"] = "cache" if use_cache else "no_cache"
            
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
            screenshot_before: Optional[bytes] = None
            screenshot_after: Optional[bytes] = None
            should_retry = False
            record_history = True
           
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
                actions, interpretation_meta = await self._interpret_step(
                    step, test_case, case_result, use_cache=use_cache
                )
                self._current_step_data["interpretation_source"] = interpretation_meta.get("interpretation_source")
                self._current_step_data["interpretation_cache_key"] = interpretation_meta.get("cache_key")
                self._current_step_data["interpretation_screenshot_hash"] = interpretation_meta.get("screenshot_hash")
                
                # Execute each action
                success = True
                action_results = []  # Store full action results
                forced_blocker_reason: Optional[str] = None
                
                for action in actions:
                    logger.debug("Executing sub-action", extra={
                        "action_type": action["type"],
                        "description": action.get("description", "")
                    })
                    
                    action_result = await self._execute_action(action, step)
                    step_result.actions_performed.append(action_result)
                    
                    # Store full action data
                    action_results.append({
                        "action": action,
                        "result": action_result,
                        "full_data": self._current_step_actions[-1] if self._current_step_actions else {}
                    })
                    
                    if forced_blocker_reason is None:
                        error_text = action_result.get("error")
                        if not error_text:
                            full_data = action_results[-1]["full_data"]
                            if isinstance(full_data, dict):
                                result_blob = full_data.get("result", {})
                                if isinstance(result_blob, dict):
                                    exec_blob = result_blob.get("execution") or {}
                                    error_text = (
                                        result_blob.get("error")
                                        or exec_blob.get("error_message")
                                    )
                        if error_text and isinstance(error_text, str) and (
                            error_text.startswith(MAX_TURN_ERROR_PREFIX)
                            or error_text.startswith(LOOP_ERROR_PREFIX)
                        ):
                            forced_blocker_reason = error_text
                            success = False
                            logger.error(
                                "Action aborted due to Computer Use limit or loop",
                                extra={
                                    "step_number": step.step_number,
                                    "action_description": action.get("description", ""),
                                    "reason": error_text,
                                },
                            )
                            self._current_step_data["blocker_reasoning"] = error_text
                            self._current_step_data["forced_blocker_reason"] = error_text
                    
                    if not action_result.get("success", False):
                        success = False
                        
                        # Determine if we should continue with remaining actions
                        if action.get("critical", True):
                            break
                
                # Capture after screenshot
                if self.browser_driver:
                    await asyncio.sleep(1)  # Brief wait for UI to stabilize
                    screenshot_after = await self.browser_driver.screenshot()
                    screenshot_path = self._save_screenshot(
                        screenshot_after,
                        f"tc{test_case.test_id}_step{step.step_number}_after"
                    )
                    step_result.screenshot_after = str(screenshot_path)
                    self._latest_screenshot_bytes = screenshot_after
                    self._latest_screenshot_path = str(screenshot_path)
                    self._latest_screenshot_origin = f"step_{step.step_number}_after"
                
                # Build execution history for this test case
                execution_history = []
                for prev_step in case_result.step_results:
                    execution_history.append({
                        "step_number": prev_step.step_number,
                        "action": prev_step.action,
                        "status": prev_step.status,
                        "actual_result": prev_step.actual_result
                    })
                
                # Get next test case for blocker determination
                next_test_case = None
                if self._current_test_plan:
                    current_idx = None
                    for idx, tc in enumerate(self._current_test_plan.test_cases):
                        if tc.case_id == test_case.case_id:
                            current_idx = idx
                            break
                    if current_idx is not None and current_idx < len(self._current_test_plan.test_cases) - 1:
                        next_test_case = self._current_test_plan.test_cases[current_idx + 1]
                
                # Always produce a verification decision for reporting
                try:
                    if forced_blocker_reason:
                        verification = {
                            "verdict": "FAIL",
                            "reasoning": forced_blocker_reason,
                            "actual_result": forced_blocker_reason,
                            "confidence": 1.0,
                            "is_blocker": True,
                            "blocker_reasoning": forced_blocker_reason,
                        }
                        self._current_step_data["verification_mode"] = "runner_short_circuit"
                    elif step.intent == StepIntent.SETUP:
                        verification = self._evaluate_setup_step(success, action_results)
                        self._current_step_data["verification_mode"] = "runner_short_circuit"
                    else:
                        verification = await self._verify_expected_outcome(
                            test_case=test_case,
                            step=step,
                            action_results=action_results,
                            screenshot_before=screenshot_before,
                            screenshot_after=screenshot_after,
                            execution_history=execution_history,
                            next_test_case=next_test_case
                        )
                        self._current_step_data["verification_mode"] = "ai"
                    
                    # Store verification result for bug reporting
                    self._current_step_data["verification_result"] = verification
                    
                    # Set step result based on verdict
                    if verification["verdict"] == "PASS":
                        step_result.status = TestStatus.PASSED
                    else:
                        step_result.status = TestStatus.FAILED
                    
                    step_result.actual_result = verification["actual_result"]
                    step_result.error_message = verification["reasoning"] if verification["verdict"] == "FAIL" else None
                    step_result.confidence = verification.get("confidence", 0.0)
                    
                    # Store blocker status
                    if verification["verdict"] == "FAIL":
                        self._current_step_data["is_blocker"] = verification.get("is_blocker", False)
                        self._current_step_data["blocker_reasoning"] = verification.get("blocker_reasoning", "")
                    
                except Exception as e:
                    # AI verification failed - this is a fatal error
                    logger.error("AI verification failed - marking test as failed", extra={
                        "error": str(e),
                        "step_number": step.step_number,
                        "test_case": test_case.name
                    })
                    step_result.status = TestStatus.FAILED
                    step_result.actual_result = "Verification failed due to AI error"
                    step_result.error_message = f"AI verification failed: {str(e)}"
                    step_result.confidence = 0.0
                    # Re-raise to trigger the outer exception handler
                    raise

                if (
                    step_result.status == TestStatus.FAILED
                    and interpretation_meta.get("interpretation_source") == "cache"
                    and self._settings.run_cache_enabled
                    and self._run_cache
                    and self._run_signature
                ):
                    should_retry = True
                    record_history = False
                
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
                if (
                    record_history
                    and
                    screenshot_after is None
                    and screenshot_before is not None
                    and step_result.screenshot_before
                ):
                    self._latest_screenshot_bytes = screenshot_before
                    self._latest_screenshot_path = step_result.screenshot_before
                    self._latest_screenshot_origin = f"step_{step.step_number}_before"
                
                if record_history:
                    self._execution_history.append({
                        "test_case": test_case.name,
                        "step": step.step_number,
                        "action": step.action,
                        "result": step_result.status.value,
                        "timestamp": step_result.completed_at
                    })

            if should_retry:
                self._invalidate_step_cache(test_case, step, "execution_failure")
                self._disable_cache_for_case[str(test_case.case_id)] = True
                use_cache = False
                attempt += 1
                continue

            last_step_result = step_result
            if step_result.status == TestStatus.FAILED:
                self._disable_cache_for_case[str(test_case.case_id)] = True
            break
        
        return last_step_result if last_step_result else step_result
    
    @staticmethod
    def _evaluate_setup_step(
        success: bool,
        action_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Derive a verification verdict for setup-intent steps without AI calls."""
        if success:
            return {
                "verdict": "PASS",
                "reasoning": "Setup-only step completed without additional verification.",
                "actual_result": "Setup actions executed successfully.",
                "confidence": 1.0,
                "is_blocker": False,
                "blocker_reasoning": "",
            }
        
        failure_reason = "Critical setup action failed."
        for action_payload in action_results:
            result_payload = action_payload.get("result", {})
            if not result_payload.get("success", True):
                failure_reason = (
                    result_payload.get("error")
                    or result_payload.get("outcome")
                    or failure_reason
                )
                break
        
        return {
            "verdict": "FAIL",
            "reasoning": failure_reason,
            "actual_result": failure_reason,
            "confidence": 0.4,
            "is_blocker": False,
            "blocker_reasoning": "",
        }
    
    async def _interpret_step(
        self,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult,
        use_cache: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Interpret a test step and decompose it into executable actions.
        
        Returns a tuple of (actions, interpretation_metadata).
        """
        # Recent execution history (for temporal awareness)
        recent_history = []
        for item in self._execution_history[-3:]:
            recent_history.append({
                "test_case": item.get("test_case", ""),
                "step": item.get("step", 0),
                "action": item.get("action", ""),
                "result": item.get("result", ""),
                "timestamp": item.get("timestamp").isoformat()
                if isinstance(item.get("timestamp"), datetime)
                else str(item.get("timestamp", "")),
            })
        recent_history_text = json.dumps(recent_history, indent=2)

        # Determine position within the test case
        total_steps = len(test_case.steps)
        step_index = None
        for idx, candidate in enumerate(test_case.steps):
            if candidate.step_id == step.step_id:
                step_index = idx
                break
        if step_index is None:
            step_index = max(0, step.step_number - 1)

        def format_step_summary(
            step_obj: TestStep,
            result_obj: Optional[StepResult],
            include_status: bool = True,
        ) -> str:
            base = f"Step {step_obj.step_number}: {step_obj.action}\n  - Expected: {step_obj.expected_result}"
            if include_status:
                if result_obj:
                    status_value = result_obj.status.value.upper()
                    actual = result_obj.actual_result or "Not recorded"
                    base += f"\n  - Status: {status_value}\n  - Actual: {actual}"
                else:
                    base += "\n  - Status: NOT_EXECUTED\n  - Actual: N/A"
            return base

        previous_step_summary = "No previous steps in this test case."
        if step_index > 0:
            previous_step = test_case.steps[step_index - 1]
            previous_result = next(
                (sr for sr in case_result.step_results if sr.step_number == previous_step.step_number),
                None,
            )
            previous_step_summary = format_step_summary(previous_step, previous_result)

        next_step_summary = "This is the final step in this test case."
        if step_index < total_steps - 1:
            next_step = test_case.steps[step_index + 1]
            next_step_summary = format_step_summary(next_step, None, include_status=False)

        previous_case_summary = "No previous test cases or steps."
        if step_index == 0 and self._test_report:
            previous_case_result: Optional[TestCaseResult] = None
            for tc_result in self._test_report.test_cases:
                if tc_result.case_id == test_case.case_id:
                    break
                previous_case_result = tc_result

            if previous_case_result:
                last_step_definition: Optional[TestStep] = None
                if self._current_test_plan:
                    for candidate_case in self._current_test_plan.test_cases:
                        if candidate_case.case_id == previous_case_result.case_id:
                            if candidate_case.steps:
                                last_step_definition = candidate_case.steps[-1]
                            break

                if previous_case_result.step_results:
                    last_result = previous_case_result.step_results[-1]
                else:
                    last_result = None

                if last_step_definition:
                    previous_case_summary = (
                        f"Previous test case '{previous_case_result.name}' ended on "
                        f"{format_step_summary(last_step_definition, last_result)}"
                    )
                else:
                    status_value = (
                        previous_case_result.status.value.upper()
                        if isinstance(previous_case_result.status, TestStatus)
                        else str(previous_case_result.status)
                    )
                    previous_case_summary = (
                        f"Previous test case '{previous_case_result.name}' completed "
                        f"with overall status {status_value}."
                    )

        case_outline_lines = [
            f"Step {case_step.step_number}: {case_step.action} "
            f"(intent: {case_step.intent.value}, expected: {case_step.expected_result})"
            for case_step in test_case.steps
        ]
        case_outline_text = "\n".join(f"- {line}" for line in case_outline_lines)

        screenshot_bytes, screenshot_path, screenshot_source = await self._get_interpretation_screenshot(
            step,
            test_case,
        )
        interpretation_meta: Dict[str, Any] = {
            "interpretation_source": "live",
            "cache_key": None,
            "screenshot_hash": None,
        }
        screenshot_b64: Optional[str] = None
        if screenshot_bytes:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        screenshot_hash = PersistentRunCache.hash_bytes(screenshot_bytes)
        interpretation_meta["screenshot_hash"] = screenshot_hash

        if use_cache and self._should_use_cache_for_case(test_case):
            try:
                cached = self._run_cache.get_step_actions(
                    signature=self._run_signature,
                    case_id=str(test_case.case_id),
                    step_number=step.step_number,
                    screenshot_hash=screenshot_hash,
                    model=self.model,
                ) if self._run_cache else None
            except Exception:
                cached = None
                logger.debug("Step cache lookup failed", exc_info=True)

            if cached:
                logger.info(
                    "Using cached step interpretation",
                    extra={
                        "test_case": test_case.name,
                        "step_number": step.step_number,
                        "cache_key": cached.get("cache_key"),
                    },
                )
                interpretation_meta["interpretation_source"] = "cache"
                interpretation_meta["cache_key"] = cached.get("cache_key")
                return cached.get("actions", []), interpretation_meta

        action_surface = (
            "desktop actions" if self._settings.desktop_mode_enabled else "browser actions"
        )
        environment_note = (
            "Environment: OS-level Linux desktop control; interact with windows and "
            "applications directly, not just a browser surface."
            if self._settings.desktop_mode_enabled
            else "Environment: Chromium browser control."
        )

        prompt = f"""You are the HAINDY Test Runner's interpretation agent. Use the current UI snapshot and scenario context to plan the minimal {action_surface} needed for the next step. You are preparing instructions for an automated Computer Use executor that will run them without further translation.

{environment_note}

Run & Screenshot Context:
- Test case: {test_case.test_id} – {test_case.name}
- Test case description: {test_case.description}
- Step position: {step.step_number} of {total_steps} (intent: {step.intent.value})
- Screenshot path: {screenshot_path or "unavailable"}
- Screenshot source: {screenshot_source or "unknown"}

Previous step summary:
{previous_step_summary}

Next step preview:
{next_step_summary}

Previous test case context:
{previous_case_summary}

Full test case outline:
{case_outline_text}

Recent execution history (most recent first):
{recent_history_text}

Computer Use executor manual (follow this precisely when writing prompts):
{self._computer_use_prompt_manual}

Guidelines:
1. Inspect the screenshot before planning navigation. If the required view is already visible, emit a single `skip_navigation` action that explains the evidence (leave computer_use_prompt empty in that case).
2. Provide high-level, outcome-focused actions. For text or form inputs, emit a single `type` action with the final value and let the Computer Use model handle focusing, clearing, or key presses—do not add helper clicks for the same control.
3. Only break a step into multiple actions when it truly touches different controls (e.g., separate date and time pickers). Otherwise, keep the entire outcome in one action so the executor can decide the mechanics.
4. Keep targets human-readable (no selectors) and ensure each action advances toward the expected result: {step.expected_result}.
5. Use the previous/next step context to stay aligned with the intended flow.
6. Every non-skip action must include a `computer_use_prompt` that is ready to send directly to the Computer Use model—no additional wrapping will be added later.

Action schema for each entry (JSON object):
- type: One of [navigate, click, type, assert, key_press, scroll_to_element, scroll_by_pixels, scroll_to_top, scroll_to_bottom, scroll_horizontal, skip_navigation].
  • Use `skip_navigation` only when navigation is already satisfied; do not provide a value.
- target: Human description of the element or high-level goal.
- value: Required only when the action type needs input (navigate URL, type text, key_press key, scroll_by_pixels amount).
- description: Outcome-focused explanation so the Action Agent knows what success looks like.
- critical: Whether failure should halt remaining actions (true/false).
- expected_outcome (optional): Override the step-level expected result only if needed.
- computer_use_prompt: String containing the final directive for the Computer Use model, constructed according to the manual above. Required unless type is `skip_navigation`.

Respond with a JSON object containing an "actions" array where every item follows this schema exactly."""

        message_content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if screenshot_b64:
            message_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                }
            )

        interpretation_context_payload = {
            "screenshot_path": screenshot_path,
            "screenshot_source": screenshot_source,
            "previous_step_summary": previous_step_summary,
            "next_step_summary": next_step_summary,
            "previous_test_case": previous_case_summary,
            "case_outline": case_outline_lines,
            "recent_history": recent_history,
        }
        if hasattr(self, "_current_step_data"):
            self._current_step_data["interpretation_context"] = interpretation_context_payload

        # Log what we're sending to AI
        logger.info("Interpreting step with AI", extra={
            "step_number": step.step_number,
            "action": step.action,
            "expected_result": step.expected_result,
            "intent": step.intent.value,
            "prompt_length": len(prompt),
            "screenshot_path": screenshot_path,
            "screenshot_source": screenshot_source,
        })
        
        try:
            
            try:
                response = await self.call_openai(
                    messages=[{"role": "user", "content": message_content}],
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
                        "response": response.get("content", {}),
                        "screenshot_path": screenshot_path,
                        "screenshot_source": screenshot_source,
                        "context": interpretation_context_payload,
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
            
            if (
                self._settings.run_cache_enabled
                and self._run_cache
                and self._run_signature
            ):
                try:
                    cache_key = self._run_cache.set_step_actions(
                        signature=self._run_signature,
                        case_id=str(test_case.case_id),
                        step_number=step.step_number,
                        screenshot_hash=screenshot_hash,
                        model=self.model,
                        actions=actions,
                    ) if self._run_cache else None
                    interpretation_meta["cache_key"] = cache_key
                except Exception:
                    logger.debug("Failed to write step cache", exc_info=True)

            logger.info("Step interpretation successful", extra={
                "step": step.step_number,
                "original_action": step.action,
                "decomposed_actions": len(actions)
            })
            
            return actions, interpretation_meta
            
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
                expected_outcome=action.get("expected_outcome", step.expected_result),
                computer_use_prompt=action.get("computer_use_prompt"),
            )
            
            # Create a temporary TestStep for the action
            action_step = TestStep(
                step_number=step.step_number,
                description=action.get("description", step.description),
                action=action.get("description", step.action),
                expected_result=action.get("expected_outcome", step.expected_result),
                action_instruction=instruction,
                optional=not action.get("critical", True),
                intent=step.intent,
            )
            
            # Build context
            test_context = {
                "test_plan_name": self._current_test_plan.name,
                "test_case_name": self._current_test_case.name,
                "step_number": step.step_number,
                "action_description": action.get("description", ""),
                "recent_actions": self._execution_history[-3:],
                "step_intent": step.intent.value,
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
        test_case: TestCase,
        step: TestStep,
        action_results: List[Dict[str, Any]],
        screenshot_before: Optional[bytes],
        screenshot_after: Optional[bytes],
        execution_history: List[Dict[str, Any]],
        next_test_case: Optional[TestCase]
    ) -> Dict[str, Any]:
        """Use AI to verify if expected outcome was achieved with full context."""
        
        # Build execution history context
        history_context = []
        for hist_item in execution_history:
            status_emoji = "✓" if hist_item.get("status") == TestStatus.PASSED else "✗"
            history_context.append(
                f"Step {hist_item.get('step_number')}: {hist_item.get('action')} - {status_emoji} {hist_item.get('status')}\n"
                f"  Result: {hist_item.get('actual_result', 'N/A')}"
            )
        
        # Build detailed action results context
        actions_context = []
        for idx, action_data in enumerate(action_results, 1):
            action = action_data.get("action", {})
            result = action_data.get("result", {})
            
            # Extract validation details
            validation = result.get("validation", {})
            ai_analysis = result.get("ai_analysis", {})
            execution = result.get("execution", {})
            
            action_detail = f"""Action {idx}: {action.get('description', 'Unknown action')}
  Type: {action.get('type', 'unknown')}
  Target: {action.get('target', 'N/A')}
  Success: {result.get('success', False)}
  
  Validation Results:"""
            
            # Add validation fields if present
            if validation:
                for key, value in validation.items():
                    if key not in ['grid_cell', 'offset']:  # Skip coordinate data
                        action_detail += f"\n    {key}: {value}"
            
            # Add AI analysis
            if ai_analysis:
                action_detail += f"\n  \n  AI Analysis:"
                action_detail += f"\n    Reasoning: {ai_analysis.get('reasoning', 'N/A')}"
                action_detail += f"\n    Actual outcome: {ai_analysis.get('actual_outcome', 'N/A')}"
                action_detail += f"\n    Confidence: {ai_analysis.get('confidence', 0.0)}"
            
            # Add execution details
            if execution:
                action_detail += f"\n  \n  Execution Details:"
                action_detail += f"\n    Duration: {execution.get('duration_ms', 'N/A')}ms"
                if execution.get('error_message'):
                    action_detail += f"\n    Error: {execution.get('error_message')}"
            
            actions_context.append(action_detail)
        
        # Build the prompt with screenshots
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""I'm executing a test case: "{test_case.name}"
Test case description: {test_case.description or 'N/A'}

Previous steps in this test case:
{chr(10).join(history_context) if history_context else 'None'}

Current step to validate:
Step {step.step_number}: {step.action}
Expected result: {step.expected_result}

Actions performed:
{chr(10).join(actions_context)}

Based on all this information:

1. Did this step achieve its intended purpose? Consider the validation results and reasoning from the action execution, not just literal text matching. Look at the overall intent of the step and whether it was accomplished.

2. Is this failure (if failed) a blocker that would prevent the next test case from running successfully?
   Next test case: {next_test_case.name if next_test_case else 'None (last test case)'}
   (Consider: Does this failure leave the system in a state where the next test case cannot execute meaningfully?)

Respond with JSON:
{{
  "verdict": "PASS" or "FAIL",
  "reasoning": "Your analysis of why the step passed or failed",
  "actual_result": "Concise description of what actually happened",
  "confidence": 0.0-1.0,
  "is_blocker": true/false,
  "blocker_reasoning": "Why this would/wouldn't block the next test case"
}}"""
                    }
                ]
            }
        ]
        
        # Add screenshots if available
        if screenshot_before:
            messages[0]["content"].insert(1, {
                "type": "text",
                "text": "\nScreenshot before actions:"
            })
            messages[0]["content"].insert(2, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(screenshot_before).decode()}"
                }
            })
        
        if screenshot_after:
            messages[0]["content"].append({
                "type": "text", 
                "text": "\nScreenshot after actions:"
            })
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(screenshot_after).decode()}"
                }
            })

        try:
            response = await self.call_openai(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.get("content", "{}")
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content
            
            # Ensure required fields and proper types
            if "verdict" not in result:
                result["verdict"] = "FAIL"
            if "reasoning" not in result:
                result["reasoning"] = "Verification failed - no reasoning provided"
            if "actual_result" not in result:
                result["actual_result"] = "Unknown outcome"
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "is_blocker" not in result:
                result["is_blocker"] = False
            if "blocker_reasoning" not in result:
                result["blocker_reasoning"] = ""
            
            # Log the verification
            logger.info("Step verification completed", extra={
                "step_number": step.step_number,
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "is_blocker": result["is_blocker"]
            })
            
            return result
            
        except Exception as e:
            logger.error("Failed to verify outcome with AI", extra={
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "step": step.step_number,
                "test_case": test_case.name
            })
            # Raise the exception - don't fallback
            raise

    async def _evaluate_bug_plan_context(
        self,
        bug_report: BugReport,
        test_case: TestCase,
        step: TestStep,
        verification_result: Dict[str, Any],
        initial_severity: BugSeverity,
    ) -> Optional[Dict[str, Any]]:
        """Ask the model to evaluate bug impact using full test plan context."""
        if not self._current_test_plan:
            return None

        plan_payload = self._current_test_plan.model_dump(mode="json")
        plan_json = json.dumps(plan_payload, indent=2)

        bug_payload = bug_report.model_dump(mode="json")
        bug_payload["severity"] = bug_report.severity.value
        bug_payload["initial_severity"] = initial_severity.value

        test_case_context = {
            "test_case_id": test_case.test_id,
            "name": test_case.name,
            "description": test_case.description,
            "priority": test_case.priority.value,
            "prerequisites": test_case.prerequisites,
            "postconditions": test_case.postconditions,
        }

        step_context = {
            "step_number": step.step_number,
            "action": step.action,
            "expected_result": step.expected_result,
            "optional": step.optional,
        }

        verification_context = {
            "verdict": verification_result.get("verdict"),
            "reasoning": verification_result.get("reasoning"),
            "actual_result": verification_result.get("actual_result"),
            "confidence": verification_result.get("confidence"),
            "is_blocker": verification_result.get("is_blocker"),
            "blocker_reasoning": verification_result.get("blocker_reasoning"),
        }

        prompt = (
            "You are a senior QA lead reviewing an automated test failure. "
            "Use the complete test plan to reason about downstream impact. "
            "If the failure prevents any remaining test cases from achieving their purpose, "
            "treat it as blocking.\n\n"
            "Test Plan (JSON):\n"
            f"{plan_json}\n\n"
            "Failed Test Case Context:\n"
            f"{json.dumps(test_case_context, indent=2)}\n\n"
            "Failed Step Context:\n"
            f"{json.dumps(step_context, indent=2)}\n\n"
            "Existing Bug Report:\n"
            f"{json.dumps(bug_payload, indent=2)}\n\n"
            "Verification Summary:\n"
            f"{json.dumps(verification_context, indent=2)}\n\n"
            "Respond with JSON using this schema:\n"
            "{\n"
            '  "severity": "critical|high|medium|low",\n'
            '  "should_block": true|false,\n'
            '  "blocker_reason": "Why later cases cannot proceed (or empty)",\n'
            '  "notes": "Additional context for the report",\n'
            '  "recommended_actions": ["Optional suggestions"]\n'
            "}"
        )

        response = await self.call_openai(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.get("content", "{}")
        if isinstance(content, str):
            return json.loads(content)
        return content
    
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
        
        # Get verification result if available
        verification_result = self._current_step_data.get("verification_result", {})
        
        # Use AI to determine error type and severity
        prompt = f"""Analyze this test failure and create a bug report:

Test Case: {test_case.name}
Failed Step: {step.action}
Expected Result: {step.expected_result}

Verification Results:
- Verdict: {verification_result.get('verdict', 'FAIL')}
- Reasoning: {verification_result.get('reasoning', step_result.error_message)}
- Actual Result: {verification_result.get('actual_result', step_result.actual_result)}
- Is Blocker: {verification_result.get('is_blocker', False)}
- Blocker Reasoning: {verification_result.get('blocker_reasoning', 'N/A')}

Error Details: {step_result.error_message}
Step is Optional: {step.optional}

Determine:
1. error_type: One of (element_not_found, assertion_failed, timeout, navigation_error, api_error, validation_error, unknown_error)
2. severity: One of (critical, high, medium, low)
   - If is_blocker=true, severity should be at least "high"
   - critical: Blocks all testing, core functionality broken
   - high: Major feature broken, blocks test case
   - medium: Feature partially working, workaround possible
   - low: Minor issue, cosmetic or edge case
3. bug_description: A clear, concise description for developers

Respond in JSON format with keys: error_type, severity, bug_description, reasoning"""

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
        for i, step_res in enumerate(case_result.step_results[-3:]):
            if step_res.status == TestStatus.PASSED:
                reproduction_steps.append(
                    f"{i+3}. Previous step completed: Step {step_res.step_number}"
                )
        
        reproduction_steps.append(
            f"{len(reproduction_steps)+1}. Execute failing step: {step.action}"
        )
        
        # Use the bug description from AI or fallback
        bug_description = result.get("bug_description", f"Step {step.step_number} failed: {step.action}")
        
        # Build comprehensive error details including verification info
        error_details_parts = []
        if verification_result.get("reasoning"):
            error_details_parts.append(f"Verification reasoning: {verification_result['reasoning']}")
        if verification_result.get("is_blocker"):
            error_details_parts.append(f"Blocker: Yes - {verification_result.get('blocker_reasoning', 'N/A')}")
        else:
            error_details_parts.append("Blocker: No")
        if step_result.error_message:
            error_details_parts.append(f"Error message: {step_result.error_message}")
        
        bug_report = BugReport(
            step_id=step.step_id,
            test_case_id=test_case.case_id,
            test_plan_id=self._current_test_plan.plan_id,
            step_number=step.step_number,
            description=bug_description,
            severity=severity,
            error_type=error_type,
            expected_result=step.expected_result,
            actual_result=step_result.actual_result,
            screenshot_path=step_result.screenshot_after,
            error_details="\n".join(error_details_parts),
            reproduction_steps=reproduction_steps
        )

        # Enrich bug report with plan-level evaluation
        plan_assessment: Optional[Dict[str, Any]] = None
        try:
            plan_assessment = await self._evaluate_bug_plan_context(
                bug_report=bug_report,
                test_case=test_case,
                step=step,
                verification_result=verification_result,
                initial_severity=severity,
            )
        except Exception as e:
            logger.error("Plan-level bug assessment failed", extra={
                "error": str(e),
                "bug_id": str(bug_report.bug_id),
            })

        if plan_assessment:
            self._current_step_data["plan_level_assessment"] = plan_assessment

            plan_severity = plan_assessment.get("severity")
            if isinstance(plan_severity, str):
                plan_severity_enum = severity_map.get(plan_severity.lower())
                if plan_severity_enum:
                    bug_report.plan_recommended_severity = plan_severity_enum
                    if self._severity_rank(plan_severity_enum) < self._severity_rank(bug_report.severity):
                        bug_report.severity = plan_severity_enum
                else:
                    logger.warning(
                        "Plan-level assessment returned unknown severity",
                        extra={"severity": plan_severity},
                    )

            blocker_flag = plan_assessment.get("should_block")
            if blocker_flag is not None:
                bug_report.plan_blocker = bool(blocker_flag)
                if bug_report.plan_blocker:
                    reasoning = plan_assessment.get("blocker_reason") or "Plan-level assessment marked this failure as blocking."
                    bug_report.plan_blocker_reason = reasoning
                    self._current_step_data["is_blocker"] = True
                    self._current_step_data["blocker_reasoning"] = reasoning
                    bug_report.error_details = (
                        f"{bug_report.error_details}\nPlan-level blocker reasoning: {reasoning}"
                        if bug_report.error_details
                        else f"Plan-level blocker reasoning: {reasoning}"
                    )
                else:
                    non_block_reason = plan_assessment.get("blocker_reason")
                    if non_block_reason:
                        bug_report.plan_blocker_reason = non_block_reason
                        bug_report.error_details = (
                            f"{bug_report.error_details}\nPlan-level blocker reasoning: {non_block_reason}"
                            if bug_report.error_details
                            else f"Plan-level blocker reasoning: {non_block_reason}"
                        )

            notes = plan_assessment.get("notes")
            if isinstance(notes, str):
                bug_report.plan_assessment_notes = notes
                bug_report.error_details = (
                    f"{bug_report.error_details}\nPlan-level notes: {notes}"
                    if bug_report.error_details
                    else f"Plan-level notes: {notes}"
                )

            recommendations = plan_assessment.get("recommended_actions")
            if isinstance(recommendations, list):
                bug_report.plan_recommendations = [str(item) for item in recommendations]

        logger.info("Bug report created", extra={
            "bug_id": str(bug_report.bug_id),
            "severity": bug_report.severity.value,
            "error_type": error_type,
            "plan_blocker": bug_report.plan_blocker
        })

        return bug_report
    
    
    
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
