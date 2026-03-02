"""
Enhanced Test Runner Agent implementation for Phase 15.

This agent orchestrates test execution with intelligent step interpretation,
living document reporting, and comprehensive failure handling.
"""

import asyncio
import base64
import json
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agents.action_agent import ActionAgent
from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import TEST_RUNNER_SYSTEM_PROMPT
from src.config.settings import get_settings
from src.core.interfaces import AutomationDriver
from src.core.types import (
    ActionInstruction,
    ActionType,
    BugReport,
    BugSeverity,
    StepIntent,
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
from src.desktop.cache import CoordinateCache
from src.desktop.execution_replay import replay_driver_actions
from src.monitoring.logger import get_logger, get_run_id
from src.runtime.evidence import EvidenceManager
from src.runtime.execution_replay_cache import (
    ExecutionReplayCache,
    ExecutionReplayCacheKey,
)
from src.runtime.task_cache import TaskPlanCache
from src.runtime.trace import RunTraceWriter, load_model_calls_for_run
from src.utils.model_logging import get_model_logger

logger = get_logger(__name__)
MAX_TURN_ERROR_PREFIX = "Computer Use max turns exceeded"
LOOP_ERROR_PREFIX = "Computer Use loop detected"
REPLAY_CACHED_ACTION_MIN_STABILIZATION_WAIT_MS = 2000
REPLAY_VALIDATION_MODEL_WAIT_BUDGET_MS = 30000
REPLAY_VALIDATION_MODEL_WAIT_FALLBACK_MS = 1000
REPLAY_VALIDATION_ONLY_ACTION_TYPES: frozenset[str] = frozenset(
    {
        ActionType.ASSERT.value,
        ActionType.SKIP_NAVIGATION.value,
        ActionType.WAIT.value,
        ActionType.SCREENSHOT.value,
    }
)

COMPUTER_USE_PROMPT_MANUAL = """Computer Use execution context:
- Executor: OpenAI Computer Use tool powered by GPT-5.2 with medium reasoning effort.
- Environment: Existing runtime UI session (desktop browser/app or Android mobile screenshot) with screenshot-driven interaction.
- Inputs: Each prompt is delivered with the latest screenshot and scenario metadata; do not capture screenshots yourself.

Prompt construction rules:
1. Begin with a concise imperative goal that states the desired outcome.
2. Identify the UI target(s) using the exact labels a human sees (no CSS/XPath speculation).
3. Provide any text to enter or keys to press when relevant.
4. Restate the expected outcome based only on what should be *immediately visible* after the action completes — for example, a screen transition, a success message, or a new UI state. Never ask the executor to navigate to a different screen, open a profile, or take any additional step to confirm results; all deeper verification is handled by a separate evaluation pass.
5. Instruct the executor to act directly without seeking confirmation from the user.
6. Require a strategy shift after three identical failures where the UI shows no visible response to the action (button appears to do nothing, screen does not change at all). If any visible response is observed — including an error message, a loading indicator, navigation away, or any UI change — do not retry; report the observed outcome immediately and stop. Retries are only for when the tap or click appears to have had no effect whatsoever.
7. Tell it to rely on the provided screenshot for context and to scroll or refocus if elements are off-screen.
8. For observation-only (`assert`) actions, explicitly forbid interactions and request a visual verification summary instead.
9. Avoid backend assumptions, hidden DOM references, or multi-step checklists—each prompt should cover one cohesive action.
10. After the primary action completes (or fails), stop. Do not take additional navigation steps to verify account details, confirm identity, or validate data that is not immediately visible on screen.
11. When a step is about entering text into one specific field (e.g. typing a verification/OTP code, entering an email, filling a single input), do NOT instruct the executor to also tap a submit/confirm/send/reset button. Just fill the field and stop. Only include button-tap instructions when the step's explicit purpose is to submit the form or the step action text says to tap that button.

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
        automation_driver: AutomationDriver | None = None,
        action_agent: ActionAgent | None = None,
        **kwargs,
    ):
        """
        Initialize the Enhanced Test Runner.

        Args:
            name: Agent name
            automation_driver: Browser driver instance
            action_agent: Action agent for executing browser actions
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_RUNNER_SYSTEM_PROMPT
        self.automation_driver = automation_driver
        self.action_agent = action_agent

        # Current execution state
        self._current_test_plan: TestPlan | None = None
        self._current_test_case: TestCase | None = None
        self._current_test_step: TestStep | None = None
        self._test_state: TestState | None = None
        self._test_report: TestReport | None = None

        # Execution context
        self._initial_url: str | None = None
        self._execution_history: list[dict[str, Any]] = []
        self._settings = get_settings()
        self._task_plan_cache = TaskPlanCache(self._settings.task_plan_cache_path)
        self._execution_replay_cache = ExecutionReplayCache(
            self._settings.execution_replay_cache_path
        )
        self._environment = "desktop"
        self._model_logger = get_model_logger(
            self._settings.model_log_path,
            max_screenshots=getattr(self._settings, "max_screenshots", None),
        )
        run_id = get_run_id()
        if run_id == "unknown":
            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid4().hex[:8]
            )
        self._trace = RunTraceWriter(run_id)
        self._evidence: EvidenceManager | None = None
        if self.automation_driver and hasattr(
            self.automation_driver, "coordinate_cache"
        ):
            self._coordinate_cache = self.automation_driver.coordinate_cache
        else:
            self._coordinate_cache = CoordinateCache(
                self._settings.desktop_coordinate_cache_path
            )
        self._initial_screenshot_bytes: bytes | None = None
        self._initial_screenshot_path: str | None = None
        self._latest_screenshot_bytes: bytes | None = None
        self._latest_screenshot_path: str | None = None
        self._latest_screenshot_origin: str | None = None

        # Action storage for Phase 17
        self._action_storage: dict[str, Any] = {
            "test_plan_id": None,
            "test_run_timestamp": None,
            "test_cases": [],
        }
        self._current_test_case_actions: dict[str, Any] | None = None
        self._current_step_actions: list[dict[str, Any]] | None = None

    def _coordinate_cache_path_for_environment(self, environment: str) -> Path:
        if str(environment or "").strip().lower() == "mobile_adb":
            return self._settings.mobile_coordinate_cache_path
        return self._settings.desktop_coordinate_cache_path

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
        """Capture and cache the initial environment screenshot."""
        if self._initial_screenshot_bytes is not None:
            return
        if not self.automation_driver:
            return

        wait_seconds = max(
            float(self._settings.actions_computer_tool_stabilization_wait_ms) / 1000.0,
            0.0,
        )
        if wait_seconds:
            await asyncio.sleep(wait_seconds)

        try:
            screenshot = await self.automation_driver.screenshot()
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
            "Captured initial environment screenshot",
            extra={"screenshot_path": self._initial_screenshot_path},
        )

    async def _get_interpretation_screenshot(
        self,
        step: TestStep,
        test_case: TestCase,
    ) -> tuple[bytes | None, str | None, str | None]:
        """
        Resolve the screenshot to send alongside step interpretation.

        Returns a tuple of (screenshot_bytes, screenshot_path, source_label).
        """
        await self._ensure_initial_screenshot()

        if self._latest_screenshot_bytes and self._latest_screenshot_path:
            source = self._latest_screenshot_origin or "cached_snapshot"
            if source == "initial_state" and step.step_number > 1:
                source = "initial_state_cached"
            return (
                self._latest_screenshot_bytes,
                self._latest_screenshot_path,
                source,
            )

        if not self.automation_driver:
            return None, None, None

        try:
            screenshot = await self.automation_driver.screenshot()
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
        self, test_state: TestState, initial_url: str | None = None
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
        logger.info(
            "Starting enhanced test plan execution",
            extra={
                "test_plan_id": str(test_plan.plan_id),
                "test_plan_name": test_plan.name,
                "total_test_cases": len(test_plan.test_cases),
            },
        )

        # Initialize execution
        self._current_test_plan = test_plan
        self._initial_url = initial_url
        self._test_state = test_state
        backend = (
            str(test_state.context.get("automation_backend") or "").strip().lower()
        )
        target_type = str(test_state.context.get("target_type") or "").strip().lower()
        if backend == "mobile_adb":
            self._environment = "mobile_adb"
        elif target_type in {"web", "browser"}:
            self._environment = "browser"
        else:
            self._environment = "desktop"
        if not (
            self.automation_driver
            and hasattr(self.automation_driver, "coordinate_cache")
        ):
            self._coordinate_cache = CoordinateCache(
                self._coordinate_cache_path_for_environment(self._environment)
            )
        self._initial_screenshot_bytes = None
        self._initial_screenshot_path = None
        self._latest_screenshot_bytes = None
        self._latest_screenshot_path = None
        self._latest_screenshot_origin = None

        # Initialize action storage for this test run
        self._action_storage = {
            "test_plan_id": str(test_plan.plan_id),
            "test_run_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_cases": [],
        }

        # Initialize test report within the test state
        self._test_report = TestReport(
            test_plan_id=test_plan.plan_id,
            test_plan_name=test_plan.name,
            started_at=datetime.now(timezone.utc),
            status=TestStatus.IN_PROGRESS,
            environment={
                "initial_url": initial_url,
                "runtime": self._environment,
                "execution_mode": "enhanced",
            },
        )

        if self._trace:
            self._trace.set_run_metadata(
                {
                    "test_plan_id": str(test_plan.plan_id),
                    "test_plan_name": test_plan.name,
                    "initial_url": initial_url,
                    "automation_backend": self._environment,
                    "model_log_path": str(self._settings.model_log_path),
                    "task_plan_cache_path": str(self._settings.task_plan_cache_path),
                    "execution_replay_cache_path": str(
                        self._settings.execution_replay_cache_path
                    ),
                    "coordinate_cache_path": str(
                        self._coordinate_cache_path_for_environment(self._environment)
                    ),
                }
            )

        if self._test_report:
            screenshots_dir = None
            from src.monitoring.debug_logger import get_debug_logger

            debug_logger = get_debug_logger()
            if debug_logger and getattr(debug_logger, "debug_dir", None):
                screenshots_dir = str(debug_logger.debug_dir)
            else:
                screenshots_dir = str(self._settings.screenshots_dir)
            self._test_report.artifacts.update(
                {
                    "model_log_path": str(self._settings.model_log_path),
                    "trace_path": str(self._trace.path) if self._trace else None,
                    "task_plan_cache_path": str(self._settings.task_plan_cache_path),
                    "execution_replay_cache_path": str(
                        self._settings.execution_replay_cache_path
                    ),
                    "coordinate_cache_path": str(
                        self._coordinate_cache_path_for_environment(self._environment)
                    ),
                    "screenshots_dir": screenshots_dir,
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
            if initial_url and self.automation_driver:
                logger.info("Navigating to initial URL", extra={"url": initial_url})
                await self.automation_driver.navigate(initial_url)

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
                    for step_data in reversed(
                        self._current_test_case_actions.get("steps", [])
                    ):
                        if step_data.get("is_blocker", False):
                            is_blocker = True
                            logger.info(
                                "Found blocker failure in test case",
                                extra={
                                    "test_case": test_case.name,
                                    "step": step_data.get("step_number"),
                                    "reasoning": step_data.get("blocker_reasoning", ""),
                                },
                            )
                            break

                    if is_blocker:
                        logger.error(
                            "Test case failure blocks further execution",
                            extra={
                                "failed_test_case": test_case.name,
                                "remaining_test_cases": len(test_plan.test_cases)
                                - i
                                - 1,
                            },
                        )

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
                                error_message=f"Blocked due to failure of test case: {test_case.name}",
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
            logger.error(
                "Test execution failed with error",
                extra={"error": str(e), "test_plan": test_plan.name},
            )
            self._test_report.status = TestStatus.FAILED
            self._test_state.status = TestStatus.FAILED
            raise

        finally:
            # Final report will be saved by the caller using TestReporter
            if self._trace:
                try:
                    self._trace.set_model_calls(
                        load_model_calls_for_run(
                            self._settings.model_log_path,
                            run_id=self._trace.run_id,
                        )
                    )
                    success = (
                        self._test_report.status == TestStatus.PASSED
                        if self._test_report
                        else False
                    )
                    self._trace.finalize(success=success)
                    self._trace.write()
                    logger.info(
                        "Run trace written", extra={"path": str(self._trace.path)}
                    )
                except Exception:
                    logger.debug("Failed to write run trace", exc_info=True)

            # Print summary to console
            self._print_summary()

        return self._test_state

    async def _execute_test_case(self, test_case: TestCase) -> TestCaseResult:
        """Execute a single test case with all its steps."""
        logger.info(
            "Starting test case execution",
            extra={
                "test_case_id": test_case.test_id,
                "test_case_name": test_case.name,
                "priority": test_case.priority.value,
                "total_steps": len(test_case.steps),
            },
        )

        self._current_test_case = test_case

        # Initialize action tracking for this test case
        self._current_test_case_actions = {
            "test_case_id": str(test_case.case_id),
            "test_case_name": test_case.name,
            "steps": [],
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
            steps_failed=0,
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
                step_result = await self._execute_test_step(
                    step, test_case, case_result
                )
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
                        logger.error(
                            "Blocker failure detected, stopping test case",
                            extra={
                                "step_number": step.step_number,
                                "test_case": test_case.name,
                                "blocker_reasoning": self._current_step_data.get(
                                    "blocker_reasoning", ""
                                ),
                            },
                        )
                        case_result.status = TestStatus.FAILED
                        break

                # Save report after each step
                # Report updates will be handled by the caller

            # Determine final test case status
            if (
                case_result.status != TestStatus.FAILED
            ):  # Not already marked as failed by blocker
                if has_failed_steps:
                    case_result.status = TestStatus.FAILED
                    if not case_result.error_message:
                        case_result.error_message = (
                            f"{case_result.steps_failed} step(s) failed"
                        )
                elif await self._verify_postconditions(test_case.postconditions):
                    case_result.status = TestStatus.PASSED
                else:
                    case_result.status = TestStatus.FAILED
                    case_result.error_message = "Postconditions not met"

        except Exception as e:
            logger.error(
                "Test case execution failed",
                extra={"error": str(e), "test_case": test_case.name},
            )
            case_result.status = TestStatus.FAILED
            case_result.error_message = str(e)

        finally:
            case_result.completed_at = datetime.now(timezone.utc)

        return case_result

    async def _execute_test_step(
        self, step: TestStep, test_case: TestCase, case_result: TestCaseResult
    ) -> StepResult:
        """Execute a single test step with intelligent interpretation."""
        logger.info(
            "Executing test step",
            extra={
                "step_number": step.step_number,
                "action": step.action,
                "test_case": test_case.name,
            },
        )

        self._current_test_step = step

        # Initialize action tracking for this step
        self._current_step_actions = []
        self._current_step_data = {
            "step_number": step.step_number,
            "step_id": str(step.step_id),
            "step_description": step.action,
            "actions": self._current_step_actions,
            "step_intent": step.intent.value,
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
            actual_result="",
        )
        screenshot_before: bytes | None = None
        screenshot_after: bytes | None = None
        attempt = 1
        plan_cache_hit = False
        replay_used = False

        try:
            # Check dependencies
            if not self._check_dependencies(step, case_result):
                step_result.status = TestStatus.SKIPPED
                step_result.actual_result = "Skipped due to unmet dependencies"
                return step_result

            # Capture before screenshot
            if self.automation_driver:
                screenshot_before = await self.automation_driver.screenshot()
                screenshot_path = self._save_screenshot(
                    screenshot_before,
                    f"tc{test_case.test_id}_step{step.step_number}_before",
                )
                step_result.screenshot_before = str(screenshot_path)

            # Build execution history and next test case context (used for replay/verification)
            execution_history = []
            for prev_step in case_result.step_results:
                execution_history.append(
                    {
                        "step_number": prev_step.step_number,
                        "action": prev_step.action,
                        "status": prev_step.status,
                        "actual_result": prev_step.actual_result,
                    }
                )

            next_test_case = None
            if self._current_test_plan:
                current_idx = None
                for idx, tc in enumerate(self._current_test_plan.test_cases):
                    if tc.case_id == test_case.case_id:
                        current_idx = idx
                        break
                if (
                    current_idx is not None
                    and current_idx < len(self._current_test_plan.test_cases) - 1
                ):
                    next_test_case = self._current_test_plan.test_cases[current_idx + 1]

            replay_result = await self._try_execution_replay(
                step=step,
                test_case=test_case,
                step_result=step_result,
                screenshot_before=screenshot_before,
                execution_history=execution_history,
                next_test_case=next_test_case,
            )
            if replay_result is not None:
                replay_used = True
                return replay_result
            latest_action_results: list[dict[str, Any]] = []

            while True:
                actions, plan_cache_hit = await self._interpret_step(
                    step, test_case, case_result, use_cache=(attempt == 1)
                )

                # Execute each action
                success = True
                action_results: list[dict[str, Any]] = []
                forced_blocker_reason = None
                step_result.actions_performed = []

                for action in actions:
                    logger.debug(
                        "Executing sub-action",
                        extra={
                            "action_type": action["type"],
                            "description": action.get("description", ""),
                        },
                    )

                    action_result = await self._execute_action(
                        action, step, record_driver_actions=self._replay_enabled(step)
                    )
                    step_result.actions_performed.append(action_result)

                    # Store full action data
                    action_results.append(
                        {
                            "action": action,
                            "result": action_result,
                            "full_data": self._current_step_actions[-1]
                            if self._current_step_actions
                            else {},
                        }
                    )

                    if forced_blocker_reason is None:
                        error_text = action_result.get("error")
                        if not error_text:
                            full_data = action_results[-1]["full_data"]
                            if isinstance(full_data, dict):
                                result_blob = full_data.get("result", {})
                                if isinstance(result_blob, dict):
                                    exec_blob = result_blob.get("execution") or {}
                                    error_text = result_blob.get(
                                        "error"
                                    ) or exec_blob.get("error_message")
                        if (
                            error_text
                            and isinstance(error_text, str)
                            and (
                                error_text.startswith(MAX_TURN_ERROR_PREFIX)
                                or error_text.startswith(LOOP_ERROR_PREFIX)
                            )
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
                            self._current_step_data["forced_blocker_reason"] = (
                                error_text
                            )

                    if not action_result.get("success", False):
                        success = False

                        # Determine if we should continue with remaining actions
                        if action.get("critical", True):
                            break

                latest_action_results = action_results
                self._current_step_data["plan_cache_hit"] = plan_cache_hit

                # Capture after screenshot
                if self.automation_driver:
                    await asyncio.sleep(1)  # Brief wait for UI to stabilize
                    screenshot_after = await self.automation_driver.screenshot()
                    screenshot_path = self._save_screenshot(
                        screenshot_after,
                        f"tc{test_case.test_id}_step{step.step_number}_after",
                    )
                    step_result.screenshot_after = str(screenshot_path)
                    self._latest_screenshot_bytes = screenshot_after
                    self._latest_screenshot_path = str(screenshot_path)
                    self._latest_screenshot_origin = f"step_{step.step_number}_after"

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
                        self._current_step_data["verification_mode"] = (
                            "runner_short_circuit"
                        )
                    elif step.intent == StepIntent.SETUP:
                        verification = self._evaluate_setup_step(
                            success, action_results
                        )
                        self._current_step_data["verification_mode"] = (
                            "runner_short_circuit"
                        )
                    else:
                        verification = await self._verify_expected_outcome(
                            test_case=test_case,
                            step=step,
                            action_results=action_results,
                            screenshot_before=screenshot_before,
                            screenshot_after=screenshot_after,
                            execution_history=execution_history,
                            next_test_case=next_test_case,
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
                    step_result.error_message = (
                        verification["reasoning"]
                        if verification["verdict"] == "FAIL"
                        else None
                    )
                    step_result.confidence = verification.get("confidence", 0.0)

                    # Store blocker status
                    if verification["verdict"] == "FAIL":
                        self._current_step_data["is_blocker"] = verification.get(
                            "is_blocker", False
                        )
                        self._current_step_data["blocker_reasoning"] = verification.get(
                            "blocker_reasoning", ""
                        )

                except Exception as e:
                    # AI verification failed - this is a fatal error
                    logger.error(
                        "AI verification failed - marking test as failed",
                        extra={
                            "error": str(e),
                            "step_number": step.step_number,
                            "test_case": test_case.name,
                        },
                    )
                    step_result.status = TestStatus.FAILED
                    step_result.actual_result = "Verification failed due to AI error"
                    step_result.error_message = f"AI verification failed: {str(e)}"
                    step_result.confidence = 0.0
                    # Re-raise to trigger the outer exception handler
                    raise

                if (
                    verification["verdict"] == "PASS"
                    or not plan_cache_hit
                    or attempt >= 2
                ):
                    break

                cache_key = self._current_step_data.get("plan_cache_key")
                cache_context = self._current_step_data.get("plan_cache_context")
                if cache_key and cache_context:
                    logger.warning(
                        "Cached plan failed; invalidating and retrying without cache",
                        extra={"step": cache_key, "attempt": attempt},
                    )
                    try:
                        self._task_plan_cache.invalidate(cache_key, cache_context)
                        if self._trace:
                            self._trace.record_cache_event(
                                {
                                    "type": "task_plan_cache_invalidate",
                                    "scenario": self._current_test_plan.name
                                    if self._current_test_plan
                                    else "",
                                    "step": cache_key,
                                    "reason": "validation_failed_with_cached_plan",
                                }
                            )
                    except Exception:
                        logger.debug(
                            "Failed to invalidate task plan cache",
                            exc_info=True,
                        )

                attempt += 1

            cache_key = self._current_step_data.get("plan_cache_key")
            cache_context = self._current_step_data.get("plan_cache_context")
            if step_result.status == TestStatus.PASSED:
                if cache_key and cache_context and not plan_cache_hit:
                    try:
                        self._task_plan_cache.store(cache_key, cache_context, actions)
                        if self._trace:
                            self._trace.record_cache_event(
                                {
                                    "type": "task_plan_cache_store",
                                    "scenario": self._current_test_plan.name
                                    if self._current_test_plan
                                    else "",
                                    "step": cache_key,
                                }
                            )
                    except Exception:
                        logger.debug("Failed to store task plan cache", exc_info=True)

                await self._store_execution_replay(
                    step, test_case, latest_action_results
                )
                await self._persist_coordinate_cache(latest_action_results)
            else:
                await self._invalidate_coordinate_cache(latest_action_results)

        except Exception as e:
            logger.error(
                "Step execution failed",
                extra={"error": str(e), "step_number": step.step_number},
            )
            step_result.status = TestStatus.FAILED
            step_result.actual_result = f"Error: {str(e)}"
            step_result.error_message = str(e)

        finally:
            step_result.completed_at = datetime.now(timezone.utc)
            if (
                screenshot_after is None
                and screenshot_before is not None
                and step_result.screenshot_before
            ):
                self._latest_screenshot_bytes = screenshot_before
                self._latest_screenshot_path = step_result.screenshot_before
                self._latest_screenshot_origin = f"step_{step.step_number}_before"

            # Add to execution history
            self._execution_history.append(
                {
                    "test_case": test_case.name,
                    "step": step.step_number,
                    "action": step.action,
                    "result": step_result.status.value,
                    "timestamp": step_result.completed_at,
                }
            )

            if self._trace and not replay_used:
                self._trace.record_step(
                    scenario_name=self._current_test_plan.name
                    if self._current_test_plan
                    else "",
                    step=step,
                    step_result=step_result,
                    attempt=attempt,
                    plan_cache_hit=plan_cache_hit,
                )

        return step_result

    @staticmethod
    def _evaluate_setup_step(
        success: bool,
        action_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Interpret a test step and decompose it into executable actions.

        Returns a list of actions to be executed sequentially.
        """
        # Recent execution history (for temporal awareness)
        recent_history = []
        for item in self._execution_history[-3:]:
            recent_history.append(
                {
                    "test_case": item.get("test_case", ""),
                    "step": item.get("step", 0),
                    "action": item.get("action", ""),
                    "result": item.get("result", ""),
                    "timestamp": item.get("timestamp").isoformat()
                    if isinstance(item.get("timestamp"), datetime)
                    else str(item.get("timestamp", "")),
                }
            )
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
            result_obj: StepResult | None,
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
                (
                    sr
                    for sr in case_result.step_results
                    if sr.step_number == previous_step.step_number
                ),
                None,
            )
            previous_step_summary = format_step_summary(previous_step, previous_result)

        next_step_summary = "This is the final step in this test case."
        if step_index < total_steps - 1:
            next_step = test_case.steps[step_index + 1]
            next_step_summary = format_step_summary(
                next_step, None, include_status=False
            )

        previous_case_summary = "No previous test cases or steps."
        if step_index == 0 and self._test_report:
            previous_case_result: TestCaseResult | None = None
            for tc_result in self._test_report.test_cases:
                if tc_result.case_id == test_case.case_id:
                    break
                previous_case_result = tc_result

            if previous_case_result:
                last_step_definition: TestStep | None = None
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
            + (" [CURRENT STEP]" if case_step.step_number == step.step_number else "")
            for case_step in test_case.steps
        ]
        prereq_lines = (
            [f"  - {p}" for p in test_case.prerequisites]
            if test_case.prerequisites
            else []
        )
        prereq_prefix = (
            "Preconditions:\n" + "\n".join(prereq_lines) + "\n\n"
            if prereq_lines
            else ""
        )
        case_outline_text = prereq_prefix + "\n".join(
            f"- {line}" for line in case_outline_lines
        )

        cache_context = {
            "test_case_id": test_case.test_id,
            "test_case_name": test_case.name,
            "step_number": step.step_number,
            "step_action": step.action,
            "expected_result": step.expected_result,
            "intent": step.intent.value,
            "previous_step_summary": previous_step_summary,
            "next_step_summary": next_step_summary,
            "previous_test_case": previous_case_summary,
            "case_outline": case_outline_lines,
        }
        step_cache_key = self._plan_cache_key(step, test_case)
        if hasattr(self, "_current_step_data"):
            self._current_step_data["plan_cache_key"] = step_cache_key
            self._current_step_data["plan_cache_context"] = cache_context
        if use_cache:
            cached_actions = self._task_plan_cache.lookup(step_cache_key, cache_context)
            if cached_actions:
                logger.info(
                    "Using cached action plan for step",
                    extra={"step": step_cache_key, "action_count": len(cached_actions)},
                )
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "task_plan_cache_hit",
                            "scenario": self._current_test_plan.name
                            if self._current_test_plan
                            else "",
                            "step": step_cache_key,
                        }
                    )
                if hasattr(self, "_current_step_data"):
                    self._current_step_data["plan_cache_hit"] = True
                    self._current_step_data["test_runner_interpretation"] = {
                        "prompt": None,
                        "response": {"actions": cached_actions},
                        "screenshot_path": None,
                        "screenshot_source": "plan_cache",
                        "context": cache_context,
                        "cache_key": step_cache_key,
                    }
                return cached_actions, True

        (
            screenshot_bytes,
            screenshot_path,
            screenshot_source,
        ) = await self._get_interpretation_screenshot(
            step,
            test_case,
        )
        screenshot_b64: str | None = None
        if screenshot_bytes:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")

        runtime_environment = self._environment or "desktop"
        viewport_hint = "unknown"
        if self.automation_driver:
            try:
                width, height = await self.automation_driver.get_viewport_size()
                viewport_hint = f"{width}x{height}"
            except Exception:
                viewport_hint = "unknown"
        interaction_hint = (
            "Android mobile application in screenshot-space coordinates"
            if runtime_environment == "mobile_adb"
            else "desktop/web UI in screenshot-space coordinates"
        )
        environment_specific_guidance = ""
        if runtime_environment == "mobile_adb":
            environment_specific_guidance = """
Mobile-specific constraints:
- This run targets Android mobile UI. Do not use desktop/browser navigation assumptions.
- Do NOT propose keyboard/browser shortcuts like Alt+Left, Alt+Right, Ctrl+L, Ctrl+Tab, or "browser back".
- Prefer tap/swipe interactions and Android-safe navigation.
- If back navigation is required, use `key_press` with value "back" or tap a visible in-app/system back control.
""".strip()

        prompt = f"""You are the HAINDY Test Runner's interpretation agent. Use the current UI snapshot and scenario context to plan the minimal actions needed for the next step. You are preparing instructions for an automated Computer Use executor that will run them without further translation.

Run & Screenshot Context:
- Test case: {test_case.test_id} – {test_case.name}
- Test case description: {test_case.description}
- Step position: {step.step_number} of {total_steps} (intent: {step.intent.value})
- Runtime backend: {runtime_environment}
- Interaction mode: {interaction_hint}
- Viewport hint: {viewport_hint}
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
{COMPUTER_USE_PROMPT_MANUAL}

Guidelines:
1. Inspect the screenshot before planning navigation. If the required view is already visible, emit a single `skip_navigation` action that explains the evidence (leave computer_use_prompt empty in that case).
2. Provide high-level, outcome-focused actions. For text or form inputs, emit a single `type` action with the final value and let the Computer Use model handle focusing, clearing, or key presses—do not add helper clicks for the same control.
3. Only break a step into multiple actions when it truly touches different controls (e.g., separate date and time pickers). Otherwise, keep the entire outcome in one action so the executor can decide the mechanics.
4. Keep targets human-readable (no selectors) and ensure each action advances toward the expected result: {step.expected_result}.
5. Use the previous/next step context to stay aligned with the intended flow.
6. Every non-skip action must include a `computer_use_prompt` that is ready to send directly to the Computer Use model—no additional wrapping will be added later.
7. You are planning actions for the step marked [CURRENT STEP] ONLY. Do not plan actions for any other step. Even if the screenshot appears to show a later step's target already populated or completed, still execute the current step's action on the correct target — the visual state may reflect autofill, prior test state, or an incorrect field.

{environment_specific_guidance}

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

        message_content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
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
            self._current_step_data["interpretation_context"] = (
                interpretation_context_payload
            )
            self._current_step_data["plan_cache_hit"] = False
            self._current_step_data["plan_cache_key"] = step_cache_key

        # Log what we're sending to AI
        logger.info(
            "Interpreting step with AI",
            extra={
                "step_number": step.step_number,
                "action": step.action,
                "expected_result": step.expected_result,
                "intent": step.intent.value,
                "prompt_length": len(prompt),
                "screenshot_path": screenshot_path,
                "screenshot_source": screenshot_source,
            },
        )

        try:
            try:
                response = await self.call_openai(
                    messages=[{"role": "user", "content": message_content}],
                    response_format={"type": "json_object"},
                )

                log_message_content = [{"type": "input_text", "text": prompt}]
                if screenshot_bytes:
                    log_message_content.append(
                        {"type": "input_image", "image_url": "<<attached screenshot>>"}
                    )
                await self._model_logger.log_call(
                    agent="test_runner.interpret_step",
                    model=self.model,
                    prompt=prompt,
                    request_payload={
                        "messages": [{"role": "user", "content": log_message_content}],
                        "response_format": {"type": "json_object"},
                    },
                    response=response,
                    screenshots=(
                        [("test_runner_interpretation", screenshot_bytes)]
                        if screenshot_bytes
                        else None
                    ),
                    metadata={
                        "step_number": step.step_number,
                        "test_case": test_case.name,
                        "cache_key": step_cache_key,
                    },
                )

                logger.debug(
                    "OpenAI API call successful",
                    extra={
                        "response_type": type(response).__name__,
                        "response_keys": list(response.keys())
                        if isinstance(response, dict)
                        else None,
                    },
                )

                # Store the Test Runner interpretation conversation
                if hasattr(self, "_current_step_data"):
                    self._current_step_data["test_runner_interpretation"] = {
                        "prompt": prompt,
                        "response": response.get("content", {}),
                        "screenshot_path": screenshot_path,
                        "screenshot_source": screenshot_source,
                        "context": interpretation_context_payload,
                    }

            except Exception as api_error:
                logger.error(
                    "OpenAI API call failed",
                    extra={
                        "api_error": str(api_error),
                        "api_error_type": type(api_error).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
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
                raise ValueError(
                    f"AI failed to provide actions for step {step.step_number}: {step.action}"
                )

            logger.info(
                "Step interpretation successful",
                extra={
                    "step": step.step_number,
                    "original_action": step.action,
                    "decomposed_actions": len(actions),
                },
            )

            return actions, False

        except Exception as e:
            logger.error(
                "Failed to interpret step with AI",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "step": step.step_number,
                    "action": step.action,
                    "traceback": traceback.format_exc(),
                },
            )
            # Re-raise - no fallback, AI failure is fatal
            raise

    async def _execute_action(
        self,
        action: dict[str, Any],
        step: TestStep,
        record_driver_actions: bool = False,
    ) -> dict[str, Any]:
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
                "test_runner_interpretation": action.copy()
                if isinstance(action, dict)
                else None,
                "action_agent_execution": None,
            },
            "automation_calls": [],
            "result": None,
            "screenshots": {},
        }

        try:
            if not self.action_agent or not self.automation_driver:
                error_result = {
                    "success": False,
                    "error": "Action agent or automation driver not available",
                }
                action_data["result"] = error_result
                action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
                self._current_step_actions.append(action_data)
                return error_result

            # If driver supports capture, start capturing calls
            if hasattr(self.automation_driver, "start_capture"):
                self.automation_driver.start_capture()

            # Create ActionInstruction for the action
            instruction = ActionInstruction(
                action_type=ActionType(action_type),
                description=action.get("description", ""),
                target=action.get("target", ""),
                value=action.get("value"),
                expected_outcome=action.get("expected_outcome", step.expected_result),
                computer_use_prompt=action.get("computer_use_prompt"),
            )
            resolved_environment = (
                str(step.environment or self._environment or "desktop").strip().lower()
            )
            if resolved_environment not in {"desktop", "browser", "mobile_adb"}:
                resolved_environment = (
                    str(self._environment or "desktop").strip().lower() or "desktop"
                )
            resolved_target_type = (
                "mobile_adb"
                if resolved_environment == "mobile_adb"
                else "web"
                if resolved_environment == "browser"
                else "desktop_app"
            )
            state_context = (
                self._test_state.context
                if self._test_state and isinstance(self._test_state.context, dict)
                else {}
            )
            context_backend = (
                str(state_context.get("automation_backend") or "").strip().lower()
            )
            context_target_type = (
                str(state_context.get("target_type") or "").strip().lower()
            )
            automation_backend = context_backend or resolved_environment
            target_type = context_target_type or resolved_target_type

            # Create a temporary TestStep for the action
            action_step = TestStep(
                step_number=step.step_number,
                description=action.get("description", step.description),
                action=action.get("description", step.action),
                expected_result=action.get("expected_outcome", step.expected_result),
                action_instruction=instruction,
                optional=not action.get("critical", True),
                intent=step.intent,
                environment=resolved_environment,
            )

            # Build context
            test_context = {
                "test_plan_name": self._current_test_plan.name,
                "test_case_name": self._current_test_case.name,
                "step_number": step.step_number,
                "action_description": action.get("description", ""),
                "recent_actions": self._execution_history[-3:],
                "step_intent": step.intent.value,
                "automation_backend": automation_backend,
                "target_type": target_type,
                "environment": resolved_environment,
            }
            test_context["cache_label"] = (
                step.cache_label or action.get("target") or action.get("description")
            )
            test_context["cache_action"] = step.cache_action or "click"

            # Get screenshot before action
            screenshot = await self.automation_driver.screenshot()

            # Execute via Action Agent
            result = await self.action_agent.execute_action(
                test_step=action_step,
                test_context=test_context,
                screenshot=screenshot,
                record_driver_actions=record_driver_actions,
            )

            # Stop capturing automation calls
            if hasattr(self.automation_driver, "stop_capture"):
                action_data["automation_calls"] = self.automation_driver.stop_capture()

            # Extract AI conversation from Action Agent
            # The action agent stores conversation history
            if hasattr(self.action_agent, "conversation_history"):
                action_data["ai_conversation"]["action_agent_execution"] = {
                    "messages": self.action_agent.conversation_history.copy(),
                    "screenshot_path": None,  # Will be filled by debug logger
                }
                # Clear conversation history for next action
                self.action_agent.conversation_history = []

            # Store comprehensive result
            action_data["result"] = {
                "success": (result.validation.valid if result.validation else False)
                and (result.execution.success if result.execution else False),
                "validation": result.validation.model_dump()
                if result.validation
                else None,
                "coordinates": result.coordinates.model_dump()
                if result.coordinates
                else None,
                "execution": result.execution.model_dump()
                if result.execution
                else None,
                "ai_analysis": result.ai_analysis.model_dump()
                if result.ai_analysis
                else None,
                "cache": {
                    "label": result.cache_label,
                    "action": result.cache_action,
                    "hit": result.cache_hit,
                    "coordinates": result.cache_coordinates,
                    "resolution": result.cache_resolution,
                },
                "driver_actions": result.driver_actions,
            }

            # Store screenshot paths
            if (
                result.environment_state_before
                and result.environment_state_before.screenshot_path
            ):
                action_data["screenshots"]["before"] = (
                    result.environment_state_before.screenshot_path
                )
            if (
                result.environment_state_after
                and result.environment_state_after.screenshot_path
            ):
                action_data["screenshots"]["after"] = (
                    result.environment_state_after.screenshot_path
                )

            # Process result for compatibility
            success = (result.validation.valid if result.validation else False) and (
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
                "confidence": result.ai_analysis.confidence
                if result.ai_analysis
                else 0.0,
                "error": result.execution.error_message
                if (result.execution and not success)
                else None,
                "cache_label": result.cache_label,
                "cache_action": result.cache_action,
                "cache_hit": result.cache_hit,
                "cache_coordinates": result.cache_coordinates,
                "cache_resolution": result.cache_resolution,
                "driver_actions": result.driver_actions,
            }

            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            self._current_step_actions.append(action_data)

            return compatibility_result

        except Exception as e:
            logger.error(
                "Action execution failed",
                extra={"error": str(e), "action_type": action_type},
            )

            # Stop capturing if needed
            if hasattr(self.automation_driver, "stop_capture"):
                action_data["automation_calls"] = self.automation_driver.stop_capture()

            error_result = {
                "success": False,
                "action_type": action_type,
                "error": str(e),
            }

            action_data["result"] = error_result
            action_data["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            self._current_step_actions.append(action_data)

            return error_result

    async def _verify_expected_outcome(
        self,
        test_case: TestCase,
        step: TestStep,
        action_results: list[dict[str, Any]],
        screenshot_before: bytes | None,
        screenshot_after: bytes | None,
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
        replay_wait_budget_ms: int | None = None,
    ) -> dict[str, Any]:
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
            # CU agent's real-time narrative observation (may contain transient UI
            # feedback such as toast text that disappears before the evaluator screenshot)
            cu_outcome = result.get("outcome", "")

            action_detail = f"""Action {idx}: {action.get("description", "Unknown action")}
  Type: {action.get("type", "unknown")}
  Target: {action.get("target", "N/A")}
  Success: {result.get("success", False)}"""

            if cu_outcome and cu_outcome != "Action completed":
                action_detail += f"\n  CU agent observation: {cu_outcome}"

            action_detail += "\n\n  Validation Results:"

            # Add validation fields if present
            if validation:
                for key, value in validation.items():
                    if key not in [
                        "target_reference",
                        "pixel_coordinates",
                        "relative_x",
                        "relative_y",
                    ]:  # Skip coordinate data
                        action_detail += f"\n    {key}: {value}"

            # Add AI analysis
            if ai_analysis:
                action_detail += "\n  \n  AI Analysis:"
                action_detail += (
                    f"\n    Reasoning: {ai_analysis.get('reasoning', 'N/A')}"
                )
                action_detail += (
                    f"\n    Actual outcome: {ai_analysis.get('actual_outcome', 'N/A')}"
                )
                action_detail += (
                    f"\n    Confidence: {ai_analysis.get('confidence', 0.0)}"
                )

            # Add execution details
            if execution:
                action_detail += "\n  \n  Execution Details:"
                action_detail += (
                    f"\n    Duration: {execution.get('duration_ms', 'N/A')}ms"
                )
                if execution.get("error_message"):
                    action_detail += f"\n    Error: {execution.get('error_message')}"

            actions_context.append(action_detail)

        replay_wait_section = ""
        replay_wait_response_fields = ""
        if replay_wait_budget_ms is not None:
            remaining_budget_ms = max(int(replay_wait_budget_ms), 0)
            replay_wait_section = f"""
3. This validation is for a replayed cached action. Decide whether this step should wait longer before final failure.
   - Request additional wait ONLY when there is clear evidence the UI is still settling (for example loading indicators, transition overlays, in-flight navigation, or a partially updated state).
   - If evidence already supports a final PASS or final FAIL, do not request more wait.
   - Remaining wait budget for this step: {remaining_budget_ms} ms.
"""
            replay_wait_response_fields = """
  "request_additional_wait": true/false,
  "recommended_wait_ms": integer (0 when no wait requested; must be <= remaining budget),
  "wait_reasoning": "Why additional wait is or is not needed",
"""

        # Build the prompt with screenshots
        prompt_text = f"""I'm executing a test case: "{test_case.name}"
Test case description: {test_case.description or "N/A"}

Previous steps in this test case:
{chr(10).join(history_context) if history_context else "None"}

Current step to validate:
Step {step.step_number}: {step.action}
Expected result: {step.expected_result}

Actions performed:
{chr(10).join(actions_context)}

Based on all this information:

IMPORTANT: The "CU agent observation" fields above are real-time descriptions captured by the executor during the action, before the final screenshot was taken. Transient UI feedback such as toast messages, snackbars, and brief success banners may have auto-dismissed by the time the screenshot was captured. If the CU agent observation describes a success message or toast, treat that as strong evidence the action succeeded even if the message is no longer visible in the screenshot.

1. Did this step achieve its intended purpose? Consider the validation results and reasoning from the action execution, not just literal text matching. Look at the overall intent of the step and whether it was accomplished.

2. Is this failure (if failed) a blocker that would prevent the next test case from running successfully?
   Next test case: {next_test_case.name if next_test_case else "None (last test case)"}
   (Consider: Does this failure leave the system in a state where the next test case cannot execute meaningfully?)
{replay_wait_section}

Respond with JSON:
{{
  "verdict": "PASS" or "FAIL",
  "reasoning": "Your analysis of why the step passed or failed",
  "actual_result": "Concise description of what actually happened",
  "confidence": 0.0-1.0,
{replay_wait_response_fields}  "is_blocker": true/false,
  "blocker_reasoning": "Why this would/wouldn't block the next test case"
}}"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ]

        # Add screenshots if available
        if screenshot_before:
            messages[0]["content"].insert(
                1, {"type": "text", "text": "\nScreenshot before actions:"}
            )
            messages[0]["content"].insert(
                2,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(screenshot_before).decode()}"
                    },
                },
            )

        if screenshot_after:
            messages[0]["content"].append(
                {"type": "text", "text": "\nScreenshot after actions:"}
            )
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(screenshot_after).decode()}"
                    },
                }
            )

        try:
            response = await self.call_openai(
                messages=messages, response_format={"type": "json_object"}
            )

            log_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
            if screenshot_before:
                log_messages[0]["content"].append(
                    {"type": "image_url", "image_url": "<<attached screenshot>>"}
                )
            if screenshot_after:
                log_messages[0]["content"].append(
                    {"type": "image_url", "image_url": "<<attached screenshot>>"}
                )

            screenshots: list[tuple[str, bytes]] = []
            if screenshot_before:
                screenshots.append(("verification_before", screenshot_before))
            if screenshot_after:
                screenshots.append(("verification_after", screenshot_after))

            await self._model_logger.log_call(
                agent="test_runner.verify_step",
                model=self.model,
                prompt=prompt_text,
                request_payload={
                    "messages": log_messages,
                    "response_format": {"type": "json_object"},
                },
                response=response,
                screenshots=screenshots or None,
                metadata={
                    "step_number": step.step_number,
                    "test_case": test_case.name,
                },
            )

            content = response.get("content", "{}")
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content

            # Ensure required fields and proper types
            if "verdict" not in result:
                result["verdict"] = "FAIL"
            result["verdict"] = str(result.get("verdict", "FAIL")).strip().upper()
            if result["verdict"] not in {"PASS", "FAIL"}:
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
            if replay_wait_budget_ms is not None:
                request_additional_wait = self._coerce_model_bool(
                    result.get("request_additional_wait", False)
                )
                recommended_wait_raw = result.get("recommended_wait_ms", 0)
                try:
                    recommended_wait_ms = int(recommended_wait_raw)
                except (TypeError, ValueError):
                    recommended_wait_ms = 0
                remaining_budget_ms = max(int(replay_wait_budget_ms), 0)
                if remaining_budget_ms <= 0:
                    request_additional_wait = False
                    recommended_wait_ms = 0
                else:
                    recommended_wait_ms = max(
                        0, min(recommended_wait_ms, remaining_budget_ms)
                    )
                    if not request_additional_wait:
                        recommended_wait_ms = 0
                result["request_additional_wait"] = request_additional_wait
                result["recommended_wait_ms"] = recommended_wait_ms
                if "wait_reasoning" not in result:
                    result["wait_reasoning"] = ""

            # Log the verification
            logger.info(
                "Step verification completed",
                extra={
                    "step_number": step.step_number,
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                    "is_blocker": result["is_blocker"],
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to verify outcome with AI",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "step": step.step_number,
                    "test_case": test_case.name,
                },
            )
            # Raise the exception - don't fallback
            raise

    async def _evaluate_bug_plan_context(
        self,
        bug_report: BugReport,
        test_case: TestCase,
        step: TestStep,
        verification_result: dict[str, Any],
        initial_severity: BugSeverity,
    ) -> dict[str, Any] | None:
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
        await self._model_logger.log_call(
            agent="test_runner.bug_plan_assessment",
            model=self.model,
            prompt=prompt,
            request_payload={
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            response=response,
            metadata={
                "step_number": step.step_number,
                "test_case": test_case.name,
            },
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
        case_result: TestCaseResult,
    ) -> BugReport | None:
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
- Verdict: {verification_result.get("verdict", "FAIL")}
- Reasoning: {verification_result.get("reasoning", step_result.error_message)}
- Actual Result: {verification_result.get("actual_result", step_result.actual_result)}
- Is Blocker: {verification_result.get("is_blocker", False)}
- Blocker Reasoning: {verification_result.get("blocker_reasoning", "N/A")}

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
                response_format={"type": "json_object"},
            )
            await self._model_logger.log_call(
                agent="test_runner.bug_report",
                model=self.model,
                prompt=prompt,
                request_payload={
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                response=response,
                metadata={
                    "step_number": step.step_number,
                    "test_case": test_case.name,
                },
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
                "low": BugSeverity.LOW,
            }
            severity = severity_map.get(
                result.get("severity", "medium").lower(), BugSeverity.MEDIUM
            )

            logger.debug(
                "AI bug classification",
                extra={
                    "error_type": error_type,
                    "severity": severity.value,
                    "reasoning": result.get("reasoning", ""),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to classify bug with AI",
                extra={"error": str(e), "step": step.step_number},
            )
            # Re-raise - AI failure is fatal
            raise

        # Build reproduction steps
        reproduction_steps = [
            f"1. Execute test case: {test_case.name}",
            f"2. Navigate to step {step.step_number}: {step.action}",
        ]

        # Add recent successful steps for context
        for i, step_res in enumerate(case_result.step_results[-3:]):
            if step_res.status == TestStatus.PASSED:
                reproduction_steps.append(
                    f"{i + 3}. Previous step completed: Step {step_res.step_number}"
                )

        reproduction_steps.append(
            f"{len(reproduction_steps) + 1}. Execute failing step: {step.action}"
        )

        # Use the bug description from AI or fallback
        bug_description = result.get(
            "bug_description", f"Step {step.step_number} failed: {step.action}"
        )

        # Build comprehensive error details including verification info
        error_details_parts = []
        if verification_result.get("reasoning"):
            error_details_parts.append(
                f"Verification reasoning: {verification_result['reasoning']}"
            )
        if verification_result.get("is_blocker"):
            error_details_parts.append(
                f"Blocker: Yes - {verification_result.get('blocker_reasoning', 'N/A')}"
            )
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
            reproduction_steps=reproduction_steps,
        )

        # Enrich bug report with plan-level evaluation
        plan_assessment: dict[str, Any] | None = None
        try:
            plan_assessment = await self._evaluate_bug_plan_context(
                bug_report=bug_report,
                test_case=test_case,
                step=step,
                verification_result=verification_result,
                initial_severity=severity,
            )
        except Exception as e:
            logger.error(
                "Plan-level bug assessment failed",
                extra={
                    "error": str(e),
                    "bug_id": str(bug_report.bug_id),
                },
            )

        if plan_assessment:
            self._current_step_data["plan_level_assessment"] = plan_assessment

            plan_severity = plan_assessment.get("severity")
            if isinstance(plan_severity, str):
                plan_severity_enum = severity_map.get(plan_severity.lower())
                if plan_severity_enum:
                    bug_report.plan_recommended_severity = plan_severity_enum
                    if self._severity_rank(plan_severity_enum) < self._severity_rank(
                        bug_report.severity
                    ):
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
                    reasoning = (
                        plan_assessment.get("blocker_reason")
                        or "Plan-level assessment marked this failure as blocking."
                    )
                    bug_report.plan_blocker_reason = reasoning
                    self._current_step_data["is_blocker"] = True
                    self._current_step_data["blocker_reasoning"] = reasoning
                    bug_report.error_details = (
                        f"{bug_report.error_details}\nPlan-level blocker reasoning: {reasoning}"
                        if bug_report.error_details
                        else f"Plan-level blocker reasoning: {reasoning}"
                    )
                else:
                    # Plan assessment explicitly says non-blocking — override any
                    # forced_blocker_reason that was set by max-turns or loop detection.
                    self._current_step_data["is_blocker"] = False
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
                bug_report.plan_recommendations = [
                    str(item) for item in recommendations
                ]

        logger.info(
            "Bug report created",
            extra={
                "bug_id": str(bug_report.bug_id),
                "severity": bug_report.severity.value,
                "error_type": error_type,
                "plan_blocker": bug_report.plan_blocker,
            },
        )

        return bug_report

    async def _verify_prerequisites(self, prerequisites: list[str]) -> bool:
        """Verify test case prerequisites are met."""
        if not prerequisites:
            return True

        # For now, log prerequisites and assume they're met
        # In future, could implement actual verification
        logger.info("Checking prerequisites", extra={"prerequisites": prerequisites})

        return True

    async def _verify_postconditions(self, postconditions: list[str]) -> bool:
        """Verify test case postconditions are met."""
        if not postconditions:
            return True

        # Take screenshot for verification
        if self.automation_driver:
            await self.automation_driver.screenshot()

            # Use AI to verify postconditions
            f"""Verify these postconditions are met based on the current state:

Postconditions:
{chr(10).join(f"- {pc}" for pc in postconditions)}

Analyze the screenshot and determine if all postconditions are satisfied.

Respond with JSON: {{"all_met": true/false, "details": ["condition: status", ...]}}"""

            # For now, assume postconditions are met
            # Full implementation would analyze screenshot with AI
            logger.info(
                "Checking postconditions", extra={"postconditions": postconditions}
            )

        return True

    def _check_dependencies(self, step: TestStep, case_result: TestCaseResult) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True

        # Check if all dependent steps completed successfully
        for dep_num in step.dependencies:
            # Find the step result for the dependency
            dep_result = next(
                (r for r in case_result.step_results if r.step_number == dep_num), None
            )

            if not dep_result or dep_result.status != TestStatus.PASSED:
                logger.warning(
                    "Step dependency not met",
                    extra={
                        "step": step.step_number,
                        "dependency": dep_num,
                        "dependency_status": dep_result.status.value
                        if dep_result
                        else "not_found",
                    },
                )
                return False

        return True

    def _determine_overall_status(self) -> TestStatus:
        """Determine overall test execution status."""
        if not self._test_report.test_cases:
            return TestStatus.FAILED

        # If any test case failed, overall status is failed
        failed_cases = [
            tc for tc in self._test_report.test_cases if tc.status == TestStatus.FAILED
        ]
        if failed_cases:
            return TestStatus.FAILED

        # If all completed, overall is completed
        all_completed = all(
            tc.status == TestStatus.PASSED for tc in self._test_report.test_cases
        )
        if all_completed:
            return TestStatus.PASSED

        # Otherwise, partial completion
        return TestStatus.SKIPPED

    def _calculate_summary(self) -> TestSummary:
        """Calculate test execution summary statistics."""
        total_cases = len(self._test_report.test_cases)
        completed_cases = sum(
            1 for tc in self._test_report.test_cases if tc.status == TestStatus.PASSED
        )
        failed_cases = sum(
            1 for tc in self._test_report.test_cases if tc.status == TestStatus.FAILED
        )

        total_steps = sum(tc.steps_total for tc in self._test_report.test_cases)
        completed_steps = sum(tc.steps_completed for tc in self._test_report.test_cases)
        failed_steps = sum(tc.steps_failed for tc in self._test_report.test_cases)

        # Count bugs by severity
        critical_bugs = sum(
            1 for bug in self._test_report.bugs if bug.severity == BugSeverity.CRITICAL
        )
        high_bugs = sum(
            1 for bug in self._test_report.bugs if bug.severity == BugSeverity.HIGH
        )
        medium_bugs = sum(
            1 for bug in self._test_report.bugs if bug.severity == BugSeverity.MEDIUM
        )
        low_bugs = sum(
            1 for bug in self._test_report.bugs if bug.severity == BugSeverity.LOW
        )

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
            execution_time_seconds=execution_time,
        )

    def _save_screenshot(self, screenshot: bytes, name: str) -> Path:
        """Save screenshot to disk and return path."""
        # Screenshots are now handled by the debug logger
        from src.monitoring.debug_logger import get_debug_logger

        debug_logger = get_debug_logger()
        if debug_logger:
            path = Path(debug_logger.save_screenshot(screenshot, name))
            self._register_evidence(path)
            return path
        # Fallback to temp directory
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(screenshot)
            path = Path(f.name)
            self._register_evidence(path)
            return path

    def _register_evidence(self, path: Path) -> None:
        if not path:
            return
        if self._evidence is None:
            self._evidence = EvidenceManager(
                path.parent, self._settings.max_screenshots
            )
        self._evidence.register([str(path)])

    @staticmethod
    def _plan_cache_key(step: TestStep, test_case: TestCase) -> str:
        return f"{test_case.test_id}:{step.step_number}:{step.action}".strip()

    @staticmethod
    def _strip_plan_fingerprint_volatile_fields(payload: Any) -> Any:
        """Remove unstable fields from plan payloads before hashing."""
        if isinstance(payload, dict):
            stripped: dict[str, Any] = {}
            for key, value in payload.items():
                if key in {"plan_id", "created_at", "case_id", "step_id"}:
                    continue
                stripped[key] = TestRunner._strip_plan_fingerprint_volatile_fields(
                    value
                )
            return stripped
        if isinstance(payload, list):
            return [
                TestRunner._strip_plan_fingerprint_volatile_fields(item)
                for item in payload
            ]
        return payload

    def _plan_fingerprint(self) -> str:
        if not self._current_test_plan:
            return ""
        payload = self._current_test_plan.model_dump(mode="json")
        stable_payload = self._strip_plan_fingerprint_volatile_fields(payload)
        serialized = json.dumps(
            stable_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_validation_only_action_result(result: dict[str, Any]) -> bool:
        action_payload = result.get("action")
        if not isinstance(action_payload, dict):
            return False
        action_type = str(action_payload.get("type") or "").strip().lower()
        if not action_type:
            return False
        return action_type in REPLAY_VALIDATION_ONLY_ACTION_TYPES

    @classmethod
    def _is_validation_only_step_result_set(
        cls, action_results: list[dict[str, Any]]
    ) -> bool:
        seen_action_type = False
        for result in action_results:
            action_payload = result.get("action")
            if not isinstance(action_payload, dict):
                continue
            action_type = str(action_payload.get("type") or "").strip().lower()
            if not action_type:
                continue
            seen_action_type = True
            if not cls._is_validation_only_action_result(result):
                return False
        return seen_action_type

    @staticmethod
    def _driver_actions_for_replay(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Accept both wrapped and direct action-result payload shapes."""
        driver_actions = result.get("driver_actions")
        if not isinstance(driver_actions, list):
            nested = result.get("result")
            if isinstance(nested, dict):
                nested_driver_actions = nested.get("driver_actions")
                if isinstance(nested_driver_actions, list):
                    driver_actions = nested_driver_actions
        if not isinstance(driver_actions, list):
            return []
        return [item for item in driver_actions if isinstance(item, dict)]

    def _replay_enabled(self, step: TestStep) -> bool:
        if not self._settings.enable_execution_replay_cache:
            return False
        if getattr(step, "loop", False):
            return False
        return self.automation_driver is not None

    def _replay_stabilization_wait_ms(self) -> int:
        """Return stabilization wait used only for replayed macro actions."""
        configured = int(
            getattr(self._settings, "actions_computer_tool_stabilization_wait_ms", 0)
        )
        return max(configured, REPLAY_CACHED_ACTION_MIN_STABILIZATION_WAIT_MS)

    @staticmethod
    def _coerce_model_bool(value: Any) -> bool:
        """Normalize model-provided boolean-like values."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return False

    async def _execution_replay_key(
        self, step: TestStep, test_case: TestCase
    ) -> ExecutionReplayCacheKey | None:
        if not self.automation_driver:
            return None
        try:
            (
                viewport_width,
                viewport_height,
            ) = await self.automation_driver.get_viewport_size()
        except Exception:
            logger.debug(
                "TestRunner: failed to read viewport for execution replay cache",
                exc_info=True,
            )
            return None
        environment = (step.environment or self._environment or "desktop").strip()
        keyboard_layout = getattr(self.automation_driver, "keyboard_layout", None) or (
            "android"
            if environment.lower() == "mobile_adb"
            else self._settings.desktop_keyboard_layout
        )
        return ExecutionReplayCacheKey(
            scenario=self._current_test_plan.name if self._current_test_plan else "",
            step=self._plan_cache_key(step, test_case),
            environment=environment,
            resolution=(int(viewport_width), int(viewport_height)),
            keyboard_layout=str(keyboard_layout),
            plan_fingerprint=self._plan_fingerprint(),
        )

    async def _try_execution_replay(
        self,
        *,
        step: TestStep,
        test_case: TestCase,
        step_result: StepResult,
        screenshot_before: bytes | None,
        execution_history: list[dict[str, Any]],
        next_test_case: TestCase | None,
    ) -> StepResult | None:
        if not self._replay_enabled(step):
            return None
        key = await self._execution_replay_key(step, test_case)
        if not key:
            return None
        entry = self._execution_replay_cache.lookup(key)
        if not entry:
            return None
        if self._trace:
            self._trace.record_cache_event(
                {
                    "type": "execution_replay_cache_hit",
                    "scenario": key.scenario,
                    "step": key.step,
                    "environment": key.environment,
                    "resolution": key.resolution,
                    "keyboard_layout": key.keyboard_layout,
                    "action_count": len(entry.actions),
                }
            )
        try:
            await replay_driver_actions(
                self.automation_driver,
                entry.actions,
                stabilization_wait_ms=self._replay_stabilization_wait_ms(),
                action_timeout_seconds=max(
                    self._settings.actions_computer_tool_action_timeout_ms / 1000.0,
                    0.5,
                ),
            )
        except Exception as exc:
            logger.warning(
                "TestRunner: execution replay failed; invalidating cache",
                extra={"step": key.step, "error": str(exc)},
            )
            try:
                self._execution_replay_cache.invalidate(key)
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "execution_replay_cache_invalidate",
                            "scenario": key.scenario,
                            "step": key.step,
                            "environment": key.environment,
                            "resolution": key.resolution,
                            "keyboard_layout": key.keyboard_layout,
                            "reason": "action_error",
                            "error": str(exc),
                        }
                    )
            except Exception:
                logger.debug(
                    "TestRunner: execution replay cache invalidation failed",
                    exc_info=True,
                )
            if self._trace:
                self._trace.record_cache_event(
                    {
                        "type": "execution_replay_fallback_to_cu",
                        "scenario": key.scenario,
                        "step": key.step,
                    }
                )
            return None

        replay_action = {
            "action": {
                "type": "execution_replay",
                "description": f"Replay cached actions for {key.step}",
            },
            "result": {
                "success": True,
                "outcome": f"Replayed {len(entry.actions)} cached driver action(s).",
                "driver_actions": entry.actions,
            },
            "full_data": {
                "action_type": "execution_replay",
                "result": {"success": True, "driver_actions": entry.actions},
            },
        }
        step_result.actions_performed = [
            {
                "success": True,
                "action_type": "execution_replay",
                "outcome": f"Replayed {len(entry.actions)} cached driver action(s).",
                "confidence": 1.0,
                "driver_actions": entry.actions,
            }
        ]
        self._current_step_actions.append(
            {
                "action_id": f"execution_replay_{uuid4()}",
                "action_type": "execution_replay",
                "target": "",
                "value": None,
                "description": replay_action["action"]["description"],
                "timestamp_start": datetime.now(timezone.utc).isoformat(),
                "timestamp_end": datetime.now(timezone.utc).isoformat(),
                "ai_conversation": {
                    "test_runner_interpretation": None,
                    "action_agent_execution": None,
                },
                "automation_calls": [],
                "result": replay_action["result"],
                "screenshots": {},
            }
        )

        screenshot_after = None
        if self.automation_driver:
            screenshot_after = await self.automation_driver.screenshot()
            screenshot_path = self._save_screenshot(
                screenshot_after,
                f"tc{test_case.test_id}_step{step.step_number}_after",
            )
            step_result.screenshot_after = str(screenshot_path)
            self._latest_screenshot_bytes = screenshot_after
            self._latest_screenshot_path = str(screenshot_path)
            self._latest_screenshot_origin = f"step_{step.step_number}_after"

        replay_wait_spent_ms = 0
        replay_wait_cycles = 0
        replay_wait_budget_ms = REPLAY_VALIDATION_MODEL_WAIT_BUDGET_MS

        while True:
            verification = await self._verify_expected_outcome(
                test_case=test_case,
                step=step,
                action_results=[replay_action],
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                execution_history=execution_history,
                next_test_case=next_test_case,
                replay_wait_budget_ms=replay_wait_budget_ms,
            )
            if verification.get("verdict") == "PASS":
                break

            request_additional_wait = self._coerce_model_bool(
                verification.get("request_additional_wait", False)
            )
            if not request_additional_wait or replay_wait_budget_ms <= 0:
                break

            recommended_wait_raw = verification.get("recommended_wait_ms", 0)
            try:
                recommended_wait_ms = int(recommended_wait_raw)
            except (TypeError, ValueError):
                recommended_wait_ms = 0
            wait_ms = recommended_wait_ms
            if wait_ms <= 0:
                wait_ms = min(
                    REPLAY_VALIDATION_MODEL_WAIT_FALLBACK_MS,
                    replay_wait_budget_ms,
                )
            wait_ms = max(0, min(wait_ms, replay_wait_budget_ms))
            if wait_ms <= 0:
                break

            logger.info(
                "Execution replay validation requested additional wait",
                extra={
                    "step": key.step,
                    "wait_ms": wait_ms,
                    "remaining_budget_before_wait_ms": replay_wait_budget_ms,
                    "wait_reasoning": verification.get("wait_reasoning", ""),
                },
            )
            await asyncio.sleep(wait_ms / 1000.0)
            replay_wait_budget_ms -= wait_ms
            replay_wait_spent_ms += wait_ms
            replay_wait_cycles += 1

            if self.automation_driver:
                screenshot_after = await self.automation_driver.screenshot()
                screenshot_path = self._save_screenshot(
                    screenshot_after,
                    (
                        f"tc{test_case.test_id}_step{step.step_number}"
                        f"_replay_wait_{replay_wait_cycles}_after"
                    ),
                )
                step_result.screenshot_after = str(screenshot_path)
                self._latest_screenshot_bytes = screenshot_after
                self._latest_screenshot_path = str(screenshot_path)
                self._latest_screenshot_origin = (
                    f"step_{step.step_number}_replay_wait_{replay_wait_cycles}_after"
                )
        self._current_step_data["verification_mode"] = "ai"

        self._current_step_data["replay_validation_wait_spent_ms"] = (
            replay_wait_spent_ms
        )
        self._current_step_data["replay_validation_wait_cycles"] = replay_wait_cycles
        self._current_step_data["replay_validation_wait_budget_remaining_ms"] = (
            replay_wait_budget_ms
        )
        self._current_step_data["verification_result"] = verification

        if verification["verdict"] == "PASS":
            step_result.status = TestStatus.PASSED
        else:
            step_result.status = TestStatus.FAILED

        step_result.actual_result = verification["actual_result"]
        step_result.error_message = (
            verification["reasoning"] if verification["verdict"] == "FAIL" else None
        )
        step_result.confidence = verification.get("confidence", 0.0)

        if verification["verdict"] == "FAIL":
            logger.info(
                "Execution replay validation failed; invalidating cache",
                extra={
                    "step": key.step,
                    "validation_reasoning": verification.get("reasoning"),
                    "replay_wait_spent_ms": replay_wait_spent_ms,
                    "replay_wait_cycles": replay_wait_cycles,
                },
            )
            try:
                self._execution_replay_cache.invalidate(key)
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "execution_replay_cache_invalidate",
                            "scenario": key.scenario,
                            "step": key.step,
                            "environment": key.environment,
                            "resolution": key.resolution,
                            "keyboard_layout": key.keyboard_layout,
                            "reason": "validation_failed",
                            "message": verification.get("reasoning"),
                            "replay_wait_spent_ms": replay_wait_spent_ms,
                            "replay_wait_cycles": replay_wait_cycles,
                        }
                    )
            except Exception:
                logger.debug(
                    "TestRunner: execution replay cache invalidation failed",
                    exc_info=True,
                )
            if self._trace:
                self._trace.record_cache_event(
                    {
                        "type": "execution_replay_fallback_to_cu",
                        "scenario": key.scenario,
                        "step": key.step,
                    }
                )
            return None

        if self._trace:
            self._trace.record_step(
                scenario_name=key.scenario,
                step=step,
                step_result=step_result,
                attempt=1,
                plan_cache_hit=None,
            )
        return step_result

    async def _store_execution_replay(
        self,
        step: TestStep,
        test_case: TestCase,
        action_results: list[dict[str, Any]],
    ) -> None:
        if not self._replay_enabled(step) or not action_results:
            return
        if self._is_validation_only_step_result_set(action_results):
            return
        recorded: list[dict] = []
        for result in action_results:
            recorded.extend(self._driver_actions_for_replay(result))
        if not recorded:
            return
        key = await self._execution_replay_key(step, test_case)
        if not key:
            return
        try:
            self._execution_replay_cache.store(key, recorded)
            if self._trace:
                self._trace.record_cache_event(
                    {
                        "type": "execution_replay_cache_store",
                        "scenario": key.scenario,
                        "step": key.step,
                        "environment": key.environment,
                        "resolution": key.resolution,
                        "keyboard_layout": key.keyboard_layout,
                        "action_count": len(recorded),
                    }
                )
        except Exception:
            logger.debug(
                "TestRunner: failed to store execution replay cache", exc_info=True
            )

    async def _persist_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        if not self._coordinate_cache or not action_results:
            return
        for result in action_results:
            label = result.get("cache_label")
            coords = result.get("cache_coordinates")
            action = result.get("cache_action") or "click"
            if not label or not coords:
                continue
            resolution = result.get("cache_resolution")
            if not resolution:
                if not self.automation_driver:
                    continue
                try:
                    resolution = await self.automation_driver.get_viewport_size()
                except Exception:
                    logger.debug(
                        "Failed to read viewport for cache persistence",
                        exc_info=True,
                    )
                    continue
            try:
                x, y = coords
                self._coordinate_cache.add(label, action, x, y, resolution)
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "coordinate_cache_add",
                            "scenario": self._current_test_plan.name
                            if self._current_test_plan
                            else "",
                            "step": self._current_step_data.get("plan_cache_key"),
                            "cache_label": label,
                            "cache_action": action,
                            "x": x,
                            "y": y,
                            "resolution": resolution,
                        }
                    )
            except Exception:
                logger.debug("Failed to append coordinate cache", exc_info=True)

    async def _invalidate_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        if not self._coordinate_cache or not action_results:
            return
        hits = [
            result
            for result in action_results
            if result.get("cache_hit") and result.get("cache_label")
        ]
        if not hits:
            return
        for result in hits:
            label = result.get("cache_label")
            action = result.get("cache_action") or "click"
            resolution = result.get("cache_resolution")
            if not resolution:
                if not self.automation_driver:
                    continue
                try:
                    resolution = await self.automation_driver.get_viewport_size()
                except Exception:
                    logger.debug(
                        "Failed to read viewport for cache invalidation",
                        exc_info=True,
                    )
                    continue
            try:
                self._coordinate_cache.invalidate(label, action, resolution)
                if self._trace:
                    self._trace.record_cache_event(
                        {
                            "type": "coordinate_cache_invalidate",
                            "scenario": self._current_test_plan.name
                            if self._current_test_plan
                            else "",
                            "step": self._current_step_data.get("plan_cache_key"),
                            "cache_label": label,
                            "cache_action": action,
                            "resolution": resolution,
                        }
                    )
            except Exception:
                logger.debug("Failed to invalidate coordinate cache", exc_info=True)

    def _print_summary(self) -> None:
        """Print test execution summary to console."""
        if not self._test_report or not self._test_report.summary:
            return

        s = self._test_report.summary

        print("\n" + "=" * 80)
        print(f"TEST EXECUTION SUMMARY: {self._test_report.test_plan_name}")
        print("=" * 80)

        print(f"\nStatus: {self._test_report.status.value.upper()}")
        print(f"Test Cases: {s.completed_test_cases}/{s.total_test_cases} completed")
        print(f"Steps: {s.completed_steps}/{s.total_steps} completed")
        print(f"Success Rate: {s.success_rate * 100:.1f}%")
        print(f"Execution Time: {s.execution_time_seconds:.1f}s")

        if self._test_report.bugs:
            print(f"\nBugs Found: {len(self._test_report.bugs)}")
            print(f"  Critical: {s.critical_bugs}")
            print(f"  High: {s.high_bugs}")
            print(f"  Medium: {s.medium_bugs}")
            print(f"  Low: {s.low_bugs}")

            # Show critical bugs
            critical = [
                b for b in self._test_report.bugs if b.severity == BugSeverity.CRITICAL
            ]
            if critical:
                print("\nCRITICAL BUGS:")
                for bug in critical:
                    print(f"  - {bug.description}")

        print("\n" + "=" * 80 + "\n")

    def get_action_storage(self) -> dict[str, Any]:
        """Return the captured action storage data."""
        return self._action_storage
