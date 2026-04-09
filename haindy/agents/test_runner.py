"""
Enhanced Test Runner Agent implementation for Phase 15.

This agent orchestrates test execution with intelligent step interpretation,
living document reporting, and comprehensive failure handling.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from haindy.agents.action_agent import ActionAgent
from haindy.agents.base_agent import BaseAgent
from haindy.agents.test_runner_artifacts import TestRunnerArtifacts
from haindy.agents.test_runner_bug_reports import (
    BugReportRequest,
    TestRunnerBugReportBuilder,
)
from haindy.agents.test_runner_step_processor import TestRunnerStepProcessor
from haindy.agents.test_runner_summary import TestRunnerSummary
from haindy.agents.test_runner_verifier import TestRunnerVerifier
from haindy.config.agent_prompts import TEST_RUNNER_SYSTEM_PROMPT
from haindy.config.settings import get_settings
from haindy.core.interfaces import AutomationDriver
from haindy.core.types import (
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
from haindy.desktop.cache import CoordinateCache
from haindy.monitoring.logger import get_logger, get_run_id
from haindy.runtime.environment import (
    coordinate_cache_path_for_environment,
    normalize_runtime_environment_name,
    resolve_runtime_environment,
)
from haindy.runtime.execution_replay_cache import (
    ExecutionReplayCache,
    ExecutionReplayCacheKey,
)
from haindy.runtime.execution_replay_service import ExecutionReplayService
from haindy.runtime.task_cache import TaskPlanCache
from haindy.runtime.trace import RunTraceWriter, load_model_calls_for_run
from haindy.utils.model_logging import get_model_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolModeTestProgress:
    """Progress snapshot for tool-call status polling."""

    current_step: str | None
    steps_total: int
    steps_completed: int
    steps_failed: int
    issues_found: dict[str, str]
    latest_screenshot_path: str | None
    actions_taken: int


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
        **kwargs: Any,
    ) -> None:
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
        if self.automation_driver and hasattr(
            self.automation_driver, "coordinate_cache"
        ):
            self._coordinate_cache = self.automation_driver.coordinate_cache
        else:
            self._coordinate_cache = CoordinateCache(
                self._settings.desktop_coordinate_cache_path
            )
        self._artifacts = TestRunnerArtifacts(self._settings, self.automation_driver)
        self._replay_service = ExecutionReplayService(
            settings=self._settings,
            automation_driver=self.automation_driver,
            execution_replay_cache=self._execution_replay_cache,
            coordinate_cache=self._coordinate_cache,
        )
        self._interpreter: Any | None = None
        self._executor: Any | None = None
        self._step_processor = TestRunnerStepProcessor(self)
        self._verifier = TestRunnerVerifier(self)
        self._summary_helper = TestRunnerSummary(self)

        # Action storage for Phase 17
        self._action_storage: dict[str, Any] = {
            "test_plan_id": None,
            "test_run_timestamp": None,
            "test_cases": [],
        }
        self._current_test_case_actions: dict[str, Any] | None = None
        self._current_step_actions: list[dict[str, Any]] | None = None
        self._current_step_data: dict[str, Any] = {}

    def _coordinate_cache_path_for_environment(self, environment: str) -> Path:
        return Path(coordinate_cache_path_for_environment(self._settings, environment))

    def _create_bug_report_builder(self) -> TestRunnerBugReportBuilder:
        """Build a bug-report helper using the runner's current model hooks."""
        return TestRunnerBugReportBuilder(
            model=self.model,
            call_model=self.call_model,
            model_logger=self._model_logger,
        )

    def _create_run_trace(self) -> RunTraceWriter:
        """Create a fresh trace writer for one execution."""
        run_id = get_run_id()
        if run_id == "unknown":
            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid4().hex[:8]
            )
        return RunTraceWriter(run_id)

    def get_tool_mode_progress(self) -> ToolModeTestProgress | None:
        """Return a lightweight execution snapshot for tool-call polling."""

        test_state = self._test_state
        if test_state is None:
            return None

        report = self._test_report
        steps_total = 0
        steps_completed = 0
        steps_failed = 0
        issues_found: dict[str, str] = {}
        latest_screenshot_path: str | None = None
        actions_taken = 0

        if report is not None:
            for case in report.test_cases:
                steps_total += case.steps_total
                steps_completed += case.steps_completed
                steps_failed += case.steps_failed
                for step in [
                    *case.setup_step_results,
                    *case.step_results,
                    *case.cleanup_step_results,
                ]:
                    actions_taken += len(step.actions_performed)
                    if step.status == TestStatus.FAILED:
                        observed = (
                            step.actual_result or step.error_message or "Step failed."
                        )
                        issues_found[f"step_{step.step_number}"] = (
                            f"Expected {step.expected_result}. Observed: {observed}"
                        )
                    latest_screenshot_path = (
                        step.screenshot_after
                        or step.screenshot_before
                        or latest_screenshot_path
                    )

        if steps_total == 0:
            steps_total = len(getattr(test_state.test_plan, "steps", []) or [])

        current_step = None
        if test_state.current_step is not None:
            current_step = (
                f"Step {test_state.current_step.step_number}: "
                f"{test_state.current_step.description}"
            )

        return ToolModeTestProgress(
            current_step=current_step,
            steps_total=steps_total,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            issues_found=issues_found,
            latest_screenshot_path=latest_screenshot_path,
            actions_taken=actions_taken,
        )

    def _reset_execution_state(self) -> None:
        """Reset run-scoped state so the runner can be reused safely."""
        self._current_test_plan = None
        self._current_test_case = None
        self._current_test_step = None
        self._test_state = None
        self._test_report = None
        self._initial_url = None
        self._execution_history = []
        self._artifacts.reset()
        self._interpreter = None
        self._executor = None
        self._current_test_case_actions = None
        self._current_step_actions = None
        self._current_step_data = {}
        self._action_storage = {
            "test_plan_id": None,
            "test_run_timestamp": None,
            "test_cases": [],
        }
        self._trace = self._create_run_trace()

    async def _ensure_initial_screenshot(self) -> None:
        """Capture and cache the initial environment screenshot."""
        await self._artifacts.ensure_initial_screenshot()

    async def _get_interpretation_screenshot(
        self,
        step: TestStep,
        test_case: TestCase,
    ) -> tuple[bytes | None, str | None, str | None]:
        """
        Resolve the screenshot to send alongside step interpretation.

        Returns a tuple of (screenshot_bytes, screenshot_path, source_label).
        """
        interpretation = await self._artifacts.get_interpretation_screenshot(
            step,
            test_case,
        )
        return (
            interpretation.screenshot_bytes,
            interpretation.screenshot_path,
            interpretation.source,
        )

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
        self._reset_execution_state()
        self._current_test_plan = test_plan
        self._initial_url = initial_url
        self._test_state = test_state
        runtime_environment = resolve_runtime_environment(
            automation_backend=test_state.context.get("automation_backend"),
            target_type=test_state.context.get("target_type"),
            default=normalize_runtime_environment_name(self._environment),
        )
        self._environment = runtime_environment.name
        if not (
            self.automation_driver
            and hasattr(self.automation_driver, "coordinate_cache")
        ):
            self._coordinate_cache = CoordinateCache(
                self._coordinate_cache_path_for_environment(self._environment)
            )
        self._artifacts.set_automation_driver(self.automation_driver)
        self._replay_service.set_automation_driver(self.automation_driver)
        self._replay_service.set_coordinate_cache(self._coordinate_cache)

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
            created_by=self.name,
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
            from haindy.monitoring.debug_logger import get_debug_logger

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
                    current_case_actions = self._current_test_case_actions or {}
                    for step_data in reversed(current_case_actions.get("steps", [])):
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

    async def _execute_setup_steps(
        self,
        test_case: TestCase,
        case_result: TestCaseResult,
    ) -> bool:
        """Execute setup_steps before the main test steps.

        Returns True if all required setup steps passed, False otherwise.
        """
        if not test_case.setup_steps:
            return True

        logger.info(
            "Executing setup steps",
            extra={
                "test_case": test_case.name,
                "num_setup_steps": len(test_case.setup_steps),
            },
        )

        for step in test_case.setup_steps:
            step_result = await self._execute_test_step(step, test_case, case_result)
            case_result.setup_step_results.append(step_result)

            if step_result.status == TestStatus.FAILED and not step.optional:
                logger.error(
                    "Required setup step failed, skipping test case",
                    extra={
                        "step_number": step.step_number,
                        "action": step.action,
                        "test_case": test_case.name,
                    },
                )
                return False

        return True

    async def _execute_cleanup_steps(
        self,
        test_case: TestCase,
        case_result: TestCaseResult,
    ) -> None:
        """Execute cleanup_steps after the main test steps.

        Cleanup failures are logged but never block subsequent test cases.
        """
        if not test_case.cleanup_steps:
            return

        logger.info(
            "Executing cleanup steps",
            extra={
                "test_case": test_case.name,
                "num_cleanup_steps": len(test_case.cleanup_steps),
            },
        )

        for step in test_case.cleanup_steps:
            step_result = await self._execute_test_step(step, test_case, case_result)
            case_result.cleanup_step_results.append(step_result)

            if step_result.status == TestStatus.FAILED:
                logger.warning(
                    "Cleanup step failed (non-blocking)",
                    extra={
                        "step_number": step.step_number,
                        "action": step.action,
                        "test_case": test_case.name,
                    },
                )

    async def _execute_test_case(self, test_case: TestCase) -> TestCaseResult:
        """Execute a single test case with all its steps."""
        logger.info(
            "Starting test case execution",
            extra={
                "test_case_id": test_case.test_id,
                "test_case_name": test_case.name,
                "priority": test_case.priority.value,
                "total_steps": len(test_case.steps),
                "setup_steps": len(test_case.setup_steps),
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
        test_report = self._test_report
        assert test_report is not None
        test_report.test_cases.append(case_result)

        try:
            # Check prerequisites
            if not await self._verify_prerequisites(test_case.prerequisites):
                case_result.status = TestStatus.SKIPPED
                case_result.error_message = "Prerequisites not met"
                return case_result

            # Execute setup steps (navigate to test case starting state)
            if not await self._execute_setup_steps(test_case, case_result):
                case_result.status = TestStatus.SKIPPED
                case_result.error_message = "Setup step failed"
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
                    assert self._current_test_plan is not None
                    bug_report_result = (
                        await self._create_bug_report_builder().build_bug_report(
                            BugReportRequest(
                                test_plan=self._current_test_plan,
                                step_result=step_result,
                                step=step,
                                test_case=test_case,
                                case_result=case_result,
                                verification_result=(
                                    self._current_step_data.get(
                                        "verification_result", {}
                                    )
                                    if isinstance(
                                        self._current_step_data.get(
                                            "verification_result", {}
                                        ),
                                        dict,
                                    )
                                    else {}
                                ),
                            )
                        )
                    )
                    if bug_report_result:
                        case_result.bugs.append(bug_report_result.bug_report)
                        test_report.bugs.append(bug_report_result.bug_report)
                        self._current_step_data["plan_level_assessment"] = (
                            bug_report_result.plan_level_assessment
                        )
                        self._current_step_data["is_blocker"] = (
                            bug_report_result.is_blocker
                        )
                        self._current_step_data["blocker_reasoning"] = (
                            bug_report_result.blocker_reasoning
                        )

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
            # Run cleanup steps regardless of test outcome
            await self._execute_cleanup_steps(test_case, case_result)
            case_result.completed_at = datetime.now(timezone.utc)

        return case_result

    async def _execute_test_step(
        self, step: TestStep, test_case: TestCase, case_result: TestCaseResult
    ) -> StepResult:
        if self._test_state is not None:
            self._test_state.current_step = step
        try:
            return await self._step_processor.execute_test_step(
                step, test_case, case_result
            )
        finally:
            if self._test_state is not None and self._test_state.current_step == step:
                self._test_state.current_step = None

    async def _interpret_step(
        self,
        step: TestStep,
        test_case: TestCase,
        case_result: TestCaseResult,
        use_cache: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        return await self._step_processor.interpret_step(
            step,
            test_case,
            case_result,
            use_cache=use_cache,
        )

    async def _execute_action(
        self,
        action: dict[str, Any],
        step: TestStep,
        record_driver_actions: bool = False,
        step_session: Any | None = None,
    ) -> dict[str, Any]:
        return await self._step_processor.execute_action(
            action,
            step,
            record_driver_actions=record_driver_actions,
            step_session=step_session,
        )

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
        return await self._verifier.verify_expected_outcome(
            test_case,
            step,
            action_results,
            screenshot_before,
            screenshot_after,
            execution_history,
            next_test_case,
            replay_wait_budget_ms=replay_wait_budget_ms,
        )

    async def _verify_prerequisites(self, prerequisites: list[str]) -> bool:
        return await self._verifier.verify_prerequisites(prerequisites)

    async def _verify_postconditions(self, postconditions: list[str]) -> bool:
        return await self._verifier.verify_postconditions(postconditions)

    def _check_dependencies(self, step: TestStep, case_result: TestCaseResult) -> bool:
        return self._verifier.check_dependencies(step, case_result)

    def _determine_overall_status(self) -> TestStatus:
        return self._summary_helper.determine_overall_status()

    def _calculate_summary(self) -> TestSummary:
        return self._summary_helper.calculate_summary()

    def _save_screenshot(self, screenshot: bytes, name: str) -> Path:
        return Path(self._artifacts.save_screenshot(screenshot, name))

    def _register_evidence(self, path: Path) -> None:
        self._artifacts.register_evidence(path)

    @staticmethod
    def _plan_cache_key(step: TestStep, test_case: TestCase) -> str:
        return TestRunnerStepProcessor.plan_cache_key(step, test_case)

    @staticmethod
    def _strip_plan_fingerprint_volatile_fields(payload: Any) -> Any:
        return TestRunnerSummary.strip_plan_fingerprint_volatile_fields(payload)

    def _plan_fingerprint(self) -> str:
        return self._summary_helper.plan_fingerprint()

    @staticmethod
    def _is_validation_only_action_result(result: dict[str, Any]) -> bool:
        return TestRunnerStepProcessor.is_validation_only_action_result(result)

    @classmethod
    def _is_validation_only_step_result_set(
        cls, action_results: list[dict[str, Any]]
    ) -> bool:
        return TestRunnerStepProcessor.is_validation_only_step_result_set(
            action_results
        )

    @staticmethod
    def _driver_actions_for_replay(result: dict[str, Any]) -> list[dict[str, Any]]:
        return TestRunnerStepProcessor.driver_actions_for_replay(result)

    def _replay_enabled(self, step: TestStep) -> bool:
        return self._step_processor.replay_enabled(step)

    def _replay_stabilization_wait_ms(self) -> int:
        return self._step_processor.replay_stabilization_wait_ms()

    @staticmethod
    def _coerce_model_bool(value: Any) -> bool:
        return TestRunnerVerifier.coerce_model_bool(value)

    async def _execution_replay_key(
        self, step: TestStep, test_case: TestCase
    ) -> ExecutionReplayCacheKey | None:
        return await self._step_processor.execution_replay_key(step, test_case)

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
        return await self._step_processor.try_execution_replay(
            step=step,
            test_case=test_case,
            step_result=step_result,
            screenshot_before=screenshot_before,
            execution_history=execution_history,
            next_test_case=next_test_case,
        )

    async def _store_execution_replay(
        self,
        step: TestStep,
        test_case: TestCase,
        action_results: list[dict[str, Any]],
    ) -> None:
        await self._step_processor.store_execution_replay(
            step, test_case, action_results
        )

    async def _persist_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        await self._step_processor.persist_coordinate_cache(action_results)

    async def _invalidate_coordinate_cache(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        await self._step_processor.invalidate_coordinate_cache(action_results)

    def _print_summary(self) -> None:
        self._summary_helper.print_summary()

    def get_action_storage(self) -> dict[str, Any]:
        """Return the captured action storage data."""
        return self._action_storage
