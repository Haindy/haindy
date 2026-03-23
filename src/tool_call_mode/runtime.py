"""Session-owned runtime for tool-call mode."""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.action_agent import ActionAgent
from src.agents.test_planner import TestPlannerAgent
from src.agents.test_runner import TestRunner
from src.config.settings import Settings, get_settings
from src.core.enhanced_types import EnhancedActionResult
from src.core.types import StepResult, TestCaseResult, TestState, TestStatus
from src.desktop.controller import DesktopController
from src.mobile.controller import MobileController
from src.mobile.ios_controller import IOSController
from src.monitoring.debug_logger import initialize_debug_logger
from src.monitoring.logger import get_logger, set_run_id
from src.runtime.agent_factory import AgentFactory
from src.runtime.environment import normalize_automation_backend
from src.security.sanitizer import set_literal_redactions

from .models import (
    CommandStatus,
    ExitReason,
    SessionMetadata,
    ToolCallEnvelope,
    ToolCallRequest,
    make_envelope,
)
from .paths import (
    get_logs_dir,
    get_screenshots_dir,
    save_session_metadata,
)
from .variables import SessionVariableStore

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolCallSessionRuntime:
    """Own one persistent session daemon runtime."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: str,
        idle_timeout_seconds: int,
        settings: Settings | None = None,
        android_serial: str | None = None,
        android_app: str | None = None,
        ios_udid: str | None = None,
        ios_app: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.settings = settings or get_settings()
        self.backend = normalize_automation_backend(backend)
        self.idle_timeout_seconds = max(int(idle_timeout_seconds), 1)
        self.android_serial = (android_serial or "").strip() or None
        self.android_app = (android_app or "").strip() or None
        self.ios_udid = (ios_udid or "").strip() or None
        self.ios_app = (ios_app or "").strip() or None

        self.controller: DesktopController | MobileController | IOSController | None = (
            None
        )
        self.action_agent: ActionAgent | None = None
        self.test_planner: TestPlannerAgent | None = None
        self.test_runner: TestRunner | None = None
        self.variables = SessionVariableStore()
        self.metadata = SessionMetadata.new(
            session_id=session_id,
            backend=self.backend,
            idle_timeout_seconds=self.idle_timeout_seconds,
            android_serial=self.android_serial,
            android_app=self.android_app,
            ios_udid=self.ios_udid,
            ios_app=self.ios_app,
        )
        self._close_requested = False
        self._command_counter = 0
        self._last_activity_monotonic = time.monotonic()

    @property
    def last_activity_monotonic(self) -> float:
        """Monotonic timestamp of the most recent handled command."""

        return self._last_activity_monotonic

    def is_close_requested(self) -> bool:
        """Return True when graceful shutdown has been requested."""

        return self._close_requested

    def set_pid(self, pid: int) -> None:
        """Attach the live daemon pid to metadata."""

        self.metadata.pid = int(pid)
        save_session_metadata(self.metadata)

    async def start(self) -> ToolCallEnvelope:
        """Start the controller, build agents, and capture the initial screenshot."""

        self._activate_command_context("session_start")
        self._sync_secret_redaction()
        self.metadata.status = "initializing"
        save_session_metadata(self.metadata)

        self.controller = self._create_controller()
        await self.controller.start()

        if self.backend == "mobile_adb" and isinstance(
            self.controller, MobileController
        ):
            await self.controller.driver.configure_target(
                adb_serial=self.android_serial,
                app_package=self.android_app,
            )
            if self.android_app:
                await self.controller.driver.launch_app(self.android_app)

        if self.backend == "mobile_ios" and isinstance(self.controller, IOSController):
            await self.controller.driver.configure_target(
                udid=self.ios_udid,
                bundle_id=self.ios_app,
            )
            if self.ios_app:
                await self.controller.driver.launch_app(self.ios_app)

        agents = AgentFactory(self.settings).create_runtime_agents(
            automation_driver=self.controller.driver
        )
        self.action_agent = agents.action_agent
        self.test_planner = agents.test_planner
        self.test_runner = agents.test_runner

        screenshot_bytes = await self.controller.driver.screenshot()
        screenshot_path = self._store_screenshot_bytes(screenshot_bytes)
        self.metadata.status = "ready"
        self.metadata.last_command_at = _utc_now().isoformat()
        save_session_metadata(self.metadata)

        response = self._build_startup_response()
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=response,
            screenshot_path=screenshot_path,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )

    async def stop(self) -> None:
        """Stop the underlying controller and persist final metadata."""

        self.metadata.status = "closed"
        self.metadata.closed_at = _utc_now().isoformat()
        save_session_metadata(self.metadata)
        if self.controller is not None:
            await self.controller.stop()

    async def handle_request(self, request: ToolCallRequest) -> ToolCallEnvelope:
        """Handle one validated daemon request."""

        started = time.perf_counter()
        self._sync_secret_redaction()
        self._last_activity_monotonic = time.monotonic()
        previous_command_name = self.metadata.last_command_name
        previous_command_at = self.metadata.last_command_at
        self.metadata.last_command_at = _utc_now().isoformat()
        save_session_metadata(self.metadata)

        if request.command == "screenshot":
            envelope = await self._handle_screenshot()
        elif request.command == "act":
            envelope = await self._handle_act(request)
        elif request.command == "test":
            envelope = await self._handle_test(request)
        elif request.command == "session_status":
            envelope = await self._handle_status(
                request,
                previous_command_name=previous_command_name,
                previous_command_at=previous_command_at,
            )
        elif request.command == "session_set":
            envelope = self._handle_set_var(request)
        elif request.command == "session_unset":
            envelope = self._handle_unset_var(request)
        elif request.command == "session_vars":
            envelope = self._handle_vars()
        elif request.command == "session_close":
            envelope = self._handle_close()
        else:
            envelope = make_envelope(
                session_id=self.session_id,
                command="session",
                status=CommandStatus.ERROR,
                response=f"Unsupported tool-call command: {request.command}",
                screenshot_path=self.metadata.latest_screenshot_path,
                exit_reason=ExitReason.AGENT_ERROR,
                duration_ms=0,
                actions_taken=0,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        envelope.meta.duration_ms = duration_ms
        self._record_command(envelope)
        return envelope

    async def _handle_status(
        self,
        request: ToolCallRequest,
        *,
        previous_command_name: str | None,
        previous_command_at: str | None,
    ) -> ToolCallEnvelope:
        del request
        self._activate_command_context("session_status")
        if self.action_agent is None or self.controller is None:
            raise RuntimeError("ActionAgent is not initialized.")

        screenshot = await self.controller.driver.screenshot()
        result = await self.action_agent.observe_current_screen(
            test_context=self._tool_context("session_status"),
            screenshot=screenshot,
        )
        screenshot_path = self._promote_result_screenshot(
            result,
            fallback_bytes=screenshot,
        )
        response_text = (
            result.final_model_output
            or (result.ai_analysis.actual_outcome if result.ai_analysis else None)
            or "Session is active."
        )
        response = self._status_response(
            description=response_text,
            previous_command_name=previous_command_name,
            previous_command_at=previous_command_at,
        )
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=(
                CommandStatus.SUCCESS
                if result.overall_success
                else CommandStatus.FAILURE
            ),
            response=self._redact(response),
            screenshot_path=screenshot_path,
            exit_reason=(
                ExitReason.COMPLETED
                if result.overall_success
                else self._action_failure_reason(result)
            ),
            duration_ms=0,
            actions_taken=1,
        )

    async def _handle_screenshot(self) -> ToolCallEnvelope:
        self._activate_command_context("screenshot")
        if self.controller is None:
            raise RuntimeError("Controller is not initialized.")

        screenshot_bytes = await self.controller.driver.screenshot()
        screenshot_path = self._store_screenshot_bytes(screenshot_bytes)
        return make_envelope(
            session_id=self.session_id,
            command="screenshot",
            status=CommandStatus.SUCCESS,
            response="Screenshot captured.",
            screenshot_path=screenshot_path,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )

    async def _handle_act(self, request: ToolCallRequest) -> ToolCallEnvelope:
        self._activate_command_context("act")
        if self.action_agent is None:
            raise RuntimeError("ActionAgent is not initialized.")

        instruction = self.variables.interpolate(request.instruction or "")
        result = await self.action_agent.execute_tool_instruction(
            instruction,
            test_context=self._tool_context("act"),
        )
        screenshot_path = self._promote_result_screenshot(result)
        actions_taken = self._count_action_result_actions(result)
        response_text = (
            result.final_model_output
            or (result.ai_analysis.actual_outcome if result.ai_analysis else None)
            or "Action completed."
        )
        return make_envelope(
            session_id=self.session_id,
            command="act",
            status=(
                CommandStatus.SUCCESS
                if result.overall_success
                else CommandStatus.FAILURE
            ),
            response=self._redact(response_text),
            screenshot_path=screenshot_path,
            exit_reason=(
                ExitReason.COMPLETED
                if result.overall_success
                else self._action_failure_reason(result)
            ),
            duration_ms=0,
            actions_taken=actions_taken,
        )

    async def _handle_test(self, request: ToolCallRequest) -> ToolCallEnvelope:
        self._activate_command_context("test")
        if self.test_planner is None or self.test_runner is None:
            raise RuntimeError("Tool-call test agents are not initialized.")

        max_steps = int(request.options.get("max_steps", 20))
        scenario = self.variables.interpolate(request.instruction or "")

        try:
            test_plan = await self.test_planner.create_tool_mode_test_plan(
                scenario,
                max_steps=max_steps,
                context=self._tool_context("test"),
            )
        except ValueError as exc:
            return make_envelope(
                session_id=self.session_id,
                command="test",
                status=CommandStatus.FAILURE,
                response=self._redact(str(exc)),
                screenshot_path=self.metadata.latest_screenshot_path,
                exit_reason=ExitReason.MAX_STEPS_REACHED,
                duration_ms=0,
                actions_taken=0,
                steps_total=max_steps,
                steps_passed=0,
                steps_failed=0,
            )

        test_state = TestState(
            test_plan=test_plan,
            context=self._tool_context("test"),
        )
        final_state = await self.test_runner.execute_test_plan(test_state)
        return self._build_test_envelope(final_state)

    def _handle_set_var(self, request: ToolCallRequest) -> ToolCallEnvelope:
        name = request.var_name or ""
        value = request.var_value or ""
        self.variables.set(name, value, secret=bool(request.var_secret))
        self._sync_secret_redaction()
        suffix = " (secret)." if request.var_secret else "."
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=f"Variable {self.variables.validate_name(name)} set{suffix}",
            screenshot_path=None,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )

    def _handle_unset_var(self, request: ToolCallRequest) -> ToolCallEnvelope:
        name = self.variables.validate_name(request.var_name or "")
        existed = self.variables.unset(name)
        self._sync_secret_redaction()
        response = (
            f"Variable {name} removed." if existed else f"Variable {name} was not set."
        )
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=response,
            screenshot_path=None,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )

    def _handle_vars(self) -> ToolCallEnvelope:
        vars_map = self.variables.as_public_map()
        if vars_map:
            rendered = ", ".join(
                f"{name} (secret)" if value == "[secret]" else name
                for name, value in vars_map.items()
            )
            response = f"{len(vars_map)} variables defined: {rendered}."
        else:
            response = "0 variables defined."
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=response,
            screenshot_path=None,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
            vars_map=vars_map,
        )

    def _handle_close(self) -> ToolCallEnvelope:
        self._close_requested = True
        return make_envelope(
            session_id=self.session_id,
            command="session",
            status=CommandStatus.SUCCESS,
            response=(
                "Session closed. "
                f"{self.metadata.actions_executed} device actions were executed during this session."
            ),
            screenshot_path=None,
            exit_reason=ExitReason.COMPLETED,
            duration_ms=0,
            actions_taken=0,
        )

    def _record_command(self, envelope: ToolCallEnvelope) -> None:
        """Persist command-level counters after one handled request."""

        self._last_activity_monotonic = time.monotonic()
        self.metadata.last_command_at = _utc_now().isoformat()
        self.metadata.last_command_name = envelope.command
        self.metadata.commands_executed += 1
        self.metadata.actions_executed += envelope.meta.actions_taken
        save_session_metadata(self.metadata)

    def _activate_command_context(self, command_name: str) -> None:
        """Set per-command logging/debug context within the live session."""

        self._command_counter += 1
        run_id = f"{self.session_id}-{self._command_counter:04d}-{command_name}"
        set_run_id(run_id)
        initialize_debug_logger(
            run_id,
            debug_dir=get_screenshots_dir(self.session_id),
            reports_dir=get_logs_dir(self.session_id),
            ai_log_path=get_logs_dir(self.session_id) / "ai_interactions.jsonl",
        )

    def _create_controller(
        self,
    ) -> DesktopController | MobileController | IOSController:
        """Instantiate the controller for this session backend."""

        if self.backend == "mobile_adb":
            return MobileController(preferred_serial=self.android_serial)
        if self.backend == "mobile_ios":
            return IOSController(preferred_udid=self.ios_udid)
        return DesktopController()

    def _tool_context(self, command_name: str) -> dict[str, Any]:
        """Build execution context for tool-mode agent calls."""

        context: dict[str, Any] = {
            "tool_mode": True,
            "tool_mode_command": command_name,
            "automation_backend": self.backend,
            "target_type": self.backend,
        }
        if self.android_serial:
            context["adb_serial"] = self.android_serial
        if self.android_app:
            context["app_package"] = self.android_app
        if self.ios_udid:
            context["ios_udid"] = self.ios_udid
        if self.ios_app:
            context["bundle_id"] = self.ios_app
        return context

    def _build_startup_response(self) -> str:
        if self.backend == "mobile_adb":
            driver = (
                self.controller.driver
                if isinstance(self.controller, MobileController)
                else None
            )
            adb_client = getattr(driver, "adb", None)
            detected_serial = getattr(adb_client, "serial", None)
            serial = self.android_serial or (
                str(detected_serial).strip() if detected_serial is not None else ""
            )
            serial_suffix = f" Device found: {serial}." if serial else ""
            return f"Session started with Android ADB backend.{serial_suffix}"
        if self.backend == "mobile_ios":
            ios_driver = (
                self.controller.driver
                if isinstance(self.controller, IOSController)
                else None
            )
            idb_client = getattr(ios_driver, "idb", None)
            detected_udid = getattr(idb_client, "udid", None)
            udid = self.ios_udid or (
                str(detected_udid).strip() if detected_udid else ""
            )
            udid_suffix = f" Device UDID: {udid}." if udid else ""
            return f"Session started with iOS idb backend.{udid_suffix}"
        return "Session started with desktop backend."

    def _status_response(
        self,
        *,
        description: str,
        previous_command_name: str | None,
        previous_command_at: str | None,
    ) -> str:
        """Add session-health context to the observe-only screen description."""

        prefix = "Session is active."
        if previous_command_name and previous_command_at:
            try:
                previous_at = datetime.fromisoformat(previous_command_at)
                elapsed_seconds = max(
                    int((_utc_now() - previous_at).total_seconds()), 0
                )
                prefix = (
                    "Session is active. "
                    f"Last completed command was '{previous_command_name}' {elapsed_seconds} seconds ago."
                )
            except ValueError:
                prefix = (
                    "Session is active. "
                    f"Last completed command was '{previous_command_name}'."
                )

        normalized_description = (description or "").strip()
        if not normalized_description:
            return prefix
        if normalized_description.lower().startswith("session is active"):
            return normalized_description
        return f"{prefix} {normalized_description}"

    def _sync_secret_redaction(self) -> None:
        """Keep the process-level log sanitizers aligned with session secrets."""

        set_literal_redactions(self.variables.secret_values())

    def _store_screenshot_bytes(self, screenshot_bytes: bytes) -> str:
        """Persist a response screenshot using the canonical sequential naming."""

        self.metadata.screenshot_count += 1
        destination = (
            get_screenshots_dir(self.session_id)
            / f"step_{self.metadata.screenshot_count:03d}.png"
        )
        destination.write_bytes(screenshot_bytes)
        self.metadata.latest_screenshot_path = str(destination.resolve())
        save_session_metadata(self.metadata)
        return str(destination.resolve())

    def _promote_existing_screenshot(self, path: str) -> str:
        """Copy an existing screenshot into the canonical sequential path."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        self.metadata.screenshot_count += 1
        destination = (
            get_screenshots_dir(self.session_id)
            / f"step_{self.metadata.screenshot_count:03d}.png"
        )
        shutil.copyfile(source, destination)
        self.metadata.latest_screenshot_path = str(destination.resolve())
        save_session_metadata(self.metadata)
        return str(destination.resolve())

    def _promote_result_screenshot(
        self,
        result: EnhancedActionResult,
        *,
        fallback_bytes: bytes | None = None,
    ) -> str | None:
        """Return the canonical response screenshot path for an action result."""

        after = result.environment_state_after
        if after is not None:
            if after.screenshot_path:
                try:
                    return self._promote_existing_screenshot(after.screenshot_path)
                except FileNotFoundError:
                    pass
            if after.screenshot is not None:
                return self._store_screenshot_bytes(after.screenshot)

        before = result.environment_state_before
        if before is not None and before.screenshot is not None:
            return self._store_screenshot_bytes(before.screenshot)
        if fallback_bytes is not None:
            return self._store_screenshot_bytes(fallback_bytes)
        return self.metadata.latest_screenshot_path

    @staticmethod
    def _count_action_result_actions(result: EnhancedActionResult) -> int:
        """Count atomic actions performed for one direct action result."""

        executed = sum(
            1 for action in result.computer_actions if action.status == "executed"
        )
        if executed:
            return executed
        if result.overall_success and result.execution and result.execution.success:
            return 1
        return 0

    def _action_failure_reason(self, result: EnhancedActionResult) -> ExitReason:
        """Map action-session failures into the public exit-reason contract."""

        code = str(result.terminal_failure_code or "").strip().lower()
        error_text = " ".join(
            filter(
                None,
                [
                    result.terminal_failure_reason,
                    result.final_model_output,
                    result.execution.error_message if result.execution else None,
                    result.ai_analysis.actual_outcome if result.ai_analysis else None,
                ],
            )
        ).lower()

        if code in {"max_turns_exceeded", "loop_detected"}:
            return ExitReason.MAX_ACTIONS_REACHED
        if code in {
            "observe_only_policy_violation",
            "google_prompt_blocked",
            "google_ambiguous_function_call_batch",
            "safety_fail_fast",
            "safety_policy",
        }:
            return ExitReason.AGENT_ERROR
        if "not find" in error_text or "not visible" in error_text:
            return ExitReason.ELEMENT_NOT_FOUND
        if "timeout" in error_text:
            return ExitReason.COMMAND_TIMEOUT
        if "driver" in error_text or "device" in error_text:
            return ExitReason.DEVICE_ERROR
        return ExitReason.ASSERTION_FAILED

    def _build_test_envelope(self, final_state: TestState) -> ToolCallEnvelope:
        """Translate a completed test-state into the tool-call response contract."""

        case_results = (
            list(final_state.test_report.test_cases) if final_state.test_report else []
        )
        steps_total = sum(case.steps_total for case in case_results)
        steps_passed = sum(case.steps_completed for case in case_results)
        steps_failed = sum(case.steps_failed for case in case_results)
        actions_taken = self._count_test_actions(case_results)
        screenshot_path = self._promote_test_screenshot(case_results)

        if final_state.status in {TestStatus.PASSED, TestStatus.COMPLETED}:
            response = (
                (
                    f"Test passed in {steps_total} steps. "
                    f"{steps_passed} steps passed with {actions_taken} device actions."
                )
                if steps_total
                else "Test passed."
            )
            return make_envelope(
                session_id=self.session_id,
                command="test",
                status=CommandStatus.SUCCESS,
                response=self._redact(response),
                screenshot_path=screenshot_path,
                exit_reason=ExitReason.COMPLETED,
                duration_ms=0,
                actions_taken=actions_taken,
                steps_total=steps_total,
                steps_passed=steps_passed,
                steps_failed=steps_failed,
            )

        failed_step, failed_case = self._first_failed_step(case_results)
        failure_reason = self._test_failure_reason(failed_step, failed_case)
        if failed_step is not None:
            observed = (
                failed_step.actual_result
                or failed_step.error_message
                or (failed_case.error_message if failed_case else None)
                or "Step failed."
            )
            response = (
                f"Test failed at step {failed_step.step_number} of {steps_total}. "
                f"{steps_passed} step(s) passed before failure. "
                f"Expected: {failed_step.expected_result}. "
                f"Observed: {observed}"
            )
        elif failed_case is not None:
            response = failed_case.error_message or "Test failed."
        else:
            response = "Test failed."

        return make_envelope(
            session_id=self.session_id,
            command="test",
            status=CommandStatus.FAILURE,
            response=self._redact(response),
            screenshot_path=screenshot_path,
            exit_reason=failure_reason,
            duration_ms=0,
            actions_taken=actions_taken,
            steps_total=steps_total,
            steps_passed=steps_passed,
            steps_failed=steps_failed,
        )

    @staticmethod
    def _count_test_actions(case_results: list[TestCaseResult]) -> int:
        total = 0
        for case in case_results:
            for step in [
                *case.setup_step_results,
                *case.step_results,
                *case.cleanup_step_results,
            ]:
                total += len(step.actions_performed)
        return total

    def _promote_test_screenshot(
        self, case_results: list[TestCaseResult]
    ) -> str | None:
        for case in reversed(case_results):
            for step in reversed(
                [
                    *case.cleanup_step_results,
                    *case.step_results,
                    *case.setup_step_results,
                ]
            ):
                if step.screenshot_after:
                    try:
                        return self._promote_existing_screenshot(step.screenshot_after)
                    except FileNotFoundError:
                        continue
                if step.screenshot_before:
                    try:
                        return self._promote_existing_screenshot(step.screenshot_before)
                    except FileNotFoundError:
                        continue
        return self.metadata.latest_screenshot_path

    @staticmethod
    def _first_failed_step(
        case_results: list[TestCaseResult],
    ) -> tuple[StepResult | None, TestCaseResult | None]:
        for case in case_results:
            for step in [
                *case.setup_step_results,
                *case.step_results,
                *case.cleanup_step_results,
            ]:
                if step.status == TestStatus.FAILED:
                    return step, case
        return None, None

    @staticmethod
    def _test_failure_reason(
        failed_step: StepResult | None,
        failed_case: TestCaseResult | None,
    ) -> ExitReason:
        text = " ".join(
            filter(
                None,
                [
                    failed_step.error_message if failed_step else None,
                    failed_step.actual_result if failed_step else None,
                    failed_case.error_message if failed_case else None,
                ],
            )
        ).lower()
        if "max turns exceeded" in text or "loop detected" in text:
            return ExitReason.MAX_ACTIONS_REACHED
        if "not find" in text or "not visible" in text:
            return ExitReason.ELEMENT_NOT_FOUND
        if "timed out" in text or "timeout" in text:
            return ExitReason.COMMAND_TIMEOUT
        if "exceeded max_steps" in text or "max_steps" in text:
            return ExitReason.MAX_STEPS_REACHED
        return ExitReason.ASSERTION_FAILED

    def _redact(self, text: str | None) -> str:
        return self.variables.redact(text) or ""
