"""Session-owned runtime for tool-call mode."""

from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from haindy.agents.action_agent import ActionAgent
from haindy.agents.awareness_agent import (
    AwarenessAgent,
    AwarenessAssessment,
    AwarenessTodoItem,
)
from haindy.agents.test_planner import TestPlannerAgent
from haindy.agents.test_runner import TestRunner
from haindy.config.settings import Settings, get_settings
from haindy.core.enhanced_types import EnhancedActionResult
from haindy.core.types import StepResult, TestCaseResult, TestState, TestStatus
from haindy.desktop.controller import DesktopController
from haindy.mobile.controller import MobileController
from haindy.mobile.ios_controller import IOSController
from haindy.monitoring.debug_logger import initialize_debug_logger
from haindy.monitoring.logger import get_logger, set_run_id
from haindy.runtime.agent_factory import AgentFactory
from haindy.runtime.environment import normalize_automation_backend
from haindy.security.sanitizer import set_literal_redactions

from .models import (
    CommandStatus,
    ExitReason,
    ExploreTaskStatus,
    ExploreTodoItem,
    SessionMetadata,
    TestTaskStatus,
    TodoStatus,
    ToolCallEnvelope,
    ToolCallRequest,
    make_envelope,
)
from .paths import (
    get_action_artifacts_dir,
    get_logs_dir,
    get_screenshots_dir,
    save_session_metadata,
)
from .variables import SessionVariableStore

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _BackgroundTaskBase:
    started_monotonic: float = field(default_factory=time.monotonic)
    run_id: str | None = None
    response: str = ""
    screenshot_path: str | None = None
    actions_taken: int = 0
    exit_reason: ExitReason = ExitReason.COMPLETED
    accounted_actions: int = 0
    latest_source_screenshot: str | None = None
    phase: str | None = None
    phase_started_at: str | None = None
    last_model_agent: str | None = None
    last_progress_at: str | None = None
    latest_action_artifact_path: str | None = None

    def elapsed_seconds(self) -> int:
        return max(int(time.monotonic() - self.started_monotonic), 0)

    def set_phase(self, phase: str, *, agent: str | None = None) -> None:
        now = _utc_now().isoformat()
        if self.phase != phase:
            self.phase = phase
            self.phase_started_at = now
        elif self.phase_started_at is None:
            self.phase_started_at = now
        self.last_progress_at = now
        if agent:
            self.last_model_agent = agent

    def mark_progress(self, *, agent: str | None = None) -> None:
        self.last_progress_at = _utc_now().isoformat()
        if agent:
            self.last_model_agent = agent


@dataclass
class _TestTaskState(_BackgroundTaskBase):
    scenario: str = ""
    status: TestTaskStatus = TestTaskStatus.IN_PROGRESS
    current_step: str | None = None
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    issues_found: dict[str, str] = field(default_factory=dict)


@dataclass
class _ExploreTaskState(_BackgroundTaskBase):
    goal: str = ""
    status: ExploreTaskStatus = ExploreTaskStatus.IN_PROGRESS
    current_focus: str | None = None
    todo: list[ExploreTodoItem] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


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
        self.awareness_agent: AwarenessAgent | None = None
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
        self._device_lock = asyncio.Lock()
        self._background_task: asyncio.Task[None] | None = None
        self._background_task_kind: str | None = None
        self._test_task_state: _TestTaskState | None = None
        self._explore_task_state: _ExploreTaskState | None = None

    @property
    def last_activity_monotonic(self) -> float:
        """Monotonic timestamp of the most recent handled command."""

        return self._last_activity_monotonic

    def is_close_requested(self) -> bool:
        """Return True when graceful shutdown has been requested."""

        return self._close_requested

    def is_background_task_active(self) -> bool:
        """Return True when a test or explore task is still running."""

        return self._background_task is not None and not self._background_task.done()

    def active_background_kind(self) -> str | None:
        """Return the active background task kind, if any."""

        if not self.is_background_task_active():
            return None
        return self._background_task_kind

    @staticmethod
    def background_command_allowed(command: str) -> bool:
        """Return True when a command is allowed during background execution."""

        return command in {
            "test_status",
            "explore_status",
            "session_set",
            "session_unset",
            "session_vars",
            "session_close",
            "screenshot",
        }

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
        self.action_agent.set_execution_lock(self._device_lock)
        self.test_planner = agents.test_planner
        self.test_runner = agents.test_runner
        self.awareness_agent = agents.awareness_agent

        screenshot_bytes = await self._take_screenshot_bytes()
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

        await self.cancel_background_task()
        self.metadata.status = "closed"
        self.metadata.closed_at = _utc_now().isoformat()
        save_session_metadata(self.metadata)
        if self.controller is not None:
            await self.controller.stop()

    async def cancel_background_task(self) -> None:
        """Cancel any active background task and wait for it to unwind."""

        task = self._background_task
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

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
            envelope = await self._handle_test_dispatch(request)
        elif request.command == "test_status":
            envelope = self._handle_test_status()
        elif request.command == "explore":
            envelope = await self._handle_explore_dispatch(request)
        elif request.command == "explore_status":
            envelope = self._handle_explore_status()
        elif request.command == "session_status":
            envelope = await self._handle_status(
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
            envelope = await self._handle_close()
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
        *,
        previous_command_name: str | None,
        previous_command_at: str | None,
    ) -> ToolCallEnvelope:
        self._activate_command_context("session_status")
        if self.action_agent is None:
            raise RuntimeError("ActionAgent is not initialized.")

        screenshot = await self._take_screenshot_bytes()
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
        screenshot_bytes = await self._take_screenshot_bytes()
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

    async def _handle_test_dispatch(self, request: ToolCallRequest) -> ToolCallEnvelope:
        self._activate_command_context("test")
        if self.test_planner is None or self.test_runner is None:
            raise RuntimeError("Tool-call test agents are not initialized.")

        scenario = self.variables.interpolate(request.instruction or "")
        max_steps = max(int(request.options.get("max_steps", 20)), 1)
        timeout_seconds = max(int(request.options.get("timeout_seconds", 300)), 1)
        screenshot_bytes = await self._take_screenshot_bytes()
        screenshot_path = self._store_screenshot_bytes(screenshot_bytes)
        run_id = self._background_run_id("test")
        self._activate_background_context(run_id)

        state = _TestTaskState(
            run_id=run_id,
            scenario=scenario,
            response="Test dispatched. Poll test-status for progress.",
            screenshot_path=screenshot_path,
            actions_taken=1,
            accounted_actions=1,
        )
        state.set_phase("planning", agent="test_planner")
        self._test_task_state = state
        self._sync_test_task_metadata(state)
        self._background_task_kind = "test"
        self._background_task = asyncio.create_task(
            self._run_test_task(
                state=state,
                scenario=scenario,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        )

        return make_envelope(
            session_id=self.session_id,
            run_id=state.run_id,
            command="test",
            status=CommandStatus.SUCCESS,
            response=state.response,
            screenshot_path=screenshot_path,
            exit_reason=ExitReason.DISPATCHED,
            duration_ms=0,
            actions_taken=1,
        )

    def _handle_test_status(self) -> ToolCallEnvelope:
        self._activate_command_context("test_status")
        state = self._test_task_state
        if state is None:
            return make_envelope(
                session_id=self.session_id,
                command="test-status",
                status=CommandStatus.ERROR,
                response="No test has been dispatched in this session.",
                screenshot_path=self.metadata.latest_screenshot_path,
                exit_reason=ExitReason.COMPLETED,
                duration_ms=0,
                actions_taken=0,
            )

        if state.status == TestTaskStatus.IN_PROGRESS:
            self._refresh_test_progress(state)
        return make_envelope(
            session_id=self.session_id,
            run_id=getattr(state, "run_id", None),
            command="test-status",
            status=CommandStatus.SUCCESS,
            response=self._redact(self._test_status_response(state)),
            screenshot_path=state.screenshot_path,
            exit_reason=state.exit_reason,
            duration_ms=0,
            actions_taken=state.actions_taken,
            test_status=state.status,
            current_step=self._redact(state.current_step),
            phase=getattr(state, "phase", None),
            phase_started_at=getattr(state, "phase_started_at", None),
            last_model_agent=getattr(state, "last_model_agent", None),
            last_progress_at=getattr(state, "last_progress_at", None),
            latest_action_artifact_path=getattr(
                state,
                "latest_action_artifact_path",
                None,
            ),
            steps_total=state.steps_total,
            steps_completed=state.steps_completed,
            steps_failed=state.steps_failed,
            issues_found=self._redact_mapping(state.issues_found),
            elapsed_time_seconds=state.elapsed_seconds(),
        )

    async def _handle_explore_dispatch(
        self,
        request: ToolCallRequest,
    ) -> ToolCallEnvelope:
        self._activate_command_context("explore")
        if self.awareness_agent is None or self.action_agent is None:
            raise RuntimeError("Explore agents are not initialized.")

        goal = self.variables.interpolate(request.instruction or "")
        max_steps = max(int(request.options.get("max_steps", 50)), 1)
        timeout_raw = request.options.get("timeout_seconds")
        if timeout_raw is None or timeout_raw == "":
            timeout_seconds = None
        else:
            timeout_seconds = max(int(timeout_raw), 1)
        screenshot_bytes = await self._take_screenshot_bytes()
        screenshot_path = self._store_screenshot_bytes(screenshot_bytes)

        state = _ExploreTaskState(
            goal=goal,
            response="Explore dispatched. Poll explore-status for progress.",
            screenshot_path=screenshot_path,
            actions_taken=1,
            accounted_actions=1,
        )
        self._explore_task_state = state
        self._background_task_kind = "explore"
        self._background_task = asyncio.create_task(
            self._run_explore_task(
                state=state,
                goal=goal,
                initial_screenshot=screenshot_bytes,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
        )

        return make_envelope(
            session_id=self.session_id,
            command="explore",
            status=CommandStatus.SUCCESS,
            response=state.response,
            screenshot_path=screenshot_path,
            exit_reason=ExitReason.DISPATCHED,
            duration_ms=0,
            actions_taken=1,
        )

    def _handle_explore_status(self) -> ToolCallEnvelope:
        self._activate_command_context("explore_status")
        state = self._explore_task_state
        if state is None:
            return make_envelope(
                session_id=self.session_id,
                command="explore-status",
                status=CommandStatus.ERROR,
                response="No explore has been dispatched in this session.",
                screenshot_path=self.metadata.latest_screenshot_path,
                exit_reason=ExitReason.COMPLETED,
                duration_ms=0,
                actions_taken=0,
            )

        return make_envelope(
            session_id=self.session_id,
            command="explore-status",
            status=CommandStatus.SUCCESS,
            response=self._redact(state.response),
            screenshot_path=state.screenshot_path,
            exit_reason=state.exit_reason,
            duration_ms=0,
            actions_taken=state.actions_taken,
            explore_status=state.status,
            current_focus=self._redact(state.current_focus),
            todo=self._redact_todo_items(state.todo),
            observations=self._redact_string_list(state.observations),
            elapsed_time_seconds=state.elapsed_seconds(),
        )

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

    async def _handle_close(self) -> ToolCallEnvelope:
        await self.cancel_background_task()
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

    async def _run_test_task(
        self,
        *,
        state: _TestTaskState,
        scenario: str,
        max_steps: int,
        timeout_seconds: int,
    ) -> None:
        if state.run_id:
            self._activate_background_context(state.run_id)
        try:
            await asyncio.wait_for(
                self._execute_test_task(
                    state=state,
                    scenario=scenario,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._refresh_test_progress(state)
            timeout_reason = self._test_timeout_reason(state.phase)
            timeout_response = self._test_timeout_response(
                state=state,
                timeout_seconds=timeout_seconds,
                timeout_reason=timeout_reason,
            )
            if self.test_runner is not None:
                self.test_runner.mark_tool_mode_timeout(
                    reason=timeout_reason,
                    message=timeout_response,
                    phase=state.phase,
                    timeout_seconds=timeout_seconds,
                )
                self._refresh_test_progress(state)
            state.status = TestTaskStatus.TIMEOUT
            state.exit_reason = ExitReason.TIMEOUT
            state.response = timeout_response
            if not state.issues_found:
                state.issues_found = {
                    "active_step": timeout_response,
                }
        except asyncio.CancelledError:
            self._refresh_test_progress(state)
            if state.status == TestTaskStatus.IN_PROGRESS:
                state.status = TestTaskStatus.ERROR
                state.exit_reason = ExitReason.AGENT_ERROR
                state.response = "Test cancelled because the session was closed."
            raise
        except Exception as exc:
            logger.exception(
                "Background test execution failed",
                extra={"session_id": self.session_id},
            )
            self._refresh_test_progress(state)
            state.status = TestTaskStatus.ERROR
            state.current_step = None
            state.exit_reason = ExitReason.AGENT_ERROR
            state.response = f"Test failed with an internal error: {exc}."
        finally:
            self._settle_background_actions(state)
            if asyncio.current_task() is self._background_task:
                self._background_task = None
                self._background_task_kind = None

    async def _execute_test_task(
        self,
        *,
        state: _TestTaskState,
        scenario: str,
        max_steps: int,
        timeout_seconds: int,
    ) -> None:
        assert self.test_planner is not None
        assert self.test_runner is not None

        state.response = "Test planning in progress."
        try:
            test_plan = await self.test_planner.create_tool_mode_test_plan(
                scenario,
                max_steps=max_steps,
                context=self._tool_context("test"),
            )
        except ValueError as exc:
            state.status = TestTaskStatus.MAX_STEPS_REACHED
            state.response = self._redact(str(exc))
            state.steps_total = max_steps
            state.steps_completed = 0
            state.steps_failed = 0
            state.issues_found = {}
            state.exit_reason = ExitReason.MAX_STEPS_REACHED
            return

        state.steps_total = len(getattr(test_plan, "steps", []) or [])
        state.response = "Test in progress."

        final_state = await self.test_runner.execute_test_plan(
            TestState(
                test_plan=test_plan,
                context={
                    **self._tool_context("test"),
                    "tool_mode_run_id": state.run_id,
                    "tool_mode_session_id": self.session_id,
                    "tool_mode_test_timeout_seconds": timeout_seconds,
                    "tool_mode_test_deadline_monotonic": (
                        state.started_monotonic + timeout_seconds
                    ),
                    "tool_mode_action_artifacts_dir": str(
                        get_action_artifacts_dir(self.session_id)
                    ),
                },
            )
        )
        self._refresh_test_progress(state)
        summary = self._summarize_test_result(final_state)
        state.status = summary["status"]
        state.response = summary["response"]
        state.current_step = None
        state.steps_total = summary["steps_total"]
        state.steps_completed = summary["steps_completed"]
        state.steps_failed = summary["steps_failed"]
        state.issues_found = summary["issues_found"]
        state.exit_reason = summary["exit_reason"]
        state.actions_taken = max(state.actions_taken, summary["actions_taken"])
        state.screenshot_path = summary["screenshot_path"] or state.screenshot_path
        state.set_phase("completed", agent="test_runner")

    async def _run_explore_task(
        self,
        *,
        state: _ExploreTaskState,
        goal: str,
        initial_screenshot: bytes,
        max_steps: int,
        timeout_seconds: int | None,
    ) -> None:
        try:
            if timeout_seconds is None:
                await self._execute_explore_task(
                    state=state,
                    goal=goal,
                    initial_screenshot=initial_screenshot,
                    max_steps=max_steps,
                )
            else:
                await asyncio.wait_for(
                    self._execute_explore_task(
                        state=state,
                        goal=goal,
                        initial_screenshot=initial_screenshot,
                        max_steps=max_steps,
                    ),
                    timeout=timeout_seconds,
                )
        except asyncio.TimeoutError:
            state.status = ExploreTaskStatus.TIMEOUT
            state.current_focus = None
            state.exit_reason = ExitReason.TIMEOUT
            state.response = f"Exploration timed out after {timeout_seconds} seconds."
        except asyncio.CancelledError:
            if state.status == ExploreTaskStatus.IN_PROGRESS:
                state.status = ExploreTaskStatus.ERROR
                state.current_focus = None
                state.exit_reason = ExitReason.AGENT_ERROR
                state.response = "Exploration cancelled because the session was closed."
            raise
        except Exception as exc:
            logger.exception(
                "Background explore execution failed",
                extra={"session_id": self.session_id},
            )
            state.status = ExploreTaskStatus.ERROR
            state.current_focus = None
            state.exit_reason = ExitReason.AGENT_ERROR
            state.response = f"Exploration failed with an internal error: {exc}."
        finally:
            self._settle_background_actions(state)
            if asyncio.current_task() is self._background_task:
                self._background_task = None
                self._background_task_kind = None

    async def _execute_explore_task(
        self,
        *,
        state: _ExploreTaskState,
        goal: str,
        initial_screenshot: bytes,
        max_steps: int,
    ) -> None:
        assert self.awareness_agent is not None
        assert self.action_agent is not None

        assessment = await self.awareness_agent.bootstrap(
            goal=goal,
            screenshot=initial_screenshot,
            context=self._tool_context("explore"),
        )
        self._apply_awareness_assessment(state, assessment)
        if state.status != ExploreTaskStatus.IN_PROGRESS:
            return

        steps_taken = 0
        while steps_taken < max_steps:
            next_action = self._next_explore_action(state.todo)
            if next_action is None:
                state.status = ExploreTaskStatus.STUCK
                state.current_focus = None
                state.exit_reason = ExitReason.STUCK
                state.response = "Exploration ended. No actionable TODO items remained."
                return

            result = await self.action_agent.execute_tool_instruction(
                next_action.action,
                test_context=self._tool_context("explore"),
            )
            state.actions_taken += self._count_action_result_actions(result)
            screenshot_bytes, screenshot_path = await self._background_result_snapshot(
                result
            )
            if screenshot_path:
                state.screenshot_path = screenshot_path

            last_action_summary = self._describe_action_result(
                result,
                instruction=next_action.action,
            )
            assessment = await self.awareness_agent.assess(
                goal=goal,
                screenshot=screenshot_bytes,
                todo=self._to_awareness_todo(state.todo),
                observations=list(state.observations),
                last_action_summary=last_action_summary,
                context=self._tool_context("explore"),
            )
            self._apply_awareness_assessment(state, assessment)
            steps_taken += 1
            if state.status != ExploreTaskStatus.IN_PROGRESS:
                return

        state.status = ExploreTaskStatus.MAX_STEPS_REACHED
        state.current_focus = None
        state.exit_reason = ExitReason.MAX_STEPS_REACHED
        state.response = (
            f"Exploration reached the configured max_steps limit of {max_steps}."
        )

    def _refresh_test_progress(self, state: _TestTaskState) -> None:
        if self.test_runner is None:
            return
        progress = self.test_runner.get_tool_mode_progress()
        if progress is None:
            return
        state.current_step = progress.current_step
        state.steps_total = max(state.steps_total, progress.steps_total)
        state.steps_completed = progress.steps_completed
        state.steps_failed = progress.steps_failed
        state.issues_found = dict(progress.issues_found)
        state.actions_taken = max(state.actions_taken, progress.actions_taken)
        if progress.phase:
            state.set_phase(progress.phase, agent=progress.last_model_agent)
        else:
            state.mark_progress(agent=progress.last_model_agent)
        if progress.phase_started_at:
            state.phase_started_at = progress.phase_started_at
        if progress.last_progress_at:
            state.last_progress_at = progress.last_progress_at
        if progress.last_model_agent:
            state.last_model_agent = progress.last_model_agent
        if progress.latest_action_artifact_path:
            state.latest_action_artifact_path = progress.latest_action_artifact_path
        if progress.latest_screenshot_path:
            state.screenshot_path = self._promote_background_artifact_screenshot(
                progress.latest_screenshot_path,
                state,
            )
        self._sync_test_task_metadata(state)

    def _apply_awareness_assessment(
        self,
        state: _ExploreTaskState,
        assessment: AwarenessAssessment,
    ) -> None:
        state.response = assessment.response or state.response
        state.current_focus = assessment.current_focus
        state.todo = self._normalize_todo(
            [
                ExploreTodoItem(
                    action=item.action,
                    status=self._todo_status_from_text(item.status),
                )
                for item in assessment.todo
                if item.action.strip()
            ],
            terminal=assessment.decision != "continue",
        )
        state.observations = list(assessment.observations)

        if assessment.decision == "goal_reached":
            state.status = ExploreTaskStatus.GOAL_REACHED
            state.current_focus = None
            state.exit_reason = ExitReason.GOAL_REACHED
        elif assessment.decision == "stuck":
            state.status = ExploreTaskStatus.STUCK
            state.current_focus = None
            state.exit_reason = ExitReason.STUCK
        elif assessment.decision == "aborted":
            state.status = ExploreTaskStatus.ABORTED
            state.current_focus = None
            state.exit_reason = ExitReason.ABORTED
        else:
            state.status = ExploreTaskStatus.IN_PROGRESS
            state.exit_reason = ExitReason.COMPLETED

    async def _background_result_snapshot(
        self,
        result: EnhancedActionResult,
    ) -> tuple[bytes, str | None]:
        screenshot_path = self._promote_result_screenshot(result)
        after = result.environment_state_after
        before = result.environment_state_before
        if after is not None and after.screenshot is not None:
            return after.screenshot, screenshot_path
        if before is not None and before.screenshot is not None:
            return before.screenshot, screenshot_path
        fresh = await self._take_screenshot_bytes()
        if screenshot_path is None:
            screenshot_path = self._store_screenshot_bytes(fresh)
        return fresh, screenshot_path

    def _summarize_test_result(self, final_state: TestState) -> dict[str, Any]:
        case_results = (
            list(final_state.test_report.test_cases) if final_state.test_report else []
        )
        steps_total = sum(case.steps_total for case in case_results)
        steps_completed = sum(case.steps_completed for case in case_results)
        steps_failed = sum(case.steps_failed for case in case_results)
        actions_taken = self._count_test_actions(case_results)
        screenshot_path = self._promote_test_screenshot(case_results)
        issues_found = self._collect_test_issues(case_results)

        if final_state.status in {TestStatus.PASSED, TestStatus.COMPLETED}:
            response = (
                (f"Test passed. All {steps_total} steps completed successfully.")
                if steps_total
                else "Test passed."
            )
            return {
                "status": TestTaskStatus.PASSED,
                "response": self._redact(response),
                "exit_reason": ExitReason.COMPLETED,
                "steps_total": steps_total,
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "issues_found": issues_found,
                "actions_taken": actions_taken,
                "screenshot_path": screenshot_path,
            }

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
                f"Steps 1-{steps_completed} passed before failure. "
                f"Expected: {failed_step.expected_result}. "
                f"Observed: {observed}"
            )
        elif failed_case is not None:
            response = failed_case.error_message or "Test failed."
        else:
            response = "Test failed."

        return {
            "status": (
                TestTaskStatus.MAX_STEPS_REACHED
                if failure_reason == ExitReason.MAX_STEPS_REACHED
                else TestTaskStatus.FAILED
            ),
            "response": self._redact(response),
            "exit_reason": failure_reason,
            "steps_total": steps_total,
            "steps_completed": steps_completed,
            "steps_failed": steps_failed,
            "issues_found": issues_found,
            "actions_taken": actions_taken,
            "screenshot_path": screenshot_path,
        }

    def _collect_test_issues(
        self,
        case_results: list[TestCaseResult],
    ) -> dict[str, str]:
        issues: dict[str, str] = {}
        for case in case_results:
            for step in [
                *case.setup_step_results,
                *case.step_results,
                *case.cleanup_step_results,
            ]:
                if step.status != TestStatus.FAILED:
                    continue
                observed = step.actual_result or step.error_message or "Step failed."
                issues[f"step_{step.step_number}"] = (
                    f"Expected {step.expected_result}. Observed: {observed}"
                )
        return issues

    def _test_status_response(self, state: _TestTaskState) -> str:
        if state.status == TestTaskStatus.IN_PROGRESS:
            progress_prefix = "Test in progress."
            if state.steps_total:
                progress_prefix = (
                    "Test in progress. "
                    f"Completed {state.steps_completed} of {state.steps_total} step(s)."
                )
            if state.phase == "planning":
                return f"{progress_prefix} Planning the scenario steps."
            if state.phase == "awaiting_step_reflection" and state.current_step:
                return (
                    f"{progress_prefix} Waiting for step reflection to finish for "
                    f"{state.current_step}."
                )
            if state.phase == "verifying" and state.current_step:
                return (
                    f"{progress_prefix} Verifying the outcome of {state.current_step}."
                )
            if state.phase == "cleanup" and state.current_step:
                return f"{progress_prefix} Cleaning up after {state.current_step}."
            if state.current_step:
                return f"{progress_prefix} Currently executing {state.current_step}."
            return progress_prefix
        return state.response

    @staticmethod
    def _test_timeout_reason(phase: str | None) -> str:
        if phase == "planning":
            return "timed_out_during_planning"
        if phase in {"awaiting_step_reflection", "verifying"}:
            return "timed_out_during_validation"
        return "timed_out_during_execution"

    @staticmethod
    def _test_timeout_response(
        *,
        state: _TestTaskState,
        timeout_seconds: int,
        timeout_reason: str,
    ) -> str:
        phase = state.phase or "executing_step"
        if timeout_reason == "timed_out_during_planning":
            return f"Test timed out after {timeout_seconds} seconds while planning the scenario."
        if state.current_step:
            return (
                f"Test timed out after {timeout_seconds} seconds during phase '{phase}' "
                f"while working on {state.current_step}."
            )
        return f"Test timed out after {timeout_seconds} seconds during phase '{phase}'."

    def _record_command(self, envelope: ToolCallEnvelope) -> None:
        """Persist command-level counters after one handled request."""

        self._last_activity_monotonic = time.monotonic()
        self.metadata.last_command_at = _utc_now().isoformat()
        self.metadata.last_command_name = envelope.command
        self.metadata.commands_executed += 1
        if envelope.command not in {"test-status", "explore-status"}:
            self.metadata.actions_executed += envelope.meta.actions_taken
        save_session_metadata(self.metadata)

    def _settle_background_actions(self, state: _BackgroundTaskBase) -> None:
        delta = max(state.actions_taken - state.accounted_actions, 0)
        if delta:
            self.metadata.actions_executed += delta
            state.accounted_actions += delta
            save_session_metadata(self.metadata)
        self._sync_test_task_metadata(state)

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

    def _background_run_id(self, task_kind: str) -> str:
        timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        return f"{self.session_id}-{task_kind}-{timestamp}-{uuid4().hex[:8]}"

    def _activate_background_context(self, run_id: str) -> None:
        """Bind the stable run context used by one background task."""
        set_run_id(run_id)
        initialize_debug_logger(
            run_id,
            debug_dir=get_screenshots_dir(self.session_id),
            reports_dir=get_logs_dir(self.session_id),
            ai_log_path=get_logs_dir(self.session_id) / "ai_interactions.jsonl",
        )

    def _sync_test_task_metadata(self, state: _BackgroundTaskBase) -> None:
        """Expose the latest background-test pointers in session.json."""
        if not isinstance(state, _TestTaskState):
            return
        self.metadata.latest_background_run_id = state.run_id
        self.metadata.latest_test_phase = state.phase
        self.metadata.latest_test_phase_started_at = state.phase_started_at
        self.metadata.latest_test_progress_at = state.last_progress_at
        self.metadata.latest_test_action_artifact_path = (
            state.latest_action_artifact_path
        )
        save_session_metadata(self.metadata)

    def _create_controller(
        self,
    ) -> DesktopController | MobileController | IOSController:
        """Instantiate the controller for this session backend."""
        import sys

        if self.backend == "mobile_adb":
            return MobileController(preferred_serial=self.android_serial)
        if self.backend == "mobile_ios":
            return IOSController(preferred_udid=self.ios_udid)
        if sys.platform == "darwin":
            from haindy.macos.controller import MacOSController

            return MacOSController()  # type: ignore[return-value]
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

    async def _take_screenshot_bytes(self) -> bytes:
        if self.controller is None:
            raise RuntimeError("Controller is not initialized.")
        async with self._device_lock:
            return await self.controller.driver.screenshot()

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

    def _promote_background_artifact_screenshot(
        self,
        source_path: str,
        state: _BackgroundTaskBase,
    ) -> str | None:
        if state.latest_source_screenshot == source_path:
            return state.screenshot_path
        try:
            promoted = self._promote_existing_screenshot(source_path)
        except FileNotFoundError:
            return state.screenshot_path
        state.latest_source_screenshot = source_path
        return promoted

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

    def _normalize_todo(
        self,
        todo: list[ExploreTodoItem],
        *,
        terminal: bool,
    ) -> list[ExploreTodoItem]:
        if not todo:
            return todo

        normalized: list[ExploreTodoItem] = []
        active_index: int | None = None
        for item in todo:
            if not item.action.strip():
                continue
            status = item.status
            if terminal and status == TodoStatus.IN_PROGRESS:
                status = TodoStatus.PENDING
            if status == TodoStatus.IN_PROGRESS and active_index is None:
                active_index = len(normalized)
            elif status == TodoStatus.IN_PROGRESS:
                status = TodoStatus.PENDING
            normalized.append(ExploreTodoItem(action=item.action, status=status))

        if not terminal and active_index is None:
            for index, item in enumerate(normalized):
                if item.status == TodoStatus.PENDING:
                    normalized[index] = ExploreTodoItem(
                        action=item.action,
                        status=TodoStatus.IN_PROGRESS,
                    )
                    break
        return normalized

    @staticmethod
    def _todo_status_from_text(value: str) -> TodoStatus:
        try:
            return TodoStatus(str(value).strip().lower())
        except ValueError:
            return TodoStatus.PENDING

    @staticmethod
    def _next_explore_action(
        todo: list[ExploreTodoItem],
    ) -> ExploreTodoItem | None:
        for item in todo:
            if item.status == TodoStatus.IN_PROGRESS:
                return item
        for item in todo:
            if item.status == TodoStatus.PENDING:
                return item
        return None

    @staticmethod
    def _to_awareness_todo(todo: list[ExploreTodoItem]) -> list[AwarenessTodoItem]:
        return [
            AwarenessTodoItem(action=item.action, status=item.status.value)
            for item in todo
        ]

    def _describe_action_result(
        self,
        result: EnhancedActionResult,
        *,
        instruction: str,
    ) -> str:
        observed = (
            result.final_model_output
            or (result.ai_analysis.actual_outcome if result.ai_analysis else None)
            or result.terminal_failure_reason
            or "No further detail available."
        )
        prefix = "Succeeded" if result.overall_success else "Failed"
        return f"{prefix} action '{instruction}'. {observed}"

    def _redact_mapping(self, values: dict[str, str]) -> dict[str, str]:
        return {key: self._redact(value) for key, value in values.items()}

    def _redact_string_list(self, values: list[str]) -> list[str]:
        return [self._redact(value) for value in values]

    def _redact_todo_items(
        self,
        todo: list[ExploreTodoItem],
    ) -> list[ExploreTodoItem]:
        return [
            ExploreTodoItem(
                action=self._redact(item.action),
                status=item.status,
            )
            for item in todo
        ]

    def _redact(self, text: str | None) -> str:
        return self.variables.redact(text) or ""
