"""Tests for tool-call runtime background task behavior."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from haindy.agents.awareness_agent import AwarenessAssessment, AwarenessTodoItem
from haindy.agents.test_runner import ToolModeTestProgress
from haindy.core.enhanced_types import (
    AIAnalysis,
    EnhancedActionResult,
    EnvironmentState,
    ExecutionResult,
    ValidationResult,
)
from haindy.core.types import TestCase as CoreTestCase
from haindy.core.types import TestCaseResult as CoreTestCaseResult
from haindy.core.types import TestPlan as CoreTestPlan
from haindy.core.types import TestReport as CoreTestReport
from haindy.core.types import TestState as CoreTestState
from haindy.core.types import TestStatus as CoreTestStatus
from haindy.core.types import TestStep as CoreTestStep
from haindy.tool_call_mode.daemon import ToolCallDaemon
from haindy.tool_call_mode.models import (
    CommandStatus,
    ExitReason,
    ExploreTaskStatus,
    TodoStatus,
    ToolCallRequest,
    make_envelope,
)
from haindy.tool_call_mode.models import (
    TestTaskStatus as ToolTestTaskStatus,
)
from haindy.tool_call_mode.paths import ensure_session_layout
from haindy.tool_call_mode.runtime import ToolCallSessionRuntime


def _make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    session_id: str = "session-123",
) -> ToolCallSessionRuntime:
    monkeypatch.setenv("HAINDY_HOME", str(tmp_path / "haindy-home"))
    ensure_session_layout(session_id)
    runtime = ToolCallSessionRuntime(
        session_id=session_id,
        backend="desktop",
        idle_timeout_seconds=1800,
    )
    runtime.metadata.status = "ready"

    async def _fake_screenshot() -> bytes:
        return b"fake-screenshot"

    monkeypatch.setattr(runtime, "_take_screenshot_bytes", _fake_screenshot)
    return runtime


def _make_plan() -> tuple[CoreTestPlan, CoreTestStep]:
    step = CoreTestStep(
        step_number=1,
        description="Open the dashboard",
        action="Tap Dashboard",
        expected_result="The dashboard is visible",
    )
    case = CoreTestCase(
        test_id="TC001",
        name="Dashboard smoke test",
        description="Verify the dashboard opens.",
        steps=[step],
    )
    plan = CoreTestPlan(
        name="Tool mode test plan",
        description="A focused tool-call mode test.",
        requirements_source="inline",
        test_cases=[case],
    )
    return plan, step


def _make_case_result(
    step: CoreTestStep,
    *,
    status: CoreTestStatus,
    screenshot_path: str | None,
    actual_result: str,
    error_message: str | None = None,
) -> CoreTestCaseResult:
    now = datetime.now(timezone.utc)
    step_result = {
        "step_id": step.step_id,
        "step_number": step.step_number,
        "status": status,
        "started_at": now,
        "completed_at": now,
        "action": step.action,
        "expected_result": step.expected_result,
        "actual_result": actual_result,
        "screenshot_after": screenshot_path,
        "error_message": error_message,
        "actions_performed": [{"action": step.action}],
    }
    return CoreTestCaseResult(
        case_id=step.step_id,
        test_id="TC001",
        name="Dashboard smoke test",
        status=status,
        started_at=now,
        completed_at=now,
        steps_total=1,
        steps_completed=1 if status == CoreTestStatus.PASSED else 0,
        steps_failed=1 if status == CoreTestStatus.FAILED else 0,
        step_results=[step_result],
        error_message=error_message,
    )


def _make_test_state(
    plan: CoreTestPlan,
    case_result: CoreTestCaseResult,
    *,
    status: CoreTestStatus,
) -> CoreTestState:
    now = datetime.now(timezone.utc)
    report = CoreTestReport(
        test_plan_id=plan.plan_id,
        test_plan_name=plan.name,
        started_at=now,
        completed_at=now,
        status=status,
        test_cases=[case_result],
    )
    return CoreTestState(
        test_plan=plan,
        status=status,
        start_time=now,
        end_time=now,
        test_report=report,
    )


def _make_action_result(
    instruction: str,
    *,
    screenshot: bytes,
    success: bool = True,
) -> EnhancedActionResult:
    step = CoreTestStep(
        step_number=1,
        description=instruction,
        action=instruction,
        expected_result="The action succeeds",
    )
    return EnhancedActionResult(
        test_step_id=step.step_id,
        test_step=step,
        test_context={"tool_mode": True},
        validation=ValidationResult(
            valid=True,
            confidence=0.99,
            reasoning="Valid action.",
        ),
        environment_state_before=EnvironmentState(
            url="app://before",
            title="Before",
            viewport_size=(1280, 720),
            screenshot=screenshot,
        ),
        environment_state_after=EnvironmentState(
            url="app://after",
            title="After",
            viewport_size=(1280, 720),
            screenshot=screenshot,
        ),
        execution=ExecutionResult(
            success=success,
            execution_time_ms=12,
        ),
        ai_analysis=AIAnalysis(
            success=success,
            confidence=0.95,
            actual_outcome="The UI moved to the requested screen.",
            matches_expected=success,
        ),
        final_model_output="Action completed.",
        overall_success=success,
    )


class _FakeTestPlanner:
    def __init__(self, plan: CoreTestPlan) -> None:
        self.plan = plan
        self.calls: list[tuple[str, int, dict[str, object]]] = []

    async def create_tool_mode_test_plan(
        self,
        scenario: str,
        *,
        max_steps: int,
        context: dict[str, object],
    ) -> CoreTestPlan:
        self.calls.append((scenario, max_steps, context))
        return self.plan


class _FakeTestRunner:
    def __init__(
        self,
        final_state: CoreTestState,
        *,
        gate: asyncio.Event,
        artifact_path: Path,
    ) -> None:
        self._final_state = final_state
        self._gate = gate
        self._artifact_path = artifact_path
        self._progress: ToolModeTestProgress | None = None

    def get_tool_mode_progress(self) -> ToolModeTestProgress | None:
        return self._progress

    async def execute_test_plan(self, state: CoreTestState) -> CoreTestState:
        step = state.test_plan.steps[0]
        state.current_step = step
        self._progress = ToolModeTestProgress(
            current_step=f"Step {step.step_number}: {step.description}",
            steps_total=1,
            steps_completed=0,
            steps_failed=0,
            issues_found={},
            latest_screenshot_path=None,
            actions_taken=0,
        )
        await self._gate.wait()
        state.current_step = None
        self._progress = ToolModeTestProgress(
            current_step=None,
            steps_total=1,
            steps_completed=1,
            steps_failed=0,
            issues_found={},
            latest_screenshot_path=str(self._artifact_path),
            actions_taken=1,
        )
        return self._final_state


class _FakeAwarenessAgent:
    async def bootstrap(
        self,
        *,
        goal: str,
        screenshot: bytes,
        context: dict[str, object] | None = None,
    ) -> AwarenessAssessment:
        del goal, screenshot, context
        return AwarenessAssessment(
            decision="continue",
            response="Exploring the settings UI.",
            current_focus="Open the notifications screen",
            todo=[AwarenessTodoItem(action="Tap Notifications", status="in_progress")],
            observations=["Settings main screen is visible."],
        )

    async def assess(
        self,
        *,
        goal: str,
        screenshot: bytes,
        todo: list[AwarenessTodoItem],
        observations: list[str],
        last_action_summary: str | None,
        context: dict[str, object] | None = None,
    ) -> AwarenessAssessment:
        del goal, screenshot, todo, last_action_summary, context
        return AwarenessAssessment(
            decision="goal_reached",
            response="Goal reached. Notifications settings are visible.",
            current_focus=None,
            todo=[AwarenessTodoItem(action="Tap Notifications", status="done")],
            observations=[*observations, "Notifications settings screen is visible."],
        )


class _FakeActionAgent:
    def __init__(self, *, gate: asyncio.Event) -> None:
        self._gate = gate
        self.started = asyncio.Event()

    async def execute_tool_instruction(
        self,
        instruction: str,
        *,
        test_context: dict[str, object],
    ) -> EnhancedActionResult:
        del test_context
        self.started.set()
        await self._gate.wait()
        return _make_action_result(instruction, screenshot=b"after-image")


class _FakeWriter:
    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_test_status_requires_dispatched_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)

    envelope = await runtime.handle_request(ToolCallRequest(command="test_status"))

    assert envelope.status == CommandStatus.ERROR
    assert envelope.command == "test-status"
    assert envelope.response == "No test has been dispatched in this session."


@pytest.mark.asyncio
async def test_runtime_test_dispatch_and_status_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    plan, step = _make_plan()
    artifact_path = tmp_path / "runner-artifact.png"
    artifact_path.write_bytes(b"runner-artifact")
    final_state = _make_test_state(
        plan,
        _make_case_result(
            step,
            status=CoreTestStatus.PASSED,
            screenshot_path=str(artifact_path),
            actual_result="The dashboard is visible.",
        ),
        status=CoreTestStatus.PASSED,
    )
    gate = asyncio.Event()

    runtime.test_planner = _FakeTestPlanner(plan)
    runtime.test_runner = _FakeTestRunner(
        final_state,
        gate=gate,
        artifact_path=artifact_path,
    )

    dispatch = await runtime.handle_request(
        ToolCallRequest(
            command="test",
            instruction="verify the dashboard appears",
            options={"max_steps": 5, "timeout_seconds": 30},
        )
    )

    assert dispatch.status == CommandStatus.SUCCESS
    assert dispatch.meta.exit_reason == ExitReason.DISPATCHED
    assert dispatch.command == "test"
    assert dispatch.screenshot_path is not None

    await asyncio.sleep(0)

    in_progress = await runtime.handle_request(ToolCallRequest(command="test_status"))

    assert in_progress.status == CommandStatus.SUCCESS
    assert in_progress.test_status == ToolTestTaskStatus.IN_PROGRESS
    assert in_progress.current_step == "Step 1: Open the dashboard"
    assert in_progress.steps_total == 1
    assert in_progress.steps_completed == 0

    gate.set()
    background_task = runtime._background_task
    assert background_task is not None
    await asyncio.wait_for(asyncio.shield(background_task), timeout=1)

    completed = await runtime.handle_request(ToolCallRequest(command="test_status"))

    assert completed.test_status == ToolTestTaskStatus.PASSED
    assert completed.meta.exit_reason == ExitReason.COMPLETED
    assert completed.steps_completed == 1
    assert completed.steps_failed == 0
    assert completed.issues_found == {}
    assert completed.screenshot_path is not None


@pytest.mark.asyncio
async def test_runtime_explore_dispatch_and_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    action_gate = asyncio.Event()

    runtime.awareness_agent = _FakeAwarenessAgent()
    runtime.action_agent = _FakeActionAgent(gate=action_gate)

    dispatch = await runtime.handle_request(
        ToolCallRequest(
            command="explore",
            instruction="find the notifications settings screen",
            options={"max_steps": 4},
        )
    )

    assert dispatch.status == CommandStatus.SUCCESS
    assert dispatch.meta.exit_reason == ExitReason.DISPATCHED
    assert dispatch.command == "explore"

    await runtime.action_agent.started.wait()

    in_progress = await runtime.handle_request(
        ToolCallRequest(command="explore_status")
    )

    assert in_progress.explore_status == ExploreTaskStatus.IN_PROGRESS
    assert in_progress.current_focus == "Open the notifications screen"
    assert in_progress.todo is not None
    assert in_progress.todo[0].status == TodoStatus.IN_PROGRESS

    action_gate.set()
    background_task = runtime._background_task
    assert background_task is not None
    await asyncio.wait_for(asyncio.shield(background_task), timeout=1)

    completed = await runtime.handle_request(ToolCallRequest(command="explore_status"))

    assert completed.explore_status == ExploreTaskStatus.GOAL_REACHED
    assert completed.meta.exit_reason == ExitReason.GOAL_REACHED
    assert completed.todo is not None
    assert completed.todo[0].status == TodoStatus.DONE
    assert "Goal reached" in completed.response


@pytest.mark.asyncio
async def test_runtime_session_close_cancels_background_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    plan, step = _make_plan()
    artifact_path = tmp_path / "runner-cancel-artifact.png"
    artifact_path.write_bytes(b"runner-artifact")
    final_state = _make_test_state(
        plan,
        _make_case_result(
            step,
            status=CoreTestStatus.PASSED,
            screenshot_path=str(artifact_path),
            actual_result="The dashboard is visible.",
        ),
        status=CoreTestStatus.PASSED,
    )
    gate = asyncio.Event()

    runtime.test_planner = _FakeTestPlanner(plan)
    runtime.test_runner = _FakeTestRunner(
        final_state,
        gate=gate,
        artifact_path=artifact_path,
    )

    await runtime.handle_request(
        ToolCallRequest(
            command="test",
            instruction="verify the dashboard appears",
            options={"max_steps": 5, "timeout_seconds": 30},
        )
    )
    await asyncio.sleep(0)

    close = await runtime.handle_request(ToolCallRequest(command="session_close"))

    assert close.status == CommandStatus.SUCCESS
    assert runtime.is_close_requested() is True
    assert runtime.is_background_task_active() is False
    assert runtime._test_task_state is not None
    assert runtime._test_task_state.status == ToolTestTaskStatus.ERROR
    assert (
        "cancelled because the session was closed" in runtime._test_task_state.response
    )


@pytest.mark.asyncio
async def test_daemon_rejects_conflicting_command_while_background_task_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class _FakeRuntime:
        def __init__(self) -> None:
            self.metadata = SimpleNamespace(latest_screenshot_path="/tmp/latest.png")
            self.handled: list[str] = []

        def is_background_task_active(self) -> bool:
            return True

        @staticmethod
        def background_command_allowed(command: str) -> bool:
            return command == "screenshot"

        async def handle_request(self, request: ToolCallRequest):
            self.handled.append(request.command)
            return make_envelope(
                session_id="session-123",
                command="screenshot",
                status=CommandStatus.SUCCESS,
                response="Screenshot captured.",
                screenshot_path="/tmp/latest.png",
                exit_reason=ExitReason.COMPLETED,
                duration_ms=0,
                actions_taken=0,
            )

        def is_close_requested(self) -> bool:
            return False

    async def _fake_read_request(_reader):
        return ToolCallRequest(command="act", instruction="tap the button")

    async def _fake_write_envelope(_writer, envelope):
        captured.append(envelope)

    daemon = ToolCallDaemon(
        session_id="session-123",
        backend="desktop",
        idle_timeout_seconds=1800,
    )
    fake_runtime = _FakeRuntime()
    daemon.runtime = fake_runtime  # type: ignore[assignment]
    monkeypatch.setattr("haindy.tool_call_mode.daemon.read_request", _fake_read_request)
    monkeypatch.setattr(
        "haindy.tool_call_mode.daemon.write_envelope",
        _fake_write_envelope,
    )

    await daemon._handle_client(asyncio.StreamReader(), _FakeWriter())

    assert len(captured) == 1
    envelope = captured[0]
    assert envelope.status == CommandStatus.ERROR
    assert envelope.meta.exit_reason == ExitReason.SESSION_BUSY
    assert fake_runtime.handled == []


@pytest.mark.asyncio
async def test_runtime_explore_status_redacts_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    runtime.variables.set("PASSWORD", "Test1234!!", secret=True)

    class _State:
        status = ExploreTaskStatus.IN_PROGRESS
        response = "Entering password Test1234!! on the login screen."
        screenshot_path = "/tmp/explore-status.png"
        exit_reason = ExitReason.COMPLETED
        actions_taken = 3
        current_focus = "Type Test1234!! into the password field"
        todo = [
            SimpleNamespace(
                action="Enter password: Test1234!!",
                status=TodoStatus.IN_PROGRESS,
            )
        ]
        observations = ["Password Test1234!! has been entered."]

        @staticmethod
        def elapsed_seconds() -> int:
            return 12

    runtime._explore_task_state = _State()

    envelope = await runtime.handle_request(ToolCallRequest(command="explore_status"))

    assert "[redacted]" in envelope.response
    assert "Test1234!!" not in envelope.response
    assert envelope.current_focus is not None
    assert "[redacted]" in envelope.current_focus
    assert envelope.todo is not None
    assert "[redacted]" in envelope.todo[0].action
    assert envelope.observations == ["Password [redacted] has been entered."]


@pytest.mark.asyncio
async def test_runtime_test_status_redacts_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(monkeypatch, tmp_path)
    runtime.variables.set("PASSWORD", "Test1234!!", secret=True)

    class _State:
        status = ToolTestTaskStatus.FAILED
        response = "Test failed after typing Test1234!!."
        screenshot_path = "/tmp/test-status.png"
        exit_reason = ExitReason.ASSERTION_FAILED
        actions_taken = 2
        current_step = "Type Test1234!! into the password field"
        steps_total = 1
        steps_completed = 0
        steps_failed = 1
        issues_found = {
            "step_1": "Expected secure login. Observed: password Test1234!! leaked."
        }

        @staticmethod
        def elapsed_seconds() -> int:
            return 8

    runtime._test_task_state = _State()

    envelope = await runtime.handle_request(ToolCallRequest(command="test_status"))

    assert "[redacted]" in envelope.response
    assert "Test1234!!" not in envelope.response
    assert envelope.current_step is not None
    assert "[redacted]" in envelope.current_step
    assert envelope.issues_found is not None
    assert "[redacted]" in envelope.issues_found["step_1"]
