"""Shared request/response models for tool-call mode."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    """Top-level command result status."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class ExitReason(str, Enum):
    """Machine-readable termination reason."""

    COMPLETED = "completed"
    DISPATCHED = "dispatched"
    ASSERTION_FAILED = "assertion_failed"
    MAX_STEPS_REACHED = "max_steps_reached"
    MAX_ACTIONS_REACHED = "max_actions_reached"
    ELEMENT_NOT_FOUND = "element_not_found"
    COMMAND_TIMEOUT = "command_timeout"
    TIMEOUT = "timeout"
    AGENT_ERROR = "agent_error"
    DEVICE_ERROR = "device_error"
    SESSION_BUSY = "session_busy"
    GOAL_REACHED = "goal_reached"
    STUCK = "stuck"
    ABORTED = "aborted"


class TestTaskStatus(str, Enum):
    """Background test status values returned by `test-status`."""

    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    MAX_STEPS_REACHED = "max_steps_reached"


class ExploreTaskStatus(str, Enum):
    """Background explore status values returned by `explore-status`."""

    IN_PROGRESS = "in_progress"
    GOAL_REACHED = "goal_reached"
    STUCK = "stuck"
    ABORTED = "aborted"
    TIMEOUT = "timeout"
    MAX_STEPS_REACHED = "max_steps_reached"
    ERROR = "error"


class TodoStatus(str, Enum):
    """TODO item status for explore polling."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


class ToolCallMeta(BaseModel):
    """Stable metadata included with every tool-call response."""

    exit_reason: ExitReason
    duration_ms: int = Field(default=0, ge=0)
    actions_taken: int = Field(default=0, ge=0)


class SessionListEntry(BaseModel):
    """Live-session summary returned by `haindy session list`."""

    session_id: str
    backend: str
    created_at: str
    steps_executed: int = Field(default=0, ge=0)
    idle_seconds: int = Field(default=0, ge=0)


class ExploreTodoItem(BaseModel):
    """One TODO item in the explore working memory."""

    action: str
    status: TodoStatus


class ToolCallEnvelope(BaseModel):
    """JSON envelope printed by tool-call CLI commands."""

    session_id: str | None
    run_id: str | None = None
    command: str
    status: CommandStatus
    response: str
    screenshot_path: str | None
    meta: ToolCallMeta
    steps_total: int | None = None
    steps_completed: int | None = None
    steps_failed: int | None = None
    test_status: TestTaskStatus | None = None
    current_step: str | None = None
    phase: str | None = None
    phase_started_at: str | None = None
    last_model_agent: str | None = None
    last_progress_at: str | None = None
    latest_action_artifact_path: str | None = None
    issues_found: dict[str, str] | None = None
    elapsed_time_seconds: int | None = None
    explore_status: ExploreTaskStatus | None = None
    current_focus: str | None = None
    todo: list[ExploreTodoItem] | None = None
    observations: list[str] | None = None
    sessions: list[SessionListEntry] | None = None
    vars: dict[str, str] | None = None


class ToolCallRequest(BaseModel):
    """NDJSON request sent from CLI clients to the session daemon."""

    command: str
    instruction: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    var_name: str | None = None
    var_value: str | None = None
    var_secret: bool = False


class SessionMetadata(BaseModel):
    """On-disk session metadata persisted in `session.json`."""

    session_id: str
    backend: str
    created_at: str
    idle_timeout_seconds: int = Field(default=1800, ge=1)
    pid: int | None = None
    status: str = "starting"
    last_command_at: str | None = None
    last_command_name: str | None = None
    closed_at: str | None = None
    latest_screenshot_path: str | None = None
    screenshot_count: int = Field(default=0, ge=0)
    commands_executed: int = Field(default=0, ge=0)
    actions_executed: int = Field(default=0, ge=0)
    android_serial: str | None = None
    android_app: str | None = None
    ios_udid: str | None = None
    ios_app: str | None = None
    notes: str | None = None
    latest_background_run_id: str | None = None
    latest_test_phase: str | None = None
    latest_test_phase_started_at: str | None = None
    latest_test_progress_at: str | None = None
    latest_test_action_artifact_path: str | None = None

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        backend: str,
        idle_timeout_seconds: int,
        android_serial: str | None = None,
        android_app: str | None = None,
        ios_udid: str | None = None,
        ios_app: str | None = None,
    ) -> SessionMetadata:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            session_id=session_id,
            backend=backend,
            created_at=now,
            last_command_at=now,
            idle_timeout_seconds=idle_timeout_seconds,
            android_serial=android_serial or None,
            android_app=android_app or None,
            ios_udid=ios_udid or None,
            ios_app=ios_app or None,
        )


def make_envelope(
    *,
    session_id: str | None,
    run_id: str | None = None,
    command: str,
    status: CommandStatus,
    response: str,
    screenshot_path: str | None,
    exit_reason: ExitReason,
    duration_ms: int,
    actions_taken: int,
    steps_total: int | None = None,
    steps_completed: int | None = None,
    steps_failed: int | None = None,
    test_status: TestTaskStatus | None = None,
    current_step: str | None = None,
    phase: str | None = None,
    phase_started_at: str | None = None,
    last_model_agent: str | None = None,
    last_progress_at: str | None = None,
    latest_action_artifact_path: str | None = None,
    issues_found: dict[str, str] | None = None,
    elapsed_time_seconds: int | None = None,
    explore_status: ExploreTaskStatus | None = None,
    current_focus: str | None = None,
    todo: list[ExploreTodoItem] | None = None,
    observations: list[str] | None = None,
    sessions: list[SessionListEntry] | None = None,
    vars_map: dict[str, str] | None = None,
) -> ToolCallEnvelope:
    """Build the stable tool-call response envelope."""

    return ToolCallEnvelope(
        session_id=session_id,
        run_id=run_id,
        command=command,
        status=status,
        response=response,
        screenshot_path=screenshot_path,
        meta=ToolCallMeta(
            exit_reason=exit_reason,
            duration_ms=max(int(duration_ms), 0),
            actions_taken=max(int(actions_taken), 0),
        ),
        steps_total=steps_total,
        steps_completed=steps_completed,
        steps_failed=steps_failed,
        test_status=test_status,
        current_step=current_step,
        phase=phase,
        phase_started_at=phase_started_at,
        last_model_agent=last_model_agent,
        last_progress_at=last_progress_at,
        latest_action_artifact_path=latest_action_artifact_path,
        issues_found=issues_found,
        elapsed_time_seconds=elapsed_time_seconds,
        explore_status=explore_status,
        current_focus=current_focus,
        todo=todo,
        observations=observations,
        sessions=sessions,
        vars=vars_map,
    )


def envelope_exit_code(envelope: ToolCallEnvelope) -> int:
    """Translate an envelope status into a process exit code."""

    return 0 if envelope.status == CommandStatus.SUCCESS else 1


def public_command_name(command: str) -> str:
    """Map daemon-internal request names onto the public JSON contract."""

    return "session" if command.startswith("session_") else command.replace("_", "-")
