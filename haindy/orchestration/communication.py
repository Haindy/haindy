"""Coordinator diagnostics helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from haindy.monitoring.logger import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    """Coordinator event types retained for diagnostics and reporting."""

    START_TEST = "start_test"
    STOP_TEST = "stop_test"
    PAUSE_TEST = "pause_test"
    RESUME_TEST = "resume_test"
    EXECUTE_STEP = "execute_step"
    DETERMINE_ACTION = "determine_action"
    PLAN_TEST = "plan_test"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    ACTION_DETERMINED = "action_determined"
    PLAN_CREATED = "plan_created"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class DiagnosticsEvent:
    """Structured in-memory record of a coordinator event."""

    event_id: UUID = field(default_factory=uuid4)
    source: str = "coordinator"
    message_type: str = MessageType.INFO
    content: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CoordinatorDiagnostics:
    """Minimal event recorder used by the workflow coordinator."""

    def __init__(self, history_limit: int = 1000) -> None:
        self._history_limit = history_limit
        self._history: list[DiagnosticsEvent] = []
        self._event_count: dict[str, int] = defaultdict(int)

    def record_event(
        self,
        *,
        message_type: MessageType,
        content: dict[str, Any],
        source: str = "coordinator",
    ) -> DiagnosticsEvent:
        """Append a diagnostics event and update aggregate counters."""
        event = DiagnosticsEvent(
            source=source,
            message_type=message_type,
            content=content,
        )
        self._history.append(event)
        self._event_count[event.message_type] += 1
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
        logger.info(
            "Coordinator diagnostics event recorded",
            extra={
                "message_type": event.message_type,
                "source": event.source,
                "event_id": str(event.event_id),
            },
        )
        return event

    def get_history(
        self,
        *,
        message_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[DiagnosticsEvent]:
        """Return recent diagnostics events with optional filtering."""
        history = self._history
        if message_type:
            history = [event for event in history if event.message_type == message_type]
        if source:
            history = [event for event in history if event.source == source]
        return history[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Expose lightweight diagnostics counters."""
        return {
            "total_messages": sum(self._event_count.values()),
            "message_counts": dict(self._event_count),
            "history_size": len(self._history),
        }

    def clear_history(self) -> None:
        """Clear recorded diagnostics history."""
        self._history.clear()
        logger.info("Coordinator diagnostics history cleared")

    async def shutdown(self) -> None:
        """Release in-memory diagnostics state."""
        self.clear_history()
