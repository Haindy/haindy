"""Tests for coordinator diagnostics helpers."""

from src.orchestration.communication import (
    CoordinatorDiagnostics,
    DiagnosticsEvent,
    MessageType,
)


def test_record_event_adds_history_and_updates_statistics() -> None:
    diagnostics = CoordinatorDiagnostics()

    event = diagnostics.record_event(
        message_type=MessageType.PLAN_CREATED,
        content={"plan": "Login flow"},
        source="test_planner",
    )

    assert isinstance(event, DiagnosticsEvent)
    assert event.message_type == MessageType.PLAN_CREATED
    assert event.source == "test_planner"
    assert event.content == {"plan": "Login flow"}
    assert diagnostics.get_statistics() == {
        "total_messages": 1,
        "message_counts": {MessageType.PLAN_CREATED: 1},
        "history_size": 1,
    }


def test_get_history_filters_by_message_type_and_source() -> None:
    diagnostics = CoordinatorDiagnostics()
    diagnostics.record_event(
        message_type=MessageType.INFO,
        content={"detail": "startup"},
        source="coordinator",
    )
    diagnostics.record_event(
        message_type=MessageType.STATUS_UPDATE,
        content={"status": "executing"},
        source="coordinator",
    )
    diagnostics.record_event(
        message_type=MessageType.STATUS_UPDATE,
        content={"status": "planning"},
        source="test_planner",
    )

    status_events = diagnostics.get_history(message_type=MessageType.STATUS_UPDATE)
    coordinator_events = diagnostics.get_history(source="coordinator")

    assert len(status_events) == 2
    assert len(coordinator_events) == 2
    assert all(event.source == "coordinator" for event in coordinator_events)


def test_history_limit_keeps_only_recent_events() -> None:
    diagnostics = CoordinatorDiagnostics(history_limit=3)

    for index in range(5):
        diagnostics.record_event(
            message_type=MessageType.INFO,
            content={"index": index},
        )

    history = diagnostics.get_history(limit=10)

    assert len(history) == 3
    assert [event.content["index"] for event in history] == [2, 3, 4]


def test_clear_history_resets_history_but_preserves_counters() -> None:
    diagnostics = CoordinatorDiagnostics()
    diagnostics.record_event(
        message_type=MessageType.INFO,
        content={"detail": "startup"},
    )

    diagnostics.clear_history()

    assert diagnostics.get_history() == []
    assert diagnostics.get_statistics() == {
        "total_messages": 1,
        "message_counts": {MessageType.INFO: 1},
        "history_size": 0,
    }
