"""Tests for tool-call awareness agent parsing helpers."""

import json

import pytest

from haindy.agents.awareness_agent import AwarenessAgent, AwarenessTodoItem


def test_awareness_agent_parse_response_accepts_json_string_payload() -> None:
    response = {
        "content": json.dumps(
            {
                "decision": "goal_reached",
                "response": "Goal reached.",
                "current_focus": None,
                "todo": [
                    {"action": "Open Settings", "status": "done"},
                    {"action": "", "status": "pending"},
                ],
                "observations": ["Settings screen visible", ""],
            }
        )
    }

    assessment = AwarenessAgent._parse_response(response)

    assert assessment.decision == "goal_reached"
    assert assessment.response == "Goal reached."
    assert assessment.current_focus is None
    assert assessment.todo == [AwarenessTodoItem(action="Open Settings", status="done")]
    assert assessment.observations == ["Settings screen visible"]


def test_awareness_agent_parse_response_accepts_dict_payload() -> None:
    response = {
        "content": {
            "decision": "continue",
            "response": "Keep exploring.",
            "current_focus": "Open the notifications screen",
            "todo": [{"action": "Tap Notifications", "status": "in_progress"}],
            "observations": ["Settings main screen visible"],
        }
    }

    assessment = AwarenessAgent._parse_response(response)

    assert assessment.decision == "continue"
    assert assessment.current_focus == "Open the notifications screen"
    assert assessment.todo[0].action == "Tap Notifications"
    assert assessment.todo[0].status == "in_progress"


def test_awareness_agent_parse_response_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="response was empty"):
        AwarenessAgent._parse_response({"content": {}})


def test_awareness_agent_parse_response_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        AwarenessAgent._parse_response({"content": {"decision": "continue"}})
