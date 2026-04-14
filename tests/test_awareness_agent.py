"""Tests for tool-call awareness agent parsing helpers."""

import json

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
