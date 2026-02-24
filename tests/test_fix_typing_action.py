"""Replay extraction tests for typing-related computer actions."""

from src.agents.action_agent import ActionAgent
from src.core.enhanced_types import ComputerToolTurn


def test_extract_driver_actions_for_type_text_at_with_clear_and_enter() -> None:
    turns = [
        ComputerToolTurn(
            call_id="c1",
            action_type="type_text_at",
            parameters={"text": "hello", "clear_before_typing": True, "press_enter": True},
            status="executed",
            metadata={"x": 200, "y": 300, "normalized_coords": False},
        )
    ]

    actions = ActionAgent._extract_driver_actions(turns)
    assert actions[0]["type"] == "click"
    assert actions[1] == {"type": "press_key", "keys": "ctrl+a"}
    assert actions[2] == {"type": "press_key", "keys": "backspace"}
    assert actions[3] == {"type": "type_text", "text": "hello"}
    assert actions[4] == {"type": "press_key", "keys": "enter"}


def test_extract_driver_actions_normalizes_key_combinations() -> None:
    turns = [
        ComputerToolTurn(
            call_id="c2",
            action_type="key_combination",
            parameters={"keys": ["ctrl", "k"]},
            status="executed",
        )
    ]

    actions = ActionAgent._extract_driver_actions(turns)
    assert actions == [{"type": "press_key", "keys": "ctrl+k"}]
