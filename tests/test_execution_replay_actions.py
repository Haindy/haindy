import pytest

from src.desktop.execution_replay import (
    DriverActionError,
    normalize_driver_action,
    normalize_driver_actions,
)


def test_normalize_click_defaults() -> None:
    assert normalize_driver_action({"type": "click", "x": "1", "y": 2}) == {
        "type": "click",
        "x": 1,
        "y": 2,
        "button": "left",
        "click_count": 1,
    }


def test_normalize_press_key_accepts_key_alias_and_list() -> None:
    assert normalize_driver_action({"type": "press_key", "key": "ctrl+l"}) == {
        "type": "press_key",
        "keys": "ctrl+l",
    }
    assert normalize_driver_action({"type": "press_key", "keys": ["ctrl", "l"]}) == {
        "type": "press_key",
        "keys": ["ctrl", "l"],
    }


def test_normalize_wait_requires_duration_ms() -> None:
    with pytest.raises(DriverActionError):
        normalize_driver_action({"type": "wait"})


def test_normalize_driver_actions_bulk() -> None:
    assert normalize_driver_actions(
        [
            {"type": "move", "x": 10, "y": 20},
            {"type": "scroll_by_pixels", "x": 0, "y": -450},
        ]
    ) == [
        {"type": "move", "x": 10, "y": 20},
        {"type": "scroll_by_pixels", "x": 0, "y": -450},
    ]
