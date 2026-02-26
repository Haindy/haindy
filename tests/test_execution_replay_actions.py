import pytest

from src.desktop.execution_replay import (
    DriverActionError,
    normalize_driver_action,
    normalize_driver_actions,
    replay_driver_actions,
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
        "keys": "ctrl+l",
    }


@pytest.mark.asyncio
async def test_replay_press_key_passes_string_to_driver() -> None:
    class DummyDriver:
        def __init__(self) -> None:
            self.keys: list[object] = []

        async def click(self, *args, **kwargs):  # pragma: no cover - not used
            return

        async def move_mouse(self, *args, **kwargs):  # pragma: no cover - not used
            return

        async def drag_mouse(self, *args, **kwargs):  # pragma: no cover - not used
            return

        async def scroll_by_pixels(
            self, *args, **kwargs
        ):  # pragma: no cover - not used
            return

        async def type_text(self, *args, **kwargs):  # pragma: no cover - not used
            return

        async def press_key(self, key):
            self.keys.append(key)

        async def wait(self, *args, **kwargs):
            return

    driver = DummyDriver()
    await replay_driver_actions(
        driver,  # type: ignore[arg-type]
        [{"type": "press_key", "keys": ["ctrl", "l"]}],
        stabilization_wait_ms=0,
    )
    assert driver.keys == ["ctrl+l"]


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
