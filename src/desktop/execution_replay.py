"""Record/replay helpers for driver-level actions."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.interfaces import BrowserDriver


class DriverActionError(ValueError):
    """Raised when a recorded driver action is invalid."""


def _require_int(value: object, *, field: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise DriverActionError(f"Invalid int for '{field}': {value!r}") from exc


def _normalize_button(value: object) -> str:
    button = str(value or "left").strip().lower()
    if button not in {"left", "right", "middle"}:
        return "left"
    return button


def _scroll_xy(direction: str, magnitude: int) -> Tuple[int, int]:
    direction_norm = str(direction or "").strip().lower()
    amount = abs(int(magnitude))
    if direction_norm == "down":
        return (0, amount)
    if direction_norm == "up":
        return (0, -amount)
    if direction_norm == "right":
        return (amount, 0)
    if direction_norm == "left":
        return (-amount, 0)
    raise DriverActionError(f"Invalid scroll direction: {direction!r}")


def normalize_driver_action(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonical driver action dict (v1 schema)."""
    if not isinstance(raw, dict):
        raise DriverActionError("Driver action must be a dict")
    action_type = str(raw.get("type") or "").strip()
    if not action_type:
        raise DriverActionError("Driver action missing 'type'")

    if action_type == "click":
        x = _require_int(raw.get("x"), field="x")
        y = _require_int(raw.get("y"), field="y")
        button = _normalize_button(raw.get("button"))
        click_count = max(_require_int(raw.get("click_count", 1), field="click_count"), 1)
        return {
            "type": "click",
            "x": x,
            "y": y,
            "button": button,
            "click_count": click_count,
        }

    if action_type == "move":
        x = _require_int(raw.get("x"), field="x")
        y = _require_int(raw.get("y"), field="y")
        return {"type": "move", "x": x, "y": y}

    if action_type == "drag":
        start_x = _require_int(raw.get("start_x"), field="start_x")
        start_y = _require_int(raw.get("start_y"), field="start_y")
        end_x = _require_int(raw.get("end_x"), field="end_x")
        end_y = _require_int(raw.get("end_y"), field="end_y")
        return {
            "type": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }

    if action_type == "scroll_by_pixels":
        x = _require_int(raw.get("x", 0), field="x")
        y = _require_int(raw.get("y", 0), field="y")
        return {"type": "scroll_by_pixels", "x": x, "y": y}

    if action_type == "type_text":
        if "text" not in raw:
            raise DriverActionError("type_text action missing 'text'")
        return {"type": "type_text", "text": str(raw.get("text") or "")}

    if action_type == "press_key":
        keys = raw.get("keys")
        if keys is None:
            keys = raw.get("key")
        if keys is None:
            raise DriverActionError("press_key action missing 'keys'/'key'")
        if isinstance(keys, (list, tuple)):
            normalized = [str(item).strip() for item in keys if str(item).strip()]
            if not normalized:
                raise DriverActionError("press_key action has empty keys list")
            return {"type": "press_key", "keys": normalized}
        normalized_single = str(keys).strip()
        if not normalized_single:
            raise DriverActionError("press_key action has empty key")
        return {"type": "press_key", "keys": normalized_single}

    if action_type == "wait":
        duration_ms = max(_require_int(raw.get("duration_ms"), field="duration_ms"), 0)
        return {"type": "wait", "duration_ms": duration_ms}

    raise DriverActionError(f"Unsupported driver action type: {action_type}")


def normalize_driver_actions(actions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_driver_action(action) for action in actions]


async def replay_driver_actions(
    driver: BrowserDriver,
    actions: List[Dict[str, Any]],
    *,
    stabilization_wait_ms: int,
    action_timeout_seconds: Optional[float] = None,
) -> None:
    """Replay recorded driver actions using the browser driver."""
    stabilization = max(int(stabilization_wait_ms), 0)
    timeout = (
        max(float(action_timeout_seconds), 0.5) if action_timeout_seconds is not None else None
    )
    normalized = normalize_driver_actions(actions)
    for action in normalized:
        action_type = action["type"]
        if action_type == "click":
            coro = driver.click(
                action["x"],
                action["y"],
                button=action.get("button", "left"),
                click_count=int(action.get("click_count", 1)),
            )
        elif action_type == "move":
            coro = driver.move_mouse(action["x"], action["y"])
        elif action_type == "drag":
            coro = driver.drag_mouse(
                action["start_x"],
                action["start_y"],
                action["end_x"],
                action["end_y"],
            )
        elif action_type == "scroll_by_pixels":
            coro = driver.scroll_by_pixels(
                x=action.get("x", 0), y=action.get("y", 0), smooth=False
            )
        elif action_type == "type_text":
            coro = driver.type_text(action.get("text", ""))
        elif action_type == "press_key":
            coro = driver.press_key(action.get("keys", ""))
        elif action_type == "wait":
            coro = driver.wait(int(action.get("duration_ms", 0)))
        else:  # pragma: no cover - normalize_driver_actions prevents this
            raise DriverActionError(f"Unsupported driver action type: {action_type}")

        if timeout is not None and action_type != "wait":
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

        if action_type != "wait" and stabilization > 0:
            await driver.wait(stabilization)
