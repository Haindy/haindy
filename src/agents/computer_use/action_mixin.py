# mypy: disable-error-code=misc
"""Action execution helpers for Computer Use sessions."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from src.core.enhanced_types import ComputerToolTurn

from .common import (
    encode_png_base64,
    normalize_coordinates,
    normalize_key_sequence,
)
from .types import ComputerUseExecutionError

logger = logging.getLogger("src.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class ComputerUseActionMixin:
    """Action execution helpers for provider loops."""

    async def _execute_tool_action(
        self: _ComputerUseSession,
        turn: ComputerToolTurn,
        metadata: dict[str, Any],
        turn_index: int,
        normalized_coords: bool = False,
        allow_unknown: bool = False,
        environment: str = "browser",
        cache_label: str | None = None,
        cache_action: str = "click",
        use_cache: bool = True,
    ) -> None:
        """Execute a single Computer Use tool action via the automation driver."""
        action = turn.parameters or {}
        raw_action_type = action.get("type") or turn.action_type
        action_type = self._canonicalize_action_type(raw_action_type)
        if action_type and action_type != raw_action_type:
            turn.metadata["normalized_action_type"] = action_type
            turn.action_type = action_type
        start = time.perf_counter()

        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
        turn.metadata["resolution"] = (viewport_width, viewport_height)
        turn.metadata["normalized_coords"] = bool(normalized_coords)
        allow_action, deny_reason = self._is_action_allowed(action_type)
        disallowed_reason = self._disallowed_action_reason(action_type, environment)
        capture_clipboard_after_action = False
        cached = None
        cache_allowed = False
        cache_hit = False

        if cache_label and self._action_matches_cache(action_type, cache_action):
            cache_allowed = True
            if use_cache:
                cached = self._coordinate_cache.lookup(
                    cache_label, cache_action, (viewport_width, viewport_height)
                )
            cache_hit = cached is not None if use_cache else False
            turn.metadata.update(
                {
                    "cache_label": cache_label,
                    "cache_action": cache_action,
                    "cache_hit": cache_hit,
                    "cache_lookup_allowed": use_cache,
                }
            )

        try:
            if not allow_action:
                turn.status = "failed"
                turn.error_message = deny_reason or "Action blocked by policy."
                turn.metadata["policy"] = "observe_only"
            elif disallowed_reason:
                turn.status = "ignored"
                turn.error_message = disallowed_reason
                turn.metadata["ignored"] = True
            else:
                implicit_modifier: str | None = None
                if action_type == "ctrl_click":
                    action_type = "click_at"
                    implicit_modifier = "ctrl"
                elif action_type == "shift_click":
                    action_type = "click_at"
                    implicit_modifier = "shift"

                await self._enforce_domain_policy(action_type)

                if (
                    self._interaction_constraints.disallow_scroll
                    and self._is_scroll_action(action_type)
                ):
                    raise ComputerUseExecutionError(
                        "Step context prohibits scrolling for this action."
                    )

                if self._pending_context_menu_selection and action_type not in {
                    "click",
                    "click_at",
                    "double_click",
                    "right_click",
                }:
                    self._pending_context_menu_selection = False

                if action_type in {"click", "click_at", "move_mouse_and_click"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                            turn=turn,
                        )
                    button = (action.get("button", "left") or "left").lower()
                    click_count = int(action.get("click_count", 1))
                    raw_modifiers = action.get("modifiers") or []
                    if isinstance(raw_modifiers, str):
                        raw_modifiers = [raw_modifiers]
                    modifiers = [str(m).lower() for m in raw_modifiers if m]
                    if implicit_modifier and implicit_modifier not in modifiers:
                        modifiers = [implicit_modifier] + modifiers
                    await asyncio.wait_for(
                        self._automation_driver.click(
                            x,
                            y,
                            button=button,
                            click_count=click_count,
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    if modifiers:
                        turn.metadata["modifiers"] = modifiers
                        turn.metadata["modifiers_applied"] = False
                    turn.status = "executed"
                    if button == "right":
                        self._pending_context_menu_selection = True
                    elif self._pending_context_menu_selection and click_count == 1:
                        capture_clipboard_after_action = True
                        self._pending_context_menu_selection = False
                    else:
                        self._pending_context_menu_selection = False
                elif action_type in {"double_click", "right_click"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                            turn=turn,
                        )
                    button = "right" if action_type == "right_click" else "left"
                    click_count = 2 if action_type == "double_click" else 1
                    await asyncio.wait_for(
                        self._automation_driver.click(
                            x, y, button=button, click_count=click_count
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                    if button == "right":
                        self._pending_context_menu_selection = True
                elif action_type in {"move", "hover_at"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                            turn=turn,
                        )
                    raw_steps = action.get("steps") or action.get("step_count")
                    steps = 1
                    if raw_steps is not None:
                        try:
                            steps = int(float(raw_steps))
                        except (TypeError, ValueError):
                            steps = 1
                    if steps <= 0:
                        steps = 1
                    await asyncio.wait_for(
                        self._automation_driver.move_mouse(x, y, steps=steps),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type == "drag":

                    def _coerce_float(value: Any) -> float | None:
                        if value is None:
                            return None
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    def _first_numeric(*keys: str) -> float | None:
                        for key in keys:
                            if key not in action:
                                continue
                            numeric = _coerce_float(action.get(key))
                            if numeric is not None:
                                return numeric
                        return None

                    start_x_raw = _first_numeric("start_x", "from_x", "x")
                    start_y_raw = _first_numeric("start_y", "from_y", "y")
                    end_x_raw = _first_numeric(
                        "end_x",
                        "to_x",
                        "destination_x",
                        "target_x",
                    )
                    end_y_raw = _first_numeric(
                        "end_y",
                        "to_y",
                        "destination_y",
                        "target_y",
                    )

                    path_points = action.get("path") or action.get("points")
                    if isinstance(path_points, list) and len(path_points) >= 2:
                        first_point = path_points[0]
                        last_point = path_points[-1]
                        if isinstance(first_point, dict):
                            if start_x_raw is None:
                                start_x_raw = _coerce_float(first_point.get("x"))
                            if start_y_raw is None:
                                start_y_raw = _coerce_float(first_point.get("y"))
                        if isinstance(last_point, dict):
                            if end_x_raw is None:
                                end_x_raw = _coerce_float(last_point.get("x"))
                            if end_y_raw is None:
                                end_y_raw = _coerce_float(last_point.get("y"))

                    if start_x_raw is None or start_y_raw is None:
                        raise ComputerUseExecutionError(
                            "Drag action missing start coordinates."
                        )

                    delta_x_raw = _first_numeric("dx", "delta_x")
                    delta_y_raw = _first_numeric("dy", "delta_y")
                    if end_x_raw is None and delta_x_raw is not None:
                        end_x_raw = start_x_raw + delta_x_raw
                    if end_y_raw is None and delta_y_raw is not None:
                        end_y_raw = start_y_raw + delta_y_raw
                    if end_x_raw is None or end_y_raw is None:
                        raise ComputerUseExecutionError(
                            "Drag action missing destination coordinates."
                        )

                    if normalized_coords:
                        start_x, start_y = (
                            self._denormalize_coordinates_for_active_frame(
                                start_x_raw,
                                start_y_raw,
                                viewport_width,
                                viewport_height,
                                turn=turn,
                                prefix="start_",
                            )
                        )
                        end_x, end_y = self._denormalize_coordinates_for_active_frame(
                            end_x_raw,
                            end_y_raw,
                            viewport_width,
                            viewport_height,
                            turn=turn,
                            prefix="end_",
                        )
                    else:
                        start_x, start_y = normalize_coordinates(
                            start_x_raw, start_y_raw, viewport_width, viewport_height
                        )
                        end_x, end_y = normalize_coordinates(
                            end_x_raw, end_y_raw, viewport_width, viewport_height
                        )

                    raw_steps = action.get("steps") or action.get("step_count")
                    steps = 1
                    if raw_steps is not None:
                        try:
                            steps = int(float(raw_steps))
                        except (TypeError, ValueError):
                            steps = 1
                    elif isinstance(path_points, list) and len(path_points) >= 2:
                        steps = max(1, len(path_points) - 1)
                    if steps <= 0:
                        steps = 1

                    await asyncio.wait_for(
                        self._automation_driver.drag_mouse(
                            start_x, start_y, end_x, end_y, steps=steps
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update(
                        {
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                        }
                    )
                    turn.status = "executed"
                elif action_type == "scroll":
                    scroll_x = int(action.get("scroll_x", 0))
                    scroll_y = int(action.get("scroll_y", 0))
                    max_pixels = int(self._settings.scroll_max_magnitude)
                    scroll_x = max(-max_pixels, min(scroll_x, max_pixels))
                    scroll_y = max(-max_pixels, min(scroll_y, max_pixels))
                    direction = self._extract_scroll_direction(action_type, action)
                    if direction:
                        turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_x"] = scroll_x
                    turn.metadata["scroll_y"] = scroll_y
                    await asyncio.wait_for(
                        self._automation_driver.scroll_by_pixels(
                            x=scroll_x, y=scroll_y, smooth=False
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.status = "executed"
                elif action_type == "scroll_document":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError(
                            "scroll_document action missing direction."
                        )
                    magnitude_raw = action.get("magnitude")
                    magnitude = (
                        abs(int(magnitude_raw))
                        if magnitude_raw is not None
                        else int(self._settings.scroll_default_magnitude)
                    )
                    magnitude = min(magnitude, int(self._settings.scroll_max_magnitude))
                    turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_magnitude"] = magnitude
                    await self._automation_driver.scroll(direction, magnitude)
                    turn.status = "executed"
                elif action_type == "scroll_at":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError(
                            "scroll_at action missing direction."
                        )
                    magnitude_raw = action.get("magnitude")
                    magnitude = (
                        abs(int(magnitude_raw))
                        if magnitude_raw is not None
                        else int(self._settings.scroll_default_magnitude)
                    )
                    magnitude = min(magnitude, int(self._settings.scroll_max_magnitude))
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                            turn=turn,
                        )
                    turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_magnitude"] = magnitude
                    await self._automation_driver.move_mouse(x, y, steps=1)
                    await self._automation_driver.scroll(direction, magnitude)
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type == "type":
                    text_payload = action.get("text")
                    if not text_payload:
                        text_payload = (
                            action.get("value")
                            or action.get("input")
                            or metadata.get("value")
                        )
                        if text_payload:
                            turn.metadata["synthetic_text_payload"] = text_payload
                    if not text_payload:
                        raise ComputerUseExecutionError(
                            "Type action missing text payload."
                        )
                    await self._automation_driver.type_text(text_payload)
                    turn.status = "executed"
                elif action_type == "type_text_at":
                    text_payload = action.get("text")
                    if text_payload is None:
                        raise ComputerUseExecutionError(
                            "type_text_at action missing text."
                        )
                    press_enter_default = not normalized_coords
                    press_enter = bool(action.get("press_enter", press_enter_default))
                    clear_before = bool(action.get("clear_before_typing", True))
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                            turn=turn,
                        )
                    tap_count = (
                        3 if (clear_before and environment == "mobile_adb") else 1
                    )
                    await self._automation_driver.click(
                        x, y, button="left", click_count=tap_count
                    )
                    if clear_before:
                        if environment == "mobile_adb":
                            pass
                        else:
                            await self._automation_driver.press_key("ctrl+a")
                            await self._automation_driver.press_key("backspace")
                    await self._automation_driver.type_text(str(text_payload))
                    if press_enter:
                        await self._automation_driver.press_key("enter")
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type in {"keypress", "key_combination"}:
                    if action_type == "keypress":
                        key_sequence = self._resolve_key_sequence(action, metadata)
                        if not key_sequence:
                            raise ComputerUseExecutionError(
                                "Key press action missing key payload."
                            )
                        if not action.get("keys"):
                            turn.metadata["synthetic_key_sequence"] = key_sequence
                        for key in key_sequence:
                            normalized = normalize_key_sequence(key)
                            await self._automation_driver.press_key(normalized)
                            if self._should_capture_clipboard_after_key_combo(
                                normalized
                            ):
                                capture_clipboard_after_action = True
                    else:
                        keys = action.get("keys")
                        if not keys:
                            raise ComputerUseExecutionError(
                                "Key combination action missing keys."
                            )
                        if isinstance(keys, (list, tuple)):
                            for key in keys:
                                normalized = normalize_key_sequence(str(key))
                                await self._automation_driver.press_key(normalized)
                                if self._should_capture_clipboard_after_key_combo(
                                    normalized
                                ):
                                    capture_clipboard_after_action = True
                        else:
                            normalized = normalize_key_sequence(str(keys))
                            await self._automation_driver.press_key(normalized)
                            if self._should_capture_clipboard_after_key_combo(
                                normalized
                            ):
                                capture_clipboard_after_action = True
                    turn.status = "executed"
                elif action_type == "read_clipboard":
                    clipboard_text = await self._maybe_read_clipboard(turn)
                    if clipboard_text is not None:
                        turn.metadata["clipboard_text"] = clipboard_text
                    turn.status = "executed"
                elif action_type == "wait":
                    duration = int(
                        action.get("duration_ms")
                        or self._settings.actions_computer_tool_stabilization_wait_ms
                    )
                    await self._automation_driver.wait(duration)
                    turn.metadata["duration_ms"] = duration
                    turn.status = "executed"
                elif action_type == "wait_5_seconds":
                    await self._automation_driver.wait(5000)
                    turn.metadata["duration_ms"] = 5000
                    turn.status = "executed"
                elif action_type == "go_back":
                    await self._automation_driver.press_key("alt+left")
                    turn.status = "executed"
                elif action_type == "go_forward":
                    await self._automation_driver.press_key("alt+right")
                    turn.status = "executed"
                elif action_type == "search":
                    await self._navigate_via_address_bar("https://www.google.com/")
                    turn.status = "executed"
                elif action_type == "navigate":
                    url = action.get("url")
                    if not url:
                        raise ComputerUseExecutionError("Navigate action missing url.")
                    await self._navigate_via_address_bar(str(url))
                    turn.status = "executed"
                elif action_type == "screenshot":
                    logger.debug(
                        "Computer Use requested screenshot action; no automation_driver operation executed."
                    )
                    turn.status = "executed"
                else:
                    if allow_unknown:
                        turn.status = "ignored"
                        turn.error_message = f"unsupported action: {action_type}"
                        turn.metadata["ignored"] = True
                    else:
                        raise ComputerUseExecutionError(
                            f"Unsupported computer action type: {action_type}"
                        )

        except ComputerUseExecutionError as policy_error:
            turn.status = "failed"
            turn.error_message = str(policy_error)
            turn.metadata["policy"] = "rejected"
            logger.warning(
                "Computer Use action rejected",
                extra={
                    "call_id": turn.call_id,
                    "action_type": action_type,
                    "raw_action_type": raw_action_type,
                    "action_keys": sorted(action.keys()),
                    "reason": turn.error_message,
                },
            )
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "",
                        cache_action,
                        (viewport_width, viewport_height),
                    )
                except Exception:
                    logger.debug("Failed to invalidate coordinate cache", exc_info=True)
        except Exception:
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "",
                        cache_action,
                        (viewport_width, viewport_height),
                    )
                except Exception:
                    logger.debug("Failed to invalidate coordinate cache", exc_info=True)
            raise
        finally:
            turn.latency_ms = (time.perf_counter() - start) * 1000
            if turn.status == "executed":
                await self._post_action_wait()
            if capture_clipboard_after_action and turn.status == "executed":
                clipboard_text = await self._maybe_read_clipboard(turn)
                if clipboard_text is not None:
                    turn.metadata["clipboard_text"] = clipboard_text
            await self._record_turn_snapshot(turn, metadata, turn_index)

    async def _record_turn_snapshot(
        self: _ComputerUseSession,
        turn: ComputerToolTurn,
        metadata: dict[str, Any],
        turn_index: int,
    ) -> None:
        """Capture screenshot and update metadata after action execution."""
        screenshot_bytes = await self._automation_driver.screenshot()
        turn.screenshot_path = self._save_turn_screenshot(
            screenshot_bytes,
            suffix=f"{turn.action_type}_{turn_index}",
            step_number=metadata.get("step_number"),
        )

        turn.metadata.update(
            {
                "screenshot_base64": encode_png_base64(screenshot_bytes),
                "current_url": await self._maybe_get_current_url(),
            }
        )

    async def _execute_click(
        self: _ComputerUseSession,
        action: dict[str, Any],
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """Execute a primary click event."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        button = action.get("button", "left")
        click_count = int(action.get("click_count", 1))
        await asyncio.wait_for(
            self._automation_driver.click(x, y, button=button, click_count=click_count),
            timeout=self._action_timeout_seconds,
        )

    async def _execute_special_click(
        self: _ComputerUseSession,
        action: dict[str, Any],
        viewport_width: int,
        viewport_height: int,
    ) -> None:
        """Execute double or right click events."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        action_type = action.get("type")
        if action_type == "double_click":
            await asyncio.wait_for(
                self._automation_driver.click(x, y, button="left", click_count=2),
                timeout=self._action_timeout_seconds,
            )
        elif action_type == "right_click":
            await asyncio.wait_for(
                self._automation_driver.click(x, y, button="right", click_count=1),
                timeout=self._action_timeout_seconds,
            )

    async def _execute_scroll(
        self: _ComputerUseSession, action: dict[str, Any]
    ) -> None:
        """Execute a scroll event via pixel deltas."""
        scroll_x = int(action.get("scroll_x", 0))
        scroll_y = int(action.get("scroll_y", 0))
        await asyncio.wait_for(
            self._automation_driver.scroll_by_pixels(
                x=scroll_x, y=scroll_y, smooth=False
            ),
            timeout=self._action_timeout_seconds,
        )
