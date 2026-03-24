"""macOS keyboard and mouse input handler using pynput."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Map from user-facing modifier name to pynput Key attribute name
_MODIFIER_TO_KEY_ATTR: dict[str, str] = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "win": "cmd",
}

# Map from user-facing key name to pynput Key attribute name
_SPECIAL_KEY_ATTRS: dict[str, str] = {
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "space": "space",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "page_up": "page_up",
    "pageup": "page_up",
    "page_down": "page_down",
    "pagedown": "page_down",
    "insert": "insert",
    "caps_lock": "caps_lock",
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "f6": "f6",
    "f7": "f7",
    "f8": "f8",
    "f9": "f9",
    "f10": "f10",
    "f11": "f11",
    "f12": "f12",
    "f13": "f13",
    "f14": "f14",
    "f15": "f15",
    "f16": "f16",
    "f17": "f17",
    "f18": "f18",
    "f19": "f19",
    "f20": "f20",
}


class MacOSInputHandler:
    """Handle keyboard and mouse input on macOS via pynput.

    Coordinates accepted by this class are in screenshot pixel space.
    They are converted to logical points before injection using the
    display scale factor detected at driver startup.
    """

    def __init__(
        self,
        logical_size: tuple[int, int],
        scale_x: float,
        scale_y: float,
        keyboard_layout: str = "us",
        key_delay_ms: int = 12,
        mouse_controller: Any | None = None,
        keyboard_controller: Any | None = None,
    ) -> None:
        self.logical_size = logical_size
        self.scale_x = max(scale_x, 0.01)
        self.scale_y = max(scale_y, 0.01)
        self.keyboard_layout = keyboard_layout
        self.key_delay_ms = max(int(key_delay_ms), 1)

        if mouse_controller is not None:
            self._mouse = mouse_controller
        else:
            from pynput.mouse import Controller as MouseController

            self._mouse = MouseController()

        if keyboard_controller is not None:
            self._keyboard = keyboard_controller
        else:
            from pynput.keyboard import Controller as KeyboardController

            self._keyboard = KeyboardController()

    def _to_logical(self, x: int, y: int) -> tuple[int, int]:
        """Convert screenshot pixel coordinates to logical display points."""
        return int(x / self.scale_x), int(y / self.scale_y)

    def _resolve_key(self, name: str) -> Any:
        """Resolve a key name string to a pynput Key enum value or character."""
        from pynput.keyboard import Key

        lower = name.strip().lower()
        if lower in _MODIFIER_TO_KEY_ATTR:
            attr = _MODIFIER_TO_KEY_ATTR[lower]
            return getattr(Key, attr, None)
        if lower in _SPECIAL_KEY_ATTRS:
            attr = _SPECIAL_KEY_ATTRS[lower]
            return getattr(Key, attr, None)
        # Single printable character
        if len(name) == 1:
            return name
        return None

    async def move(self, x: int, y: int) -> None:
        """Move the mouse pointer to absolute screenshot pixel coordinates."""
        lx, ly = self._to_logical(x, y)
        self._mouse.position = (lx, ly)

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        """Click at screenshot pixel coordinates with optional modifiers."""
        from pynput.mouse import Button

        lx, ly = self._to_logical(x, y)
        self._mouse.position = (lx, ly)

        btn_map = {
            "left": Button.left,
            "right": Button.right,
            "middle": Button.middle,
        }
        btn = btn_map.get(str(button).lower(), Button.left)

        pressed_mods: list[Any] = []
        if modifiers:
            for mod in modifiers:
                key = self._resolve_key(mod)
                if key is not None:
                    self._keyboard.press(key)
                    pressed_mods.append(key)

        for i in range(max(1, click_count)):
            self._mouse.click(btn)
            if click_count > 1 and i < click_count - 1:
                await asyncio.sleep(0.05)

        for key in reversed(pressed_mods):
            self._keyboard.release(key)

    async def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        steps: int = 1,
    ) -> None:
        """Drag from start to end in screenshot pixel coordinates."""
        from pynput.mouse import Button

        lx1, ly1 = self._to_logical(*start)
        lx2, ly2 = self._to_logical(*end)
        self._mouse.position = (lx1, ly1)
        self._mouse.press(Button.left)
        await asyncio.sleep(0.05)

        steps = max(1, steps)
        for i in range(1, steps + 1):
            ix = int(lx1 + (lx2 - lx1) * i / steps)
            iy = int(ly1 + (ly2 - ly1) * i / steps)
            self._mouse.position = (ix, iy)
            await asyncio.sleep(0.01)

        self._mouse.release(Button.left)

    async def scroll(self, x: int = 0, y: int = 0) -> None:
        """Scroll by pixel deltas (positive y = scroll down)."""
        # pynput scroll: positive dy = scroll up, negative = scroll down
        # Our convention: positive y = scroll down (same as desktop driver)
        # Convert pixels to fractional ticks (120 pixels per tick)
        x_ticks = x / 120.0
        y_ticks = -(y / 120.0)
        self._mouse.scroll(x_ticks, y_ticks)

    async def type_text(self, text: str) -> None:
        """Type text using pynput keyboard controller."""
        delay = self.key_delay_ms / 1000.0
        for char in text:
            self._keyboard.type(char)
            if delay > 0:
                await asyncio.sleep(delay)

    async def press_key(self, key: str) -> None:
        """Press a key or key combination (e.g. 'enter', 'cmd+c', 'shift+tab')."""
        delay = max(self.key_delay_ms, 5) / 1000.0
        parts = [p.strip().lower() for p in key.split("+") if p.strip()]

        modifiers: list[Any] = []
        primary: str | None = None

        for part in parts:
            if part in _MODIFIER_TO_KEY_ATTR:
                resolved = self._resolve_key(part)
                if resolved is not None:
                    modifiers.append(resolved)
            else:
                primary = part

        for mod in modifiers:
            self._keyboard.press(mod)
            await asyncio.sleep(delay)

        if primary is not None:
            resolved_primary = self._resolve_key(primary)
            if resolved_primary is not None:
                self._keyboard.press(resolved_primary)
                await asyncio.sleep(delay)
                self._keyboard.release(resolved_primary)

        for mod in reversed(modifiers):
            self._keyboard.release(mod)
            await asyncio.sleep(delay)
