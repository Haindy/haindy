"""Virtual keyboard and mouse using uinput."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional, Tuple

from evdev import AbsInfo, UInput, ecodes

logger = logging.getLogger(__name__)


class VirtualInput:
    """Manage a virtual keyboard/mouse backed by /dev/uinput."""

    def __init__(
        self,
        viewport: Tuple[int, int],
        device_name: str = "haindy-virtual-input",
    ) -> None:
        width, height = viewport
        abs_caps = [
            # Use full-screen absolute axes so desktop environments recognize a pointer.
            (ecodes.ABS_X, AbsInfo(0, 0, max(width - 1, 0), 0, 0, 0)),
            (ecodes.ABS_Y, AbsInfo(0, 0, max(height - 1, 0), 0, 0, 0)),
        ]

        key_codes = self._keyboard_keys()
        capabilities = {
            ecodes.EV_KEY: key_codes,
            ecodes.EV_ABS: abs_caps,
            ecodes.EV_REL: [ecodes.REL_WHEEL, ecodes.REL_HWHEEL],
            ecodes.EV_MSC: [ecodes.MSC_SCAN],
        }

        self._ui = UInput(capabilities, name=device_name, bustype=0x03)
        self._viewport = viewport
        logger.info(
            "Initialized virtual input device",
            extra={"device": device_name, "viewport": f"{viewport[0]}x{viewport[1]}"},
        )

    async def move(self, x: int, y: int, steps: int = 1) -> None:
        """Move pointer to absolute coordinates."""
        x_clamped, y_clamped = self._clamp(x, y)
        for _ in range(max(steps, 1)):
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_X, x_clamped)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_Y, y_clamped)
            self._ui.syn()
            if steps > 1:
                await asyncio.sleep(0.01)

    async def click(self, x: int, y: int, button: str = "left", click_count: int = 1) -> None:
        """Click at absolute coordinates."""
        code = self._button_code(button)
        x_clamped, y_clamped = self._clamp(x, y)
        for _ in range(max(click_count, 1)):
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_X, x_clamped)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_Y, y_clamped)
            self._ui.syn()
            self._ui.write(ecodes.EV_KEY, code, 1)
            self._ui.syn()
            self._ui.write(ecodes.EV_KEY, code, 0)
            self._ui.syn()
            await asyncio.sleep(0.02)

    async def drag(self, start: Tuple[int, int], end: Tuple[int, int], steps: int = 1) -> None:
        """Drag from start to end coordinates."""
        start_x, start_y = self._clamp(*start)
        end_x, end_y = self._clamp(*end)
        self._ui.write(ecodes.EV_ABS, ecodes.ABS_X, start_x)
        self._ui.write(ecodes.EV_ABS, ecodes.ABS_Y, start_y)
        self._ui.syn()
        self._ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
        self._ui.syn()

        for step in range(max(steps, 1)):
            progress = (step + 1) / max(steps, 1)
            x = int(start_x + (end_x - start_x) * progress)
            y = int(start_y + (end_y - start_y) * progress)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_X, x)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_Y, y)
            self._ui.syn()
            await asyncio.sleep(0.01)

        self._ui.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
        self._ui.syn()

    async def scroll(self, x: int = 0, y: int = 0) -> None:
        """Scroll by pixel deltas using wheel events."""
        if y:
            self._ui.write(ecodes.EV_REL, ecodes.REL_WHEEL, self._scroll_delta(y))
        if x:
            self._ui.write(ecodes.EV_REL, ecodes.REL_HWHEEL, self._scroll_delta(x))
        self._ui.syn()
        await asyncio.sleep(0.01)

    async def type_text(self, text: str) -> None:
        """Type text using key events."""
        for char in text:
            await self._emit_char(char)

    async def press_key(self, key: str | Iterable[str]) -> None:
        """Press a key or key combination."""
        sequence: List[str] = []
        if isinstance(key, str):
            if "+" in key:
                sequence = [part.strip() for part in key.split("+") if part.strip()]
            else:
                sequence = [key.strip()]
        else:
            sequence = [k for k in key]

        if not sequence:
            return

        codes = [self._lookup_key_code(item) for item in sequence]
        codes = [c for c in codes if c is not None]
        if not codes:
            return

        for code in codes:
            self._ui.write(ecodes.EV_KEY, code, 1)
        self._ui.syn()
        await asyncio.sleep(0.01)
        for code in reversed(codes):
            self._ui.write(ecodes.EV_KEY, code, 0)
        self._ui.syn()

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(milliseconds / 1000.0)

    def _emit_char(self, char: str) -> None:
        code, needs_shift = self._char_to_key(char)
        if code is None:
            logger.debug("Skipping unsupported character", extra={"char": repr(char)})
            return
        if needs_shift:
            self._ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1)
        self._ui.write(ecodes.EV_KEY, code, 1)
        self._ui.syn()
        self._ui.write(ecodes.EV_KEY, code, 0)
        if needs_shift:
            self._ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 0)
        self._ui.syn()

    @staticmethod
    def _button_code(button: str) -> int:
        normalized = (button or "left").lower()
        if normalized == "right":
            return ecodes.BTN_RIGHT
        if normalized == "middle":
            return ecodes.BTN_MIDDLE
        return ecodes.BTN_LEFT

    @staticmethod
    def _scroll_delta(pixels: int) -> int:
        # Wheel ticks are coarse; approximate 120 pixels per notch.
        if pixels == 0:
            return 0
        direction = 1 if pixels < 0 else -1
        return direction * max(abs(pixels) // 120, 1)

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        width, height = self._viewport
        x_clamped = max(0, min(int(x), max(width - 1, 0)))
        y_clamped = max(0, min(int(y), max(height - 1, 0)))
        return x_clamped, y_clamped

    @staticmethod
    def _keyboard_keys() -> List[int]:
        keys: List[int] = [
            ecodes.BTN_LEFT,
            ecodes.BTN_RIGHT,
            ecodes.BTN_MIDDLE,
            ecodes.KEY_ESC,
            ecodes.KEY_TAB,
            ecodes.KEY_ENTER,
            ecodes.KEY_SPACE,
            ecodes.KEY_BACKSPACE,
            ecodes.KEY_LEFTSHIFT,
            ecodes.KEY_RIGHTSHIFT,
            ecodes.KEY_LEFTCTRL,
            ecodes.KEY_RIGHTCTRL,
            ecodes.KEY_LEFTALT,
            ecodes.KEY_RIGHTALT,
            ecodes.KEY_CAPSLOCK,
            ecodes.KEY_LEFT,
            ecodes.KEY_RIGHT,
            ecodes.KEY_UP,
            ecodes.KEY_DOWN,
            ecodes.KEY_HOME,
            ecodes.KEY_END,
            ecodes.KEY_PAGEUP,
            ecodes.KEY_PAGEDOWN,
            ecodes.KEY_DELETE,
        ]
        keys.extend([getattr(ecodes, f"KEY_{i}") for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        keys.extend([getattr(ecodes, f"KEY_{i}") for i in range(10)])
        keys.extend(
            [
                ecodes.KEY_MINUS,
                ecodes.KEY_EQUAL,
                ecodes.KEY_LEFTBRACE,
                ecodes.KEY_RIGHTBRACE,
                ecodes.KEY_BACKSLASH,
                ecodes.KEY_SEMICOLON,
                ecodes.KEY_APOSTROPHE,
                ecodes.KEY_GRAVE,
                ecodes.KEY_COMMA,
                ecodes.KEY_DOT,
                ecodes.KEY_SLASH,
            ]
        )
        return keys

    def _char_to_key(self, char: str) -> Tuple[Optional[int], bool]:
        """Map ASCII character to keycode and shift requirement."""
        if not char:
            return None, False
        if char.isalpha():
            key = getattr(ecodes, f"KEY_{char.upper()}", None)
            return key, char.isupper()
        if char.isdigit():
            key = getattr(ecodes, f"KEY_{char}", None)
            return key, False

        mapping = {
            " ": (ecodes.KEY_SPACE, False),
            "\n": (ecodes.KEY_ENTER, False),
            "\t": (ecodes.KEY_TAB, False),
            "-": (ecodes.KEY_MINUS, False),
            "_": (ecodes.KEY_MINUS, True),
            "=": (ecodes.KEY_EQUAL, False),
            "+": (ecodes.KEY_EQUAL, True),
            "[": (ecodes.KEY_LEFTBRACE, False),
            "]": (ecodes.KEY_RIGHTBRACE, False),
            "{": (ecodes.KEY_LEFTBRACE, True),
            "}": (ecodes.KEY_RIGHTBRACE, True),
            "\\": (ecodes.KEY_BACKSLASH, False),
            "|": (ecodes.KEY_BACKSLASH, True),
            ";": (ecodes.KEY_SEMICOLON, False),
            ":": (ecodes.KEY_SEMICOLON, True),
            "'": (ecodes.KEY_APOSTROPHE, False),
            '"': (ecodes.KEY_APOSTROPHE, True),
            "`": (ecodes.KEY_GRAVE, False),
            "~": (ecodes.KEY_GRAVE, True),
            ",": (ecodes.KEY_COMMA, False),
            "<": (ecodes.KEY_COMMA, True),
            ".": (ecodes.KEY_DOT, False),
            ">": (ecodes.KEY_DOT, True),
            "/": (ecodes.KEY_SLASH, False),
            "?": (ecodes.KEY_SLASH, True),
        }
        return mapping.get(char, (None, False))

    @staticmethod
    def _lookup_key_code(name: str) -> Optional[int]:
        normalized = name.strip().lower()
        alias_map = {
            "ctrl": ecodes.KEY_LEFTCTRL,
            "control": ecodes.KEY_LEFTCTRL,
            "alt": ecodes.KEY_LEFTALT,
            "shift": ecodes.KEY_LEFTSHIFT,
            "meta": ecodes.KEY_LEFTMETA if hasattr(ecodes, "KEY_LEFTMETA") else None,
            "cmd": ecodes.KEY_LEFTMETA if hasattr(ecodes, "KEY_LEFTMETA") else None,
            "enter": ecodes.KEY_ENTER,
            "return": ecodes.KEY_ENTER,
            "tab": ecodes.KEY_TAB,
            "esc": ecodes.KEY_ESC,
            "escape": ecodes.KEY_ESC,
            "space": ecodes.KEY_SPACE,
            "backspace": ecodes.KEY_BACKSPACE,
            "delete": ecodes.KEY_DELETE,
            "home": ecodes.KEY_HOME,
            "end": ecodes.KEY_END,
            "pageup": ecodes.KEY_PAGEUP,
            "pagedown": ecodes.KEY_PAGEDOWN,
            "up": ecodes.KEY_UP,
            "down": ecodes.KEY_DOWN,
            "left": ecodes.KEY_LEFT,
            "right": ecodes.KEY_RIGHT,
        }
        if normalized in alias_map:
            return alias_map[normalized]

        upper = normalized.upper()
        if len(upper) == 1 and upper.isalpha():
            return getattr(ecodes, f"KEY_{upper}", None)
        if len(upper) == 1 and upper.isdigit():
            return getattr(ecodes, f"KEY_{upper}", None)

        return None
