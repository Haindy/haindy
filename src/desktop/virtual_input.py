"""Virtual keyboard and mouse using uinput."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from evdev import AbsInfo, UInput, ecodes

logger = logging.getLogger(__name__)

# Scancode map (set 1 / Linux input) keyed by EV_KEY code. Extended (E0-prefixed)
# scancodes are encoded as single integers, e.g., 0xE01D for right control.
DEFAULT_SCANCODE_MAP: Dict[int, int] = {
    ecodes.KEY_ESC: 0x01,
    ecodes.KEY_1: 0x02,
    ecodes.KEY_2: 0x03,
    ecodes.KEY_3: 0x04,
    ecodes.KEY_4: 0x05,
    ecodes.KEY_5: 0x06,
    ecodes.KEY_6: 0x07,
    ecodes.KEY_7: 0x08,
    ecodes.KEY_8: 0x09,
    ecodes.KEY_9: 0x0A,
    ecodes.KEY_0: 0x0B,
    ecodes.KEY_MINUS: 0x0C,
    ecodes.KEY_EQUAL: 0x0D,
    ecodes.KEY_BACKSPACE: 0x0E,
    ecodes.KEY_TAB: 0x0F,
    ecodes.KEY_Q: 0x10,
    ecodes.KEY_W: 0x11,
    ecodes.KEY_E: 0x12,
    ecodes.KEY_R: 0x13,
    ecodes.KEY_T: 0x14,
    ecodes.KEY_Y: 0x15,
    ecodes.KEY_U: 0x16,
    ecodes.KEY_I: 0x17,
    ecodes.KEY_O: 0x18,
    ecodes.KEY_P: 0x19,
    ecodes.KEY_LEFTBRACE: 0x1A,
    ecodes.KEY_RIGHTBRACE: 0x1B,
    ecodes.KEY_ENTER: 0x1C,
    ecodes.KEY_LEFTCTRL: 0x1D,
    ecodes.KEY_A: 0x1E,
    ecodes.KEY_S: 0x1F,
    ecodes.KEY_D: 0x20,
    ecodes.KEY_F: 0x21,
    ecodes.KEY_G: 0x22,
    ecodes.KEY_H: 0x23,
    ecodes.KEY_J: 0x24,
    ecodes.KEY_K: 0x25,
    ecodes.KEY_L: 0x26,
    ecodes.KEY_SEMICOLON: 0x27,
    ecodes.KEY_APOSTROPHE: 0x28,
    ecodes.KEY_GRAVE: 0x29,
    ecodes.KEY_LEFTSHIFT: 0x2A,
    ecodes.KEY_BACKSLASH: 0x2B,
    ecodes.KEY_Z: 0x2C,
    ecodes.KEY_X: 0x2D,
    ecodes.KEY_C: 0x2E,
    ecodes.KEY_V: 0x2F,
    ecodes.KEY_B: 0x30,
    ecodes.KEY_N: 0x31,
    ecodes.KEY_M: 0x32,
    ecodes.KEY_COMMA: 0x33,
    ecodes.KEY_DOT: 0x34,
    ecodes.KEY_SLASH: 0x35,
    ecodes.KEY_RIGHTSHIFT: 0x36,
    ecodes.KEY_KPASTERISK: 0x37,
    ecodes.KEY_LEFTALT: 0x38,
    ecodes.KEY_SPACE: 0x39,
    ecodes.KEY_CAPSLOCK: 0x3A,
    ecodes.KEY_F1: 0x3B,
    ecodes.KEY_F2: 0x3C,
    ecodes.KEY_F3: 0x3D,
    ecodes.KEY_F4: 0x3E,
    ecodes.KEY_F5: 0x3F,
    ecodes.KEY_F6: 0x40,
    ecodes.KEY_F7: 0x41,
    ecodes.KEY_F8: 0x42,
    ecodes.KEY_F9: 0x43,
    ecodes.KEY_F10: 0x44,
    ecodes.KEY_NUMLOCK: 0x45,
    ecodes.KEY_SCROLLLOCK: 0x46,
    ecodes.KEY_KP7: 0x47,
    ecodes.KEY_KP8: 0x48,
    ecodes.KEY_KP9: 0x49,
    ecodes.KEY_KPMINUS: 0x4A,
    ecodes.KEY_KP4: 0x4B,
    ecodes.KEY_KP5: 0x4C,
    ecodes.KEY_KP6: 0x4D,
    ecodes.KEY_KPPLUS: 0x4E,
    ecodes.KEY_KP1: 0x4F,
    ecodes.KEY_KP2: 0x50,
    ecodes.KEY_KP3: 0x51,
    ecodes.KEY_KP0: 0x52,
    ecodes.KEY_KPDOT: 0x53,
    ecodes.KEY_102ND: 0x56,
    ecodes.KEY_F11: 0x57,
    ecodes.KEY_F12: 0x58,
    # Extended keys (E0-prefixed)
    ecodes.KEY_KPENTER: 0xE01C,
    ecodes.KEY_RIGHTCTRL: 0xE01D,
    ecodes.KEY_KPSLASH: 0xE035,
    ecodes.KEY_RIGHTALT: 0xE038,
    ecodes.KEY_HOME: 0xE047,
    ecodes.KEY_UP: 0xE048,
    ecodes.KEY_PAGEUP: 0xE049,
    ecodes.KEY_LEFT: 0xE04B,
    ecodes.KEY_RIGHT: 0xE04D,
    ecodes.KEY_END: 0xE04F,
    ecodes.KEY_DOWN: 0xE050,
    ecodes.KEY_PAGEDOWN: 0xE051,
    ecodes.KEY_INSERT: 0xE052,
    ecodes.KEY_DELETE: 0xE053,
    ecodes.KEY_LEFTMETA: 0xE05B,
    ecodes.KEY_RIGHTMETA: 0xE05C,
    ecodes.KEY_MENU: 0xE05D,
    ecodes.KEY_COMPOSE: 0xE05D,
    ecodes.KEY_POWER: 0xE05E,
    ecodes.KEY_SLEEP: 0xE05F,
    ecodes.KEY_SYSRQ: 0xE037,
    ecodes.KEY_PAUSE: 0xE11D45,
}

# Additional function key scancodes (F13–F24) follow the set 1 convention used
# on extended keyboards. They are not always present on commodity keyboards but
# are included for completeness.
for idx, code in enumerate(range(13, 25), start=0):
    attr = f"KEY_F{code}"
    if hasattr(ecodes, attr):
        DEFAULT_SCANCODE_MAP[getattr(ecodes, attr)] = 0x64 + idx


class VirtualInput:
    """Manage a virtual keyboard/mouse backed by /dev/uinput."""

    def __init__(
        self,
        viewport: Tuple[int, int],
        device_name: str = "haindy-virtual-input",
        keyboard_layout: str = "us",
        emit_scancodes: bool = True,
        key_delay_ms: int = 12,
        uinput_device: Optional[UInput] = None,
    ) -> None:
        width, height = viewport
        abs_caps = [
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

        self._ui = uinput_device or UInput(capabilities, name=device_name, bustype=0x03)
        self._viewport = viewport
        self._keyboard_layout = self._normalize_layout(keyboard_layout)
        self._emit_scancodes = emit_scancodes
        self._key_delay = max(key_delay_ms, 0) / 1000.0
        self._scancode_map = DEFAULT_SCANCODE_MAP
        self._missing_scancodes: set[int] = set()
        logger.info(
            "Initialized virtual input device",
            extra={
                "device": device_name,
                "viewport": f"{viewport[0]}x{viewport[1]}",
                "keyboard_layout": self._keyboard_layout,
                "scancodes": self._emit_scancodes,
            },
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

    async def click(
        self, x: int, y: int, button: str = "left", click_count: int = 1
    ) -> None:
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

    async def drag(
        self, start: Tuple[int, int], end: Tuple[int, int], steps: int = 1
    ) -> None:
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
            self._emit_char(char)
            await asyncio.sleep(max(self._key_delay, 0.005))

    async def press_key(self, key: str | Iterable[str]) -> None:
        """Press a key or key combination."""
        sequence = self._normalize_key_sequence(key)
        codes = [self._lookup_key_code(item) for item in sequence]
        codes = [c for c in codes if c is not None]
        if not codes:
            logger.debug("No valid keycodes found for press_key", extra={"key": key})
            return

        modifiers, primary = codes[:-1], codes[-1]

        for code in modifiers:
            self._write_key_event(code, 1)
        self._write_key_event(primary, 1)
        self._ui.syn()
        await asyncio.sleep(max(self._key_delay, 0.01))
        self._write_key_event(primary, 0)
        for code in reversed(modifiers):
            self._write_key_event(code, 0)
        self._ui.syn()

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(milliseconds / 1000.0)

    def _emit_char(self, char: str) -> None:
        code, modifiers = self._char_to_key(char)
        if code is None:
            logger.debug("Skipping unsupported character", extra={"char": repr(char)})
            return
        for mod in modifiers:
            self._write_key_event(mod, 1)
        self._write_key_event(code, 1)
        self._ui.syn()
        self._write_key_event(code, 0)
        for mod in reversed(modifiers):
            self._write_key_event(mod, 0)
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
        if pixels == 0:
            return 0
        direction = 1 if pixels < 0 else -1
        return direction * max(abs(pixels) // 120, 1)

    def _lookup_scancode(self, key_code: int) -> Optional[int]:
        if not self._emit_scancodes:
            return None
        scancode = self._scancode_map.get(key_code)
        if scancode is None:
            if key_code not in self._missing_scancodes:
                self._missing_scancodes.add(key_code)
                logger.debug(
                    "No scancode mapping for keycode", extra={"keycode": key_code}
                )
            return None
        return scancode

    def _write_key_event(self, key_code: int, value: int) -> None:
        scancode = self._lookup_scancode(key_code)
        if scancode is not None:
            self._ui.write(ecodes.EV_MSC, ecodes.MSC_SCAN, scancode)
        self._ui.write(ecodes.EV_KEY, key_code, value)

    @staticmethod
    def _normalize_key_sequence(key: str | Iterable[str]) -> List[str]:
        sequence: List[str] = []
        if isinstance(key, str):
            if "+" in key:
                sequence = [part.strip() for part in key.split("+") if part.strip()]
            elif key.strip():
                sequence = [key.strip()]
        else:
            sequence = [k for k in key if k]
        return sequence

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        width, height = self._viewport
        x_clamped = max(0, min(int(x), max(width - 1, 0)))
        y_clamped = max(0, min(int(y), max(height - 1, 0)))
        return x_clamped, y_clamped

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        normalized = (layout or "us").lower()
        if normalized not in {"us", "es"}:
            logger.warning("Unknown keyboard layout; defaulting to US", extra={"layout": layout})
            return "us"
        return normalized

    @staticmethod
    def _keyboard_keys() -> List[int]:
        keys = {
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
            ecodes.KEY_NUMLOCK,
            ecodes.KEY_SCROLLLOCK,
            ecodes.KEY_SYSRQ,
            ecodes.KEY_PAUSE,
            ecodes.KEY_LEFT,
            ecodes.KEY_RIGHT,
            ecodes.KEY_UP,
            ecodes.KEY_DOWN,
            ecodes.KEY_HOME,
            ecodes.KEY_END,
            ecodes.KEY_PAGEUP,
            ecodes.KEY_PAGEDOWN,
            ecodes.KEY_INSERT,
            ecodes.KEY_DELETE,
            ecodes.KEY_102ND,
        }

        for attr in (
            "KEY_LEFTMETA",
            "KEY_RIGHTMETA",
            "KEY_MENU",
            "KEY_COMPOSE",
            "KEY_POWER",
            "KEY_SLEEP",
        ):
            if hasattr(ecodes, attr):
                keys.add(getattr(ecodes, attr))

        for fn in range(1, 25):
            code = getattr(ecodes, f"KEY_F{fn}", None)
            if code:
                keys.add(code)

        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            keys.add(getattr(ecodes, f"KEY_{letter}"))

        for digit in "0123456789":
            keys.add(getattr(ecodes, f"KEY_{digit}"))

        keys.update(
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

        keypad_keys = [
            ecodes.KEY_KP0,
            ecodes.KEY_KP1,
            ecodes.KEY_KP2,
            ecodes.KEY_KP3,
            ecodes.KEY_KP4,
            ecodes.KEY_KP5,
            ecodes.KEY_KP6,
            ecodes.KEY_KP7,
            ecodes.KEY_KP8,
            ecodes.KEY_KP9,
            ecodes.KEY_KPASTERISK,
            ecodes.KEY_KPSLASH,
            ecodes.KEY_KPMINUS,
            ecodes.KEY_KPPLUS,
            ecodes.KEY_KPDOT,
            ecodes.KEY_KPENTER,
        ]
        keys.update(k for k in keypad_keys if k is not None)

        return sorted(keys)

    def _char_to_key(self, char: str) -> Tuple[Optional[int], Tuple[int, ...]]:
        """Map character to keycode and modifier sequence for the configured layout."""
        if self._keyboard_layout == "es":
            return self._char_to_key_es(char)
        return self._char_to_key_us(char)

    @staticmethod
    def _char_to_key_us(char: str) -> Tuple[Optional[int], Tuple[int, ...]]:
        """US layout mapping (default)."""
        if not char:
            return None, ()
        if char.isalpha():
            key = getattr(ecodes, f"KEY_{char.upper()}", None)
            return key, (ecodes.KEY_LEFTSHIFT,) if char.isupper() else ()
        if char.isdigit():
            key = getattr(ecodes, f"KEY_{char}", None)
            return key, ()

        digit_shift_map: Dict[str, Tuple[int, Tuple[int, ...]]] = {
            "!": (ecodes.KEY_1, (ecodes.KEY_LEFTSHIFT,)),
            "@": (ecodes.KEY_2, (ecodes.KEY_LEFTSHIFT,)),
            "#": (ecodes.KEY_3, (ecodes.KEY_LEFTSHIFT,)),
            "$": (ecodes.KEY_4, (ecodes.KEY_LEFTSHIFT,)),
            "%": (ecodes.KEY_5, (ecodes.KEY_LEFTSHIFT,)),
            "^": (ecodes.KEY_6, (ecodes.KEY_LEFTSHIFT,)),
            "&": (ecodes.KEY_7, (ecodes.KEY_LEFTSHIFT,)),
            "*": (ecodes.KEY_8, (ecodes.KEY_LEFTSHIFT,)),
            "(": (ecodes.KEY_9, (ecodes.KEY_LEFTSHIFT,)),
            ")": (ecodes.KEY_0, (ecodes.KEY_LEFTSHIFT,)),
        }
        if char in digit_shift_map:
            return digit_shift_map[char]

        mapping: Dict[str, Tuple[int, Tuple[int, ...]]] = {
            " ": (ecodes.KEY_SPACE, ()),
            "\n": (ecodes.KEY_ENTER, ()),
            "\t": (ecodes.KEY_TAB, ()),
            "-": (ecodes.KEY_MINUS, ()),
            "_": (ecodes.KEY_MINUS, (ecodes.KEY_LEFTSHIFT,)),
            "=": (ecodes.KEY_EQUAL, ()),
            "+": (ecodes.KEY_EQUAL, (ecodes.KEY_LEFTSHIFT,)),
            "[": (ecodes.KEY_LEFTBRACE, ()),
            "]": (ecodes.KEY_RIGHTBRACE, ()),
            "{": (ecodes.KEY_LEFTBRACE, (ecodes.KEY_LEFTSHIFT,)),
            "}": (ecodes.KEY_RIGHTBRACE, (ecodes.KEY_LEFTSHIFT,)),
            "\\": (ecodes.KEY_BACKSLASH, ()),
            "|": (ecodes.KEY_BACKSLASH, (ecodes.KEY_LEFTSHIFT,)),
            ";": (ecodes.KEY_SEMICOLON, ()),
            ":": (ecodes.KEY_SEMICOLON, (ecodes.KEY_LEFTSHIFT,)),
            "'": (ecodes.KEY_APOSTROPHE, ()),
            '"': (ecodes.KEY_APOSTROPHE, (ecodes.KEY_LEFTSHIFT,)),
            "`": (ecodes.KEY_GRAVE, ()),
            "~": (ecodes.KEY_GRAVE, (ecodes.KEY_LEFTSHIFT,)),
            ",": (ecodes.KEY_COMMA, ()),
            "<": (ecodes.KEY_COMMA, (ecodes.KEY_LEFTSHIFT,)),
            ".": (ecodes.KEY_DOT, ()),
            ">": (ecodes.KEY_DOT, (ecodes.KEY_LEFTSHIFT,)),
            "/": (ecodes.KEY_SLASH, ()),
            "?": (ecodes.KEY_SLASH, (ecodes.KEY_LEFTSHIFT,)),
        }
        return mapping.get(char, (None, ()))

    @staticmethod
    def _char_to_key_es(char: str) -> Tuple[Optional[int], Tuple[int, ...]]:
        """Spanish (Spain) layout mapping using AltGr where required."""
        if not char:
            return None, ()
        shift = ecodes.KEY_LEFTSHIFT
        altgr = ecodes.KEY_RIGHTALT

        if char.isalpha():
            if char.lower() == "\u00f1":
                return ecodes.KEY_SEMICOLON, (shift,) if char.isupper() else ()
            if char.lower() == "\u00e7":
                return ecodes.KEY_BACKSLASH, (shift,) if char.isupper() else ()
            key = getattr(ecodes, f"KEY_{char.upper()}", None)
            return key, (shift,) if char.isupper() else ()

        if char.isdigit():
            key = getattr(ecodes, f"KEY_{char}", None)
            return key, ()

        mapping: Dict[str, Tuple[int, Tuple[int, ...]]] = {
            " ": (ecodes.KEY_SPACE, ()),
            "\n": (ecodes.KEY_ENTER, ()),
            "\t": (ecodes.KEY_TAB, ()),
            "!": (ecodes.KEY_1, (shift,)),
            "\u00a1": (ecodes.KEY_EQUAL, ()),
            '"': (ecodes.KEY_2, (shift,)),
            "@": (ecodes.KEY_2, (altgr,)),
            "#": (ecodes.KEY_3, (altgr,)),
            "$": (ecodes.KEY_4, (shift,)),
            "%": (ecodes.KEY_5, (shift,)),
            "&": (ecodes.KEY_6, (shift,)),
            "/": (ecodes.KEY_7, (shift,)),
            "(": (ecodes.KEY_8, (shift,)),
            ")": (ecodes.KEY_9, (shift,)),
            "=": (ecodes.KEY_0, (shift,)),
            "'": (ecodes.KEY_MINUS, ()),
            "?": (ecodes.KEY_MINUS, (shift,)),
            "\u00bf": (ecodes.KEY_EQUAL, (shift,)),
            "\\": (ecodes.KEY_MINUS, (altgr,)),
            "|": (ecodes.KEY_1, (altgr,)),
            "~": (ecodes.KEY_4, (altgr,)),
            "{": (ecodes.KEY_7, (altgr,)),
            "}": (ecodes.KEY_0, (altgr,)),
            "[": (ecodes.KEY_8, (altgr,)),
            "]": (ecodes.KEY_9, (altgr,)),
            "+": (ecodes.KEY_RIGHTBRACE, ()),
            "*": (ecodes.KEY_RIGHTBRACE, (shift,)),
            "-": (ecodes.KEY_SLASH, ()),
            "_": (ecodes.KEY_SLASH, (shift,)),
            ",": (ecodes.KEY_COMMA, ()),
            ";": (ecodes.KEY_COMMA, (shift,)),
            ".": (ecodes.KEY_DOT, ()),
            ":": (ecodes.KEY_DOT, (shift,)),
            "<": (ecodes.KEY_102ND, ()),
            ">": (ecodes.KEY_102ND, (shift,)),
            "\u20ac": (ecodes.KEY_E, (altgr,)),
            "\u00ba": (ecodes.KEY_GRAVE, ()),
            "\u00aa": (ecodes.KEY_GRAVE, (shift,)),
            "\u00b7": (ecodes.KEY_3, (shift,)),
        }
        return mapping.get(char, (None, ()))

    @staticmethod
    def _lookup_key_code(name: str) -> Optional[int]:
        normalized = name.strip().lower()
        meta_key = getattr(ecodes, "KEY_LEFTMETA", None)
        right_meta_key = getattr(ecodes, "KEY_RIGHTMETA", None)
        menu_key = getattr(ecodes, "KEY_MENU", getattr(ecodes, "KEY_COMPOSE", None))
        alias_map = {
            "ctrl": ecodes.KEY_LEFTCTRL,
            "control": ecodes.KEY_LEFTCTRL,
            "lctrl": ecodes.KEY_LEFTCTRL,
            "rctrl": ecodes.KEY_RIGHTCTRL,
            "alt": ecodes.KEY_LEFTALT,
            "lalt": ecodes.KEY_LEFTALT,
            "altgr": ecodes.KEY_RIGHTALT,
            "ralt": ecodes.KEY_RIGHTALT,
            "shift": ecodes.KEY_LEFTSHIFT,
            "lshift": ecodes.KEY_LEFTSHIFT,
            "rshift": ecodes.KEY_RIGHTSHIFT,
            "meta": meta_key,
            "cmd": meta_key,
            "super": meta_key,
            "win": meta_key,
            "leftmeta": meta_key,
            "rightmeta": right_meta_key,
            "menu": menu_key,
            "compose": getattr(ecodes, "KEY_COMPOSE", None),
            "enter": ecodes.KEY_ENTER,
            "return": ecodes.KEY_ENTER,
            "kpenter": ecodes.KEY_KPENTER,
            "tab": ecodes.KEY_TAB,
            "esc": ecodes.KEY_ESC,
            "escape": ecodes.KEY_ESC,
            "space": ecodes.KEY_SPACE,
            "backspace": ecodes.KEY_BACKSPACE,
            "delete": ecodes.KEY_DELETE,
            "del": ecodes.KEY_DELETE,
            "insert": ecodes.KEY_INSERT,
            "ins": ecodes.KEY_INSERT,
            "home": ecodes.KEY_HOME,
            "end": ecodes.KEY_END,
            "pageup": ecodes.KEY_PAGEUP,
            "pgup": ecodes.KEY_PAGEUP,
            "pagedown": ecodes.KEY_PAGEDOWN,
            "pgdn": ecodes.KEY_PAGEDOWN,
            "up": ecodes.KEY_UP,
            "down": ecodes.KEY_DOWN,
            "left": ecodes.KEY_LEFT,
            "right": ecodes.KEY_RIGHT,
            "numlock": ecodes.KEY_NUMLOCK,
            "scrolllock": ecodes.KEY_SCROLLLOCK,
            "capslock": ecodes.KEY_CAPSLOCK,
            "printscreen": ecodes.KEY_SYSRQ,
            "prtsc": ecodes.KEY_SYSRQ,
            "sysrq": ecodes.KEY_SYSRQ,
            "pause": ecodes.KEY_PAUSE,
            "break": ecodes.KEY_PAUSE,
            "kpplus": ecodes.KEY_KPPLUS,
            "kpminus": ecodes.KEY_KPMINUS,
            "kpdivide": ecodes.KEY_KPSLASH,
            "kpmultiply": ecodes.KEY_KPASTERISK,
            "kpdot": ecodes.KEY_KPDOT,
        }
        alias_map = {k: v for k, v in alias_map.items() if v is not None}
        if normalized in alias_map:
            return alias_map[normalized]

        if normalized.startswith("kp") and normalized[2:].isdigit():
            code = getattr(ecodes, f"KEY_KP{normalized[2:]}", None)
            if code:
                return code

        if normalized.startswith("f") and normalized[1:].isdigit():
            idx = int(normalized[1:])
            if 1 <= idx <= 24:
                return getattr(ecodes, f"KEY_F{idx}", None)

        upper = normalized.upper()
        if len(upper) == 1 and upper.isalpha():
            return getattr(ecodes, f"KEY_{upper}", None)
        if len(upper) == 1 and upper.isdigit():
            return getattr(ecodes, f"KEY_{upper}", None)

        return None
