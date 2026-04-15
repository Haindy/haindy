"""Virtual keyboard and mouse using uinput."""

from __future__ import annotations

import asyncio
import logging
import shutil
from asyncio import subprocess as aio_subprocess
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from evdev import UInput as EvdevUInput

_AbsInfoCtor: Any
_UInputCtor: Any
_UInputErrorType: type[Exception]

try:
    from evdev import AbsInfo as _AbsInfoCtor  # type: ignore[no-redef]
    from evdev import UInput as _UInputCtor  # type: ignore[no-redef]
    from evdev import ecodes
    from evdev.uinput import UInputError as _UInputErrorType  # type: ignore[no-redef]

    _EVDEV_AVAILABLE = True
    _EVDEV_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    _AbsInfoCtor = None
    _UInputCtor = None
    ecodes = cast(Any, None)
    _UInputErrorType = RuntimeError
    _EVDEV_AVAILABLE = False
    _EVDEV_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


def _virtual_input_backend_error() -> RuntimeError:
    message = (
        "Virtual input backend unavailable: the desktop input backend requires "
        "Linux evdev or xdotool."
    )
    if _EVDEV_IMPORT_ERROR is None:
        return RuntimeError(message)
    return RuntimeError(f"{message} ({_EVDEV_IMPORT_ERROR})")


def _build_default_scancode_map() -> dict[int, int]:
    if not _EVDEV_AVAILABLE:
        return {}

    # Scancode map (set 1 / Linux input) keyed by EV_KEY code. Extended
    # (E0-prefixed) scancodes are encoded as single integers, e.g., 0xE01D for
    # right control.
    default_scancode_map: dict[int, int] = {
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

    # Additional function key scancodes (F13-F24) follow the set 1 convention
    # used on extended keyboards. They are not always present on commodity
    # keyboards but are included for completeness.
    for idx, code in enumerate(range(13, 25), start=0):
        attr = f"KEY_F{code}"
        if hasattr(ecodes, attr):
            default_scancode_map[getattr(ecodes, attr)] = 0x64 + idx
    return default_scancode_map


DEFAULT_SCANCODE_MAP = _build_default_scancode_map()


class VirtualInput:
    """Manage a virtual keyboard/mouse backed by /dev/uinput."""

    def __init__(
        self,
        viewport: tuple[int, int],
        device_name: str = "haindy-virtual-input",
        keyboard_layout: str = "us",
        emit_scancodes: bool = True,
        key_delay_ms: int = 12,
        uinput_device: EvdevUInput | None = None,
    ) -> None:
        self._ui: EvdevUInput | None = None
        self._xdotool_binary: str | None = None
        if uinput_device is not None:
            self._ui = uinput_device
        elif _EVDEV_AVAILABLE:
            width, height = viewport
            abs_caps = [
                (ecodes.ABS_X, _AbsInfoCtor(0, 0, max(width - 1, 0), 0, 0, 0)),
                (ecodes.ABS_Y, _AbsInfoCtor(0, 0, max(height - 1, 0), 0, 0, 0)),
            ]

            key_codes = self._keyboard_keys()
            capabilities = cast(
                dict[int, Sequence[int]],
                {
                    ecodes.EV_KEY: key_codes,
                    ecodes.EV_ABS: abs_caps,
                    ecodes.EV_REL: [ecodes.REL_WHEEL, ecodes.REL_HWHEEL],
                    ecodes.EV_MSC: [ecodes.MSC_SCAN],
                },
            )
            try:
                self._ui = _UInputCtor(capabilities, name=device_name, bustype=0x03)
            except (PermissionError, OSError, _UInputErrorType) as exc:
                xdotool_binary = shutil.which("xdotool")
                if not xdotool_binary:
                    raise _virtual_input_backend_error() from exc
                self._xdotool_binary = xdotool_binary
                logger.warning(
                    "uinput unavailable; falling back to xdotool input backend",
                    extra={"error": str(exc), "binary": xdotool_binary},
                )
        else:
            xdotool_binary = shutil.which("xdotool")
            if not xdotool_binary:
                raise _virtual_input_backend_error()
            self._xdotool_binary = xdotool_binary
            logger.warning(
                "evdev unavailable; falling back to xdotool input backend",
                extra={
                    "error": str(_EVDEV_IMPORT_ERROR) if _EVDEV_IMPORT_ERROR else None,
                    "binary": xdotool_binary,
                },
            )

        self._viewport = viewport
        self._last_x: int = viewport[0] // 2
        self._last_y: int = viewport[1] // 2
        self._keyboard_layout = self._normalize_layout(keyboard_layout)
        self._emit_scancodes = emit_scancodes
        self._key_delay = max(key_delay_ms, 0) / 1000.0
        self._scancode_map = dict(DEFAULT_SCANCODE_MAP)
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

    async def _interpolated_move(self, target_x: int, target_y: int) -> None:
        """Move pointer from current position to target via intermediate ABS
        events so that toolkits (GTK, Qt) register proper enter/leave crossing
        events -- required for hover-triggered submenus, tooltips, etc."""
        assert self._ui is not None
        dx = target_x - self._last_x
        dy = target_y - self._last_y
        distance = max(abs(dx), abs(dy))
        step_size = 8
        num_steps = max(distance // step_size, 1)
        for i in range(1, num_steps + 1):
            progress = i / num_steps
            ix = int(self._last_x + dx * progress)
            iy = int(self._last_y + dy * progress)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_X, ix)
            self._ui.write(ecodes.EV_ABS, ecodes.ABS_Y, iy)
            self._ui.syn()
            await asyncio.sleep(0.002)
        self._last_x = target_x
        self._last_y = target_y

    async def move(self, x: int, y: int, steps: int = 1) -> None:
        """Move pointer to absolute coordinates."""
        x_clamped, y_clamped = self._clamp(x, y)
        if self._ui is None:
            await self._xdotool_move(x_clamped, y_clamped)
            self._last_x, self._last_y = x_clamped, y_clamped
            return

        await self._interpolated_move(x_clamped, y_clamped)

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        """Click at absolute coordinates, optionally holding modifier keys."""
        x_clamped, y_clamped = self._clamp(x, y)
        logger.debug(
            "click dispatch: (%d, %d) -> clamped (%d, %d), button=%s, count=%d, uinput=%s",
            x,
            y,
            x_clamped,
            y_clamped,
            button,
            click_count,
            self._ui is not None,
        )
        if self._ui is None:
            await self._xdotool_click(
                x_clamped,
                y_clamped,
                button=button,
                click_count=click_count,
                modifiers=modifiers,
            )
            return

        await self._interpolated_move(x_clamped, y_clamped)

        code = self._button_code(button)
        modifier_codes = self._modifier_codes(modifiers)
        for mc in modifier_codes:
            self._ui.write(ecodes.EV_KEY, mc, 1)
            self._ui.syn()
        for _ in range(max(click_count, 1)):
            self._ui.write(ecodes.EV_KEY, code, 1)
            self._ui.syn()
            self._ui.write(ecodes.EV_KEY, code, 0)
            self._ui.syn()
            await asyncio.sleep(0.02)
        for mc in reversed(modifier_codes):
            self._ui.write(ecodes.EV_KEY, mc, 0)
            self._ui.syn()

    async def drag(
        self, start: tuple[int, int], end: tuple[int, int], steps: int = 1
    ) -> None:
        """Drag from start to end coordinates."""
        start_x, start_y = self._clamp(*start)
        end_x, end_y = self._clamp(*end)
        if self._ui is None:
            await self._xdotool_drag((start_x, start_y), (end_x, end_y), steps=steps)
            self._last_x, self._last_y = end_x, end_y
            return

        await self._interpolated_move(start_x, start_y)
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
        self._last_x, self._last_y = end_x, end_y

    async def scroll(self, x: int = 0, y: int = 0) -> None:
        """Scroll by pixel deltas using wheel events."""
        if self._ui is None:
            await self._xdotool_scroll(x=x, y=y)
            return

        if y:
            self._ui.write(ecodes.EV_REL, ecodes.REL_WHEEL, self._scroll_delta(y))
        if x:
            self._ui.write(ecodes.EV_REL, ecodes.REL_HWHEEL, self._scroll_delta(x))
        self._ui.syn()
        await asyncio.sleep(0.01)

    async def type_text(self, text: str) -> None:
        """Type text using key events."""
        if self._ui is None:
            await self._xdotool_type(text)
            return

        for char in text:
            self._emit_char(char)
            await asyncio.sleep(max(self._key_delay, 0.005))

    async def press_key(self, key: str | Iterable[str]) -> None:
        """Press a key or key combination."""
        if self._ui is None:
            await self._xdotool_press_key(key)
            return

        sequence = self._normalize_key_sequence(key)
        raw_codes = [self._lookup_key_code(item) for item in sequence]
        codes: list[int] = [code for code in raw_codes if code is not None]
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
        ui = self._ui
        if ui is None:
            raise RuntimeError("Virtual input device is not initialized")
        for mod in modifiers:
            self._write_key_event(mod, 1)
        self._write_key_event(code, 1)
        ui.syn()
        self._write_key_event(code, 0)
        for mod in reversed(modifiers):
            self._write_key_event(mod, 0)
        ui.syn()

    @staticmethod
    def _button_code(button: str) -> int:
        normalized = (button or "left").lower()
        if normalized == "right":
            return int(ecodes.BTN_RIGHT)
        if normalized == "middle":
            return int(ecodes.BTN_MIDDLE)
        return int(ecodes.BTN_LEFT)

    @staticmethod
    def _scroll_delta(pixels: int) -> int:
        if pixels == 0:
            return 0
        direction = 1 if pixels < 0 else -1
        return direction * max(abs(pixels) // 120, 1)

    def _lookup_scancode(self, key_code: int) -> int | None:
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
        if self._ui is None:
            return
        scancode = self._lookup_scancode(key_code)
        if scancode is not None:
            self._ui.write(ecodes.EV_MSC, ecodes.MSC_SCAN, scancode)
        self._ui.write(ecodes.EV_KEY, key_code, value)

    @staticmethod
    def _modifier_codes(modifiers: list[str] | None) -> list[int]:
        """Map modifier name strings to evdev key codes."""
        if not modifiers:
            return []
        mapping: dict[str, int] = {
            "ctrl": ecodes.KEY_LEFTCTRL,
            "control": ecodes.KEY_LEFTCTRL,
            "shift": ecodes.KEY_LEFTSHIFT,
            "alt": ecodes.KEY_LEFTALT,
            "meta": ecodes.KEY_LEFTMETA,
            "super": ecodes.KEY_LEFTMETA,
        }
        return [mapping[m] for m in modifiers if m in mapping]

    @staticmethod
    def _modifier_xdotool_names(modifiers: list[str] | None) -> list[str]:
        """Map modifier name strings to xdotool key names."""
        if not modifiers:
            return []
        mapping: dict[str, str] = {
            "ctrl": "ctrl",
            "control": "ctrl",
            "shift": "shift",
            "alt": "alt",
            "meta": "super",
            "super": "super",
        }
        return [mapping[m] for m in modifiers if m in mapping]

    async def _run_xdotool(self, *args: str) -> None:
        if not self._xdotool_binary:
            raise RuntimeError(
                "Virtual input backend unavailable: /dev/uinput failed and xdotool is not installed."
            )
        process = await asyncio.create_subprocess_exec(
            self._xdotool_binary,
            *args,
            stdout=aio_subprocess.DEVNULL,
            stderr=aio_subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            message = (stderr or b"").decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"xdotool command failed ({' '.join(args)}): {message or 'unknown error'}"
            )

    async def _xdotool_move(self, x: int, y: int) -> None:
        await self._run_xdotool("mousemove", "--sync", str(x), str(y))

    async def _xdotool_click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        button_map = {"left": 1, "middle": 2, "right": 3}
        button_code = button_map.get((button or "left").lower(), 1)
        mod_key_names = self._modifier_xdotool_names(modifiers)
        for k in mod_key_names:
            await self._run_xdotool("keydown", k)
        await self._xdotool_move(x, y)
        await self._run_xdotool(
            "click",
            "--repeat",
            str(max(click_count, 1)),
            str(button_code),
        )
        for k in reversed(mod_key_names):
            await self._run_xdotool("keyup", k)
        await asyncio.sleep(0.02)

    async def _xdotool_drag(
        self, start: tuple[int, int], end: tuple[int, int], steps: int = 1
    ) -> None:
        start_x, start_y = start
        end_x, end_y = end
        await self._xdotool_move(start_x, start_y)
        await self._run_xdotool("mousedown", "1")
        for step in range(max(steps, 1)):
            progress = (step + 1) / max(steps, 1)
            x = int(start_x + (end_x - start_x) * progress)
            y = int(start_y + (end_y - start_y) * progress)
            await self._xdotool_move(x, y)
            if steps > 1:
                await asyncio.sleep(0.01)
        await self._run_xdotool("mouseup", "1")

    async def _xdotool_scroll(self, x: int = 0, y: int = 0) -> None:
        if y:
            vertical_button = "5" if y > 0 else "4"
            repeats = max(abs(y) // 120, 1)
            await self._run_xdotool("click", "--repeat", str(repeats), vertical_button)
        if x:
            horizontal_button = "7" if x > 0 else "6"
            repeats = max(abs(x) // 120, 1)
            await self._run_xdotool(
                "click", "--repeat", str(repeats), horizontal_button
            )
        await asyncio.sleep(0.01)

    async def _xdotool_type(self, text: str) -> None:
        if text == "":
            return
        delay_ms = max(int(self._key_delay * 1000), 5)
        await self._run_xdotool(
            "type",
            "--delay",
            str(delay_ms),
            "--clearmodifiers",
            "--",
            text,
        )

    async def _xdotool_press_key(self, key: str | Iterable[str]) -> None:
        sequence = self._normalize_key_sequence(key)
        translated = [
            translated_key
            for translated_key in (
                self._translate_xdotool_key_name(item) for item in sequence
            )
            if translated_key
        ]
        if not translated:
            logger.debug(
                "No valid xdotool keys found for press_key", extra={"key": key}
            )
            return
        await self._run_xdotool("key", "--clearmodifiers", "+".join(translated))
        await asyncio.sleep(max(self._key_delay, 0.01))

    @staticmethod
    def _translate_xdotool_key_name(name: str) -> str | None:
        normalized = (name or "").strip().lower()
        if not normalized:
            return None
        alias_map = {
            "ctrl": "ctrl",
            "control": "ctrl",
            "lctrl": "ctrl",
            "rctrl": "ctrl",
            "alt": "alt",
            "lalt": "alt",
            "ralt": "alt",
            "altgr": "ISO_Level3_Shift",
            "shift": "shift",
            "lshift": "shift",
            "rshift": "shift",
            "meta": "super",
            "cmd": "super",
            "super": "super",
            "win": "super",
            "leftmeta": "super",
            "rightmeta": "super_R",
            "menu": "Menu",
            "compose": "Multi_key",
            "enter": "Return",
            "return": "Return",
            "kpenter": "KP_Enter",
            "tab": "Tab",
            "esc": "Escape",
            "escape": "Escape",
            "space": "space",
            "backspace": "BackSpace",
            "delete": "Delete",
            "del": "Delete",
            "insert": "Insert",
            "ins": "Insert",
            "home": "Home",
            "end": "End",
            "pageup": "Page_Up",
            "pgup": "Page_Up",
            "pagedown": "Page_Down",
            "pgdn": "Page_Down",
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
            "numlock": "Num_Lock",
            "scrolllock": "Scroll_Lock",
            "capslock": "Caps_Lock",
            "printscreen": "Print",
            "prtsc": "Print",
            "sysrq": "Sys_Req",
            "pause": "Pause",
            "break": "Break",
            "kpplus": "KP_Add",
            "kpminus": "KP_Subtract",
            "kpdivide": "KP_Divide",
            "kpmultiply": "KP_Multiply",
            "kpdot": "KP_Decimal",
        }
        if normalized in alias_map:
            return alias_map[normalized]
        if normalized.startswith("kp") and normalized[2:].isdigit():
            return f"KP_{normalized[2:]}"
        if normalized.startswith("f") and normalized[1:].isdigit():
            index = int(normalized[1:])
            if 1 <= index <= 24:
                return f"F{index}"
        if len(normalized) == 1 and (normalized.isalpha() or normalized.isdigit()):
            return normalized
        return name.strip()

    @staticmethod
    def _normalize_key_sequence(key: str | Iterable[str]) -> list[str]:
        sequence: list[str] = []
        if isinstance(key, str):
            if "+" in key:
                sequence = [part.strip() for part in key.split("+") if part.strip()]
            elif key.strip():
                sequence = [key.strip()]
        else:
            sequence = [k for k in key if k]
        return sequence

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        width, height = self._viewport
        x_clamped = max(0, min(int(x), max(width - 1, 0)))
        y_clamped = max(0, min(int(y), max(height - 1, 0)))
        return x_clamped, y_clamped

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        normalized = (layout or "us").lower()
        if normalized not in {"us", "es"}:
            logger.warning(
                "Unknown keyboard layout; defaulting to US", extra={"layout": layout}
            )
            return "us"
        return normalized

    @staticmethod
    def _keyboard_keys() -> list[int]:
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

    def _char_to_key(self, char: str) -> tuple[int | None, tuple[int, ...]]:
        """Map character to keycode and modifier sequence for the configured layout."""
        if self._keyboard_layout == "es":
            return self._char_to_key_es(char)
        return self._char_to_key_us(char)

    @staticmethod
    def _char_to_key_us(char: str) -> tuple[int | None, tuple[int, ...]]:
        """US layout mapping (default)."""
        if not char:
            return None, ()
        if char.isalpha():
            key = getattr(ecodes, f"KEY_{char.upper()}", None)
            return key, (ecodes.KEY_LEFTSHIFT,) if char.isupper() else ()
        if char.isdigit():
            key = getattr(ecodes, f"KEY_{char}", None)
            return key, ()

        digit_shift_map: dict[str, tuple[int, tuple[int, ...]]] = {
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

        mapping: dict[str, tuple[int, tuple[int, ...]]] = {
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
    def _char_to_key_es(char: str) -> tuple[int | None, tuple[int, ...]]:
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

        mapping: dict[str, tuple[int, tuple[int, ...]]] = {
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
    def _lookup_key_code(name: str) -> int | None:
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
        alias_lookup: dict[str, int] = {
            k: int(v) for k, v in alias_map.items() if v is not None
        }
        if normalized in alias_lookup:
            return alias_lookup[normalized]

        if normalized.startswith("kp") and normalized[2:].isdigit():
            code = getattr(ecodes, f"KEY_KP{normalized[2:]}", None)
            if code:
                return int(code)

        if normalized.startswith("f") and normalized[1:].isdigit():
            idx = int(normalized[1:])
            if 1 <= idx <= 24:
                code = getattr(ecodes, f"KEY_F{idx}", None)
                return int(code) if code is not None else None

        upper = normalized.upper()
        if len(upper) == 1 and upper.isalpha():
            code = getattr(ecodes, f"KEY_{upper}", None)
            return int(code) if code is not None else None
        if len(upper) == 1 and upper.isdigit():
            code = getattr(ecodes, f"KEY_{upper}", None)
            return int(code) if code is not None else None

        return None
