"""Android automation driver backed by adb."""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import ClassVar

from src.core.interfaces import AutomationDriver
from src.mobile.adb_client import ADBClient

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_WM_SIZE_PATTERN = re.compile(r"(?P<width>\d+)\s*x\s*(?P<height>\d+)")
_ADB_TEXT_ESCAPE_PATTERN = re.compile(r"([\\\"'&<>|;()$`])")


class MobileDriver(AutomationDriver):
    """Automation driver for Android devices using adb input commands."""

    _ANDROID_KEYCODES: ClassVar[dict[str, int]] = {
        "enter": 66,
        "return": 66,
        "backspace": 67,
        "delete": 67,
        "del": 67,
        "tab": 61,
        "space": 62,
        "escape": 111,
        "esc": 111,
        "home": 3,
        "back": 4,
        "menu": 82,
        "recent": 187,
        "app_switch": 187,
        "recent_apps": 187,
        "up": 19,
        "down": 20,
        "left": 21,
        "right": 22,
        "page_up": 92,
        "page_down": 93,
        "end": 123,
        "ctrl": 113,
        "control": 113,
        "shift": 59,
        "alt": 57,
        "meta": 117,
        "cmd": 117,
        "volume_up": 24,
        "volume_down": 25,
        "power": 26,
        ".": 56,
        ",": 55,
        "/": 76,
        "@": 77,
    }

    def __init__(
        self,
        preferred_serial: str | None = None,
        adb_timeout_seconds: float = 15.0,
        adb_client: ADBClient | None = None,
    ) -> None:
        self.adb = adb_client or ADBClient(timeout_seconds=adb_timeout_seconds)
        self._preferred_serial = (preferred_serial or "").strip() or None
        self._app_package: str | None = None
        self._app_activity: str | None = None
        self._started = False
        self._last_screenshot_size: tuple[int, int] | None = None
        self._capturing = False
        self._captured_calls: list[dict[str, object]] = []

    async def start(self) -> None:
        if self._started:
            return
        serial = await self.adb.resolve_serial(self._preferred_serial)
        self.adb.serial = serial
        await self.adb.run_adb("get-state")
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def configure_target(
        self,
        adb_serial: str | None = None,
        app_package: str | None = None,
        app_activity: str | None = None,
    ) -> None:
        """Update target device/app details before entrypoint setup."""
        normalized_serial = (adb_serial or "").strip() or None
        if normalized_serial:
            self._preferred_serial = normalized_serial
            self.adb.serial = normalized_serial
        normalized_package = (app_package or "").strip() or None
        if normalized_package:
            self._app_package = normalized_package
        normalized_activity = (app_activity or "").strip() or None
        if normalized_activity:
            self._app_activity = normalized_activity

    async def run_adb_commands(self, commands: list[str]) -> None:
        """Run context-provided adb commands in sequence."""
        await self._ensure_ready()
        for raw_command in commands:
            command_text = str(raw_command or "").strip()
            if not command_text:
                continue
            parts = shlex.split(command_text)
            if not parts:
                continue
            if parts[0] != "adb":
                raise ValueError(f"Only adb commands are allowed: {command_text!r}")
            await self.adb.run_command(parts)
            self._capture_call("adb_command", {"command": command_text})

    async def force_stop_app(self, app_package: str | None = None) -> None:
        """Force-stop an Android application."""
        await self._ensure_ready()
        package = (app_package or "").strip() or self._app_package
        if not package:
            raise RuntimeError("Cannot force-stop: missing Android package name.")
        await self.adb.run_adb("shell", "am", "force-stop", package)
        self._capture_call("force_stop_app", {"app_package": package})

    async def clear_app_data(self, app_package: str | None = None) -> None:
        """Clear all data and cache for an Android application (pm clear).

        This wipes login sessions, databases, shared preferences, and cache.
        The app will behave as if freshly installed after this call.
        """
        await self._ensure_ready()
        package = (app_package or "").strip() or self._app_package
        if not package:
            raise RuntimeError("Cannot clear app data: missing Android package name.")
        await self.adb.run_adb("shell", "pm", "clear", package)
        self._capture_call("clear_app_data", {"app_package": package})

    async def _is_app_foreground(self, package: str) -> bool:
        """Return True if the package is the current top resumed activity."""
        result = await self.adb.run_adb(
            "shell", "dumpsys", "activity", "activities", check=False
        )
        output = (result.stdout or b"").decode(errors="replace")
        # API 29+: mResumedActivity / topResumedActivity lines contain the package
        for line in output.splitlines():
            if (
                "mResumedActivity" in line or "topResumedActivity" in line
            ) and package in line:
                return True
        return False

    async def launch_app(
        self,
        app_package: str,
        app_activity: str | None = None,
    ) -> None:
        """Launch an Android application via package/activity or monkey fallback."""
        await self._ensure_ready()
        package = str(app_package or "").strip() or self._app_package
        if not package:
            raise RuntimeError("Cannot launch app: missing Android package name.")
        activity = str(app_activity or "").strip() or self._app_activity

        if activity:
            component = activity if "/" in activity else f"{package}/{activity}"
            await self.adb.run_adb("shell", "am", "start", "-n", component)
        else:
            if await self._is_app_foreground(package):
                self._capture_call(
                    "launch_app",
                    {"app_package": package, "app_activity": "", "skipped": True},
                )
                return
            await self.adb.run_adb("shell", "monkey", "-p", package, "1")
        self._capture_call(
            "launch_app",
            {"app_package": package, "app_activity": activity or ""},
        )

    async def navigate(self, url: str) -> None:
        await self._ensure_ready()
        await self.adb.run_adb(
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            url,
        )
        self._capture_call("navigate", {"url": url})

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        await self._ensure_ready()
        normalized_button = str(button or "left").strip().lower()
        if normalized_button not in {"left", "right", "middle"}:
            raise ValueError(f"Unsupported button for mobile tap: {button!r}")
        tap_count = max(int(click_count), 1)
        mapped_x, mapped_y = await self._map_point_to_device(x, y)
        for _ in range(tap_count):
            await self.adb.run_adb(
                "shell",
                "input",
                "tap",
                str(mapped_x),
                str(mapped_y),
            )
        self._capture_call(
            "click",
            {
                "x": x,
                "y": y,
                "button": normalized_button,
                "click_count": tap_count,
                "mapped_x": mapped_x,
                "mapped_y": mapped_y,
            },
        )

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        # Touch-only surfaces do not expose a persistent pointer position.
        self._capture_call("move_mouse", {"x": x, "y": y, "steps": steps, "noop": True})

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        await self._ensure_ready()
        mapped_start_x, mapped_start_y = await self._map_point_to_device(
            start_x, start_y
        )
        mapped_end_x, mapped_end_y = await self._map_point_to_device(end_x, end_y)
        duration_ms = max(int(steps) * 16, 120)
        await self.adb.run_adb(
            "shell",
            "input",
            "swipe",
            str(mapped_start_x),
            str(mapped_start_y),
            str(mapped_end_x),
            str(mapped_end_y),
            str(duration_ms),
        )
        self._capture_call(
            "drag_mouse",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "steps": steps,
                "mapped_start_x": mapped_start_x,
                "mapped_start_y": mapped_start_y,
                "mapped_end_x": mapped_end_x,
                "mapped_end_y": mapped_end_y,
            },
        )

    async def type_text(self, text: str) -> None:
        await self._ensure_ready()
        escaped = self._escape_text_for_adb(text)
        if escaped:
            await self.adb.run_adb("shell", "input", "text", escaped)
        self._capture_call("type_text", {"length": len(text)})

    async def press_key(self, key: str) -> None:
        await self._ensure_ready()
        tokens = [
            self._normalize_key_token(part)
            for part in str(key or "").split("+")
            if part.strip()
        ]
        if not tokens:
            raise ValueError("Key must not be empty.")
        keycodes = [self._keycode_for_token(token) for token in tokens]
        await self.adb.run_adb(
            "shell",
            "input",
            "keyevent",
            *[str(code) for code in keycodes],
        )
        self._capture_call("press_key", {"key": key, "keycodes": keycodes})

    async def scroll(self, direction: str, amount: int) -> None:
        normalized_direction = str(direction or "").strip().lower()
        if normalized_direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Invalid scroll direction: {direction!r}")
        magnitude = max(abs(int(amount)), 1)
        delta = magnitude if normalized_direction in {"down", "right"} else -magnitude
        await self.scroll_by_pixels(
            y=delta if normalized_direction in {"up", "down"} else 0,
            x=delta if normalized_direction in {"left", "right"} else 0,
        )

    async def scroll_by_pixels(
        self,
        x: int = 0,
        y: int = 0,
        smooth: bool = True,
    ) -> None:
        await self._ensure_ready()
        width, height = await self.get_viewport_size()
        scaled_x, scaled_y = self._scale_deltas_to_device(x, y, (width, height))

        start_x = width // 2
        start_y = height // 2
        end_x = self._clamp(start_x - scaled_x, 0, max(width - 1, 0))
        end_y = self._clamp(start_y - scaled_y, 0, max(height - 1, 0))

        if (start_x, start_y) == (end_x, end_y) and (scaled_x != 0 or scaled_y != 0):
            if scaled_y != 0 and height > 1:
                end_y = self._clamp(
                    start_y - (1 if scaled_y > 0 else -1), 0, height - 1
                )
            elif scaled_x != 0 and width > 1:
                end_x = self._clamp(start_x - (1 if scaled_x > 0 else -1), 0, width - 1)

        duration_ms = 220 if smooth else 100
        await self.adb.run_adb(
            "shell",
            "input",
            "swipe",
            str(start_x),
            str(start_y),
            str(end_x),
            str(end_y),
            str(duration_ms),
        )
        self._capture_call("scroll_by_pixels", {"x": x, "y": y, "smooth": smooth})

    async def screenshot(self) -> bytes:
        await self._ensure_ready()
        result = await self.adb.run_adb("exec-out", "screencap", "-p")
        screenshot_bytes = result.stdout

        if not screenshot_bytes.startswith(_PNG_SIGNATURE):
            normalized = screenshot_bytes.replace(b"\r\n", b"\n")
            if normalized.startswith(_PNG_SIGNATURE):
                screenshot_bytes = normalized

        size = self._parse_png_size(screenshot_bytes)
        if size is not None:
            self._last_screenshot_size = size
        self._capture_call("screenshot", {"label": "mobile"})
        return screenshot_bytes

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(max(int(milliseconds), 0) / 1000.0)
        self._capture_call("wait", {"milliseconds": milliseconds})

    async def get_viewport_size(self) -> tuple[int, int]:
        await self._ensure_ready()
        result = await self.adb.run_adb("shell", "wm", "size")
        output = result.stdout_text.strip()
        override_size: tuple[int, int] | None = None
        physical_size: tuple[int, int] | None = None
        fallback_size: tuple[int, int] | None = None

        for raw_line in output.splitlines():
            line = raw_line.strip()
            match = _WM_SIZE_PATTERN.search(line)
            if not match:
                continue
            width = int(match.group("width"))
            height = int(match.group("height"))
            parsed = (width, height)
            lowered = line.lower()
            if "override size" in lowered:
                override_size = parsed
            elif "physical size" in lowered:
                physical_size = parsed
            elif fallback_size is None:
                fallback_size = parsed

        resolved = override_size or physical_size or fallback_size
        if resolved is None:
            raise RuntimeError(
                f"Unable to parse viewport size from adb output: {output!r}"
            )
        return resolved

    async def get_page_url(self) -> str:
        return ""

    async def get_page_title(self) -> str:
        return "Android Device"

    def start_capture(self) -> None:
        self._capturing = True
        self._captured_calls = []

    def stop_capture(self) -> list[dict[str, object]]:
        calls = self._captured_calls.copy()
        self._capturing = False
        self._captured_calls = []
        return calls

    async def _ensure_ready(self) -> None:
        if not self._started:
            await self.start()

    async def _map_point_to_device(self, x: int, y: int) -> tuple[int, int]:
        width, height = await self.get_viewport_size()
        source = self._last_screenshot_size
        mapped_x = int(x)
        mapped_y = int(y)
        if source and source != (width, height) and source[0] > 0 and source[1] > 0:
            mapped_x = round(mapped_x * width / source[0])
            mapped_y = round(mapped_y * height / source[1])
        mapped_x = self._clamp(mapped_x, 0, max(width - 1, 0))
        mapped_y = self._clamp(mapped_y, 0, max(height - 1, 0))
        return mapped_x, mapped_y

    def _scale_deltas_to_device(
        self,
        x: int,
        y: int,
        device_size: tuple[int, int],
    ) -> tuple[int, int]:
        scaled_x = int(x)
        scaled_y = int(y)
        source = self._last_screenshot_size
        if source and source != device_size and source[0] > 0 and source[1] > 0:
            scaled_x = round(scaled_x * device_size[0] / source[0])
            scaled_y = round(scaled_y * device_size[1] / source[1])
            if x != 0 and scaled_x == 0:
                scaled_x = 1 if x > 0 else -1
            if y != 0 and scaled_y == 0:
                scaled_y = 1 if y > 0 else -1
        max_x = max(device_size[0] - 1, 1)
        max_y = max(device_size[1] - 1, 1)
        scaled_x = self._clamp(scaled_x, -max_x, max_x)
        scaled_y = self._clamp(scaled_y, -max_y, max_y)
        return scaled_x, scaled_y

    @staticmethod
    def _parse_png_size(data: bytes) -> tuple[int, int] | None:
        if len(data) < 24 or not data.startswith(_PNG_SIGNATURE):
            return None
        if data[12:16] != b"IHDR":
            return None
        width = int.from_bytes(data[16:20], byteorder="big")
        height = int.from_bytes(data[20:24], byteorder="big")
        if width <= 0 or height <= 0:
            return None
        return width, height

    @classmethod
    def _escape_text_for_adb(cls, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\n", "%s").replace(" ", "%s")
        return _ADB_TEXT_ESCAPE_PATTERN.sub(r"\\\1", normalized)

    def _keycode_for_token(self, token: str) -> int:
        if token in self._ANDROID_KEYCODES:
            return self._ANDROID_KEYCODES[token]
        if len(token) == 1:
            if "a" <= token <= "z":
                return 29 + (ord(token) - ord("a"))
            if "0" <= token <= "9":
                return 7 + (ord(token) - ord("0"))
        raise ValueError(
            "Unsupported Android key "
            f"{token!r}. Use known names like enter/back/tab/home/ctrl or single "
            "alphanumeric characters."
        )

    @staticmethod
    def _normalize_key_token(token: str) -> str:
        normalized = token.strip().lower().replace("-", "_")
        alias_map = {
            "arrowleft": "left",
            "arrowright": "right",
            "arrowup": "up",
            "arrowdown": "down",
        }
        return alias_map.get(normalized, normalized)

    def _capture_call(self, method: str, params: dict[str, object]) -> None:
        if not self._capturing:
            return
        self._captured_calls.append({"method": method, "params": params})

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))
