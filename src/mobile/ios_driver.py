"""iOS automation driver backed by idb (iOS Development Bridge)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from typing import ClassVar

from src.core.interfaces import AutomationDriver
from src.mobile.idb_client import IDBClient, IDBClientProtocol

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Matches the Python-repr output from `idb describe`:
# screen_dimensions=ScreenDimensions(width=2064, height=2752, ..., width_points=1032, height_points=1376)
_SCREEN_DIMENSIONS_PATTERN = re.compile(
    r"width_points=(?P<width>\d+).*?height_points=(?P<height>\d+)"
)


class IOSDriver(AutomationDriver):
    """Automation driver for iOS devices and simulators using idb commands."""

    # USB HID keyboard keycodes (standard HID usage page 0x07)
    _IOS_KEYCODES: ClassVar[dict[str, int]] = {
        "enter": 40,
        "return": 40,
        "escape": 41,
        "esc": 41,
        "backspace": 42,
        "delete": 42,
        "del": 42,
        "tab": 43,
        "space": 44,
        "up": 82,
        "arrowup": 82,
        "down": 81,
        "arrowdown": 81,
        "left": 80,
        "arrowleft": 80,
        "right": 79,
        "arrowright": 79,
        "home": 74,
        "end": 77,
        "page_up": 75,
        "page_down": 78,
        "forward_delete": 76,
        # Modifier keys (left-hand variants)
        "ctrl": 224,
        "control": 224,
        "shift": 225,
        "alt": 226,
        "option": 226,
        "cmd": 227,
        "meta": 227,
        "command": 227,
        # Function keys
        "f1": 58,
        "f2": 59,
        "f3": 60,
        "f4": 61,
        "f5": 62,
        "f6": 63,
        "f7": 64,
        "f8": 65,
        "f9": 66,
        "f10": 67,
        "f11": 68,
        "f12": 69,
    }

    def __init__(
        self,
        preferred_udid: str | None = None,
        idb_timeout_seconds: float = 15.0,
        idb_client: IDBClientProtocol | None = None,
    ) -> None:
        self.idb: IDBClientProtocol = idb_client or IDBClient(
            timeout_seconds=idb_timeout_seconds
        )
        self._preferred_udid = (preferred_udid or "").strip() or None
        self._bundle_id: str | None = None
        self._started = False
        self._last_screenshot_size: tuple[int, int] | None = None
        # Logical point dimensions (what idb commands expect); cached after first describe.
        self._logical_size: tuple[int, int] | None = None
        self._capturing = False
        self._captured_calls: list[dict[str, object]] = []

    async def start(self) -> None:
        if self._started:
            return
        udid = await self.idb.resolve_udid(self._preferred_udid)
        self.idb.udid = udid
        # Validate connectivity by describing the device
        await self.idb.run_idb("describe")
        self._started = True

    async def stop(self) -> None:
        self._started = False
        await self.idb.run_idb("kill", check=False)

    async def configure_target(
        self,
        udid: str | None = None,
        bundle_id: str | None = None,
    ) -> None:
        """Update target device/app details before entrypoint setup."""
        normalized_udid = (udid or "").strip() or None
        if normalized_udid:
            self._preferred_udid = normalized_udid
            self.idb.udid = normalized_udid
        normalized_bundle = (bundle_id or "").strip() or None
        if normalized_bundle:
            self._bundle_id = normalized_bundle

    async def launch_app(self, bundle_id: str | None = None) -> None:
        """Launch an iOS application by bundle ID."""
        await self._ensure_ready()
        bundle = (bundle_id or "").strip() or self._bundle_id
        if not bundle:
            raise RuntimeError("Cannot launch app: missing iOS bundle ID.")
        await self.idb.run_idb("launch", bundle)
        self._capture_call("launch_app", {"bundle_id": bundle})

    async def force_stop_app(self, bundle_id: str | None = None) -> None:
        """Terminate an iOS application."""
        await self._ensure_ready()
        bundle = (bundle_id or "").strip() or self._bundle_id
        if not bundle:
            raise RuntimeError("Cannot terminate app: missing iOS bundle ID.")
        await self.idb.run_idb("terminate", bundle)
        self._capture_call("force_stop_app", {"bundle_id": bundle})

    async def navigate(self, url: str) -> None:
        await self._ensure_ready()
        await self.idb.run_idb("open", url)
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
            raise ValueError(f"Unsupported button for iOS tap: {button!r}")
        tap_count = max(int(click_count), 1)
        mapped_x, mapped_y = await self._map_point_to_device(x, y)
        for _ in range(tap_count):
            await self.idb.run_idb("ui", "tap", str(mapped_x), str(mapped_y))
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
        duration_secs = duration_ms / 1000.0
        await self.idb.run_idb(
            "ui",
            "swipe",
            str(mapped_start_x),
            str(mapped_start_y),
            str(mapped_end_x),
            str(mapped_end_y),
            "--duration",
            f"{duration_secs:.3f}",
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
        if text:
            await self.idb.run_idb("ui", "text", text)
        self._capture_call("type_text", {"length": len(text)})

    # Special virtual keys that map to idb ui button rather than HID keycodes.
    _IOS_BUTTON_KEYS: frozenset[str] = frozenset({"home", "lock", "siri"})

    async def press_key(self, key: str) -> None:
        await self._ensure_ready()
        normalized = str(key or "").strip().lower()
        if normalized in self._IOS_BUTTON_KEYS:
            button_name = normalized.upper()
            await self.idb.run_idb("ui", "button", button_name)
            self._capture_call("press_key", {"key": key, "button": button_name})
            return
        tokens = [
            self._normalize_key_token(part)
            for part in str(key or "").split("+")
            if part.strip()
        ]
        if not tokens:
            raise ValueError("Key must not be empty.")
        keycodes = [self._keycode_for_token(token) for token in tokens]
        await self.idb.run_idb(
            "ui",
            "key-sequence",
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
        # Use logical point dimensions for swipe coordinates sent to idb.
        width, height = await self._fetch_logical_size()
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
        duration_secs = duration_ms / 1000.0
        await self.idb.run_idb(
            "ui",
            "swipe",
            str(start_x),
            str(start_y),
            str(end_x),
            str(end_y),
            "--duration",
            f"{duration_secs:.3f}",
        )
        self._capture_call("scroll_by_pixels", {"x": x, "y": y, "smooth": smooth})

    async def screenshot(self) -> bytes:
        await self._ensure_ready()
        screenshot_bytes = await self._screenshot_idb()
        if screenshot_bytes is None:
            screenshot_bytes = await self._screenshot_simctl()
        if not screenshot_bytes.startswith(_PNG_SIGNATURE):
            raise RuntimeError(
                f"iOS screenshot did not produce a valid PNG (got {len(screenshot_bytes)} bytes)"
            )
        size = self._parse_png_size(screenshot_bytes)
        if size is not None:
            self._last_screenshot_size = size
        self._capture_call("screenshot", {"label": "ios"})
        return screenshot_bytes

    async def _screenshot_idb(self) -> bytes | None:
        """Try idb screenshot; return None if it fails or produces no image."""
        tmp_path = tempfile.mktemp(suffix=".png")
        try:
            result = await self.idb.run_idb("screenshot", tmp_path, check=False)
            if result.returncode != 0:
                return None
            with open(tmp_path, "rb") as fh:
                data = fh.read()
            return data if data.startswith(_PNG_SIGNATURE) else None
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _screenshot_simctl(self) -> bytes:
        """Fall back to xcrun simctl io screenshot (works on newer iOS betas)."""
        udid = self.idb.udid
        if not udid:
            raise RuntimeError("No UDID set; cannot use simctl screenshot fallback.")
        tmp_path = tempfile.mktemp(suffix=".png")
        try:
            proc = await asyncio.create_subprocess_exec(
                "xcrun",
                "simctl",
                "io",
                udid,
                "screenshot",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"simctl screenshot failed: {stderr.decode('utf-8', errors='replace').strip()}"
                )
            with open(tmp_path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(max(int(milliseconds), 0) / 1000.0)
        self._capture_call("wait", {"milliseconds": milliseconds})

    async def get_viewport_size(self) -> tuple[int, int]:
        """Return physical screenshot dimensions for CU model coordinate alignment.

        The CU model normalises its coordinates against the viewport we report here.
        Returning physical (Retina) pixel dimensions ensures that, after de-
        normalisation, the AI's coordinates land in screenshot pixel space, which
        _map_point_to_device then correctly scales to logical points for idb.
        """
        if self._last_screenshot_size:
            return self._last_screenshot_size
        # Before the first screenshot, fall back to logical size; it will be
        # replaced once a screenshot is taken.
        return await self._fetch_logical_size()

    async def _fetch_logical_size(self) -> tuple[int, int]:
        """Return logical point dimensions from idb describe (for idb commands).

        Results are cached in _logical_size after the first call.
        """
        await self._ensure_ready()
        if self._logical_size:
            return self._logical_size

        result = await self.idb.run_idb("describe", check=False)
        text = result.stdout_text.strip()

        if text:
            # Try JSON first (future idb versions may output JSON)
            try:
                data = json.loads(text)
                screen = data.get("screen", {})
                width = int(screen.get("width_points") or screen.get("width") or 0)
                height = int(screen.get("height_points") or screen.get("height") or 0)
                if width > 0 and height > 0:
                    self._logical_size = (width, height)
                    return self._logical_size
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
            # Fallback: parse Python-repr output from current idb versions
            # e.g. "screen_dimensions=ScreenDimensions(..., width_points=1032, height_points=1376)"
            match = _SCREEN_DIMENSIONS_PATTERN.search(text)
            if match:
                width = int(match.group("width"))
                height = int(match.group("height"))
                if width > 0 and height > 0:
                    self._logical_size = (width, height)
                    return self._logical_size

        raise RuntimeError(
            f"Unable to parse logical size from idb describe output: {text!r}"
        )

    async def get_page_url(self) -> str:
        return ""

    async def get_page_title(self) -> str:
        return "iOS Device"

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
        # Target is always logical points (what idb commands expect).
        width, height = await self._fetch_logical_size()
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

    def _keycode_for_token(self, token: str) -> int:
        if token in self._IOS_KEYCODES:
            return self._IOS_KEYCODES[token]
        if len(token) == 1:
            # Letters: a=4 through z=29
            if "a" <= token <= "z":
                return 4 + (ord(token) - ord("a"))
            # Digits: 1=30 through 9=38, 0=39
            if "1" <= token <= "9":
                return 30 + (ord(token) - ord("1"))
            if token == "0":
                return 39
        raise ValueError(
            f"Unsupported iOS key {token!r}. Use known names like enter/escape/tab/space/"
            "backspace/up/down/left/right/cmd/shift/ctrl or single alphanumeric characters."
        )

    @staticmethod
    def _normalize_key_token(token: str) -> str:
        normalized = token.strip().lower().replace("-", "_")
        alias_map = {
            "arrowleft": "arrowleft",
            "arrowright": "arrowright",
            "arrowup": "arrowup",
            "arrowdown": "arrowdown",
        }
        return alias_map.get(normalized, normalized)

    def _capture_call(self, method: str, params: dict[str, object]) -> None:
        if not self._capturing:
            return
        self._captured_calls.append({"method": method, "params": params})

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))
