"""macOS desktop driver implementing the AutomationDriver interface."""

from __future__ import annotations

import asyncio
import logging
import struct
from asyncio import subprocess as aio_subprocess
from pathlib import Path

from haindy.core.interfaces import AutomationDriver
from haindy.desktop.cache import CoordinateCache
from haindy.macos.input_handler import MacOSInputHandler
from haindy.macos.screen_capture import MacOSScreenCapture

logger = logging.getLogger(__name__)


def _parse_png_size(data: bytes) -> tuple[int, int]:
    """Parse width and height from a PNG header.

    PNG layout: 8-byte signature + IHDR chunk (4-byte len + 4-byte type +
    4-byte width + 4-byte height + ...).
    """
    if len(data) < 24:
        raise ValueError("Data too short to be a valid PNG")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    return int(width), int(height)


class MacOSDriver(AutomationDriver):
    """macOS desktop automation driver using mss (screenshots) and pynput (input)."""

    def __init__(
        self,
        screenshot_dir: Path,
        cache_path: Path,
        keyboard_layout: str = "us",
        keyboard_key_delay_ms: int = 12,
        clipboard_timeout_seconds: float = 3.0,
        clipboard_hold_seconds: float = 15.0,
        max_screenshots: int | None = None,
    ) -> None:
        self.screenshot_dir = screenshot_dir
        self.max_screenshots = max_screenshots
        self.keyboard_layout = keyboard_layout
        self.keyboard_key_delay_ms = keyboard_key_delay_ms
        self.coordinate_cache = CoordinateCache(cache_path)
        self.screen_capture: MacOSScreenCapture | None = None
        self.input_handler: MacOSInputHandler | None = None
        self._started = False
        self._clipboard_timeout_seconds = max(float(clipboard_timeout_seconds), 0.5)
        self._clipboard_hold_seconds = max(float(clipboard_hold_seconds), 0.5)
        self._capturing = False
        self._captured_calls: list[dict[str, object]] = []
        # Pixel dimensions of the primary display (set during start)
        self._pixel_width: int = 0
        self._pixel_height: int = 0

    async def start(self) -> None:
        """Detect display geometry and initialise the input handler."""
        if self._started and self.input_handler is not None:
            return

        self.screen_capture = MacOSScreenCapture(
            screenshot_dir=self.screenshot_dir,
            max_screenshots=self.max_screenshots,
        )

        logical_w, logical_h = self.screen_capture.get_logical_size()
        init_bytes, _ = self.screen_capture.capture("init")
        pixel_w, pixel_h = _parse_png_size(init_bytes)

        self._pixel_width = pixel_w
        self._pixel_height = pixel_h

        scale_x = pixel_w / logical_w if logical_w > 0 else 1.0
        scale_y = pixel_h / logical_h if logical_h > 0 else 1.0

        try:
            self.input_handler = MacOSInputHandler(
                logical_size=(logical_w, logical_h),
                scale_x=scale_x,
                scale_y=scale_y,
                keyboard_layout=self.keyboard_layout,
                key_delay_ms=self.keyboard_key_delay_ms,
            )
        except Exception:
            self.screen_capture = None
            self.input_handler = None
            self._started = False
            raise

        self._started = True
        logger.info(
            "MacOSDriver started",
            extra={
                "logical": f"{logical_w}x{logical_h}",
                "pixels": f"{pixel_w}x{pixel_h}",
                "scale": f"{scale_x:.2f}x{scale_y:.2f}",
            },
        )

    async def stop(self) -> None:
        """Tear down the driver."""
        if not self._started and self.input_handler is None:
            return
        self.input_handler = None
        self.screen_capture = None
        self._started = False

    async def __aenter__(self) -> MacOSDriver:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()

    async def navigate(self, url: str) -> None:
        """Best-effort navigation by opening Spotlight/URL bar with Cmd+L."""
        await self._ensure_ready()
        await self.press_key("cmd+l")
        await self.type_text(url)
        await self.press_key("enter")

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        if modifiers:
            await handler.click(
                x, y, button=button, click_count=click_count, modifiers=modifiers
            )
        else:
            await handler.click(x, y, button=button, click_count=click_count)
        call_info: dict[str, object] = {
            "x": x,
            "y": y,
            "button": button,
            "click_count": click_count,
        }
        if modifiers:
            call_info["modifiers"] = modifiers
        self._capture_call("click", call_info)

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        await handler.move(x, y)
        self._capture_call("move_mouse", {"x": x, "y": y, "steps": steps})

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        await handler.drag((start_x, start_y), (end_x, end_y), steps=steps)
        self._capture_call(
            "drag_mouse",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "steps": steps,
            },
        )

    async def type_text(self, text: str) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        await handler.type_text(text)
        self._capture_call("type_text", {"length": len(text)})

    async def press_key(self, key: str) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        await handler.press_key(key)
        self._capture_call("press_key", {"key": key})

    async def scroll(
        self,
        direction: str,
        amount: int,
        origin: tuple[int, int] | None = None,
    ) -> None:
        normalized = str(direction or "").strip().lower()
        if normalized not in {"up", "down", "left", "right"}:
            raise ValueError(f"Invalid scroll direction: {direction!r}")
        magnitude = abs(int(amount))
        delta = magnitude if normalized in {"down", "right"} else -magnitude
        await self.scroll_by_pixels(
            y=delta if normalized in {"up", "down"} else 0,
            x=delta if normalized in {"left", "right"} else 0,
        )

    async def scroll_by_pixels(
        self,
        x: int = 0,
        y: int = 0,
        smooth: bool = True,
        origin: tuple[int, int] | None = None,
    ) -> None:
        await self._ensure_ready()
        handler = self.input_handler
        assert handler is not None
        await handler.scroll(x=x, y=y)
        self._capture_call("scroll_by_pixels", {"x": x, "y": y, "smooth": smooth})

    async def screenshot(self) -> bytes:
        bytes_, _ = await self._capture("macos")
        self._capture_call("screenshot", {"label": "macos"})
        return bytes_

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(milliseconds / 1000.0)
        self._capture_call("wait", {"milliseconds": milliseconds})

    async def get_viewport_size(self) -> tuple[int, int]:
        """Return the native pixel dimensions of the primary display."""
        if self._pixel_width > 0 and self._pixel_height > 0:
            return self._pixel_width, self._pixel_height
        # Fallback: take a screenshot to detect size
        if self.screen_capture is None:
            self.screen_capture = MacOSScreenCapture(
                screenshot_dir=self.screenshot_dir,
                max_screenshots=self.max_screenshots,
            )
        init_bytes, _ = self.screen_capture.capture("viewport_probe")
        w, h = _parse_png_size(init_bytes)
        self._pixel_width, self._pixel_height = w, h
        return w, h

    async def get_page_url(self) -> str:
        return ""

    async def get_page_title(self) -> str:
        return "Desktop Session"

    async def save_screenshot(self, path: Path) -> None:
        bytes_, _ = await self._capture("save")
        path.write_bytes(bytes_)
        self._capture_call("save_screenshot", {"path": str(path)})

    async def wait_for_load_state(self, state: str = "networkidle") -> None:
        return

    async def capture(self, label: str) -> tuple[bytes, str]:
        """Capture a screenshot with the given label."""
        bytes_, saved_path = await self._capture(label)
        self._capture_call("capture", {"label": label, "path": saved_path})
        return bytes_, saved_path

    def start_capture(self) -> None:
        """Begin collecting automation call traces."""
        self._capturing = True
        self._captured_calls = []

    def stop_capture(self) -> list[dict[str, object]]:
        """Stop trace collection and return captured calls."""
        calls = self._captured_calls.copy()
        self._capturing = False
        self._captured_calls = []
        return calls

    async def read_clipboard(self) -> str:
        """Return the current clipboard contents via pbpaste."""
        proc = await asyncio.create_subprocess_exec(
            "pbpaste",
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._clipboard_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Clipboard read timed out.") from exc
        if proc.returncode != 0:
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"Clipboard read failed: {stderr_text.strip()}")
        return (stdout or b"").decode("utf-8", errors="ignore").strip()

    async def write_clipboard(self, text: str) -> None:
        """Set clipboard contents via pbcopy."""
        proc = await asyncio.create_subprocess_exec(
            "pbcopy",
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.DEVNULL,
            stderr=aio_subprocess.PIPE,
        )
        if not proc.stdin:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Clipboard write failed: unable to open stdin.")
        proc.stdin.write(text.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._clipboard_timeout_seconds)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Clipboard write timed out.") from exc
        if proc.returncode != 0:
            _, stderr = await proc.communicate()
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"Clipboard write failed: {stderr_text.strip()}")

    async def _capture(self, label: str) -> tuple[bytes, str]:
        await self._ensure_ready()
        assert self.screen_capture is not None
        return self.screen_capture.capture(label)

    def _capture_call(self, method: str, params: dict[str, object]) -> None:
        if not self._capturing:
            return
        self._captured_calls.append({"method": method, "params": params})

    async def _ensure_ready(self) -> None:
        if not self._started or self.input_handler is None:
            await self.start()
