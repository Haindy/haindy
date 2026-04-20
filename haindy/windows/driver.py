"""Windows desktop driver stub.

Real implementation lands in Milestone 2 inside a Windows VM. This stub exists
so ``haindy.windows`` imports cleanly on Linux/macOS dev boxes and so the
platform dispatch in ``haindy.main`` and ``haindy.tool_call_mode.runtime``
compiles. All runtime methods raise ``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path

from haindy.core.coordinate_cache import CoordinateCache
from haindy.core.interfaces import AutomationDriver


class WindowsDriver(AutomationDriver):
    """Windows desktop automation driver (stub)."""

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
        self._clipboard_timeout_seconds = max(float(clipboard_timeout_seconds), 0.5)
        self._clipboard_hold_seconds = max(float(clipboard_hold_seconds), 0.5)
        self._pixel_width: int = 0
        self._pixel_height: int = 0

    async def start(self) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def stop(self) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def navigate(self, url: str) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def type_text(self, text: str) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def press_key(self, key: str) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def scroll(
        self,
        direction: str,
        amount: int,
        origin: tuple[int, int] | None = None,
    ) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def scroll_by_pixels(
        self,
        x: int = 0,
        y: int = 0,
        smooth: bool = True,
        origin: tuple[int, int] | None = None,
    ) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def screenshot(self) -> bytes:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def wait(self, milliseconds: int) -> None:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def get_viewport_size(self) -> tuple[int, int]:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def get_page_url(self) -> str:
        raise NotImplementedError("Windows driver lands in Milestone 2")

    async def get_page_title(self) -> str:
        raise NotImplementedError("Windows driver lands in Milestone 2")
