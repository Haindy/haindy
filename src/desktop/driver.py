"""Desktop driver implementing the BrowserDriver interface using OS-level input."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.core.interfaces import BrowserDriver
from src.desktop.cache import CoordinateCache
from src.desktop.resolution_manager import ResolutionManager
from src.desktop.screen_capture import ScreenCapture
from src.desktop.virtual_input import VirtualInput

logger = logging.getLogger(__name__)


class DesktopDriver(BrowserDriver):
    """OS-level driver that controls an existing desktop session."""

    def __init__(
        self,
        screenshot_dir: Path,
        cache_path: Path,
        prefer_resolution: Tuple[int, int] = (1920, 1080),
        enable_resolution_switch: bool = False,
        display: Optional[str] = None,
    ) -> None:
        self.resolution_manager = ResolutionManager(
            preferred_width=prefer_resolution[0],
            preferred_height=prefer_resolution[1],
            enable_switch=enable_resolution_switch,
        )
        self.screen_capture = ScreenCapture(
            resolution_manager=self.resolution_manager,
            screenshot_dir=screenshot_dir,
            display=display,
        )
        self.coordinate_cache = CoordinateCache(cache_path)
        self.virtual_input: Optional[VirtualInput] = None
        self._started = False

    async def start(self) -> None:
        """Initialize resolution and virtual input device."""
        self.resolution_manager.maybe_downshift()
        viewport = self.resolution_manager.viewport_size()
        self.virtual_input = VirtualInput(viewport=viewport)
        self._started = True

    async def stop(self) -> None:
        """Restore resolution if changed."""
        self.resolution_manager.restore()
        self._started = False

    async def navigate(self, url: str) -> None:
        raise RuntimeError("Navigation is not supported in desktop mode.")

    async def click(self, x: int, y: int, button: str = "left", click_count: int = 1) -> None:
        await self._ensure_ready()
        await self.virtual_input.click(x, y, button=button, click_count=click_count)

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        await self._ensure_ready()
        await self.virtual_input.move(x, y, steps=steps)

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        await self._ensure_ready()
        await self.virtual_input.drag((start_x, start_y), (end_x, end_y), steps=steps)

    async def type_text(self, text: str) -> None:
        await self._ensure_ready()
        await self.virtual_input.type_text(text)

    async def press_key(self, key: str) -> None:
        await self._ensure_ready()
        await self.virtual_input.press_key(key)

    async def scroll(self, direction: str, amount: int) -> None:
        delta = amount if direction in {"down", "right"} else -amount
        await self.scroll_by_pixels(y=delta if direction in {"up", "down"} else 0, x=delta if direction in {"left", "right"} else 0)

    async def scroll_by_pixels(self, x: int = 0, y: int = 0, smooth: bool = True) -> None:
        await self._ensure_ready()
        await self.virtual_input.scroll(x=x, y=y)

    async def screenshot(self) -> bytes:
        bytes_, _ = await self._capture("desktop")
        return bytes_

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(milliseconds / 1000.0)

    async def get_viewport_size(self) -> Tuple[int, int]:
        return self.resolution_manager.viewport_size()

    async def get_page_url(self) -> str:
        return ""

    async def get_page_title(self) -> str:
        return "Desktop Session"

    async def save_screenshot(self, path: Path) -> None:
        bytes_, _ = await self._capture("save")
        path.write_bytes(bytes_)

    async def wait_for_load_state(self, state: str = "networkidle") -> None:
        return

    async def _capture(self, label: str) -> Tuple[bytes, str]:
        await self._ensure_ready()
        return self.screen_capture.capture(label)

    async def _ensure_ready(self) -> None:
        if not self._started:
            await self.start()
        if not self.virtual_input:
            viewport = self.resolution_manager.viewport_size()
            self.virtual_input = VirtualInput(viewport=viewport)
