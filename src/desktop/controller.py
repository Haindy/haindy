"""High-level controller for desktop automation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from src.desktop.driver import DesktopDriver


class DesktopController:
    """Wrapper that mirrors BrowserController shape for desktop mode."""

    def __init__(
        self,
        screenshot_dir: Path,
        cache_path: Path,
        prefer_resolution: Tuple[int, int] = (1920, 1080),
        enable_resolution_switch: bool = False,
        display: Optional[str] = None,
    ) -> None:
        self.driver = DesktopDriver(
            screenshot_dir=screenshot_dir,
            cache_path=cache_path,
            prefer_resolution=prefer_resolution,
            enable_resolution_switch=enable_resolution_switch,
            display=display,
        )
        self._started = False

    async def start(self) -> None:
        await self.driver.start()
        self._started = True

    async def stop(self) -> None:
        await self.driver.stop()
        self._started = False

    async def __aenter__(self) -> "DesktopController":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
