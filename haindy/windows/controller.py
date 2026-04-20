"""High-level Windows controller stub.

Real implementation lands in Milestone 2 inside a Windows VM.
"""

from __future__ import annotations

from haindy.config.settings import get_settings
from haindy.monitoring.logger import get_logger
from haindy.windows.driver import WindowsDriver


class WindowsController:
    """High-level controller for Windows desktop automation (stub)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("windows.controller")
        self.driver = WindowsDriver(
            screenshot_dir=self.settings.desktop_screenshot_dir,
            cache_path=self.settings.windows_coordinate_cache_path,
            max_screenshots=self.settings.max_screenshots,
        )
        self._initialized = False

    async def start(self) -> None:
        raise NotImplementedError("Windows controller lands in Milestone 2")

    async def stop(self) -> None:
        raise NotImplementedError("Windows controller lands in Milestone 2")
