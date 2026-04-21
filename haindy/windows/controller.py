"""High-level Windows controller with plain coordinate interactions."""

from __future__ import annotations

from pathlib import Path

from haindy.config.settings import get_settings
from haindy.monitoring.logger import get_logger
from haindy.windows.driver import WindowsDriver


class WindowsController:
    """High-level controller for Windows desktop automation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("windows.controller")
        self.driver = WindowsDriver(
            screenshot_dir=self.settings.windows_screenshot_dir,
            cache_path=self.settings.windows_coordinate_cache_path,
            keyboard_layout=self.settings.windows_keyboard_layout,
            keyboard_key_delay_ms=self.settings.windows_keyboard_key_delay_ms,
            clipboard_timeout_seconds=self.settings.windows_clipboard_timeout_seconds,
            clipboard_hold_seconds=self.settings.windows_clipboard_hold_seconds,
            max_screenshots=self.settings.max_screenshots,
        )
        self._initialized = False

    async def start(self) -> None:
        await self.driver.start()
        self._initialized = True
        self.logger.info("Windows controller started")

    async def stop(self) -> None:
        await self.driver.stop()
        self._initialized = False
        self.logger.info("Windows controller stopped")

    async def press_key(self, key: str) -> None:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")
        await self.driver.press_key(key)

    async def click_at_coordinates(
        self, x: int, y: int, button: str = "left", click_count: int = 1
    ) -> tuple[int, int]:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")
        await self.driver.click(x, y, button=button, click_count=click_count)
        return x, y

    async def capture_screenshot(self, save_path: Path | None = None) -> bytes:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")
        screenshot = await self.driver.screenshot()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(screenshot)
        return screenshot

    async def get_current_state(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        width, height = await self.driver.get_viewport_size()
        return {
            "initialized": True,
            "url": await self.driver.get_page_url(),
            "title": await self.driver.get_page_title(),
            "viewport": {"width": width, "height": height},
        }

    async def __aenter__(self) -> WindowsController:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()
