"""High-level desktop controller with plain coordinate interactions."""

from __future__ import annotations

from pathlib import Path

from haindy.config.settings import get_settings
from haindy.linux.driver import DesktopDriver
from haindy.monitoring.logger import get_logger


class DesktopController:
    """High-level controller for desktop automation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger("desktop.controller")
        self.driver = DesktopDriver(
            screenshot_dir=self.settings.desktop_screenshot_dir,
            cache_path=self.settings.linux_coordinate_cache_path,
            prefer_resolution=self.settings.desktop_prefer_resolution,
            enable_resolution_switch=self.settings.desktop_enable_resolution_switch,
            display=self.settings.desktop_display,
            keyboard_layout=self.settings.desktop_keyboard_layout,
            keyboard_emit_scancodes=self.settings.desktop_enable_keyboard_scancodes,
            keyboard_key_delay_ms=self.settings.desktop_keyboard_key_delay_ms,
            clipboard_timeout_seconds=self.settings.desktop_clipboard_timeout_seconds,
            clipboard_hold_seconds=self.settings.desktop_clipboard_hold_seconds,
            max_screenshots=self.settings.max_screenshots,
        )
        self._initialized = False

    async def start(self) -> None:
        await self.driver.start()
        self._initialized = True
        self.logger.info("Desktop controller started")

    async def stop(self) -> None:
        await self.driver.stop()
        self._initialized = False
        self.logger.info("Desktop controller stopped")

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

    async def __aenter__(self) -> DesktopController:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()
