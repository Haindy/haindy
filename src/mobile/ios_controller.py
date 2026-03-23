"""High-level iOS controller with coordinate interactions."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from src.config.settings import get_settings
from src.mobile.ios_driver import IOSDriver
from src.monitoring.logger import get_logger


class IOSController:
    """High-level controller for iOS automation."""

    def __init__(self, preferred_udid: str | None = None) -> None:
        self.settings = get_settings()
        self.logger = get_logger("ios.controller")
        resolved_udid = (preferred_udid or "").strip() or (
            self.settings.ios_default_device_udid.strip() or None
        )
        self.driver = IOSDriver(
            preferred_udid=resolved_udid,
            idb_timeout_seconds=self.settings.ios_idb_timeout_seconds,
        )
        self._initialized = False

    async def start(self) -> None:
        await self.driver.start()
        self._initialized = True
        self.logger.info("iOS controller started")

    async def stop(self) -> None:
        await self.driver.stop()
        self._initialized = False
        self.logger.info("iOS controller stopped")

    async def press_key(self, key: str) -> None:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")
        await self.driver.press_key(key)

    async def click_at_coordinates(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
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

    async def get_current_state(self) -> dict[str, object]:
        if not self._initialized:
            return {"initialized": False}
        width, height = await self.driver.get_viewport_size()
        return {
            "initialized": True,
            "url": await self.driver.get_page_url(),
            "title": await self.driver.get_page_title(),
            "viewport": {"width": width, "height": height},
        }

    async def __aenter__(self) -> IOSController:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()
