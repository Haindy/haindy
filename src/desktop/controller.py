"""High-level desktop controller with grid support."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import get_settings
from src.core.types import GridCoordinate
from src.desktop.driver import DesktopDriver
from src.grid.overlay import GridOverlay
from src.grid.refinement import GridRefinement
from src.monitoring.logger import get_logger


class DesktopController:
    """High-level controller for desktop automation with grid support."""

    def __init__(self, grid_size: int | None = None) -> None:
        self.settings = get_settings()
        self.logger = get_logger("desktop.controller")
        self.driver = DesktopDriver(
            screenshot_dir=self.settings.desktop_screenshot_dir,
            cache_path=self.settings.desktop_coordinate_cache_path,
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
        self.grid = GridOverlay(grid_size=grid_size)
        self.refinement = GridRefinement(self.grid)
        self._initialized = False

    async def start(self) -> None:
        await self.driver.start()
        width, height = await self.driver.get_viewport_size()
        self.grid.initialize(width, height)
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

    async def click_at_grid(
        self,
        coord: GridCoordinate,
        refine: bool = True,
    ) -> GridCoordinate:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")

        if refine and self.refinement.should_refine(coord):
            screenshot = await self.driver.screenshot()
            coord = self.refinement.refine_coordinate(
                screenshot,
                coord,
                target_description="click target",
            )

        x, y = self.grid.coordinate_to_pixels(coord)
        await self.driver.click(x, y)
        return coord

    async def screenshot_with_grid(
        self,
        save_path: Path | None = None,
        show_overlay: bool | None = None,
    ) -> bytes:
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")

        screenshot = await self.driver.screenshot()
        if show_overlay is None:
            show_overlay = self.settings.enable_grid_overlay
        if show_overlay:
            screenshot = self.grid.create_overlay_image(screenshot)

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
            "grid": {
                "size": self.grid.grid_size,
                "cell_size": {
                    "width": self.grid.cell_width,
                    "height": self.grid.cell_height,
                },
            },
        }

    async def __aenter__(self) -> DesktopController:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
