"""
High-level browser controller combining browser and grid functionality.
"""

from pathlib import Path
from typing import Optional

from src.browser.driver import PlaywrightDriver
from src.browser.instrumented_driver import InstrumentedBrowserDriver
from src.config.settings import get_settings
from src.core.types import GridCoordinate
from src.grid.overlay import GridOverlay
from src.grid.refinement import GridRefinement


class BrowserController:
    """High-level controller for browser automation with grid support."""

    def __init__(
        self,
        headless: Optional[bool] = None,
        grid_size: Optional[int] = None,
    ) -> None:
        """
        Initialize browser controller.

        Args:
            headless: Run browser in headless mode
            grid_size: Grid size (defaults to config)
        """
        self.settings = get_settings()
        # Lazy import to avoid circular dependency
        from src.monitoring.logger import get_logger
        self.logger = get_logger("browser.controller")

        # Initialize components
        # Use InstrumentedBrowserDriver for action tracking
        self.driver = InstrumentedBrowserDriver(headless=headless)
        self.grid = GridOverlay(grid_size=grid_size)
        self.refinement = GridRefinement(self.grid)

        self._initialized = False

    async def start(self) -> None:
        """Start the browser and initialize grid."""
        await self.driver.start()

        # Get viewport size and initialize grid
        width, height = await self.driver.get_viewport_size()
        self.grid.initialize(width, height)

        self._initialized = True
        self.logger.info("Browser controller started")

    async def stop(self) -> None:
        """Stop the browser."""
        await self.driver.stop()
        self._initialized = False
        self.logger.info("Browser controller stopped")

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self._initialized:
            await self.start()

        await self.driver.navigate(url)

    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")
        
        await self.driver.press_key(key)

    async def click_at_grid(
        self,
        coord: GridCoordinate,
        refine: bool = True,
    ) -> GridCoordinate:
        """
        Click at a grid coordinate.

        Args:
            coord: Grid coordinate
            refine: Whether to apply refinement if confidence is low

        Returns:
            Final coordinate used (may be refined)
        """
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")

        # Check if refinement is needed
        if refine and self.refinement.should_refine(coord):
            screenshot = await self.driver.screenshot()
            coord = self.refinement.refine_coordinate(
                screenshot,
                coord,
                target_description="click target",
            )

        # Convert to pixels and click
        x, y = self.grid.coordinate_to_pixels(coord)
        await self.driver.click(x, y)

        self.logger.info(
            f"Clicked at grid coordinate",
            extra={
                "cell": coord.cell,
                "offset": f"({coord.offset_x:.2f}, {coord.offset_y:.2f})",
                "pixel": f"({x}, {y})",
                "refined": coord.refined,
            },
        )

        return coord

    async def screenshot_with_grid(
        self,
        save_path: Optional[Path] = None,
        show_overlay: Optional[bool] = None,
    ) -> bytes:
        """
        Take a screenshot, optionally with grid overlay.

        Args:
            save_path: Optional path to save the screenshot
            show_overlay: Show grid overlay (defaults to config)

        Returns:
            Screenshot bytes
        """
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")

        # Take screenshot
        screenshot = await self.driver.screenshot()

        # Add grid overlay if requested
        if show_overlay is None:
            show_overlay = self.settings.enable_grid_overlay

        if show_overlay:
            screenshot = self.grid.create_overlay_image(screenshot)

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(screenshot)
            self.logger.info(f"Screenshot saved", extra={"path": str(save_path)})

        return screenshot

    async def screenshot_refinement_region(
        self,
        cell: str,
        save_path: Optional[Path] = None,
    ) -> bytes:
        """
        Take a screenshot of a refinement region with fine grid.

        Args:
            cell: Center cell for refinement region
            save_path: Optional path to save the screenshot

        Returns:
            Cropped screenshot with fine grid overlay
        """
        if not self._initialized:
            raise RuntimeError("Controller not started. Call start() first.")

        # Take full screenshot
        screenshot = await self.driver.screenshot()

        # Crop to refinement region
        cropped, bounds = self.refinement.crop_refinement_region(screenshot, cell)

        # Add fine grid overlay
        with_overlay = self.refinement.create_refinement_overlay(cropped)

        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(with_overlay)
            self.logger.info(
                f"Refinement region saved",
                extra={"path": str(save_path), "cell": cell},
            )

        return with_overlay

    async def get_current_state(self) -> dict:
        """
        Get current browser and grid state.

        Returns:
            Dictionary with state information
        """
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

    async def __aenter__(self) -> "BrowserController":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()