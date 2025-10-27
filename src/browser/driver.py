"""
Playwright browser driver implementation.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from src.config.settings import get_settings
from src.core.interfaces import BrowserDriver
from src.monitoring.logger import get_logger, log_performance_metric


class PlaywrightDriver(BrowserDriver):
    """Playwright-based browser automation driver."""

    def __init__(
        self,
        headless: Optional[bool] = None,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize the Playwright driver.

        Args:
            headless: Run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Default timeout in milliseconds
        """
        settings = get_settings()
        self.headless = headless if headless is not None else settings.browser_headless
        self.viewport_width = viewport_width or settings.browser_viewport_width
        self.viewport_height = viewport_height or settings.browser_viewport_height
        self.timeout = timeout or settings.browser_timeout

        self.logger = get_logger("browser.driver")
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self) -> None:
        """Start the browser and create a page."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()

        if self._browser is None:
            self.logger.info(
                f"Starting browser",
                extra={
                    "headless": self.headless,
                    "viewport": f"{self.viewport_width}x{self.viewport_height}",
                },
            )
            launch_args = [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-file-system",
            ]

            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=launch_args,
                env=os.environ,
                chromium_sandbox=True,
            )

        if self._context is None:
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.viewport_width,
                    "height": self.viewport_height,
                },
                screen={
                    "width": self.viewport_width,
                    "height": self.viewport_height,
                },
            )
            self._context.set_default_timeout(self.timeout)

        if self._page is None:
            self._page = await self._context.new_page()

    async def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        if self._page:
            await self._page.close()
            self._page = None

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self.logger.info("Browser stopped")

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self._page:
            await self.start()

        self.logger.info(f"Navigating to URL", extra={"url": url})
        start_time = asyncio.get_event_loop().time()

        await self._page.goto(url, wait_until="networkidle")

        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        log_performance_metric("page_navigation", elapsed_ms, context={"url": url})

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
    ) -> None:
        """Click at absolute coordinates."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        if button not in {"left", "right", "middle"}:
            button = "left"

        self.logger.debug(
            "Clicking at coordinates",
            extra={"x": x, "y": y, "button": button, "click_count": click_count},
        )
        await self._page.mouse.click(x, y, button=button, click_count=click_count)

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        """Move mouse pointer to absolute coordinates without clicking."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            normalized_steps = max(1, int(steps))
        except (TypeError, ValueError):
            normalized_steps = 1
        self.logger.debug(
            "Moving mouse",
            extra={"x": x, "y": y, "steps": normalized_steps},
        )
        await self._page.mouse.move(x, y, steps=normalized_steps)

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        """Drag mouse pointer from start coordinates to end coordinates."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            normalized_steps = max(1, int(steps))
        except (TypeError, ValueError):
            normalized_steps = 1

        self.logger.debug(
            "Dragging mouse",
            extra={
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "steps": normalized_steps,
            },
        )

        await self._page.mouse.move(start_x, start_y, steps=1)
        await self._page.mouse.down(button="left")
        await self._page.mouse.move(end_x, end_y, steps=normalized_steps)
        await self._page.mouse.up(button="left")

    async def type_text(self, text: str) -> None:
        """Type text at current focus."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        self.logger.debug(f"Typing text", extra={"length": len(text)})
        await self._page.keyboard.type(text)
    
    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        self.logger.debug(f"Pressing key", extra={"key": key})
        await self._page.keyboard.press(key)

    async def scroll(self, direction: str, amount: int) -> None:
        """Scroll in given direction."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        if direction not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid scroll direction: {direction}")

        self.logger.debug(
            f"Scrolling", extra={"direction": direction, "amount": amount}
        )

        # Calculate scroll delta based on direction
        delta_x = 0
        delta_y = 0
        if direction == "up":
            delta_y = -amount
        elif direction == "down":
            delta_y = amount
        elif direction == "left":
            delta_x = -amount
        elif direction == "right":
            delta_x = amount

        # Get current viewport center
        viewport_width = self._page.viewport_size["width"]
        viewport_height = self._page.viewport_size["height"]
        center_x = viewport_width // 2
        center_y = viewport_height // 2

        # Perform scroll
        await self._page.mouse.wheel(delta_x, delta_y)
    
    async def scroll_by_pixels(self, x: int = 0, y: int = 0, smooth: bool = True) -> None:
        """
        Scroll by a specific number of pixels.
        
        Args:
            x: Horizontal scroll amount (positive = right, negative = left)
            y: Vertical scroll amount (positive = down, negative = up)
            smooth: Use smooth scrolling animation
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        self.logger.debug(f"Scrolling by pixels", extra={"x": x, "y": y, "smooth": smooth})
        
        behavior = "smooth" if smooth else "auto"
        await self._page.evaluate(
            f"""
            window.scrollBy({{
                left: {x},
                top: {y},
                behavior: '{behavior}'
            }});
            """
        )
        
    async def scroll_to_top(self) -> None:
        """Scroll to the top of the page."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        self.logger.debug("Scrolling to top of page")
        await self._page.evaluate("window.scrollTo(0, 0)")
        
    async def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the page."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        self.logger.debug("Scrolling to bottom of page")
        await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    
    async def get_scroll_position(self) -> Tuple[int, int]:
        """
        Get current scroll position.
        
        Returns:
            Tuple of (x, y) scroll position
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        position = await self._page.evaluate(
            "({x: window.pageXOffset || document.documentElement.scrollLeft, y: window.pageYOffset || document.documentElement.scrollTop})"
        )
        return position["x"], position["y"]
    
    async def get_page_dimensions(self) -> Tuple[int, int, int, int]:
        """
        Get page dimensions including scrollable area.
        
        Returns:
            Tuple of (viewport_width, viewport_height, page_width, page_height)
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        dimensions = await self._page.evaluate(
            """({
                viewportWidth: window.innerWidth,
                viewportHeight: window.innerHeight,
                pageWidth: document.documentElement.scrollWidth,
                pageHeight: document.documentElement.scrollHeight
            })"""
        )
        return (
            dimensions["viewportWidth"],
            dimensions["viewportHeight"],
            dimensions["pageWidth"],
            dimensions["pageHeight"]
        )

    async def screenshot(self) -> bytes:
        """Take a screenshot and return as bytes."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        self.logger.debug("Taking screenshot")
        screenshot_bytes = await self._page.screenshot(type="png", full_page=False)
        return screenshot_bytes

    async def wait(self, milliseconds: int) -> None:
        """Wait for specified duration."""
        self.logger.debug(f"Waiting", extra={"milliseconds": milliseconds})
        await self._page.wait_for_timeout(milliseconds)

    async def get_viewport_size(self) -> Tuple[int, int]:
        """Get current viewport dimensions."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        viewport = self._page.viewport_size
        return viewport["width"], viewport["height"]

    async def save_screenshot(self, path: Path) -> None:
        """
        Save a screenshot to file.

        Args:
            path: Path to save the screenshot
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        self.logger.info(f"Saving screenshot", extra={"path": str(path)})
        await self._page.screenshot(path=str(path), type="png", full_page=False)

    async def get_page_title(self) -> str:
        """Get the current page title."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        return await self._page.title()

    async def get_page_url(self) -> str:
        """Get the current page URL."""
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        return self._page.url
    
    @property
    def page(self) -> Optional[Page]:
        """Get the current page object (for advanced operations)."""
        return self._page

    async def wait_for_load_state(self, state: str = "networkidle") -> None:
        """
        Wait for a specific load state.

        Args:
            state: Load state to wait for (load, domcontentloaded, networkidle)
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")

        await self._page.wait_for_load_state(state)

    async def __aenter__(self) -> "PlaywrightDriver":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
