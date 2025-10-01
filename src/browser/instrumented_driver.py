"""
Instrumented browser driver that captures all browser interactions.

This wrapper captures method calls to the browser driver for action storage.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from src.browser.driver import PlaywrightDriver
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class InstrumentedBrowserDriver(PlaywrightDriver):
    """
    Browser driver wrapper that captures all method calls for action storage.
    
    This wrapper intercepts and logs all browser automation calls while
    delegating the actual execution to the underlying PlaywrightDriver.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the instrumented driver."""
        super().__init__(*args, **kwargs)
        self.captured_calls: List[Dict[str, Any]] = []
        self._capturing = False
    
    def start_capture(self) -> None:
        """Start capturing browser calls."""
        self._capturing = True
        self.captured_calls = []
        logger.debug("Started capturing browser calls")
    
    def stop_capture(self) -> List[Dict[str, Any]]:
        """Stop capturing and return captured calls."""
        self._capturing = False
        calls = self.captured_calls.copy()
        self.captured_calls = []
        logger.debug(f"Stopped capturing browser calls, captured {len(calls)} calls")
        return calls
    
    def _capture_call(self, method_name: str, parameters: Dict[str, Any], duration_ms: float) -> None:
        """Capture a browser method call."""
        if self._capturing:
            call_data = {
                "method": method_name,
                "parameters": parameters,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms
            }
            self.captured_calls.append(call_data)
            logger.debug(f"Captured browser call: {method_name}", extra=call_data)
    
    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        start_time = time.time()
        await super().navigate(url)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.goto", {"url": url}, duration_ms)
    
    async def click(self, x: int, y: int) -> None:
        """Click at absolute coordinates."""
        start_time = time.time()
        await super().click(x, y)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.mouse.click", {"x": x, "y": y}, duration_ms)
    
    async def type_text(self, text: str) -> None:
        """Type text at current focus."""
        start_time = time.time()
        await super().type_text(text)
        duration_ms = (time.time() - start_time) * 1000
        # Don't store the actual text for security reasons, just length
        self._capture_call("page.keyboard.type", {"text_length": len(text)}, duration_ms)
    
    async def press_key(self, key: str) -> None:
        """Press a keyboard key."""
        start_time = time.time()
        await super().press_key(key)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.keyboard.press", {"key": key}, duration_ms)
    
    async def scroll(self, direction: str, amount: int) -> None:
        """Scroll in given direction."""
        start_time = time.time()
        await super().scroll(direction, amount)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.mouse.wheel", {"direction": direction, "amount": amount}, duration_ms)
    
    async def scroll_by_pixels(self, x: int = 0, y: int = 0, smooth: bool = True) -> None:
        """Scroll by a specific number of pixels."""
        start_time = time.time()
        await super().scroll_by_pixels(x, y, smooth)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("window.scrollBy", {"x": x, "y": y, "smooth": smooth}, duration_ms)
    
    async def scroll_to_top(self) -> None:
        """Scroll to the top of the page."""
        start_time = time.time()
        await super().scroll_to_top()
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("window.scrollTo", {"position": "top"}, duration_ms)
    
    async def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the page."""
        start_time = time.time()
        await super().scroll_to_bottom()
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("window.scrollTo", {"position": "bottom"}, duration_ms)
    
    async def wait(self, milliseconds: int) -> None:
        """Wait for specified duration."""
        start_time = time.time()
        await super().wait(milliseconds)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.wait_for_timeout", {"milliseconds": milliseconds}, duration_ms)
    
    async def screenshot(self) -> bytes:
        """Take a screenshot and return as bytes."""
        start_time = time.time()
        result = await super().screenshot()
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.screenshot", {"type": "png", "full_page": False}, duration_ms)
        return result
    
    async def save_screenshot(self, path: Path) -> None:
        """Save a screenshot to file."""
        start_time = time.time()
        await super().save_screenshot(path)
        duration_ms = (time.time() - start_time) * 1000
        self._capture_call("page.screenshot", {"path": str(path), "type": "png"}, duration_ms)