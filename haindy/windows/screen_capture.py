"""Windows screen capture stub.

Real implementation (``mss``-based, mirroring macOS screen capture) lands in
Milestone 2.
"""

from __future__ import annotations

from pathlib import Path


class WindowsScreenCapture:
    """Windows screen capture (stub)."""

    def __init__(
        self,
        screenshot_dir: Path,
        max_screenshots: int | None = None,
    ) -> None:
        self.screenshot_dir = screenshot_dir
        self.max_screenshots = max_screenshots
        raise NotImplementedError("Windows screen capture lands in Milestone 2")
