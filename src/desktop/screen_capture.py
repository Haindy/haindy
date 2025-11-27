"""Screen capture helper that shells out to ffmpeg."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
import os
from typing import Optional, Tuple

from src.desktop.resolution_manager import ResolutionManager

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Capture full-screen images using ffmpeg."""

    def __init__(
        self,
        resolution_manager: ResolutionManager,
        screenshot_dir: Path,
        display: Optional[str] = None,
    ) -> None:
        self.resolution_manager = resolution_manager
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.display = display or os.environ.get("DISPLAY", ":0")

    def capture(self, label: str) -> Tuple[bytes, str]:
        """Capture a single frame and persist it for debugging."""
        width, height = self.resolution_manager.viewport_size()
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "x11grab",
            "-video_size",
            f"{width}x{height}",
            "-i",
            f"{self.display}+0,0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]

        logger.debug("Capturing desktop screenshot", extra={"width": width, "height": height})
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError(f"ffmpeg capture failed: {result.stderr.decode('utf-8', errors='ignore')}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.png"
        path = self.screenshot_dir / filename
        path.write_bytes(result.stdout)
        return result.stdout, str(path)
