"""macOS screen capture using mss."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.runtime.evidence import EvidenceManager

logger = logging.getLogger(__name__)


class MacOSScreenCapture:
    """Capture full-screen images on macOS using mss."""

    def __init__(
        self,
        screenshot_dir: Path,
        max_screenshots: int | None = None,
    ) -> None:
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._evidence: EvidenceManager | None = None
        if max_screenshots is not None and int(max_screenshots) > 0:
            self._evidence = EvidenceManager(self.screenshot_dir, int(max_screenshots))

    def get_logical_size(self) -> tuple[int, int]:
        """Return the logical (point) dimensions of the primary display.

        mss.monitors[1] reports the primary monitor bounds in logical points,
        which is the coordinate space pynput uses for input injection.
        """
        import mss as _mss

        with _mss.mss() as sct:
            mon = sct.monitors[1]
            return int(mon["width"]), int(mon["height"])

    def capture(self, label: str = "screenshot") -> tuple[bytes, str]:
        """Capture the primary screen and return (PNG bytes, saved path)."""
        import mss as _mss
        import mss.tools

        logger.debug("Capturing macOS screenshot")
        with _mss.mss() as sct:
            mon = sct.monitors[1]
            screenshot = sct.grab(mon)
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        return self._persist(png_bytes, label)

    def _persist(self, image_bytes: bytes, label: str) -> tuple[bytes, str]:
        if not image_bytes:
            raise RuntimeError("Screen capture produced empty image data.")
        safe_label = str(label or "macos").replace(" ", "_").replace("/", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{safe_label}_{timestamp}_{uuid4().hex[:8]}.png"
        path = self.screenshot_dir / filename
        path.write_bytes(image_bytes)
        if self._evidence:
            self._evidence.register([str(path)])
        return image_bytes, str(path)
