"""Screen capture helper that shells out to ffmpeg."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.desktop.resolution_manager import ResolutionManager
from src.runtime.evidence import EvidenceManager

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Capture full-screen images using ffmpeg."""

    def __init__(
        self,
        resolution_manager: ResolutionManager,
        screenshot_dir: Path,
        display: str | None = None,
        max_screenshots: int | None = None,
    ) -> None:
        self.resolution_manager = resolution_manager
        self.screenshot_dir = screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.display = display or os.environ.get("DISPLAY", ":0")
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._imagemagick_import_path = shutil.which("import")
        self._evidence: EvidenceManager | None = None
        if max_screenshots is not None and int(max_screenshots) > 0:
            self._evidence = EvidenceManager(self.screenshot_dir, int(max_screenshots))

    def capture(self, label: str) -> tuple[bytes, str]:
        """Capture a single frame and persist it for debugging."""
        width, height = self.resolution_manager.viewport_size()
        capture_attempt_errors: list[str] = []

        logger.debug(
            "Capturing desktop screenshot", extra={"width": width, "height": height}
        )
        for backend_name, cmd in self._capture_commands(width=width, height=height):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                capture_attempt_errors.append(f"{backend_name}: {exc}")
                continue

            if result.returncode == 0 and result.stdout:
                return self._persist_capture(result.stdout, label)

            error_detail = result.stderr.decode("utf-8", errors="ignore").strip()
            if not error_detail:
                error_detail = f"empty output (exit={result.returncode})"
            capture_attempt_errors.append(f"{backend_name}: {error_detail}")

        if capture_attempt_errors:
            raise RuntimeError(
                "Screen capture failed. Attempts: " + " | ".join(capture_attempt_errors)
            )
        raise RuntimeError(
            "No screen capture backend available. Install ffmpeg or ImageMagick ('import')."
        )

    def _persist_capture(self, image_bytes: bytes, label: str) -> tuple[bytes, str]:
        """Persist screenshot bytes to disk and evidence retention."""
        if not image_bytes:
            raise RuntimeError("Screen capture produced empty image data.")

        safe_label = str(label or "desktop").replace(" ", "_").replace("/", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{safe_label}_{timestamp}_{uuid4().hex[:8]}.png"
        path = self.screenshot_dir / filename
        path.write_bytes(image_bytes)
        if self._evidence:
            self._evidence.register([str(path)])
        return image_bytes, str(path)

    def _capture_commands(self, width: int, height: int) -> list[tuple[str, list[str]]]:
        """Build preferred capture commands in execution order."""
        commands: list[tuple[str, list[str]]] = []
        if self._ffmpeg_path:
            commands.append(
                (
                    "ffmpeg",
                    [
                        self._ffmpeg_path,
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
                    ],
                )
            )

        if self._imagemagick_import_path:
            commands.append(
                (
                    "import",
                    [
                        self._imagemagick_import_path,
                        "-silent",
                        "-display",
                        self.display,
                        "-window",
                        "root",
                        "png:-",
                    ],
                )
            )

        return commands
