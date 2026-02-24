"""Helpers to control GNOME's built-in screencast recorder via gdbus."""

from __future__ import annotations

import datetime
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ScreenRecorderError(RuntimeError):
    """Raised when recording cannot be started or stopped."""


class ScreenRecorder:
    """Starts and stops GNOME Shell's screen recorder via the D-Bus API."""

    def __init__(
        self,
        output_dir: Path,
        framerate: int = 30,
        draw_cursor: bool = True,
        filename_prefix: str = "haindy-agent",
    ) -> None:
        self.output_dir = output_dir
        self.framerate = max(1, framerate)
        self.draw_cursor = draw_cursor
        self.filename_prefix = filename_prefix
        self._session_path: str | None = None
        self._file_path: Path | None = None
        self._gdbus_path: str | None = shutil.which("gdbus")

    def start(self) -> Path:
        """Start recording and return the destination file path."""
        if self._session_path:
            raise ScreenRecorderError("Screen recording already in progress.")

        gdbus = self._resolve_gdbus()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = self.output_dir / f"{self.filename_prefix}-{timestamp}.webm"
        file_uri = file_path.resolve().as_uri()
        options = self._options_arg(file_uri)
        cmd = [
            gdbus,
            "call",
            "--session",
            "--dest",
            "org.gnome.Shell.Screencast",
            "--object-path",
            "/org/gnome/Shell/Screencast",
            "--method",
            "org.gnome.Shell.Screencast.Screencast",
            options,
            "@a{sv} {}",
        ]
        logger.debug(
            "ScreenRecorder: starting GNOME screencast", extra={"path": str(file_path)}
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Failed to start gdbus screencast."
            )
            raise ScreenRecorderError(message)

        session_path = self._parse_session_path(result.stdout)
        if not session_path:
            raise ScreenRecorderError(
                "Unable to parse screencast session path from gdbus output."
            )

        self._session_path = session_path
        self._file_path = file_path
        return file_path

    def stop(self) -> Path | None:
        """Stop recording and return the recorded file path."""
        if not self._session_path:
            return self._file_path

        gdbus = self._resolve_gdbus()
        cmd = [
            gdbus,
            "call",
            "--session",
            "--dest",
            "org.gnome.Shell.Screencast",
            "--object-path",
            "/org/gnome/Shell/Screencast",
            "--method",
            "org.gnome.Shell.Screencast.StopScreencast",
            f"@o {self._session_path}",
        ]
        logger.debug(
            "ScreenRecorder: stopping GNOME screencast",
            extra={
                "session": self._session_path,
                "path": str(self._file_path) if self._file_path else None,
            },
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Failed to stop gdbus screencast."
            )
            raise ScreenRecorderError(message)

        file_path = self._file_path
        self._session_path = None
        self._file_path = None
        return file_path

    def _resolve_gdbus(self) -> str:
        if self._gdbus_path:
            return self._gdbus_path
        path = shutil.which("gdbus")
        if not path:
            raise ScreenRecorderError(
                "gdbus command not found; install gdbus to enable --record."
            )
        self._gdbus_path = path
        return path

    @staticmethod
    def _parse_session_path(output: str) -> str | None:
        matches = re.findall(r"'([^']+)'", output)
        if len(matches) >= 2:
            return matches[1]
        return None

    def _options_arg(self, file_uri: str) -> str:
        draw_cursor = "true" if self.draw_cursor else "false"
        return (
            "@a{sv} {"
            f"'file_path': <'{file_uri}'>, "
            f"'framerate': <uint32 {self.framerate}>, "
            f"'draw-cursor': <{draw_cursor}>"
            "}"
        )
