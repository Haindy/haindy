"""Resolution management helpers for desktop automation."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DisplayMode:
    """Represents a display mode."""

    width: int
    height: int
    refresh: Optional[float] = None
    raw: Optional[str] = None


class ResolutionManager:
    """Manage resolution detection and optional downshifts."""

    def __init__(
        self,
        preferred_width: int = 1920,
        preferred_height: int = 1080,
        enable_switch: bool = False,
    ) -> None:
        self.preferred_width = preferred_width
        self.preferred_height = preferred_height
        self.enable_switch = enable_switch
        self._original_mode: Optional[str] = None
        self._original_output: Optional[str] = None
        self._current_mode: Optional[DisplayMode] = None

    def detect_current_mode(self) -> DisplayMode:
        """Detect the current resolution using xrandr."""
        output = self._run(["xrandr", "--query"])
        primary_line = next((line for line in output.splitlines() if " primary " in line), None)
        if not primary_line:
            primary_line = next((line for line in output.splitlines() if " connected" in line), "")

        mode_match = re.search(r"(\d+)x(\d+)", primary_line)
        width = int(mode_match.group(1)) if mode_match else 1920
        height = int(mode_match.group(2)) if mode_match else 1080

        refresh_match = re.search(r"(\d+\.\d+)\*", primary_line)
        refresh = float(refresh_match.group(1)) if refresh_match else None

        mode = DisplayMode(width=width, height=height, refresh=refresh, raw=primary_line)
        self._current_mode = mode
        return mode

    def maybe_downshift(self) -> Optional[DisplayMode]:
        """Optionally switch to preferred resolution when allowed."""
        if not self.enable_switch:
            return self.detect_current_mode()

        current = self.detect_current_mode()
        if (current.width, current.height) == (self.preferred_width, self.preferred_height):
            logger.info("Resolution already at preferred mode", extra={"resolution": f"{current.width}x{current.height}"})
            return current

        output, mode_str = self._resolve_primary_output()
        if not output or not mode_str:
            logger.warning("Unable to determine primary output for resolution switch; continuing with current mode.")
            return current

        self._original_mode = mode_str
        self._original_output = output

        target_mode = f"{self.preferred_width}x{self.preferred_height}"
        logger.info(
            "Switching resolution for desktop automation",
            extra={"from": f"{current.width}x{current.height}", "to": target_mode},
        )
        try:
            self._run(["xrandr", "--output", output, "--mode", target_mode])
            self._current_mode = DisplayMode(
                width=self.preferred_width,
                height=self.preferred_height,
                refresh=None,
                raw=f"{output} {target_mode}",
            )
            return self._current_mode
        except Exception as exc:  # pragma: no cover - side-effectful
            logger.warning(
                "Failed to switch resolution; continuing with current mode",
                extra={"error": str(exc)},
            )
            self._current_mode = current
            return current

    def restore(self) -> None:
        """Restore the original resolution if it was changed."""
        if not self.enable_switch:
            return
        if not self._original_output or not self._original_mode:
            return
        try:
            self._run(["xrandr", "--output", self._original_output, "--mode", self._original_mode])
            logger.info(
                "Restored original resolution",
                extra={"mode": self._original_mode, "output": self._original_output},
            )
        except Exception as exc:  # pragma: no cover - side-effectful
            logger.warning("Failed to restore original resolution", extra={"error": str(exc)})

    def viewport_size(self) -> Tuple[int, int]:
        """Return the current viewport size."""
        if self._current_mode:
            return self._current_mode.width, self._current_mode.height
        mode = self.detect_current_mode()
        return mode.width, mode.height

    def _resolve_primary_output(self) -> Tuple[Optional[str], Optional[str]]:
        output = self._run(["xrandr", "--query"])
        for line in output.splitlines():
            if " primary " in line:
                parts = line.split()
                if len(parts) >= 2:
                    output_name = parts[0]
                    mode_match = re.search(r"(\d+x\d+)", line)
                    return output_name, mode_match.group(1) if mode_match else None
        return None, None

    @staticmethod
    def _run(command: list[str]) -> str:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(command)} :: {result.stderr.strip()}")
        return result.stdout
