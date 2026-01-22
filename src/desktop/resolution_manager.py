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
        primary_line = next(
            (line for line in output.splitlines() if " primary " in line), None
        )
        if not primary_line:
            primary_line = next(
                (line for line in output.splitlines() if " connected" in line), ""
            )

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

        try:
            output_text = self._run(["xrandr", "--query"])
        except Exception as exc:
            logger.warning(
                "Failed to query display modes; continuing with current mode",
                extra={"error": str(exc)},
            )
            return self.detect_current_mode()

        current = self._parse_current_mode(output_text)
        if (current.width, current.height) == (
            self.preferred_width,
            self.preferred_height,
        ):
            logger.info(
                "Resolution already at preferred mode",
                extra={"resolution": f"{current.width}x{current.height}"},
            )
            return current

        output, mode_str = self._resolve_primary_output(output_text)
        if not output or not mode_str:
            logger.warning(
                "Unable to determine primary output for resolution switch; continuing with current mode."
            )
            return current

        self._original_mode = mode_str
        self._original_output = output

        modes = self._available_modes(output_text, output)
        target_mode = f"{self.preferred_width}x{self.preferred_height}"
        target_candidate = (
            target_mode if target_mode in modes else self._fallback_mode(modes)
        )
        if not target_candidate:
            logger.warning("Preferred resolution not available; continuing with current mode.")
            self._current_mode = current
            return current

        logger.info(
            "Switching resolution for desktop automation",
            extra={
                "from": f"{current.width}x{current.height}",
                "to": target_candidate,
            },
        )
        try:
            self._run(["xrandr", "--output", output, "--mode", target_candidate])
            self._current_mode = DisplayMode(
                width=int(target_candidate.split("x")[0]),
                height=int(target_candidate.split("x")[1]),
                refresh=None,
                raw=f"{output} {target_candidate}",
            )
            return self._current_mode
        except Exception as exc:
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
            self._run(
                ["xrandr", "--output", self._original_output, "--mode", self._original_mode]
            )
            logger.info(
                "Restored original resolution",
                extra={
                    "mode": self._original_mode,
                    "output": self._original_output,
                },
            )
        except Exception as exc:
            logger.warning("Failed to restore original resolution", extra={"error": str(exc)})

    def viewport_size(self) -> Tuple[int, int]:
        """Return the current viewport size."""
        if self._current_mode:
            return self._current_mode.width, self._current_mode.height
        mode = self.detect_current_mode()
        return mode.width, mode.height

    def _resolve_primary_output(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        for line in output.splitlines():
            if " primary " in line:
                parts = line.split()
                if len(parts) >= 2:
                    output_name = parts[0]
                    mode_match = re.search(r"(\d+x\d+)", line)
                    return output_name, mode_match.group(1) if mode_match else None
        return None, None

    @staticmethod
    def _available_modes(output_text: str, output_name: str) -> list[str]:
        """Return available modes for the given output."""
        lines = output_text.splitlines()
        modes: list[str] = []
        try:
            start_idx = next(
                i for i, line in enumerate(lines) if line.strip().startswith(output_name)
            )
        except StopIteration:
            return modes
        for line in lines[start_idx + 1 :]:
            if not line.startswith(" "):
                break
            match = re.search(r"(\d+x\d+)", line)
            if match:
                modes.append(match.group(1))
        return modes

    def _fallback_mode(self, modes: list[str]) -> Optional[str]:
        """Choose the closest available mode not exceeding the preferred size."""
        if not modes:
            return None
        preferred_area = self.preferred_width * self.preferred_height
        candidates: list[tuple[int, str]] = []
        for mode in modes:
            try:
                w, h = [int(p) for p in mode.split("x")]
            except Exception:
                continue
            area = w * h
            candidates.append((abs(preferred_area - area), mode))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _parse_current_mode(self, output_text: str) -> DisplayMode:
        primary_line = next(
            (line for line in output_text.splitlines() if " primary " in line), None
        )
        if not primary_line:
            primary_line = next(
                (line for line in output_text.splitlines() if " connected" in line),
                "",
            )

        mode_match = re.search(r"(\d+)x(\d+)", primary_line)
        width = int(mode_match.group(1)) if mode_match else 1920
        height = int(mode_match.group(2)) if mode_match else 1080

        refresh_match = re.search(r"(\d+\.\d+)\*", primary_line)
        refresh = float(refresh_match.group(1)) if refresh_match else None

        mode = DisplayMode(width=width, height=height, refresh=refresh, raw=primary_line)
        self._current_mode = mode
        return mode

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
            raise RuntimeError(
                f"Command failed: {' '.join(command)} :: {result.stderr.strip()}"
            )
        return result.stdout
