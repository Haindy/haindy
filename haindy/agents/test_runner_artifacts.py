"""Screenshot and evidence helpers for TestRunner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haindy.core.interfaces import AutomationDriver
from haindy.core.types import TestCase, TestStep
from haindy.monitoring.logger import get_logger
from haindy.runtime.evidence import EvidenceManager

logger = get_logger(__name__)


@dataclass(frozen=True)
class InterpretationScreenshot:
    """Screenshot bundle used when interpreting a step."""

    screenshot_bytes: bytes | None
    screenshot_path: str | None
    source: str | None


@dataclass(frozen=True)
class CapturedScreenshot:
    """Persisted screenshot plus in-memory bytes."""

    screenshot_bytes: bytes
    screenshot_path: str


class TestRunnerArtifacts:
    """Owns screenshot persistence and in-memory screenshot state."""

    def __init__(
        self, settings: Any, automation_driver: AutomationDriver | None
    ) -> None:
        self._settings = settings
        self._automation_driver = automation_driver
        self._evidence: EvidenceManager | None = None
        self._initial_screenshot_bytes: bytes | None = None
        self._initial_screenshot_path: str | None = None
        self._latest_screenshot_bytes: bytes | None = None
        self._latest_screenshot_path: str | None = None
        self._latest_screenshot_origin: str | None = None

    def set_automation_driver(self, automation_driver: AutomationDriver | None) -> None:
        self._automation_driver = automation_driver

    def reset(self) -> None:
        self._initial_screenshot_bytes = None
        self._initial_screenshot_path = None
        self._latest_screenshot_bytes = None
        self._latest_screenshot_path = None
        self._latest_screenshot_origin = None
        self._evidence = None

    @property
    def latest_screenshot_bytes(self) -> bytes | None:
        return self._latest_screenshot_bytes

    @property
    def latest_screenshot_path(self) -> str | None:
        return self._latest_screenshot_path

    @property
    def latest_screenshot_origin(self) -> str | None:
        return self._latest_screenshot_origin

    def update_latest_snapshot(
        self,
        screenshot: bytes | None,
        screenshot_path: str | None,
        origin: str | None,
    ) -> None:
        if screenshot is None or screenshot_path is None:
            return
        self._latest_screenshot_bytes = screenshot
        self._latest_screenshot_path = str(screenshot_path)
        self._latest_screenshot_origin = origin
        if self._initial_screenshot_bytes is None:
            self._initial_screenshot_bytes = screenshot
            self._initial_screenshot_path = str(screenshot_path)

    async def ensure_initial_screenshot(self) -> None:
        """Capture and cache the initial environment screenshot."""
        if self._initial_screenshot_bytes is not None:
            return
        if not self._automation_driver:
            return

        wait_seconds = max(
            float(self._settings.actions_computer_tool_stabilization_wait_ms) / 1000.0,
            0.0,
        )
        if wait_seconds:
            await asyncio.sleep(wait_seconds)

        try:
            screenshot = await self._automation_driver.screenshot()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "Failed to capture initial screenshot",
                extra={"error": str(exc)},
            )
            return

        screenshot_path = self.save_screenshot(screenshot, "initial_state")
        self._initial_screenshot_bytes = screenshot
        self._initial_screenshot_path = str(screenshot_path)
        self._latest_screenshot_bytes = screenshot
        self._latest_screenshot_path = str(screenshot_path)
        self._latest_screenshot_origin = "initial_state"

        logger.info(
            "Captured initial environment screenshot",
            extra={"screenshot_path": self._initial_screenshot_path},
        )

    async def get_interpretation_screenshot(
        self,
        step: TestStep,
        test_case: TestCase,
    ) -> InterpretationScreenshot:
        """
        Resolve the screenshot to send alongside step interpretation.

        Returns screenshot bytes plus persisted path/source metadata.
        """
        await self.ensure_initial_screenshot()

        if self._latest_screenshot_bytes and self._latest_screenshot_path:
            source = self._latest_screenshot_origin or "cached_snapshot"
            if source == "initial_state" and step.step_number > 1:
                source = "initial_state_cached"
            return InterpretationScreenshot(
                screenshot_bytes=self._latest_screenshot_bytes,
                screenshot_path=self._latest_screenshot_path,
                source=source,
            )

        if not self._automation_driver:
            return InterpretationScreenshot(None, None, None)

        try:
            screenshot = await self._automation_driver.screenshot()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "Failed to capture screenshot for interpretation",
                extra={"error": str(exc)},
            )
            return InterpretationScreenshot(None, None, None)

        screenshot_path = self.save_screenshot(
            screenshot,
            f"tc{test_case.test_id}_step{step.step_number}_context",
        )
        self.update_latest_snapshot(
            screenshot,
            str(screenshot_path),
            "fresh_capture",
        )
        return InterpretationScreenshot(
            screenshot_bytes=screenshot,
            screenshot_path=str(screenshot_path),
            source="fresh_capture",
        )

    async def capture_screenshot(
        self,
        name: str,
        *,
        origin: str | None = None,
        update_latest: bool = False,
    ) -> CapturedScreenshot | None:
        """Capture, persist, and optionally register a screenshot as latest."""
        if not self._automation_driver:
            return None

        try:
            screenshot = await self._automation_driver.screenshot()
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning(
                "Failed to capture screenshot",
                extra={"error": str(exc), "name": name},
            )
            return None

        screenshot_path = self.save_screenshot(screenshot, name)
        screenshot_path_str = str(screenshot_path)
        if update_latest:
            self.update_latest_snapshot(screenshot, screenshot_path_str, origin or name)

        return CapturedScreenshot(
            screenshot_bytes=screenshot,
            screenshot_path=screenshot_path_str,
        )

    async def capture_test_step_screenshot(
        self,
        *,
        test_case: TestCase,
        step: TestStep,
        suffix: str,
        origin: str | None = None,
        update_latest: bool = False,
    ) -> CapturedScreenshot | None:
        """Capture a screenshot using the standard test-step naming scheme."""
        return await self.capture_screenshot(
            f"tc{test_case.test_id}_step{step.step_number}_{suffix}",
            origin=origin,
            update_latest=update_latest,
        )

    def save_screenshot(self, screenshot: bytes, name: str) -> Path:
        """Persist a screenshot and register it as evidence."""
        from haindy.monitoring.debug_logger import get_debug_logger

        debug_logger = get_debug_logger()
        if debug_logger:
            path = Path(debug_logger.save_screenshot(screenshot, name))
            self.register_evidence(path)
            return path

        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as handle:
            handle.write(screenshot)
            path = Path(handle.name)
            self.register_evidence(path)
            return path

    def register_evidence(self, path: Path) -> None:
        if not path:
            return
        if self._evidence is None:
            max_screenshots = getattr(self._settings, "max_screenshots", None)
            if max_screenshots is None:
                return
            self._evidence = EvidenceManager(
                path.parent,
                max_screenshots,
            )
        self._evidence.register([str(path)])
