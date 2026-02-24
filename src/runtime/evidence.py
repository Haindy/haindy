"""Helpers for screenshot retention."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


class EvidenceManager:
    """Cleans up screenshot directories to keep them lightweight."""

    def __init__(self, screenshot_dir: Path, max_items: int) -> None:
        self.screenshot_dir = screenshot_dir
        self.max_items = max_items
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def register(self, _paths: Iterable[str]) -> None:
        """Record new screenshots and prune older artifacts."""
        self.prune_many([self.screenshot_dir], self.max_items)

    def prune(self) -> None:
        self.prune_many([self.screenshot_dir], self.max_items)

    @staticmethod
    def prune_many(directories: Iterable[Path], max_items: int) -> None:
        """Prune screenshot files in one or more directories."""
        if max_items <= 0:
            return
        for directory in directories:
            EvidenceManager._prune_dir(directory, max_items)

    @staticmethod
    def _prune_dir(directory: Path, max_items: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        files: list[Path] = [
            p for p in directory.glob("*.png") if p.is_file()
        ]
        if len(files) <= max_items:
            return

        files.sort(key=lambda p: p.stat().st_mtime)
        to_delete = files[: -max_items]
        for path in to_delete:
            try:
                path.unlink()
            except Exception:
                logger.debug(
                    "Failed to delete old screenshot",
                    exc_info=True,
                    extra={"path": str(path)},
                )
