"""Helpers for screenshot retention."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)


class EvidenceManager:
    """Cleans up screenshot directories to keep them lightweight."""

    def __init__(self, screenshot_dir: Path, max_items: int) -> None:
        self.screenshot_dir = screenshot_dir
        self.max_items = max_items
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def register(self, paths: Iterable[str]) -> None:
        """Record new screenshots and prune older artifacts."""
        self.prune()

    def prune(self) -> None:
        files: List[Path] = [
            p for p in self.screenshot_dir.glob("*.png") if p.is_file()
        ]
        if len(files) <= self.max_items:
            return

        files.sort(key=lambda p: p.stat().st_mtime)
        to_delete = files[: -self.max_items]
        for path in to_delete:
            try:
                path.unlink()
            except Exception:
                logger.debug(
                    "Failed to delete old screenshot",
                    exc_info=True,
                    extra={"path": str(path)},
                )
