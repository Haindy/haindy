"""Append-only coordinate cache shared by all desktop automation drivers."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CachedCoordinate:
    """Represents a cached absolute coordinate for a specific target/action."""

    label: str
    action: str
    x: int
    y: int
    resolution: tuple[int, int]
    screenshot_hash: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> CachedCoordinate:
        return cls(
            label=str(payload.get("label", "")).strip(),
            action=str(payload.get("action", "")).strip(),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
            resolution=tuple(payload.get("resolution", (0, 0))),  # type: ignore[arg-type]
            screenshot_hash=payload.get("screenshot_hash"),
        )


class CoordinateCache:
    """Disk-backed cache keyed by label/action/resolution."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[CachedCoordinate]:
        if not self.cache_path.exists():
            return []
        try:
            raw = json.loads(self.cache_path.read_text())
        except Exception:
            logger.debug(
                "Failed to parse coordinate cache; starting empty.", exc_info=True
            )
            return []

        entries: list[CachedCoordinate] = []
        if isinstance(raw, list):
            for item in raw:
                try:
                    entries.append(CachedCoordinate.from_dict(item))
                except Exception:
                    logger.debug("Skipping malformed cache entry", exc_info=True)
                    continue
        return entries

    def _save(self, entries: list[CachedCoordinate]) -> None:
        serializable = [asdict(entry) for entry in entries]
        self.cache_path.write_text(json.dumps(serializable, indent=2))

    def lookup(
        self,
        label: str,
        action: str,
        resolution: tuple[int, int],
        screenshot_bytes: bytes | None = None,
    ) -> CachedCoordinate | None:
        """Return the newest cached coordinate matching the lookup key."""
        screenshot_hash = (
            self._hash_bytes(screenshot_bytes) if screenshot_bytes else None
        )
        candidates: list[CachedCoordinate] = []
        for entry in self._load():
            if entry.label.lower().strip() != label.lower().strip():
                continue
            if entry.action.lower().strip() != action.lower().strip():
                continue
            if tuple(entry.resolution) != tuple(resolution):
                continue
            if (
                screenshot_hash
                and entry.screenshot_hash
                and entry.screenshot_hash != screenshot_hash
            ):
                continue
            candidates.append(entry)

        if not candidates:
            return None
        return candidates[-1]

    def add(
        self,
        label: str,
        action: str,
        x: int,
        y: int,
        resolution: tuple[int, int],
        screenshot_bytes: bytes | None = None,
    ) -> CachedCoordinate:
        """Persist a new coordinate without truncation."""
        entries = self._load()
        entry = CachedCoordinate(
            label=label,
            action=action,
            x=x,
            y=y,
            resolution=resolution,
            screenshot_hash=(
                self._hash_bytes(screenshot_bytes) if screenshot_bytes else None
            ),
        )
        entries.append(entry)
        self._save(entries)
        return entry

    def invalidate(self, label: str, action: str, resolution: tuple[int, int]) -> None:
        """Remove entries matching the failing label/action/resolution."""
        entries = [
            item
            for item in self._load()
            if not (
                item.label.lower().strip() == label.lower().strip()
                and item.action.lower().strip() == action.lower().strip()
                and tuple(item.resolution) == tuple(resolution)
            )
        ]
        self._save(entries)

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
