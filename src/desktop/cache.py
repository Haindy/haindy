"""Simple coordinate cache for desktop automation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedCoordinate:
    """Represents a cached absolute coordinate for a specific target/action."""

    label: str
    action: str
    x: int
    y: int
    resolution: Tuple[int, int]
    timestamp: float
    screenshot_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "CachedCoordinate":
        return cls(
            label=payload.get("label", ""),
            action=payload.get("action", ""),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
            resolution=tuple(payload.get("resolution", (0, 0))),  # type: ignore[arg-type]
            timestamp=float(payload.get("timestamp", 0)),
            screenshot_hash=payload.get("screenshot_hash"),
        )


class CoordinateCache:
    """Append-only JSON cache for desktop coordinates."""

    def __init__(self, cache_path: Path, ttl_seconds: int = 86_400) -> None:
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[CachedCoordinate]:
        if not self.cache_path.exists():
            return []
        try:
            raw = json.loads(self.cache_path.read_text())
            if not isinstance(raw, list):
                return []
            entries: List[CachedCoordinate] = []
            for item in raw:
                try:
                    entries.append(CachedCoordinate.from_dict(item))
                except Exception:
                    continue
            return entries
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("Failed to load coordinate cache", exc_info=True, extra={"error": str(exc)})
            return []

    def _save(self, entries: List[CachedCoordinate]) -> None:
        serializable = [asdict(entry) for entry in entries]
        self.cache_path.write_text(json.dumps(serializable, indent=2))

    def lookup(
        self,
        label: str,
        action: str,
        resolution: Tuple[int, int],
        screenshot_bytes: Optional[bytes] = None,
    ) -> Optional[CachedCoordinate]:
        """Return the most recent valid cache entry for the given label/action."""
        now = time.time()
        screenshot_hash = self._hash_bytes(screenshot_bytes) if screenshot_bytes else None
        candidates: List[CachedCoordinate] = []
        for entry in self._load():
            if now - entry.timestamp > self.ttl_seconds:
                continue
            if entry.label.lower().strip() != label.lower().strip():
                continue
            if entry.action.lower().strip() != action.lower().strip():
                continue
            if tuple(entry.resolution) != tuple(resolution):
                continue
            if screenshot_hash and entry.screenshot_hash and entry.screenshot_hash != screenshot_hash:
                continue
            candidates.append(entry)

        if not candidates:
            return None
        # Return most recent
        candidates.sort(key=lambda c: c.timestamp, reverse=True)
        return candidates[0]

    def add(
        self,
        label: str,
        action: str,
        x: int,
        y: int,
        resolution: Tuple[int, int],
        screenshot_bytes: Optional[bytes] = None,
    ) -> CachedCoordinate:
        """Append a new cache entry and persist the store."""
        entries = self._load()
        entry = CachedCoordinate(
            label=label,
            action=action,
            x=x,
            y=y,
            resolution=resolution,
            timestamp=time.time(),
            screenshot_hash=self._hash_bytes(screenshot_bytes) if screenshot_bytes else None,
        )
        entries.append(entry)
        # Compact by dropping stale entries
        now = time.time()
        entries = [item for item in entries if now - item.timestamp <= self.ttl_seconds]
        self._save(entries)
        return entry

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
