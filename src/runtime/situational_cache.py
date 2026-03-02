"""Disk-backed cache for situational assessment outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SITUATIONAL_CACHE_VERSION = 1


def build_situational_cache_key_payload(
    requirements: object | None = None,
    context_text: object | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a stable payload used for situational cache key hashing."""
    payload: dict[str, Any] = {"_cache_version": SITUATIONAL_CACHE_VERSION}
    if requirements is not None:
        payload["requirements"] = requirements
    if context_text is not None:
        payload["context_text"] = context_text
    payload.update(kwargs)
    return payload


def hash_situational_cache_key(payload: object) -> str:
    """Return a SHA-256 hash of a canonicalized situational cache-key payload."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _coerce_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


@dataclass
class SituationalCacheEntry:
    """Persisted situational assessment payload for a key hash."""

    cache_version: int
    key_hash: str
    cached_at_epoch_seconds: int
    assessment_payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SituationalCacheEntry:
        return cls(
            cache_version=int(payload.get("cache_version", SITUATIONAL_CACHE_VERSION)),
            key_hash=str(payload.get("key_hash", "")),
            cached_at_epoch_seconds=int(payload.get("cached_at_epoch_seconds") or 0),
            assessment_payload=_coerce_dict(payload.get("assessment_payload")),
        )


class SituationalCache:
    """JSON cache storing situational assessments by key hash."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def lookup(self, key_hash: str) -> SituationalCacheEntry | None:
        matches = [
            entry
            for entry in self._load()
            if entry.cache_version == SITUATIONAL_CACHE_VERSION
            and entry.key_hash == key_hash
        ]
        if not matches:
            return None
        return matches[-1]

    def store(
        self,
        key_hash: str,
        assessment_payload: dict[str, Any],
    ) -> SituationalCacheEntry:
        entries = [
            entry
            for entry in self._load()
            if not (
                entry.cache_version == SITUATIONAL_CACHE_VERSION
                and entry.key_hash == key_hash
            )
        ]
        entry = SituationalCacheEntry(
            cache_version=SITUATIONAL_CACHE_VERSION,
            key_hash=key_hash,
            cached_at_epoch_seconds=int(time.time()),
            assessment_payload=_coerce_dict(assessment_payload),
        )
        entries.append(entry)
        self._save(entries)
        return entry

    def invalidate(self, key_hash: str) -> None:
        entries = self._load()
        new_entries = [entry for entry in entries if entry.key_hash != key_hash]
        if len(new_entries) == len(entries):
            return
        self._save(new_entries)

    def _load(self) -> list[SituationalCacheEntry]:
        if not self._path.exists():
            return []
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "SituationalCache: failed to parse cache file; starting empty.",
                exc_info=True,
            )
            return []
        if not isinstance(raw, list):
            return []
        entries: list[SituationalCacheEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(SituationalCacheEntry.from_dict(item))
            except Exception:
                logger.debug(
                    "SituationalCache: skipping malformed cache entry",
                    exc_info=True,
                )
        return entries

    def _save(self, entries: list[SituationalCacheEntry]) -> None:
        serializable = [asdict(entry) for entry in entries]
        self._path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
