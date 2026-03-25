"""Disk-backed cache for scope triage and test plan generation outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLANNING_CACHE_VERSION = 1


def build_planning_cache_key_payload(
    requirements: object | None = None,
    context: object | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a stable payload used for planning cache key hashing."""
    payload: dict[str, Any] = {"_cache_version": PLANNING_CACHE_VERSION}
    if requirements is not None:
        payload["requirements"] = requirements
    if context is not None:
        payload["context"] = context
    payload.update(kwargs)
    return payload


def hash_planning_cache_key(payload: object) -> str:
    """Return a SHA-256 hash of a canonicalized cache-key payload."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return None


@dataclass
class PlanningCacheEntry:
    """Persisted triage + planning result payload for a key hash."""

    cache_version: int
    key_hash: str
    cached_at_epoch_seconds: int
    triage_payload: dict[str, Any]
    test_plan_payload: dict[str, Any] | None
    has_blockers: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanningCacheEntry:
        return cls(
            cache_version=int(payload.get("cache_version", PLANNING_CACHE_VERSION)),
            key_hash=str(payload.get("key_hash", "")),
            cached_at_epoch_seconds=int(payload.get("cached_at_epoch_seconds") or 0),
            triage_payload=_coerce_dict(payload.get("triage_payload")),
            test_plan_payload=_coerce_optional_dict(payload.get("test_plan_payload")),
            has_blockers=_coerce_bool(payload.get("has_blockers", False)),
        )


class PlanningCache:
    """JSON cache storing planning outcomes by key hash."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def lookup(self, key_hash: str) -> PlanningCacheEntry | None:
        matches = [
            entry
            for entry in self._load()
            if entry.cache_version == PLANNING_CACHE_VERSION
            and entry.key_hash == key_hash
        ]
        if not matches:
            return None
        return matches[-1]

    def store(
        self,
        key_hash: str,
        triage_payload: dict[str, Any],
        test_plan_payload: dict[str, Any] | None = None,
        has_blockers: bool = False,
    ) -> PlanningCacheEntry:
        entries = [
            entry
            for entry in self._load()
            if not (
                entry.cache_version == PLANNING_CACHE_VERSION
                and entry.key_hash == key_hash
            )
        ]
        entry = PlanningCacheEntry(
            cache_version=PLANNING_CACHE_VERSION,
            key_hash=key_hash,
            cached_at_epoch_seconds=int(time.time()),
            triage_payload=_coerce_dict(triage_payload),
            test_plan_payload=_coerce_optional_dict(test_plan_payload),
            has_blockers=has_blockers,
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

    def _load(self) -> list[PlanningCacheEntry]:
        if not self._path.exists():
            return []
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "PlanningCache: failed to parse cache file; starting empty.",
                exc_info=True,
            )
            return []
        if not isinstance(raw, list):
            return []
        entries: list[PlanningCacheEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(PlanningCacheEntry.from_dict(item))
            except Exception:
                logger.debug(
                    "PlanningCache: skipping malformed cache entry",
                    exc_info=True,
                )
        return entries

    def _save(self, entries: list[PlanningCacheEntry]) -> None:
        serializable = [asdict(entry) for entry in entries]
        self._path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
