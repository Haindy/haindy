"""Disk-backed execution replay cache for step-level driver action recording."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXECUTION_REPLAY_CACHE_VERSION = 1


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class ExecutionReplayCacheKey:
    """Key for looking up step-level replay recordings."""

    scenario: str
    step: str
    environment: str
    resolution: tuple[int, int]
    keyboard_layout: str = "us"
    plan_fingerprint: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> ExecutionReplayCacheKey:
        resolution_raw = (
            payload.get("resolution")
            or payload.get("viewport_resolution")
            or payload.get("viewport")
            or (0, 0)
        )
        resolution: tuple[int, int] = (0, 0)
        if isinstance(resolution_raw, (list, tuple)) and len(resolution_raw) == 2:
            try:
                resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
            except Exception:
                resolution = (0, 0)
        return cls(
            scenario=str(
                payload.get("scenario") or payload.get("scenario_name") or ""
            ).strip(),
            step=str(payload.get("step") or payload.get("step_name") or "").strip(),
            environment=str(payload.get("environment") or "desktop").strip(),
            resolution=resolution,
            keyboard_layout=str(payload.get("keyboard_layout") or "us").strip(),
            plan_fingerprint=str(payload.get("plan_fingerprint") or "").strip(),
        )

    def matches(self, other: ExecutionReplayCacheKey) -> bool:
        return (
            _norm(self.scenario) == _norm(other.scenario)
            and _norm(self.step) == _norm(other.step)
            and _norm(self.environment) == _norm(other.environment)
            and tuple(self.resolution) == tuple(other.resolution)
            and _norm(self.keyboard_layout) == _norm(other.keyboard_layout)
            and _norm(self.plan_fingerprint) == _norm(other.plan_fingerprint)
        )


@dataclass
class ExecutionReplayEntry:
    """Recorded driver actions for a key."""

    cache_version: int
    key: ExecutionReplayCacheKey
    recorded_at_epoch_seconds: int
    actions: list[dict]

    @classmethod
    def from_dict(cls, payload: dict) -> ExecutionReplayEntry:
        key_payload = payload.get("key") or {}
        actions = payload.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        return cls(
            cache_version=int(
                payload.get("cache_version", EXECUTION_REPLAY_CACHE_VERSION)
            ),
            key=ExecutionReplayCacheKey.from_dict(
                key_payload if isinstance(key_payload, dict) else {}
            ),
            recorded_at_epoch_seconds=int(
                payload.get("recorded_at_epoch_seconds") or 0
            ),
            actions=[item for item in actions if isinstance(item, dict)],
        )


class ExecutionReplayCache:
    """Append-only JSON cache that stores replay recordings."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def lookup(self, key: ExecutionReplayCacheKey) -> ExecutionReplayEntry | None:
        candidates = [
            entry
            for entry in self._load()
            if entry.cache_version == EXECUTION_REPLAY_CACHE_VERSION
            and entry.key.matches(key)
        ]
        if not candidates:
            return None
        return candidates[-1]

    def store(
        self, key: ExecutionReplayCacheKey, actions: list[dict]
    ) -> ExecutionReplayEntry:
        entries = self._load()
        entry = ExecutionReplayEntry(
            cache_version=EXECUTION_REPLAY_CACHE_VERSION,
            key=key,
            recorded_at_epoch_seconds=int(time.time()),
            actions=[item for item in actions if isinstance(item, dict)],
        )
        entries.append(entry)
        self._save(entries)
        return entry

    def invalidate(self, key: ExecutionReplayCacheKey) -> None:
        entries = self._load()
        new_entries = [
            entry
            for entry in entries
            if not (
                entry.cache_version == EXECUTION_REPLAY_CACHE_VERSION
                and entry.key.matches(key)
            )
        ]
        if len(new_entries) == len(entries):
            return
        self._save(new_entries)

    def _load(self) -> list[ExecutionReplayEntry]:
        if not self._path.exists():
            return []
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "ExecutionReplayCache: failed to parse cache file; starting empty.",
                exc_info=True,
            )
            return []
        if not isinstance(raw, list):
            return []
        entries: list[ExecutionReplayEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(ExecutionReplayEntry.from_dict(item))
            except Exception:
                logger.debug(
                    "ExecutionReplayCache: skipping malformed cache entry",
                    exc_info=True,
                )
        return entries

    def _save(self, entries: list[ExecutionReplayEntry]) -> None:
        serializable = [asdict(entry) for entry in entries]
        self._path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
