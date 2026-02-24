"""Cache for task runner action planning."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

TASK_PLAN_CACHE_VERSION = 3


def _hash_context(context: object) -> str:
    payload = json.dumps(
        {"_cache_version": TASK_PLAN_CACHE_VERSION, "context": context},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedPlan:
    """Persisted planned actions for a step/context."""

    step_name: str
    context_hash: str
    actions: List[dict]

    @classmethod
    def from_dict(cls, data: dict) -> "CachedPlan":
        return cls(
            step_name=str(data.get("step_name", "")),
            context_hash=str(data.get("context_hash", "")),
            actions=data.get("actions") or [],
        )


class TaskPlanCache:
    """Disk-backed cache to reuse planned actions per step/context."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def lookup(self, step_name: str, context: object) -> Optional[List[dict]]:
        context_hash = _hash_context(context)
        for entry in self._load():
            if entry.step_name == step_name and entry.context_hash == context_hash:
                actions = [item for item in entry.actions if isinstance(item, dict)]
                if actions:
                    logger.info("Task plan cache hit", extra={"step": step_name})
                    return actions
        return None

    def store(self, step_name: str, context: object, actions: List[dict]) -> None:
        context_hash = _hash_context(context)
        entries = self._load()
        entries = [
            entry
            for entry in entries
            if not (entry.step_name == step_name and entry.context_hash == context_hash)
        ]
        entries.append(
            CachedPlan(
                step_name=step_name,
                context_hash=context_hash,
                actions=[item for item in actions if isinstance(item, dict)],
            )
        )
        self._save(entries)
        logger.info("Task plan cached", extra={"step": step_name})

    def invalidate(self, step_name: str, context: object) -> None:
        context_hash = _hash_context(context)
        entries = self._load()
        new_entries = [
            entry
            for entry in entries
            if not (entry.step_name == step_name and entry.context_hash == context_hash)
        ]
        if len(new_entries) != len(entries):
            self._save(new_entries)
            logger.info("Task plan cache invalidated", extra={"step": step_name})

    def _load(self) -> List[CachedPlan]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text())
        except Exception:
            logger.debug("Failed to read task plan cache", exc_info=True)
            return []
        entries: List[CachedPlan] = []
        if isinstance(raw, list):
            for item in raw:
                try:
                    entries.append(CachedPlan.from_dict(item))
                except Exception:
                    logger.debug("Skipping malformed task plan cache entry", exc_info=True)
        return entries

    def _save(self, entries: List[CachedPlan]) -> None:
        serializable = [entry.__dict__ for entry in entries]
        self._path.write_text(json.dumps(serializable, indent=2))
