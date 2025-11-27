"""Persistent run-level cache for plans, cases, and step interpretations."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _stable_hash(payload: Any) -> str:
    """Return a stable sha256 hash for any JSON-serializable payload."""
    try:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except Exception:
        serialized = str(payload).encode("utf-8", errors="ignore")
    return hashlib.sha256(serialized).hexdigest()


class PersistentRunCache:
    """JSON-backed run cache scoped by a deterministic signature."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_signature(payload: Dict[str, Any]) -> str:
        """Compute a deterministic signature for the supplied payload."""
        return _stable_hash(payload)

    @staticmethod
    def hash_bytes(data: Optional[bytes]) -> Optional[str]:
        """Return sha256 for bytes, or None when no data is provided."""
        if data is None:
            return None
        return hashlib.sha256(data).hexdigest()

    def _path(self, signature: str) -> Path:
        return self.cache_dir / f"{signature}.json"

    def _load(self, signature: str) -> Dict[str, Any]:
        path = self._path(signature)
        if not path.exists():
            return {"metadata": {}, "plan": None, "cases": {}, "steps": {}}
        try:
            return json.loads(path.read_text())
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to read run cache; returning empty", exc_info=True)
            return {"metadata": {}, "plan": None, "cases": {}, "steps": {}}

    def _save(self, signature: str, data: Dict[str, Any]) -> None:
        path = self._path(signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))

    # ------------------------------------------------------------------ #
    # Plan-level operations
    # ------------------------------------------------------------------ #
    def get_plan(self, signature: str, model: Optional[str]) -> Optional[Dict[str, Any]]:
        cache = self._load(signature)
        plan_entry = cache.get("plan")
        if not plan_entry:
            return None
        if plan_entry.get("status") != "valid":
            return None
        if model and plan_entry.get("model") and plan_entry.get("model") != model:
            return None
        return plan_entry

    def set_plan(
        self,
        signature: str,
        plan_payload: Dict[str, Any],
        triage_payload: Optional[Dict[str, Any]],
        model: Optional[str],
        signature_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        cache = self._load(signature)
        cache["metadata"] = signature_payload or cache.get("metadata", {})
        cache["plan"] = {
            "status": "valid",
            "model": model,
            "data": plan_payload,
            "triage": triage_payload or {},
            "updated_at": time.time(),
        }
        self._save(signature, cache)

    def invalidate_plan(self, signature: str, reason: str) -> None:
        cache = self._load(signature)
        plan_entry = cache.get("plan") or {}
        plan_entry["status"] = "invalidated"
        plan_entry["invalidated_at"] = time.time()
        plan_entry["reason"] = reason
        cache["plan"] = plan_entry
        self._save(signature, cache)

    # ------------------------------------------------------------------ #
    # Case-level operations
    # ------------------------------------------------------------------ #
    def mark_case_invalid(self, signature: str, case_id: str, reason: str) -> None:
        cache = self._load(signature)
        case_map: Dict[str, Any] = cache.get("cases") or {}
        case_entry = case_map.get(case_id, {})
        case_entry.update(
            {
                "status": "invalidated",
                "invalidated_at": time.time(),
                "reason": reason,
            }
        )
        case_map[case_id] = case_entry
        cache["cases"] = case_map

        # Drop step entries for this case
        step_map: Dict[str, Any] = cache.get("steps") or {}
        step_map = {
            key: value for key, value in step_map.items() if not key.startswith(f"{case_id}::")
        }
        cache["steps"] = step_map
        self._save(signature, cache)

    # ------------------------------------------------------------------ #
    # Step-level operations
    # ------------------------------------------------------------------ #
    def get_step_actions(
        self,
        signature: str,
        case_id: str,
        step_number: int,
        screenshot_hash: Optional[str],
        model: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        cache = self._load(signature)
        key = f"{case_id}::{step_number}"
        entry = (cache.get("steps") or {}).get(key)
        if not entry:
            return None
        if entry.get("status") != "valid":
            return None
        if model and entry.get("model") and entry.get("model") != model:
            return None
        cached_hash = entry.get("screenshot_hash")
        if screenshot_hash and cached_hash and cached_hash != screenshot_hash:
            return None
        return {"cache_key": key, "actions": entry.get("actions", [])}

    def set_step_actions(
        self,
        signature: str,
        case_id: str,
        step_number: int,
        screenshot_hash: Optional[str],
        model: Optional[str],
        actions: List[Dict[str, Any]],
    ) -> str:
        cache = self._load(signature)
        steps: Dict[str, Any] = cache.get("steps") or {}
        key = f"{case_id}::{step_number}"
        steps[key] = {
            "status": "valid",
            "model": model,
            "actions": actions,
            "screenshot_hash": screenshot_hash,
            "updated_at": time.time(),
        }
        cache["steps"] = steps
        self._save(signature, cache)
        return key

    def invalidate_step(self, signature: str, case_id: str, step_number: int, reason: str) -> None:
        cache = self._load(signature)
        steps: Dict[str, Any] = cache.get("steps") or {}
        key = f"{case_id}::{step_number}"
        entry = steps.get(key, {})
        entry.update(
            {
                "status": "invalidated",
                "invalidated_at": time.time(),
                "reason": reason,
            }
        )
        steps[key] = entry
        cache["steps"] = steps
        self._save(signature, cache)
