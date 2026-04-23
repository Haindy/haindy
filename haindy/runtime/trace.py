"""Run trace artifact writer."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haindy.core.types import StepResult, TestStep

logger = logging.getLogger(__name__)

TRACE_VERSION = 1


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _sanitize_for_json(value: Any, *, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            length = len(value)
        except Exception:
            length = -1
        return f"<<bytes:{length}>>"
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _sanitize_for_json(asdict(value), _seen=_seen)
        except Exception:
            return str(value)

    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return "<<cycle>>"
    _seen.add(value_id)

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized[str(key)] = _sanitize_for_json(item, _seen=_seen)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item, _seen=_seen) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _sanitize_for_json(value.model_dump(), _seen=_seen)
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _sanitize_for_json(value.to_dict(), _seen=_seen)
        except Exception:
            return str(value)

    return str(value)


def load_model_calls_for_run(log_path: Path, *, run_id: str) -> list[dict[str, Any]]:
    """Read model_calls.jsonl and return entries for the provided run id."""
    if not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if parsed.get("run_id") == run_id and isinstance(parsed, dict):
                    entries.append(parsed)
    except Exception:
        logger.debug("RunTraceWriter: failed to read model call log", exc_info=True)
        return []
    return entries


class RunTraceWriter:
    """Collects a per-run trace and writes it to disk."""

    def __init__(self, run_id: str, *, trace_dir: Path | None = None) -> None:
        self.run_id = run_id
        self._path = (trace_dir or _default_trace_dir()) / f"{run_id}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "trace_version": TRACE_VERSION,
            "run_id": run_id,
            "started_at": _now_iso(),
            "ended_at": None,
            "success": None,
            "run_metadata": {},
            "steps": [],
            "cache_events": [],
            "model_calls": [],
        }

    @property
    def path(self) -> Path:
        return self._path

    def set_run_metadata(self, metadata: dict[str, Any]) -> None:
        self._data["run_metadata"] = _sanitize_for_json(metadata)

    def record_step(
        self,
        *,
        scenario_name: str,
        step: TestStep,
        step_result: StepResult,
        attempt: int | None = None,
        plan_cache_hit: bool | None = None,
    ) -> None:
        try:
            entry = {
                "timestamp": _now_iso(),
                "scenario": scenario_name,
                "step_number": step.step_number,
                "step_action": step.action,
                "expected_result": step.expected_result,
                "intent": step.intent.value,
                "environment": step.environment,
                "loop": step.loop,
                "attempt": attempt,
                "plan_cache_hit": plan_cache_hit,
                "step_result": _sanitize_for_json(step_result),
            }
            self._data["steps"].append(entry)
        except Exception:
            logger.debug("RunTraceWriter: failed to record step", exc_info=True)

    def update_last_step(
        self,
        *,
        step_number: int | None = None,
        step_result: StepResult | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Update the most recent recorded step entry in-place."""
        try:
            for entry in reversed(self._data["steps"]):
                if step_number is not None and entry.get("step_number") != step_number:
                    continue
                if step_result is not None:
                    entry["step_result"] = _sanitize_for_json(step_result)
                if extra:
                    entry.update(_sanitize_for_json(extra))
                return True
        except Exception:
            logger.debug("RunTraceWriter: failed to update last step", exc_info=True)
        return False

    def record_cache_event(self, event: dict[str, Any]) -> None:
        try:
            event_with_timestamp = dict(event)
            event_with_timestamp.setdefault("timestamp", _now_iso())
            self._data["cache_events"].append(_sanitize_for_json(event_with_timestamp))
        except Exception:
            logger.debug("RunTraceWriter: failed to record cache event", exc_info=True)

    def set_model_calls(self, model_calls: list[dict[str, Any]]) -> None:
        self._data["model_calls"] = _sanitize_for_json(model_calls)

    def finalize(self, *, success: bool, ended_at: str | None = None) -> None:
        self._data["ended_at"] = ended_at or _now_iso()
        self._data["success"] = bool(success)

    def write(self) -> None:
        try:
            serialized = json.dumps(self._data, ensure_ascii=False, indent=2)
            self._path.write_text(serialized, encoding="utf-8")
        except Exception:
            logger.debug("RunTraceWriter: failed to write trace file", exc_info=True)


def _default_trace_dir() -> Path:
    from haindy.config.settings import get_settings

    return get_settings().data_dir / "traces"
