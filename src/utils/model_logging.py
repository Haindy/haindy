"""Utilities for recording raw model prompts and responses."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime.evidence import EvidenceManager
from src.security.sanitizer import sanitize_dict, sanitize_string

logger = logging.getLogger(__name__)

try:
    from src.monitoring.logger import get_run_id
except Exception:  # pragma: no cover - defensive import

    def get_run_id() -> str:
        return "unknown"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _normalize_response_obj(response: Any) -> Any:
    if response is None:
        return None
    if isinstance(response, (str, int, float, bool)):
        return response
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(warnings="none")
        except TypeError:
            try:
                return response.model_dump()
            except Exception:
                return str(response)
        except Exception:
            return str(response)
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:
            return str(response)
    return str(response)


def _sanitize_for_json(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Recursively coerce values into JSON-serializable shapes."""
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
            return _sanitize_for_json(
                value.model_dump(warnings="none"),
                _seen=_seen,
            )
        except TypeError:
            try:
                return _sanitize_for_json(value.model_dump(), _seen=_seen)
            except Exception:
                return str(value)
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _sanitize_for_json(value.to_dict(), _seen=_seen)
        except Exception:
            return str(value)

    return str(value)


def _redact_sensitive(value: Any) -> Any:
    """Recursively sanitize strings inside JSON-safe values."""

    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, dict):
        sanitized_dict = sanitize_dict({str(key): item for key, item in value.items()})
        return {key: _redact_sensitive(item) for key, item in sanitized_dict.items()}
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


class ModelCallLogger:
    """Append-only logger for model inputs/outputs with optional screenshots."""

    def __init__(self, log_path: Path, max_screenshots: int | None = None) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._screenshot_dir = self._log_path.parent / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._evidence: EvidenceManager | None = None
        if max_screenshots is not None and int(max_screenshots) > 0:
            self._evidence = EvidenceManager(self._screenshot_dir, int(max_screenshots))

    async def log_call(
        self,
        *,
        agent: str,
        model: str,
        prompt: str,
        request_payload: Any | None,
        response: Any,
        screenshots: Sequence[tuple[str, bytes]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        screenshot_entries: list[dict[str, str]] = []
        screenshot_errors: list[dict[str, str]] = []
        if screenshots:
            for label, data in screenshots:
                if not data:
                    continue
                try:
                    path = self._persist_screenshot(label, data)
                except Exception as exc:  # pragma: no cover - best-effort logging
                    screenshot_errors.append({"label": label, "error": str(exc)})
                    continue
                else:
                    screenshot_entries.append({"label": label, "path": path})

        entry = {
            "timestamp": _now_iso(),
            "run_id": get_run_id(),
            "agent": agent,
            "model": model,
            "prompt": sanitize_string(prompt),
            "request_payload": _redact_sensitive(_sanitize_for_json(request_payload)),
            "prompt_has_screenshot": bool(screenshot_entries),
            "attached_screenshots": screenshot_entries,
            "response": _redact_sensitive(
                _sanitize_for_json(_normalize_response_obj(response))
            ),
            "metadata": _redact_sensitive(_sanitize_for_json(metadata or {})),
        }
        if screenshot_errors:
            entry["metadata"]["screenshot_errors"] = screenshot_errors

        try:
            serialized = json.dumps(entry, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - logging should never crash a run
            fallback = {
                "timestamp": entry.get("timestamp"),
                "agent": agent,
                "model": model,
                "prompt": sanitize_string(prompt),
                "prompt_has_screenshot": bool(screenshot_entries),
                "attached_screenshots": screenshot_entries,
                "serialization_error": str(exc),
            }
            serialized = json.dumps(fallback, ensure_ascii=False, default=str)

        try:
            async with self._lock:
                with self._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.write("\n")
        except Exception:  # pragma: no cover - logging must be non-fatal
            logger.debug("ModelCallLogger failed to write log entry", exc_info=True)
            return

    def _persist_screenshot(self, label: str, data: bytes) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_label = label.replace(" ", "_").replace("/", "-")
        filename = f"{safe_label}_{timestamp}.png"
        path = self._screenshot_dir / filename
        path.write_bytes(data)
        if self._evidence:
            self._evidence.register([str(path)])
        return str(path)


_LOGGER_CACHE: dict[tuple[Path, int | None], ModelCallLogger] = {}


def get_model_logger(
    log_path: Path, max_screenshots: int | None = None
) -> ModelCallLogger:
    """Return a singleton logger instance per log file path."""
    resolved = log_path.resolve()
    normalized_max: int | None = None
    if max_screenshots is not None:
        try:
            candidate = int(max_screenshots)
            if candidate > 0:
                normalized_max = candidate
        except Exception:
            normalized_max = None
    cache_key = (resolved, normalized_max)
    cached = _LOGGER_CACHE.get(cache_key)
    if cached:
        return cached
    logger_instance = ModelCallLogger(resolved, max_screenshots=normalized_max)
    _LOGGER_CACHE[cache_key] = logger_instance
    return logger_instance
