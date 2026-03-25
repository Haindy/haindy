"""Utilities for recording raw model prompts and responses."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haindy.runtime.evidence import EvidenceManager
from haindy.security.sanitizer import sanitize_dict, sanitize_string

logger = logging.getLogger(__name__)

try:
    from haindy.monitoring.logger import get_run_id
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


@dataclass(frozen=True)
class ModelCallFailure:
    """Normalized details for one failed model-call attempt."""

    failure_kind: str
    error: dict[str, Any]
    response: Any | None
    suppressed: bool = False


def _extract_error_payload(value: Any) -> Any | None:
    """Best-effort normalization for provider error payloads."""
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value

    response_body = getattr(value, "body", None)
    if response_body is not None:
        return _normalize_response_obj(response_body)

    error_payload = getattr(value, "error", None)
    if error_payload is not None:
        return _normalize_response_obj(error_payload)

    errors_payload = getattr(value, "errors", None)
    if errors_payload is not None:
        return _normalize_response_obj(errors_payload)

    json_method = getattr(value, "json", None)
    if callable(json_method):
        try:
            return _normalize_response_obj(json_method())
        except Exception:
            pass

    text_value = getattr(value, "text", None)
    if isinstance(text_value, str) and text_value.strip():
        return text_value

    content_value = getattr(value, "content", None)
    if content_value is not None:
        return _normalize_response_obj(content_value)

    normalized = _normalize_response_obj(value)
    if normalized == str(value):
        return None
    return normalized


def _extract_exception_details(exc: BaseException) -> tuple[dict[str, Any], Any | None]:
    """Return structured exception metadata plus any provider error payload."""
    error: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }

    for attr_name in ("status_code", "request_id"):
        value = getattr(exc, attr_name, None)
        if value is not None:
            error[attr_name] = value

    code = getattr(exc, "code", None)
    if code is not None:
        error["provider_code"] = code

    payload: Any | None = None
    for candidate in (
        getattr(exc, "body", None),
        getattr(exc, "error", None),
        getattr(exc, "errors", None),
        getattr(exc, "response", None),
    ):
        payload = _extract_error_payload(candidate)
        if payload is not None:
            break

    if payload is not None:
        error["raw_error_payload"] = payload

    return error, payload


def _serialized_failure_text(exc: BaseException, payload: Any | None) -> str:
    """Build a normalized text blob for retry/noise classification."""
    parts = [str(exc)]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(str(status_code))
    code = getattr(exc, "code", None)
    if code is not None:
        parts.append(str(code))
    if payload is not None:
        try:
            parts.append(json.dumps(_sanitize_for_json(payload), ensure_ascii=False))
        except Exception:
            parts.append(str(payload))
    return " ".join(part for part in parts if part).lower()


def is_suppressed_retryable_failure(
    exc: BaseException,
    *,
    response: Any | None = None,
) -> bool:
    """Return True for excluded retry-noise failures such as rate limits."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 429:
        return True

    code = getattr(exc, "code", None)
    if isinstance(code, int) and code == 429:
        return True
    if isinstance(code, str):
        normalized_code = code.strip().lower()
        if normalized_code in {
            "429",
            "resource_exhausted",
            "rate_limit",
            "rate_limit_exceeded",
            "too_many_requests",
        }:
            return True

    _, payload = _extract_exception_details(exc)
    payload_to_check = response if response is not None else payload
    message = _serialized_failure_text(exc, payload_to_check)
    markers = (
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "too_many_requests",
        "429",
    )
    return any(marker in message for marker in markers)


def classify_model_call_failure(
    exc: BaseException,
    *,
    response: Any | None = None,
    failure_kind: str | None = None,
) -> ModelCallFailure:
    """Normalize a failed model-call attempt for durable logging."""
    error, provider_payload = _extract_exception_details(exc)
    effective_response = response if response is not None else provider_payload

    normalized_failure_kind = failure_kind
    if normalized_failure_kind is None:
        if isinstance(exc, json.JSONDecodeError):
            normalized_failure_kind = "response_parse_error"
        elif getattr(exc, "status_code", None) is not None:
            normalized_failure_kind = "provider_http_error"
        elif getattr(exc, "code", None) is not None:
            normalized_failure_kind = "provider_sdk_error"
        else:
            normalized_failure_kind = "unknown_error"

    suppressed = is_suppressed_retryable_failure(exc, response=effective_response)
    return ModelCallFailure(
        failure_kind=normalized_failure_kind,
        error=error,
        response=effective_response,
        suppressed=suppressed,
    )


async def log_model_call_failure(
    model_logger: ModelCallLogger,
    *,
    agent: str,
    model: str,
    prompt: str,
    request_payload: Any | None,
    exception: BaseException,
    response: Any | None = None,
    screenshots: Sequence[tuple[str, bytes]] | None = None,
    metadata: dict[str, Any] | None = None,
    failure_kind: str | None = None,
) -> bool:
    """Persist one failed model-call attempt unless policy suppresses it."""
    failure = classify_model_call_failure(
        exception,
        response=response,
        failure_kind=failure_kind,
    )
    if failure.suppressed:
        return False

    await model_logger.log_outcome(
        agent=agent,
        model=model,
        prompt=prompt,
        request_payload=request_payload,
        response=failure.response,
        error=failure.error,
        screenshots=screenshots,
        metadata=metadata,
        outcome="failure",
        failure_kind=failure.failure_kind,
    )
    return True


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
        await self.log_outcome(
            agent=agent,
            model=model,
            prompt=prompt,
            request_payload=request_payload,
            response=response,
            screenshots=screenshots,
            metadata=metadata,
            outcome="success",
        )

    async def log_outcome(
        self,
        *,
        agent: str,
        model: str,
        prompt: str,
        request_payload: Any | None,
        response: Any,
        error: Any | None = None,
        screenshots: Sequence[tuple[str, bytes]] | None = None,
        metadata: dict[str, Any] | None = None,
        outcome: str = "success",
        failure_kind: str | None = None,
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
            "outcome": outcome,
            "prompt": sanitize_string(prompt),
            "request_payload": _redact_sensitive(_sanitize_for_json(request_payload)),
            "prompt_has_screenshot": bool(screenshot_entries),
            "attached_screenshots": screenshot_entries,
            "response": _redact_sensitive(
                _sanitize_for_json(_normalize_response_obj(response))
            ),
            "metadata": _redact_sensitive(_sanitize_for_json(metadata or {})),
        }
        if error is not None:
            entry["error"] = _redact_sensitive(
                _sanitize_for_json(_normalize_response_obj(error))
            )
        if failure_kind is not None:
            entry["failure_kind"] = failure_kind
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
