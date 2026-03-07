"""Provider-neutral Computer Use follow-up result models and builders."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.enhanced_types import ComputerToolTurn

from .common import encode_png_base64

_GOOGLE_PROVIDER_METADATA_KEYS = (
    "google_function_call_id",
    "google_function_call_name",
    "google_function_call_sequence",
    "google_correlation_mode",
    "google_function_call_fallback_id",
)


@dataclass
class ComputerUseActionResult:
    """Normalized result facts for one executed Computer Use action."""

    action_type: str
    status: str
    x: int | None = None
    y: int | None = None
    start_x: int | None = None
    start_y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    clipboard_text: str | None = None
    clipboard_truncated: bool | None = None
    clipboard_error: str | None = None
    error_message: str | None = None


@dataclass
class ComputerUseCallResult:
    """Provider-neutral follow-up facts for one model call."""

    call_id: str
    actions: list[ComputerUseActionResult] = field(default_factory=list)
    pending_safety_checks: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_safety_checks: list[dict[str, str]] = field(default_factory=list)
    requires_safety_acknowledgement: bool = False
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputerUseFollowUpBatch:
    """Shared follow-up state for one completed provider model turn."""

    calls: list[ComputerUseCallResult] = field(default_factory=list)
    screenshot_bytes: bytes = b""
    screenshot_base64: str = ""
    current_url: str = "desktop://"
    grounding_text: str | None = None
    reminder_text: str | None = None
    error_text: str | None = None


def build_action_result(turn: ComputerToolTurn) -> ComputerUseActionResult:
    """Normalize one executed turn into provider-neutral action facts."""
    return ComputerUseActionResult(
        action_type=str(turn.action_type or "").strip() or "unknown",
        status=str(turn.status or "").strip() or "pending",
        x=_coerce_int(turn.metadata.get("x")),
        y=_coerce_int(turn.metadata.get("y")),
        start_x=_coerce_int(turn.metadata.get("start_x")),
        start_y=_coerce_int(turn.metadata.get("start_y")),
        end_x=_coerce_int(turn.metadata.get("end_x")),
        end_y=_coerce_int(turn.metadata.get("end_y")),
        clipboard_text=_coerce_optional_str(turn.metadata.get("clipboard_text")),
        clipboard_truncated=_coerce_bool(turn.metadata.get("clipboard_truncated")),
        clipboard_error=_coerce_optional_str(turn.metadata.get("clipboard_error")),
        error_message=_coerce_optional_str(turn.error_message),
    )


def build_call_result(turns: Sequence[ComputerToolTurn]) -> ComputerUseCallResult:
    """Normalize one grouped provider call into provider-neutral follow-up facts."""
    if not turns:
        raise ValueError(
            "Expected at least one ComputerToolTurn to build a call result."
        )

    first_turn = turns[0]
    provider_metadata = {
        key: value
        for key in _GOOGLE_PROVIDER_METADATA_KEYS
        if (value := first_turn.metadata.get(key)) is not None
    }
    pending_safety_checks = _normalize_dict_list(first_turn.pending_safety_checks)
    acknowledged_safety_checks = _normalize_acknowledged_safety_checks(
        first_turn.metadata.get("acknowledged_safety_checks")
    )

    return ComputerUseCallResult(
        call_id=str(first_turn.call_id or "").strip(),
        actions=[build_action_result(turn) for turn in turns],
        pending_safety_checks=pending_safety_checks,
        acknowledged_safety_checks=acknowledged_safety_checks,
        requires_safety_acknowledgement=_requires_safety_acknowledgement(
            pending_safety_checks
        ),
        provider_metadata=provider_metadata,
    )


def build_follow_up_batch(
    call_groups: Sequence[Sequence[ComputerToolTurn]],
    *,
    screenshot_bytes: bytes,
    current_url: str,
    interaction_mode: str | None = None,
) -> ComputerUseFollowUpBatch:
    """Build the shared follow-up batch from completed call groups and one fresh capture."""
    calls = [build_call_result(group) for group in call_groups if group]
    normalized_mode = str(interaction_mode or "").strip().lower()
    return ComputerUseFollowUpBatch(
        calls=calls,
        screenshot_bytes=screenshot_bytes,
        screenshot_base64=encode_png_base64(screenshot_bytes),
        current_url=current_url or "desktop://",
        grounding_text=build_grounding_text(calls, current_url or "desktop://"),
        reminder_text=build_reminder_text(normalized_mode),
        error_text=extract_first_error_text(calls),
    )


def build_grounding_text(
    calls: Sequence[ComputerUseCallResult],
    current_url: str,
) -> str:
    """Render deterministic provider-neutral grounding text for OpenAI and Anthropic."""
    lines = [f"current_url={_format_value(current_url or 'desktop://')}"]
    for call in calls:
        for action_index, action in enumerate(call.actions, start=1):
            parts = [
                f"call_id={_format_value(call.call_id)}",
                f"action_index={action_index}",
                f"action={_format_value(action.action_type)}",
                f"status={_format_value(action.status)}",
            ]
            if action.x is not None:
                parts.append(f"x={action.x}")
            if action.y is not None:
                parts.append(f"y={action.y}")
            if action.start_x is not None:
                parts.append(f"start_x={action.start_x}")
            if action.start_y is not None:
                parts.append(f"start_y={action.start_y}")
            if action.end_x is not None:
                parts.append(f"end_x={action.end_x}")
            if action.end_y is not None:
                parts.append(f"end_y={action.end_y}")
            if action.clipboard_text is not None:
                parts.append(f"clipboard_text={_format_value(action.clipboard_text)}")
            if action.clipboard_truncated is not None:
                parts.append(
                    f"clipboard_truncated={str(action.clipboard_truncated).lower()}"
                )
            if action.clipboard_error is not None:
                parts.append(f"clipboard_error={_format_value(action.clipboard_error)}")
            if action.error_message is not None:
                parts.append(f"error={_format_value(action.error_message)}")
            if action_index == 1 and call.requires_safety_acknowledgement:
                parts.append("safety_acknowledgement=true")
            lines.append(" ".join(parts))
    return "\n".join(lines)


def build_reminder_text(interaction_mode: str) -> str | None:
    """Return the shared follow-up reminder text for the current interaction mode."""
    if interaction_mode == "observe_only":
        return (
            "Reminder: Observe-only mode is active. Do not interact with the UI. "
            "Do not call click_at, type_text_at, key_combination, or drag actions. "
            "Only inspect and report findings."
        )
    if interaction_mode:
        return (
            "Reminder: Execute mode is active. Complete the requested interaction "
            "directly without asking for confirmation."
        )
    return None


def extract_first_error_text(calls: Sequence[ComputerUseCallResult]) -> str | None:
    """Return the first shared execution error summary for a follow-up batch."""
    for call in calls:
        for action in call.actions:
            if action.status != "executed" and action.error_message:
                return f"Execution error: {action.error_message}"
    return None


def _normalize_dict_list(value: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _normalize_acknowledged_safety_checks(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if not isinstance(item, dict):
            continue
        payload: dict[str, str] = {}
        for key in ("id", "code", "message"):
            raw = item.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                payload[key] = text
        if payload:
            normalized.append(payload)
    return normalized


def _requires_safety_acknowledgement(checks: Sequence[dict[str, Any]]) -> bool:
    for check in checks:
        if not isinstance(check, dict):
            continue
        decision = str(check.get("decision") or check.get("code") or "").strip()
        if decision.lower() == "require_confirmation":
            return True
    return False


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _format_value(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


__all__ = [
    "ComputerUseActionResult",
    "ComputerUseCallResult",
    "ComputerUseFollowUpBatch",
    "build_action_result",
    "build_call_result",
    "build_follow_up_batch",
    "build_grounding_text",
    "build_reminder_text",
    "extract_first_error_text",
]
