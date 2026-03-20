"""Shared helper functions for Computer Use orchestration."""

from __future__ import annotations

import base64
import re
from types import SimpleNamespace
from typing import Any, cast

from src.core.enhanced_types import ComputerToolTurn

from .types import ComputerUseExecutionError, GoogleFunctionCallEnvelope


def encode_png_base64(data: bytes) -> str:
    """Encode PNG bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def normalize_response(response: Any) -> dict[str, Any]:
    """Normalize provider response objects into standard dictionaries."""
    if response is None:
        return {}

    payload: dict[str, Any] | None = None
    if isinstance(response, dict):
        payload = dict(response)
    elif hasattr(response, "model_dump"):
        try:
            payload = cast(dict[str, Any], response.model_dump(warnings="none"))
        except TypeError:
            payload = cast(dict[str, Any], response.model_dump())
    elif hasattr(response, "to_dict"):
        try:
            payload = cast(dict[str, Any], response.to_dict())
        except Exception:
            pass
    if payload is None:
        raise ComputerUseExecutionError(f"Unsupported response type: {type(response)}")

    # Some Google SDK response objects keep candidate payloads on attributes even when
    # model_dump()/to_dict() omits them. Preserve those fields so downstream parsers can
    # continue handling both object-backed fixtures and SDK objects consistently.
    if not isinstance(response, dict):
        for key in (
            "id",
            "status",
            "output",
            "outputs",
            "content",
            "candidates",
            "prompt_feedback",
        ):
            if not hasattr(response, key):
                continue
            value = getattr(response, key)
            if key not in payload or payload[key] in (None, [], {}):
                payload[key] = value

    return payload


def normalize_key_sequence(key: str) -> str:
    """
    Convert Computer Use key strings to Automation-compatible sequences.

    The model commonly emits uppercase tokens (e.g., "ENTER") or modifier
    combinations like "CTRL+ENTER". Automation expects specific casing, so we
    normalize each segment before execution.
    """
    if not key:
        return key

    def normalize_single(token: str) -> str:
        mapping = {
            "ENTER": "Enter",
            "RETURN": "Enter",
            "ESC": "Escape",
            "ESCAPE": "Escape",
            "TAB": "Tab",
            "SPACE": "Space",
            "BACKSPACE": "Backspace",
            "DELETE": "Delete",
            "DEL": "Delete",
            "HOME": "Home",
            "END": "End",
            "PAGEUP": "PageUp",
            "PAGEDOWN": "PageDown",
            "ARROWUP": "ArrowUp",
            "ARROWDOWN": "ArrowDown",
            "ARROWLEFT": "ArrowLeft",
            "ARROWRIGHT": "ArrowRight",
            "LEFT": "ArrowLeft",
            "RIGHT": "ArrowRight",
            "UP": "ArrowUp",
            "DOWN": "ArrowDown",
            "CTRL": "Control",
            "CONTROL": "Control",
            "ALT": "Alt",
            "OPTION": "Alt",
            "SHIFT": "Shift",
            "META": "Meta",
            "CMD": "Meta",
            "COMMAND": "Meta",
            "CAPSLOCK": "CapsLock",
            "NUMLOCK": "NumLock",
            "SCROLLLOCK": "ScrollLock",
        }
        token_upper = token.upper()
        if token_upper in mapping:
            return mapping[token_upper]

        if token_upper.startswith("F") and token_upper[1:].isdigit():
            return token_upper

        if len(token) == 1:
            return token

        return token.capitalize()

    if "+" in key:
        parts = [part.strip() for part in key.split("+") if part.strip()]
        normalized_parts = [normalize_single(part) for part in parts]
        return "+".join(normalized_parts)

    return normalize_single(key.strip())


def extract_computer_calls(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract OpenAI computer_call items from a response."""
    return [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "computer_call"
    ]


def extract_computer_call_actions(call: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract executable actions from an OpenAI computer_call item.

    GPT-5.4 emits batched `actions[]` per `computer_call`. A legacy singular
    `action` payload is still accepted here so tests and fixtures can evolve
    without breaking the internal execution contract.
    """
    actions = call.get("actions")
    if isinstance(actions, list):
        normalized = [item for item in actions if isinstance(item, dict)]
        if normalized:
            return normalized

    legacy_action = call.get("action")
    if isinstance(legacy_action, dict) and legacy_action:
        return [legacy_action]

    # A screenshot-only turn may omit executable actions but still expects
    # the harness to return the latest screenshot state.
    return [{"type": "screenshot"}]


def extract_google_function_calls(response_obj: Any) -> list[Any]:
    """Extract function_call parts from a Google response object."""
    return [
        envelope.function_call
        for envelope in extract_google_function_call_envelopes(response_obj)
    ]


def extract_google_function_call_envelopes(
    response_obj: Any,
    *,
    candidate_index: int = 0,
) -> list[GoogleFunctionCallEnvelope]:
    """Extract ordered function_call parts from a selected Google candidate."""
    envelopes: list[GoogleFunctionCallEnvelope] = []
    response_dict = (
        response_obj
        if isinstance(response_obj, dict)
        else normalize_response(response_obj)
    )

    outputs = response_dict.get("outputs") or []
    if isinstance(outputs, list):
        sequence = 0
        for output_index, output in enumerate(outputs):
            if not isinstance(output, dict):
                continue
            if str(output.get("type") or "").strip().lower() != "function_call":
                continue
            raw_arguments = output.get("arguments") or output.get("args") or {}
            if not isinstance(raw_arguments, dict):
                raw_arguments = {}
            sequence += 1
            envelopes.append(
                GoogleFunctionCallEnvelope(
                    function_call=SimpleNamespace(
                        name=output.get("name"),
                        args=raw_arguments,
                        id=output.get("id"),
                    ),
                    sequence=sequence,
                    candidate_index=0,
                    part_index=output_index,
                )
            )
        if envelopes:
            return envelopes

    candidates = response_dict.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return envelopes

    selected_index = candidate_index
    if selected_index < 0 or selected_index >= len(candidates):
        selected_index = 0

    candidate = candidates[selected_index]
    if isinstance(candidate, dict):
        content = candidate.get("content")
    else:
        content = getattr(candidate, "content", None)
    if not content:
        return envelopes

    sequence = 0
    if isinstance(content, dict):
        parts = content.get("parts") or []
    else:
        parts = getattr(content, "parts", []) or []
    for part_index, part in enumerate(parts):
        if isinstance(part, dict):
            func_call = part.get("function_call") or part.get("functionCall")
        else:
            func_call = getattr(part, "function_call", None)
        if not func_call:
            continue
        if isinstance(func_call, dict):
            raw_arguments = func_call.get("args") or func_call.get("arguments") or {}
            if not isinstance(raw_arguments, dict):
                raw_arguments = {}
            func_call = SimpleNamespace(
                name=func_call.get("name"),
                args=raw_arguments,
                id=func_call.get("id"),
            )
        sequence += 1
        envelopes.append(
            GoogleFunctionCallEnvelope(
                function_call=func_call,
                sequence=sequence,
                candidate_index=selected_index,
                part_index=part_index,
            )
        )
    return envelopes


def extract_google_computer_calls(
    response_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract function/call items from a Google response."""
    calls: list[dict[str, Any]] = []
    outputs = response_dict.get("outputs") or []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if output.get("type") != "function_call":
            continue
        arguments = output.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            {
                "id": output.get("id") or output.get("name"),
                "call_id": output.get("id") or output.get("name"),
                "action": {
                    "type": output.get("name"),
                    **arguments,
                },
            }
        )
    if calls:
        return calls

    output_items = response_dict.get("output", [])
    for item in output_items:
        if item.get("type") in {"function_call", "computer_call"}:
            calls.append(item)
    if calls:
        return calls

    candidates = response_dict.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            fn_call = part.get("functionCall") or part.get("function_call")
            if not fn_call:
                continue
            calls.append(
                {
                    "id": fn_call.get("id") or fn_call.get("name"),
                    "call_id": fn_call.get("id") or fn_call.get("name"),
                    "action": {
                        "type": fn_call.get("name"),
                        **(fn_call.get("args") or {}),
                    },
                }
            )
    return calls


def extract_anthropic_computer_calls(
    response_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract Anthropic computer tool-use calls from a response."""
    calls: list[dict[str, Any]] = []
    for item in response_dict.get("content", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use":
            continue
        tool_name = str(item.get("name") or "").strip().lower()
        if tool_name and tool_name != "computer":
            continue
        action_payload = item.get("input")
        if not isinstance(action_payload, dict):
            action_payload = {}
        calls.append(
            {
                "id": item.get("id") or "",
                "name": item.get("name") or "computer",
                "action": action_payload,
            }
        )
    return calls


def extract_assistant_text(response_dict: dict[str, Any]) -> str | None:
    """Extract assistant text output from a response."""
    texts: list[str] = []

    for output in response_dict.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        if output.get("type") != "text":
            continue
        text = output.get("text")
        if isinstance(text, str) and text:
            texts.append(text)

    messages = [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "message"
    ]
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    if not texts:
        candidates = response_dict.get("candidates") or []
        for candidate in candidates:
            if isinstance(candidate, dict):
                content = candidate.get("content") or {}
            else:
                content = getattr(candidate, "content", None) or {}
            if isinstance(content, dict):
                parts = content.get("parts") or []
            else:
                parts = getattr(content, "parts", []) or []
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("output_text")
                else:
                    text = getattr(part, "text", None) or getattr(
                        part, "output_text", None
                    )
                if text:
                    texts.append(text)
    if not texts:
        for content in response_dict.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") != "text":
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return _select_preferred_assistant_text(texts)


def _select_preferred_assistant_text(texts: list[str]) -> str | None:
    """Prefer the final meaningful assistant text block over prefixed fragments."""
    candidates = [
        text.strip() for text in texts if isinstance(text, str) and text.strip()
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for text in reversed(candidates):
        if _looks_like_json_payload(text) or _looks_like_natural_language(text):
            return text

    return candidates[-1]


def _looks_like_json_payload(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_natural_language(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.lower() in {"ok", "done", "success", "completed", "failed", "error"}:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped)
    if not words:
        return False
    if len(words) >= 2:
        return True
    if any(char.isspace() for char in stripped):
        return True
    return stripped.endswith((".", "!", "?"))


def _inject_context_metadata(turn: ComputerToolTurn, metadata: dict[str, Any]) -> None:
    """Copy high-level context into the turn metadata for observability."""
    if not isinstance(turn.metadata, dict):
        return
    for key in ("step_number", "test_plan_name", "test_case_name", "target", "value"):
        if metadata.get(key) is not None:
            turn.metadata[key] = metadata[key]
    if metadata.get("allow_safety_auto_approve") is not None:
        turn.metadata["allow_safety_auto_approve"] = bool(
            metadata["allow_safety_auto_approve"]
        )
    if metadata.get("safety_identifier") is not None:
        turn.metadata["safety_identifier"] = metadata["safety_identifier"]
    if metadata.get("interaction_mode") is not None:
        turn.metadata["interaction_mode"] = metadata["interaction_mode"]


def normalize_coordinates(
    x: float | None,
    y: float | None,
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """Normalize coordinates into the viewport bounds."""

    def _clamp(value: float, maximum: int) -> int:
        return max(0, min(int(round(value)), max(0, maximum - 1)))

    if x is None or y is None:
        return viewport_width // 2, viewport_height // 2
    return _clamp(x, viewport_width), _clamp(y, viewport_height)


def denormalize_coordinates(
    x: float | None,
    y: float | None,
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """Convert normalized 0-999 coordinates to absolute pixels."""

    def _clamp(value: float, maximum: int) -> int:
        return max(0, min(int(round(value)), max(0, maximum - 1)))

    if x is None or y is None:
        return viewport_width // 2, viewport_height // 2
    return _clamp(float(x) * viewport_width / 999.0, viewport_width), _clamp(
        float(y) * viewport_height / 999.0, viewport_height
    )


__all__ = [
    "_inject_context_metadata",
    "denormalize_coordinates",
    "encode_png_base64",
    "extract_anthropic_computer_calls",
    "extract_assistant_text",
    "extract_computer_call_actions",
    "extract_computer_calls",
    "extract_google_computer_calls",
    "extract_google_function_call_envelopes",
    "extract_google_function_calls",
    "normalize_coordinates",
    "normalize_key_sequence",
    "normalize_response",
]
