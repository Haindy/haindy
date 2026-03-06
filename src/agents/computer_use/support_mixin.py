# mypy: disable-error-code=misc
"""Shared support methods for Computer Use sessions."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from src.core.enhanced_types import ComputerToolTurn, SafetyEvent
from src.runtime.environment import (
    RuntimeEnvironmentName,
    normalize_runtime_environment_name,
    runtime_environment_spec,
)

from .common import (
    _inject_context_metadata,
    denormalize_coordinates,
    normalize_coordinates,
)
from .types import ComputerUseExecutionError, ComputerUseSessionResult

logger = logging.getLogger("src.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class ComputerUseSupportMixin:
    """Support helpers shared across provider implementations."""

    def _update_loop_history(
        self: _ComputerUseSession,
        turn: ComputerToolTurn,
        history: deque[tuple[tuple[str, ...], str]],
        window: int,
    ) -> dict[str, Any] | None:
        """Track repeated Computer Use turns and detect loops."""
        if window < 2:
            return None
        if turn.status != "executed":
            history.clear()
            return None

        signature = self._compute_turn_signature(turn)
        screenshot_hash = self._hash_base64(turn.metadata.get("screenshot_base64"))
        if not signature or not screenshot_hash:
            history.clear()
            return None

        history.append((signature, screenshot_hash))
        if len(history) < window:
            return None

        if all(sig == signature and hash_ == screenshot_hash for sig, hash_ in history):
            action_label = signature[0] if signature else "unknown"
            message = (
                f"Computer Use loop detected: action '{action_label}' "
                f"repeated {window} times without visible change."
            )
            return {
                "message": message,
                "signature": signature,
                "screenshot_hash": screenshot_hash,
                "loop_window": window,
            }

        if all(hash_ == screenshot_hash for _, hash_ in history):
            action_label = signature[0] if signature else "unknown"
            message = (
                f"Computer Use loop detected: screen unchanged after "
                f"{window} consecutive '{action_label}' actions."
            )
            return {
                "message": message,
                "signature": signature,
                "screenshot_hash": screenshot_hash,
                "loop_window": window,
            }

        return None

    def _compute_turn_signature(
        self: _ComputerUseSession, turn: ComputerToolTurn
    ) -> tuple[str, ...] | None:
        """Build a lightweight signature representing the Computer Use turn."""
        action_type_raw = turn.action_type or turn.parameters.get("type")
        if not action_type_raw:
            return None
        action_type = str(action_type_raw).strip().lower()
        if not action_type:
            return None

        params = turn.parameters or {}
        signature: list[str] = [action_type]

        def append_coord(label: str, x_value: Any, y_value: Any) -> None:
            coord = self._format_coordinate(x_value, y_value)
            if coord:
                signature.append(f"{label}:{coord}")

        raw_coord = params.get("coordinate") or params.get("coordinates")
        if isinstance(raw_coord, (list, tuple)) and len(raw_coord) == 2:
            append_coord("xy", raw_coord[0], raw_coord[1])
        else:
            append_coord("xy", params.get("x"), params.get("y"))
        append_coord(
            "start",
            params.get("start_x") or params.get("from_x"),
            params.get("start_y") or params.get("from_y"),
        )
        append_coord(
            "end",
            params.get("end_x")
            or params.get("to_x")
            or params.get("destination_x")
            or params.get("target_x"),
            params.get("end_y")
            or params.get("to_y")
            or params.get("destination_y")
            or params.get("target_y"),
        )

        if action_type == "type":
            text_value = params.get("text") or params.get("value") or ""
            signature.append(str(text_value).strip().lower()[:64])
        elif action_type in {"key_press", "keypress", "key_combination"}:
            key_value = params.get("key") or params.get("keys")
            if isinstance(key_value, list):
                key_value = "+".join(str(part) for part in key_value)
            signature.append(str(key_value).strip().lower())

        if "scroll_x" in params or "scroll_y" in params:
            signature.append(
                f"scroll:{params.get('scroll_x', 0)}:{params.get('scroll_y', 0)}"
            )

        return tuple(component for component in signature if component)

    @staticmethod
    def _format_coordinate(x_value: Any, y_value: Any) -> str | None:
        """Format coordinate pairs for signature comparison."""
        if x_value is None or y_value is None:
            return None
        try:
            x_float = round(float(x_value), 2)
            y_float = round(float(y_value), 2)
            return f"{x_float}:{y_float}"
        except (TypeError, ValueError):
            return f"{x_value}:{y_value}"

    @staticmethod
    def _hash_base64(value: str | None) -> str | None:
        """Hash base64 strings for quick comparison without storing raw data."""
        if not value or not isinstance(value, str):
            return None
        try:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        except Exception:
            return None

    async def _ensure_automation_driver_ready(
        self: _ComputerUseSession,
    ) -> None:
        """Ensure the automation_driver session is started before execution."""
        await self._automation_driver.start()

    async def invalidate_cache(
        self: _ComputerUseSession, cache_label: str, cache_action: str
    ) -> None:
        """Invalidate a cached coordinate for the current resolution."""
        logger.info("ComputerUseSession: invalidating coordinate cache after failure")
        try:
            resolution = await self._automation_driver.get_viewport_size()
            self._coordinate_cache.invalidate(cache_label, cache_action, resolution)
        except Exception:
            logger.debug(
                "Failed to invalidate coordinate cache entry",
                exc_info=True,
                extra={"cache_label": cache_label, "cache_action": cache_action},
            )

    async def _post_action_wait(self: _ComputerUseSession) -> None:
        """Wait for the configured stabilization interval."""
        wait_ms = self._settings.actions_computer_tool_stabilization_wait_ms
        if wait_ms > 0:
            await self._automation_driver.wait(wait_ms)

    async def _maybe_get_current_url(
        self: _ComputerUseSession,
    ) -> str | None:
        """Attempt to retrieve the current page URL from the automation driver."""
        get_url = getattr(self._automation_driver, "get_page_url", None)
        if callable(get_url):
            try:
                value = await get_url()
                if value is None:
                    return None
                return str(value)
            except Exception:
                logger.debug(
                    "Failed to retrieve current URL from automation driver",
                    exc_info=True,
                )
        return None

    @staticmethod
    def _canonicalize_action_type(action_type: str | None) -> str | None:
        """Normalize provider action types to the automation driver's expectations."""
        if not action_type:
            return action_type

        normalized = action_type.replace("-", "_").lower()
        alias_map = {
            "left_click": "click",
            "middle_click": "click",
            "key_press": "keypress",
            "keypress": "keypress",
            "press_key": "keypress",
            "press": "keypress",
            "key": "keypress",
            "type_text": "type",
            "input": "type",
            "doubleclick": "double_click",
            "rightclick": "right_click",
            "scroll_to_element": "scroll",
            "scroll_vertical": "scroll",
            "pointer_move": "move",
            "mouse_move": "move",
            "hover": "move",
            "pointer_drag": "drag",
            "left_click_drag": "drag",
            "drag_and_drop": "drag",
            "dragdrop": "drag",
            "mouse_drag": "drag",
        }
        return alias_map.get(normalized, normalized)

    def _resolve_key_sequence(
        self: _ComputerUseSession,
        action: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[str]:
        """Resolve keyboard keys to press from the tool payload or fallback metadata."""
        keys = action.get("keys")
        if isinstance(keys, list) and keys:
            return [str(key) for key in keys if isinstance(key, str) and key.strip()]
        if isinstance(keys, str) and keys.strip():
            return [keys.strip()]

        candidate_fields = [
            action.get("key"),
            action.get("value"),
            action.get("text"),
            metadata.get("value"),
        ]
        for candidate in candidate_fields:
            if isinstance(candidate, str) and candidate.strip():
                return [candidate.strip()]

        combination = action.get("key_combination") or action.get("shortcut")
        if isinstance(combination, str) and combination.strip():
            return [combination.strip()]

        return []

    @staticmethod
    def _looks_like_confirmation_request(message: str) -> bool:
        """Detect if the assistant is asking for permission to continue."""
        text = message.strip().lower()
        if not text.endswith("?"):
            return False

        confirmation_markers = [
            "should i",
            "should we",
            "would you like",
            "want me to",
            "do you want",
            "shall i",
            "ok to",
            "is it okay",
            "proceed",
            "continue",
            "ready to",
        ]
        return any(marker in text for marker in confirmation_markers)

    @staticmethod
    def _apply_interaction_mode_guidance(goal: str, metadata: dict[str, Any]) -> str:
        """Inject explicit interaction-mode instructions into provider prompts."""
        interaction_mode = str(metadata.get("interaction_mode") or "").strip().lower()
        if interaction_mode != "observe_only":
            return goal

        guidance = (
            "CRITICAL OBSERVE-ONLY MODE:\n"
            "- Do NOT interact with the UI.\n"
            "- Do NOT call click_at, type_text_at, key_combination, drag_and_drop, or any mutating action.\n"
            "- Use only observation actions (screenshot/wait/scroll) if needed.\n"
            "- When verification is complete, respond with findings and stop."
        )
        if guidance in goal:
            return goal
        return f"{goal}\n\n{guidance}"

    async def _build_confirmation_request(
        self: _ComputerUseSession,
        previous_response_id: str | None,
        metadata: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Construct a follow-up request that confirms execution should proceed."""
        confirmation_text = (
            "Yes, proceed. Execute the requested action now without asking for "
            "additional confirmation."
        )
        target_text = metadata.get("target")
        if target_text:
            confirmation_text += f" Focus on: {target_text}."

        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "previous_response_id": previous_response_id,
            "tools": [{"type": "computer"}],
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": confirmation_text}],
                }
            ],
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        return payload

    def _is_action_allowed(
        self: _ComputerUseSession, action_type: str | None
    ) -> tuple[bool, str | None]:
        """Determine if the requested action type is permitted in the current mode."""
        if not action_type:
            return False, "Computer Use response omitted action type information."
        if self._allowed_actions is None:
            return True, None
        if action_type in self._allowed_actions:
            return True, None
        return False, f"Action '{action_type}' is not permitted in observe-only mode."

    def _should_abort_on_safety(
        self: _ComputerUseSession,
        turn: ComputerToolTurn,
        result: ComputerUseSessionResult,
    ) -> bool:
        """Handle pending safety checks based on configured policy."""
        if not turn.pending_safety_checks:
            return False

        policy = self._settings.cu_safety_policy
        fail_fast = bool(
            getattr(self._settings, "actions_computer_tool_fail_fast_on_safety", True)
        )
        allow_auto_approve_override = bool(
            turn.metadata.get("allow_safety_auto_approve")
        )
        if policy == "auto_approve" and (not fail_fast or allow_auto_approve_override):
            return False

        safety_payload = turn.pending_safety_checks[0]
        safety_event = SafetyEvent(
            call_id=turn.call_id,
            code=safety_payload.get("code", "unknown"),
            message=safety_payload.get("message", ""),
            acknowledged=False,
            response_id=turn.response_id,
        )
        result.safety_events.append(safety_event)
        turn.status = "failed"
        if fail_fast:
            turn.error_message = (
                "Safety check triggered; action halted (fail-fast safety setting)."
            )
            failure_code = "safety_fail_fast"
        else:
            turn.error_message = (
                "Safety check triggered; action halted (policy enforcement)."
            )
            failure_code = "safety_policy"
        turn.metadata.update(
            {
                "safety_policy": policy,
                "fail_fast_on_safety": fail_fast,
            }
        )
        result.actions.append(turn)
        result.terminal_status = "failed"
        result.terminal_failure_reason = turn.error_message
        result.terminal_failure_code = failure_code
        result.final_output = turn.error_message
        logger.warning(
            "Computer Use safety check triggered; aborting action execution",
            extra={
                "call_id": turn.call_id,
                "code": safety_event.code,
                "safety_message": safety_event.message,
                "policy": policy,
                "fail_fast": fail_fast,
            },
        )
        return True

    def _append_terminal_failure_turn(
        self: _ComputerUseSession,
        *,
        result: ComputerUseSessionResult,
        metadata: dict[str, Any],
        reason: str,
        code: str,
        response_id: str | None,
        parameters: dict[str, Any] | None = None,
        metadata_updates: dict[str, Any] | None = None,
        call_id_prefix: str = "terminal",
    ) -> ComputerToolTurn:
        """Append a terminal failed system notice turn and mark the result as failed."""
        params = {"type": "system_notice", "reason": code}
        if parameters:
            params.update(parameters)
        failure_turn = ComputerToolTurn(
            call_id=f"{call_id_prefix}-{uuid4()}",
            action_type="system_notice",
            parameters=params,
            response_id=response_id,
        )
        failure_turn.status = "failed"
        failure_turn.error_message = reason
        _inject_context_metadata(failure_turn, metadata)
        if metadata_updates:
            failure_turn.metadata.update(metadata_updates)
        result.actions.append(failure_turn)
        result.final_output = reason
        result.terminal_status = "failed"
        result.terminal_failure_reason = reason
        result.terminal_failure_code = code
        return failure_turn

    @staticmethod
    def _mark_terminal_failure(
        *,
        result: ComputerUseSessionResult,
        reason: str,
        code: str,
    ) -> None:
        """Mark a session result as terminally failed without appending a turn."""
        result.final_output = reason
        result.terminal_status = "failed"
        result.terminal_failure_reason = reason
        result.terminal_failure_code = code

    @staticmethod
    def _is_observe_only_policy_violation(turn: ComputerToolTurn) -> bool:
        """Return True when a failed turn is caused by observe-only enforcement."""
        if turn.status != "failed":
            return False
        metadata = turn.metadata if isinstance(turn.metadata, dict) else {}
        return str(metadata.get("policy") or "").strip().lower() == "observe_only"

    async def _enforce_domain_policy(
        self: _ComputerUseSession, action_type: str | None
    ) -> None:
        """Prevent interactions with domains outside defined allow/block lists."""
        if (not self._allowed_domains and not self._blocked_domains) or not action_type:
            return
        if action_type not in self._stateful_actions:
            return

        current_url = await self._maybe_get_current_url()
        if not current_url:
            return

        hostname = urlparse(current_url).hostname or ""
        if not hostname:
            return

        normalized_host = hostname.lower()
        if self._allowed_domains and not any(
            self._domain_matches(normalized_host, domain)
            for domain in self._allowed_domains
        ):
            raise ComputerUseExecutionError(
                f"Current domain '{hostname}' is not in the allowlist."
            )

        if self._blocked_domains and any(
            self._domain_matches(normalized_host, domain)
            for domain in self._blocked_domains
        ):
            raise ComputerUseExecutionError(
                f"Current domain '{hostname}' is blocked by policy."
            )

    @classmethod
    def _disallowed_action_reason(
        cls: type[_ComputerUseSession], action_type: str | None, environment: str
    ) -> str | None:
        if not action_type:
            return None
        env_mode = cls._normalize_environment_name(environment)
        normalized = action_type.lower().strip()
        if normalized in cls._DISALLOWED_NAVIGATION_ACTIONS and env_mode in {
            "desktop",
            "browser",
        }:
            return (
                f"{normalized} is disabled; use direct UI interactions in the "
                "existing window instead."
            )
        return None

    def _resolve_coordinates(
        self: _ComputerUseSession,
        action: dict[str, Any],
        viewport_width: int,
        viewport_height: int,
        *,
        prefix: str = "",
        normalized: bool = False,
    ) -> tuple[int, int]:
        x = action.get(f"{prefix}x")
        y = action.get(f"{prefix}y")

        if prefix == "start_":
            if x is None:
                x = action.get("from_x")
            if y is None:
                y = action.get("from_y")
            if x is None:
                x = action.get("x")
            if y is None:
                y = action.get("y")
        if prefix == "end_":
            if x is None:
                x = action.get("to_x")
            if y is None:
                y = action.get("to_y")

        if normalized:
            return denormalize_coordinates(x, y, viewport_width, viewport_height)
        return normalize_coordinates(x, y, viewport_width, viewport_height)

    @staticmethod
    def _action_matches_cache(action_type: str | None, cache_action: str) -> bool:
        if not action_type:
            return False
        normalized_action = action_type.lower().strip()
        normalized_cache = (cache_action or "").lower().strip()
        if normalized_action == normalized_cache:
            return True
        if normalized_cache in {"click", "click_at"}:
            return normalized_action in {
                "click",
                "click_at",
                "move_mouse_and_click",
                "type_text_at",
                "hover_at",
            }
        return False

    @staticmethod
    def _extract_scroll_direction(
        action_type: str, action: dict[str, Any]
    ) -> str | None:
        normalized = (action_type or "").lower()
        if normalized in {"scroll_document", "scroll_at", "scroll_window"}:
            direction = (action.get("direction") or "").lower()
            return direction if direction in {"up", "down", "left", "right"} else None
        if normalized == "scroll":
            try:
                scroll_x = int(action.get("scroll_x", 0))
                scroll_y = int(action.get("scroll_y", 0))
            except Exception:
                return None
            if abs(scroll_y) >= abs(scroll_x):
                if scroll_y > 0:
                    return "down"
                if scroll_y < 0:
                    return "up"
                return None
            if scroll_x > 0:
                return "right"
            if scroll_x < 0:
                return "left"
        return None

    @staticmethod
    def _should_capture_clipboard_after_key_combo(keys: Any) -> bool:
        """Return True when the key combo likely performed a copy action."""
        try:
            if isinstance(keys, str):
                keys_str = keys
            else:
                keys_str = "+".join([str(k) for k in keys])
        except Exception:
            keys_str = str(keys)
        normalized = keys_str.lower().replace(" ", "")
        if normalized in {"ctrl+c", "control+c"}:
            return True
        if (
            ("ctrl+" in normalized) or ("control+" in normalized)
        ) and normalized.endswith("+c"):
            return True
        return False

    async def _maybe_read_clipboard(
        self: _ComputerUseSession, turn: ComputerToolTurn
    ) -> str | None:
        reader = getattr(self._automation_driver, "read_clipboard", None)
        if not callable(reader):
            return None
        try:
            clipboard_text = await reader()
        except Exception as exc:
            turn.metadata["clipboard_error"] = str(exc)
            logger.debug("Failed to read clipboard", exc_info=True)
            return None
        if not clipboard_text:
            return ""
        max_chars = 6000
        truncated = len(clipboard_text) > max_chars
        turn.metadata["clipboard_truncated"] = truncated
        return str(clipboard_text)[:max_chars]

    async def _navigate_via_address_bar(self: _ComputerUseSession, url: str) -> None:
        await self._automation_driver.press_key("ctrl+l")
        await self._automation_driver.type_text(url)
        await self._automation_driver.press_key("enter")

    def _save_turn_screenshot(
        self: _ComputerUseSession,
        screenshot_bytes: bytes,
        suffix: str,
        step_number: int | None,
    ) -> str | None:
        """Persist a screenshot for observability if a debug logger is available."""
        if not self._debug_logger:
            return None
        name = f"computer_use_turn_{suffix}"
        screenshot_path = self._debug_logger.save_screenshot(
            screenshot_bytes, name=name, step_number=step_number
        )
        return cast(str | None, screenshot_path)

    def _wrap_goal_for_google(
        self: _ComputerUseSession, goal: str, env_mode: str
    ) -> str:
        """Wrap the goal with context for Google CU per environment."""
        requires_json = self._goal_requires_json(goal)
        completion_instruction = (
            "When the task is complete, STOP issuing function calls and reply ONLY "
            "with the requested JSON (no prose).\n\n"
            if requires_json
            else "When the task is complete, stop issuing function calls and reply "
            "with a short confirmation of what you accomplished.\n\n"
        )
        if env_mode == "browser":
            browser_context = (
                "IMPORTANT: You are controlling a single Firefox browser window on Linux. "
                "Stay inside the browser tab that is already open. Do NOT use alt+tab or open other desktop apps. "
                "Use only these actions to interact with the page:\n"
                "- click_at for buttons, inputs, and toggles (set button='right' for context clicks or click_count=2 for double clicks)\n"
                "- type_text_at after clicking into inputs (set press_enter=true only when instructed)\n"
                "- scroll_at / scroll_document to move within the page\n"
                "- key_combination for shortcuts like ctrl+l, ctrl+c, etc.\n"
                "- read_clipboard to read copied text/URLs after using Copy actions\n"
                "- hover_at or drag_and_drop when explicitly needed\n\n"
                "Never issue browser navigation commands like navigate/open_web_browser/search/go_back/go_forward; instead, control the existing tab visually. "
                + completion_instruction
                + "YOUR TASK: "
            )
            return browser_context + goal

        if env_mode == "mobile_adb":
            mobile_context = (
                "IMPORTANT: You are controlling an Android mobile app through ADB-backed screenshots. "
                "Treat coordinates as mobile screen positions. Avoid desktop assumptions (dock, alt-tab, windows).\n\n"
                "SCREEN DETAILS:\n"
                "- Screenshot orientation and resolution may vary per device; always rely on the provided screenshot.\n"
                "- Prefer tap and type interactions; use scroll_document/scroll_at for vertical movement.\n"
                "- Do not use open_web_browser/search/go_back/go_forward desktop-browser actions unless explicitly required.\n\n"
                "AVAILABLE ACTIONS:\n"
                "- click_at: Tap at screen coordinates\n"
                "- type_text_at: Tap and type text into focused field\n"
                "- key_combination: Android key events - valid values include: 'home' (go to home screen), 'back' (navigate back), 'app_switch' (open recent apps switcher), 'enter', 'delete'\n"
                "- scroll_at / scroll_document: Scroll app content\n"
                "- drag_and_drop: Swipe/drag gestures when necessary\n\n"
                "APP SWITCHING ON ANDROID:\n"
                "- To switch to a recently used app: use key_combination: 'app_switch' to open the recents overlay, then tap the app card you want.\n"
                "- Do NOT use swipe/drag gestures from the bottom edge to open recents - use key_combination: 'app_switch' instead.\n"
                "- To go to the home screen: use key_combination: 'home'.\n\n"
                "TEXT INPUT ON ANDROID:\n"
                "- To replace existing text in a field: use type_text_at directly - it automatically selects all existing content before typing.\n"
                "- Do NOT use key_combination with ctrl+a or any other desktop shortcut to select or clear text; these do not work on Android.\n"
                "- For type_text_at, set press_enter=true ONLY when the task explicitly says to submit the form or press Enter. Do NOT set press_enter=true just because you are typing into a field - doing so submits the form prematurely.\n"
                "- Do NOT use key_combination or press_key with 'enter' or 'return' after typing into a field unless the task explicitly says to submit the form. Pressing Enter submits the whole form - if there are more fields to fill, use tap/type_text_at for the next field instead. Only tap the on-screen submit button (e.g. 'Reset Password', 'Sign In') when all fields are filled.\n"
                "- For password fields that mask input with dots or asterisks: a single type_text_at call is sufficient. Do NOT retry just because you cannot read the entered text. Seeing masked characters (dots) after typing confirms the field is populated - stop immediately.\n\n"
                + completion_instruction
                + "YOUR TASK: "
            )
            return mobile_context + goal

        desktop_context = (
            "IMPORTANT: You are controlling an UBUNTU LINUX desktop with GNOME shell. "
            "The screenshot shows native applications (Slack, Firefox, terminals, etc.). "
            "Do NOT use built-in browser navigation commands like open_web_browser/navigate/search/go_back/go_forward; "
            "interact by clicking, typing, scrolling, or key combinations within the existing windows.\n\n"
            "SWITCHING APPLICATIONS: The left edge of the screen has the Ubuntu dock/sidebar with application icons. "
            "To switch apps, PREFER clicking the app icon in the sidebar over using alt+tab. "
            "Look for Slack (purple icon) or Firefox (orange/red icon) in the sidebar.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- click_at: Click at screen coordinates (button='right' for context clicks, click_count=2 for double clicks)\n"
            "- type_text_at: Click and type text (press Enter only if instructed)\n"
            "- key_combination: Keyboard shortcuts like ctrl+c, enter\n"
            "- read_clipboard: Read copied text/URLs after using Copy actions\n"
            "- scroll_at / scroll_document: Scroll content\n"
            "- hover_at / drag_and_drop: When required\n\n"
            + completion_instruction
            + "YOUR TASK: "
        )
        return desktop_context + goal

    @staticmethod
    def _goal_requires_json(goal: str) -> bool:
        """Heuristic: treat goals that request JSON output as strict-JSON tasks."""
        raw_text = goal or ""
        text = raw_text.lower()
        if "json" not in text:
            return False
        markers = [
            "return json",
            "return a json",
            "respond only",
            "respond with json",
            "reply with json",
            "output json",
            "return only with json",
            "json like",
            "in a json",
            "in json format",
        ]
        if any(marker in text for marker in markers):
            return True
        return '{"' in raw_text or "[{" in raw_text

    @staticmethod
    def _normalize_environment_name(
        environment: str | None,
    ) -> RuntimeEnvironmentName:
        return normalize_runtime_environment_name(environment)

    @staticmethod
    def _map_openai_environment(env_mode: str) -> str:
        return str(
            runtime_environment_spec(
                normalize_runtime_environment_name(env_mode)
            ).openai_computer_environment
        )

    @staticmethod
    def _map_google_environment(env_mode: str) -> Any:
        from google.genai import types  # type: ignore

        environment_name = runtime_environment_spec(
            normalize_runtime_environment_name(env_mode)
        ).google_computer_environment_name
        return getattr(
            types.Environment,
            environment_name,
            types.Environment.ENVIRONMENT_UNSPECIFIED,
        )

    @staticmethod
    def _wrap_goal_for_mobile(
        goal: str,
        env_mode: str,
        viewport_width: int,
        viewport_height: int,
    ) -> str:
        if env_mode != "mobile_adb":
            return goal
        orientation = "portrait" if viewport_height >= viewport_width else "landscape"
        mobile_context = (
            "MOBILE EXECUTION CONTEXT:\n"
            f"- Runtime: Android app via ADB screenshot loop\n"
            f"- Resolution: {viewport_width}x{viewport_height}\n"
            f"- Orientation: {orientation}\n"
            "- Use mobile UI assumptions (tap, type, swipe/scroll) and avoid desktop/window-management instructions.\n"
            "- Coordinates must be interpreted against the provided mobile screenshot.\n\n"
            "TASK:\n"
        )
        return f"{mobile_context}{goal}"

    @staticmethod
    def _google_modifier_click_tools() -> Any:
        """Return a Tool with custom modifier-click function declarations."""
        from google.genai import types  # type: ignore

        coord_props = {
            "x": types.Schema(
                type=types.Type.NUMBER,
                description="X coordinate (0-999 scale)",
            ),
            "y": types.Schema(
                type=types.Type.NUMBER,
                description="Y coordinate (0-999 scale)",
            ),
        }
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="ctrl_click",
                    description=(
                        "Performs a Ctrl+Click at the given coordinates. "
                        "Use this to add items to an existing selection, e.g. "
                        "selecting multiple files in a file picker."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=coord_props,
                        required=["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="shift_click",
                    description=(
                        "Performs a Shift+Click at the given coordinates. "
                        "Use this to extend a contiguous selection, e.g. "
                        "selecting a range of files in a file picker."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=coord_props,
                        required=["x", "y"],
                    ),
                ),
            ]
        )

    @staticmethod
    def _parse_betas(raw: str) -> list[str]:
        values = [part.strip() for part in str(raw or "").split(",")]
        return [value for value in values if value]

    @staticmethod
    def _is_scroll_action(action_type: str | None) -> bool:
        if not action_type:
            return False
        normalized = action_type.lower()
        return normalized in {"scroll", "scroll_at", "scroll_document", "scroll_window"}

    @staticmethod
    def _normalize_domain_set(domains: list[str]) -> set[str]:
        normalized: set[str] = set()
        for domain in domains or []:
            if not domain:
                continue
            value = domain.strip().lower().lstrip(".")
            if value:
                normalized.add(value)
        return normalized

    @staticmethod
    def _domain_matches(hostname: str, domain: str) -> bool:
        hostname = hostname.strip().lower()
        domain = domain.strip().lower()
        return hostname == domain or hostname.endswith(f".{domain}")

    @property
    def _action_timeout_seconds(self: _ComputerUseSession) -> float:
        return max(
            float(self._settings.actions_computer_tool_action_timeout_ms) / 1000.0,
            0.5,
        )
