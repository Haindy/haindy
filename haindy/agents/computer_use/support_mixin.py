# mypy: disable-error-code=misc
"""Shared support methods for Computer Use sessions."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from haindy.core.enhanced_types import ComputerToolTurn, SafetyEvent
from haindy.runtime.environment import (
    RuntimeEnvironmentName,
    normalize_runtime_environment_name,
    runtime_environment_spec,
)

from .common import (
    _inject_context_metadata,
    denormalize_coordinates,
    normalize_coordinates,
)
from .turn_result import ComputerUseFollowUpBatch, build_follow_up_batch
from .types import ComputerUseExecutionError, ComputerUseSessionResult
from .visual_pipeline import VisualStatePlanner
from .visual_state import (
    CARTOGRAPHY_BLOCK_END,
    CARTOGRAPHY_BLOCK_START,
    VisualFrame,
    attach_cartography,
    build_keyframe,
    extract_cartography_payload,
    parse_cartography_payload,
)

logger = logging.getLogger("haindy.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class ComputerUseSupportMixin:
    """Support helpers shared across provider implementations."""

    _GOOGLE_MOBILE_EXCLUDED_PREDEFINED_FUNCTIONS: tuple[str, ...] = (
        "open_web_browser",
        "search",
        "navigate",
        "hover_at",
        "go_forward",
        "scroll_document",
        "key_combination",
        "drag_and_drop",
    )

    _current_keyframe: VisualFrame | None
    _last_visual_frame: VisualFrame | None
    _visual_state_planner: VisualStatePlanner
    _turns_since_keyframe: int
    _turns_since_cartography_refresh: int

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

        action_label = signature[0] if signature else "unknown"
        if all(hash_ == screenshot_hash for _, hash_ in history) and all(
            bool(sig) and sig[0] == action_label for sig, _ in history
        ):
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

    def _maybe_seed_initial_keyframe(
        self: _ComputerUseSession,
        initial_screenshot: bytes | None,
    ) -> None:
        """Seed the first step keyframe from the caller-provided screenshot."""
        if self._current_keyframe is not None or initial_screenshot is None:
            return

        self._current_keyframe = build_keyframe(
            initial_screenshot,
            source="initial_screenshot",
        )
        self._turns_since_keyframe = 0
        self._turns_since_cartography_refresh = 0
        logger.info(
            "Computer Use seeded initial keyframe",
            extra={
                "visual_frame_id": self._current_keyframe.frame_id,
                "visual_frame_kind": self._current_keyframe.kind,
                "has_cartography": False,
            },
        )

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
            "long_press": "long_press_at",
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
            reporting_guidance = ComputerUseSupportMixin._build_reporting_guidance(
                metadata
            )
            if not reporting_guidance or reporting_guidance in goal:
                return goal
            return f"{goal}\n\n{reporting_guidance}"

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

    @staticmethod
    def _build_reporting_guidance(metadata: dict[str, Any]) -> str | None:
        """Constrain how much detail the executor should include in final replies."""
        scope = str(metadata.get("response_reporting_scope") or "").strip().lower()
        if scope != "state_only":
            return None
        return (
            "REPORTING RULES:\n"
            "- In your final confirmation, summarize only the requested visible state change.\n"
            "- Do NOT quote or infer exact field values, email addresses, usernames, codes, IDs, or other on-screen text unless the task explicitly asks you to verify or read that exact text.\n"
            "- For navigation/opening steps, confirm that the target screen or section is visible and stop."
        )

    @staticmethod
    def _build_execute_follow_up_reporting_reminder(
        metadata: dict[str, Any],
    ) -> str | None:
        """Return a compact execute-mode reminder for follow-up turns."""
        scope = str(metadata.get("response_reporting_scope") or "").strip().lower()
        if scope != "state_only":
            return None
        return (
            "In your confirmation, summarize only the requested visible state change. "
            "Do not quote exact field values or other on-screen text unless the task explicitly asks for that text."
        )

    def _prime_initial_visual_state_for_request(
        self: _ComputerUseSession,
        screenshot_bytes: bytes | None,
        *,
        source: str,
    ) -> None:
        """Record the full-frame screenshot that the provider is about to see."""
        if screenshot_bytes is None:
            return
        if self._current_keyframe is None:
            self._current_keyframe = build_keyframe(
                screenshot_bytes,
                source=source,
            )
        self._last_visual_frame = self._current_keyframe

    @staticmethod
    def _build_localization_protocol_text(target_text: str) -> str:
        """Return the shared in-session localization protocol."""
        return (
            "LOCALIZATION MAP RULES:\n"
            f"- When you have a full-screen reference screenshot, append exactly one cartography block at the end of your reply for the current target {target_text!r}.\n"
            f"- Use this exact wrapper: {CARTOGRAPHY_BLOCK_START}"
            '{"targets":[{"target_id":"target_1","label":"visible label or short descriptor","bbox":{"x":0,"y":0,"width":0,"height":0},"interaction_point":{"x":0,"y":0},"confidence":0.0}]}'
            f"{CARTOGRAPHY_BLOCK_END}\n"
            "- Use only elements actually visible in the screenshot.\n"
            "- Use absolute full-screen pixel coordinates.\n"
            '- If the target is not visible, return {"targets":[]}.\n'
            "- Do not emit a cartography block for cropped patch screenshots."
        )

    @classmethod
    def _apply_localization_protocol_guidance(
        cls,
        goal: str,
        metadata: dict[str, Any],
    ) -> str:
        """Append the shared localization protocol when the step has a target."""
        target_text = str(metadata.get("target") or "").strip()
        if not target_text:
            return goal
        guidance = cls._build_localization_protocol_text(target_text)
        if guidance in goal:
            return goal
        return f"{goal}\n\n{guidance}"

    @classmethod
    def _build_follow_up_localization_prompt(
        cls,
        follow_up_batch: ComputerUseFollowUpBatch,
        metadata: dict[str, Any],
    ) -> str | None:
        """Return a refresh prompt when the session needs a new target map."""
        if not follow_up_batch.request_localization:
            return None
        target_text = str(metadata.get("target") or "").strip()
        if not target_text:
            return None
        reason = str(follow_up_batch.localization_reason or "").strip() or "refresh"
        return (
            "This screenshot is a full-screen reference. Refresh the session cartography "
            f"now because: {reason}.\n"
            f"{cls._build_localization_protocol_text(target_text)}"
        )

    def _consume_localization_response(
        self: _ComputerUseSession,
        assistant_text: str | None,
        *,
        metadata: dict[str, Any],
        provider: str,
        model: str | None,
    ) -> str | None:
        """Extract session-local cartography from assistant text and persist it."""
        cleaned_text, payload_text = extract_cartography_payload(assistant_text)
        if payload_text is None:
            return cleaned_text

        frame = self._last_visual_frame
        if frame is None or frame.kind != "keyframe":
            logger.debug(
                "Ignoring cartography block without an active keyframe",
                extra={"provider": provider},
            )
            return cleaned_text

        target_text = str(metadata.get("target") or "").strip()
        if not target_text:
            logger.debug(
                "Ignoring cartography block without a target description",
                extra={"provider": provider, "frame_id": frame.frame_id},
            )
            return cleaned_text

        cartography = parse_cartography_payload(
            payload_text,
            frame=frame,
            provider=provider,
            model=model,
            fallback_label=target_text,
        )
        if cartography is None:
            logger.debug(
                "Ignoring invalid cartography payload",
                extra={"provider": provider, "frame_id": frame.frame_id},
            )
            return cleaned_text

        updated_frame = attach_cartography(frame, cartography)
        if (
            self._current_keyframe is not None
            and self._current_keyframe.frame_id == frame.frame_id
        ):
            self._current_keyframe = updated_frame
        if (
            self._last_visual_frame is not None
            and self._last_visual_frame.frame_id == frame.frame_id
        ):
            self._last_visual_frame = updated_frame
        self._turns_since_cartography_refresh = 0
        logger.info(
            "Computer Use cartography refreshed in-session",
            extra={
                "provider": provider,
                "model": model,
                "frame_id": frame.frame_id,
                "target": target_text,
                "cartography_target_count": len(cartography.targets),
                "cartography_labels": [
                    target.label for target in cartography.targets if target.label
                ]
                or None,
                "step_number": metadata.get("step_number"),
            },
        )
        return cleaned_text

    async def _build_follow_up_batch(
        self: _ComputerUseSession,
        *,
        call_groups: list[list[ComputerToolTurn]],
        metadata: dict[str, Any],
        skip_localization: bool = False,
    ) -> ComputerUseFollowUpBatch:
        """Capture one fresh follow-up state and build the shared batch model."""
        screenshot_bytes = await self._automation_driver.screenshot()
        current_url = await self._maybe_get_current_url()
        if not current_url:
            current_url = "desktop://"
        interaction_mode = str(metadata.get("interaction_mode") or "").strip().lower()
        action_types = [
            str(turn.action_type or "").strip().lower()
            for group in call_groups
            for turn in group
            if turn.action_type
        ]
        plan = await self._visual_state_planner.build_follow_up_frame(
            screenshot_bytes=screenshot_bytes,
            metadata=metadata,
            action_types=action_types,
            previous_keyframe=self._current_keyframe,
            turns_since_keyframe=self._turns_since_keyframe,
            turns_since_cartography_refresh=self._turns_since_cartography_refresh,
            cartography=(
                self._current_keyframe.cartography
                if self._current_keyframe is not None
                else None
            ),
            skip_localization=skip_localization,
        )
        self._current_keyframe = plan.current_keyframe
        self._last_visual_frame = plan.visual_frame
        if plan.visual_frame.kind == "keyframe":
            self._turns_since_keyframe = 0
        else:
            self._turns_since_keyframe += 1
        self._turns_since_cartography_refresh += 1
        follow_up_batch = build_follow_up_batch(
            call_groups,
            screenshot_bytes=plan.visual_frame.image_bytes,
            current_url=current_url,
            interaction_mode=interaction_mode,
            visual_frame=plan.visual_frame,
            artifact_frame=plan.artifact_frame,
        )
        follow_up_batch.request_localization = plan.request_localization
        follow_up_batch.localization_reason = plan.localization_reason
        if plan.request_localization:
            logger.info(
                "Computer Use cartography refresh requested",
                extra={
                    "provider": self._provider,
                    "reason": plan.localization_reason,
                    "frame_id": plan.current_keyframe.frame_id,
                    "step_number": metadata.get("step_number"),
                },
            )
        execute_reporting_reminder = self._build_execute_follow_up_reporting_reminder(
            metadata
        )
        if execute_reporting_reminder:
            existing_reminder = str(follow_up_batch.reminder_text or "").strip()
            if existing_reminder:
                follow_up_batch.reminder_text = (
                    f"{existing_reminder}\n{execute_reporting_reminder}"
                )
            else:
                follow_up_batch.reminder_text = execute_reporting_reminder
        return follow_up_batch

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
        turn: ComputerToolTurn | None = None,
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
            return self._denormalize_coordinates_for_active_frame(
                x,
                y,
                viewport_width,
                viewport_height,
                turn=turn,
                prefix=prefix,
            )
        return normalize_coordinates(x, y, viewport_width, viewport_height)

    def _denormalize_coordinates_for_active_frame(
        self: _ComputerUseSession,
        x: float | None,
        y: float | None,
        viewport_width: int,
        viewport_height: int,
        *,
        turn: ComputerToolTurn | None = None,
        prefix: str = "",
    ) -> tuple[int, int]:
        """Resolve normalized coordinates against the latest model-visible frame."""
        frame = self._last_visual_frame
        normalized_prefix = f"{prefix}normalized_"
        if turn is not None:
            turn.metadata[f"{normalized_prefix}x"] = x
            turn.metadata[f"{normalized_prefix}y"] = y

        if self._provider != "google" or frame is None or frame.kind != "patch":
            resolved = denormalize_coordinates(x, y, viewport_width, viewport_height)
            if turn is not None:
                turn.metadata[f"{prefix}coordinate_frame_kind"] = (
                    frame.kind if frame is not None else "viewport"
                )
            return resolved

        patch_bounds = frame.bounds
        patch_x, patch_y = denormalize_coordinates(
            x,
            y,
            max(patch_bounds.width, 1),
            max(patch_bounds.height, 1),
        )
        resolved = normalize_coordinates(
            patch_bounds.x + patch_x,
            patch_bounds.y + patch_y,
            viewport_width,
            viewport_height,
        )
        if turn is not None:
            turn.metadata[f"{prefix}coordinate_frame_kind"] = "patch"
            turn.metadata[f"{prefix}patch_bounds"] = patch_bounds.as_tuple()
            turn.metadata[f"{prefix}patch_coordinate"] = (patch_x, patch_y)
            turn.metadata[f"{prefix}full_screen_coordinate"] = resolved
            turn.metadata["visual_frame_id"] = frame.frame_id
        return resolved

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
        self: _ComputerUseSession,
        goal: str,
        env_mode: str,
        viewport_width: int = 0,
        viewport_height: int = 0,
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
            goal = self._strip_mobile_goal_wrapper(goal)
            orientation = (
                "portrait" if viewport_height >= viewport_width else "landscape"
            )
            device_line = ""
            if viewport_width and viewport_height:
                device_line = f"- Device: Android phone, {viewport_width}x{viewport_height}, {orientation}\n"
            mobile_context = (
                "IMPORTANT: You are controlling an Android phone through ADB-backed screenshots.\n"
                + device_line
                + "- Treat coordinates as positions on the provided mobile screenshot.\n"
                "- Use mobile interactions only: click_at, type_text_at, scroll_at, wait_5_seconds, and the custom helpers long_press_at, go_home, and open_app when needed.\n"
                "- Browser-style actions such as open_web_browser, search, navigate, hover_at, go_forward, scroll_document, drag_and_drop, and key_combination are unavailable in this mobile flow.\n"
                "\n"
                "GESTURE NAVIGATION:\n"
                "- To go to the home screen, use the go_home helper.\n"
                "- To go back, use scroll_at with direction=right at the left edge of the screen.\n"
                "- There are no on-screen navigation buttons; this device uses gesture navigation.\n"
                "\n"
                "SWIPE GESTURES:\n"
                "- scroll_at is also used for swipe gestures on list items (e.g. swipe left/right on an email row).\n"
                "- When swiping a specific list item, set the scroll_at coordinates to the vertical center of that item's row. Precise Y targeting matters.\n"
                "\n"
                "GENERAL RULES:\n"
                "- Do not use desktop assumptions, browser navigation actions, or desktop shortcuts like ctrl+a.\n"
                "- For text entry, use type_text_at directly. Set press_enter=true only when the task explicitly says to submit or press Enter.\n"
                "- If masked password dots are visible after typing, treat the field as filled and stop retrying.\n\n"
                + completion_instruction
                + "YOUR TASK: "
            )
            return mobile_context + goal

        if env_mode == "mobile_ios":
            goal = self._strip_mobile_goal_wrapper(goal)
            ios_context = (
                "IMPORTANT: You are controlling an iOS device (iPhone or iPad) through idb-backed screenshots.\n"
                "- Treat coordinates as positions on the provided iOS screenshot.\n"
                "- Use mobile interactions only: click_at, type_text_at, scroll_at, wait_5_seconds, and the custom helpers long_press_at and go_home when needed.\n"
                "- Browser-style actions such as open_web_browser, search, navigate, hover_at, go_forward, scroll_document, drag_and_drop, and key_combination are unavailable in this mobile flow.\n"
                "- For system navigation, use the go_home helper to return to the iOS home screen.\n"
                "- Do not use desktop assumptions, browser navigation actions, or desktop shortcuts.\n"
                "- For text entry, use type_text_at directly. Set press_enter=true only when the task explicitly says to submit or press Enter.\n\n"
                + completion_instruction
                + "YOUR TASK: "
            )
            return ios_context + goal

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
    def _strip_mobile_goal_wrapper(goal: str) -> str:
        """Remove the generic mobile goal prefix when a provider adds its own wrapper."""
        marker = "MOBILE EXECUTION CONTEXT:"
        text = str(goal or "").strip()
        if not text.startswith(marker):
            return text

        task_marker = "\n\nTASK:\n"
        task_index = text.find(task_marker)
        if task_index == -1:
            return text

        stripped = text[task_index + len(task_marker) :].strip()
        return stripped or text

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
    def _map_google_interaction_environment(env_mode: str) -> str | None:
        spec = runtime_environment_spec(normalize_runtime_environment_name(env_mode))
        return "browser" if (spec.is_browser or spec.is_mobile) else None

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
    def _google_mobile_excluded_predefined_functions() -> list[str]:
        """Return Google Computer Use actions that should be unavailable on mobile."""
        return list(
            ComputerUseSupportMixin._GOOGLE_MOBILE_EXCLUDED_PREDEFINED_FUNCTIONS
        )

    @staticmethod
    def _google_mobile_custom_tools() -> Any:
        """Return a Tool with the documented mobile helper functions."""
        from google.genai import types  # type: ignore

        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="open_app",
                    description=(
                        "Open the configured Android app under test, or launch a "
                        "deep link when one is provided. Use this when the task "
                        "explicitly requires opening or foregrounding the app."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "app_name": types.Schema(
                                type=types.Type.STRING,
                                description=(
                                    "Name of the app to open. Prefer the app "
                                    "currently under test."
                                ),
                            ),
                            "intent": types.Schema(
                                type=types.Type.STRING,
                                description=(
                                    "Optional deep link or Android intent URL to "
                                    "launch."
                                ),
                            ),
                        },
                        required=["app_name"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="long_press_at",
                    description=(
                        "Long-press at the given mobile screen coordinates. Use "
                        "for context menus or press-and-hold interactions."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "x": types.Schema(
                                type=types.Type.NUMBER,
                                description="X coordinate (0-999 scale)",
                            ),
                            "y": types.Schema(
                                type=types.Type.NUMBER,
                                description="Y coordinate (0-999 scale)",
                            ),
                            "duration_ms": types.Schema(
                                type=types.Type.INTEGER,
                                description="Optional hold duration in milliseconds.",
                            ),
                        },
                        required=["x", "y"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="go_home",
                    description="Navigate to the Android home screen.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={},
                    ),
                ),
            ]
        )

    @staticmethod
    def _google_mobile_function_tools() -> list[dict[str, Any]]:
        """Return Interactions API mobile-only function declarations."""
        return [
            {
                "type": "function",
                "name": "open_app",
                "description": (
                    "Open the configured Android app under test, or launch a deep "
                    "link when one is provided. Use this when the task explicitly "
                    "requires opening or foregrounding the app."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": (
                                "Name of the app to open. Prefer the app currently "
                                "under test."
                            ),
                        },
                        "intent": {
                            "type": "string",
                            "description": (
                                "Optional deep link or Android intent URL to launch."
                            ),
                        },
                    },
                    "required": ["app_name"],
                },
            },
            {
                "type": "function",
                "name": "long_press_at",
                "description": (
                    "Long-press at the given mobile screen coordinates. Use for "
                    "context menus or press-and-hold interactions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "X coordinate (0-999 scale)",
                        },
                        "y": {
                            "type": "number",
                            "description": "Y coordinate (0-999 scale)",
                        },
                        "duration_ms": {
                            "type": "integer",
                            "description": "Optional hold duration in milliseconds.",
                        },
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "type": "function",
                "name": "go_home",
                "description": "Navigate to the Android home screen.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    @staticmethod
    def _google_modifier_click_function_tools() -> list[dict[str, Any]]:
        """Return Interactions API tool declarations for modifier-click actions."""
        coord_schema = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X coordinate (0-999 scale)",
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate (0-999 scale)",
                },
            },
            "required": ["x", "y"],
        }
        return [
            {
                "type": "function",
                "name": "ctrl_click",
                "description": (
                    "Performs a Ctrl+Click at the given coordinates. "
                    "Use this to add items to an existing selection, e.g. "
                    "selecting multiple files in a file picker."
                ),
                "parameters": coord_schema,
            },
            {
                "type": "function",
                "name": "shift_click",
                "description": (
                    "Performs a Shift+Click at the given coordinates. "
                    "Use this to extend a contiguous selection, e.g. "
                    "selecting a range of files in a file picker."
                ),
                "parameters": coord_schema,
            },
        ]

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
