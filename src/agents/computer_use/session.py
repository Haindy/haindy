"""Computer Use tool orchestration for the Action Agent."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from openai import AsyncOpenAI

from src.config.settings import Settings
from src.core.enhanced_types import ComputerToolTurn, SafetyEvent
from src.core.interfaces import BrowserDriver
from src.monitoring.debug_logger import DebugLogger


logger = logging.getLogger(__name__)


class ComputerUseExecutionError(RuntimeError):
    """Raised when the Computer Use orchestration fails irrecoverably."""


@dataclass
class ComputerUseSessionResult:
    """Result of executing a Computer Use session."""

    actions: List[ComputerToolTurn] = field(default_factory=list)
    safety_events: List[SafetyEvent] = field(default_factory=list)
    final_output: Optional[str] = None
    response_ids: List[str] = field(default_factory=list)
    last_response: Optional[Dict[str, Any]] = None


class ComputerUseSession:
    """Wraps the OpenAI Computer Use tool and orchestrates action execution."""

    def __init__(
        self,
        client: AsyncOpenAI,
        browser: BrowserDriver,
        settings: Settings,
        debug_logger: Optional[DebugLogger] = None,
        model: str = "computer-use-preview",
        environment: str = "browser",
    ) -> None:
        self._client = client
        self._browser = browser
        self._settings = settings
        self._debug_logger = debug_logger
        self._model = model
        self._environment = self._normalize_environment(environment)
        self._allowed_actions: Optional[Set[str]] = None
        self._allowed_domains: Set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_allowed_domains
        )
        self._blocked_domains: Set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_blocked_domains
        )
        self._auto_ack_safety: bool = settings.actions_computer_tool_auto_ack_safety
        self._auto_ack_codes: Optional[Set[str]] = self._normalize_safety_code_set(
            settings.actions_computer_tool_auto_ack_codes
        )
        self._stateful_actions: Set[str] = {
            "click",
            "double_click",
            "right_click",
            "move",
            "type",
            "keypress",
            "drag",
            "navigate",
        }

    @staticmethod
    def _normalize_environment(environment: str) -> str:
        """
        Ensure the Computer Use environment value matches supported options.

        Maps legacy or shorthand values to the OpenAI-supported set and
        defaults to "browser" when an unexpected value is provided.
        """
        normalized = (environment or "browser").strip().lower()
        alias_map = {
            "desktop": "linux",
            "os": "linux",
        }
        normalized = alias_map.get(normalized, normalized)
        allowed = {"browser", "windows", "mac", "linux"}
        if normalized not in allowed:
            logger.warning(
                "Unsupported Computer Use environment '%s'; defaulting to 'browser'.",
                environment,
            )
            return "browser"
        return normalized

    async def run(
        self,
        goal: str,
        initial_screenshot: Optional[bytes],
        metadata: Optional[Dict[str, Any]] = None,
        allowed_actions: Optional[Set[str]] = None,
    ) -> ComputerUseSessionResult:
        """
        Execute a Computer Use loop until completion or failure.

        Args:
            goal: Natural language instruction for the model.
            initial_screenshot: Screenshot bytes representing the current state.
            metadata: Optional context (step number, plan/case names).

        Returns:
            ComputerUseSessionResult with action traces and final output.
        """
        metadata = metadata or {}
        self._allowed_actions = allowed_actions
        result = ComputerUseSessionResult()

        await self._ensure_browser_ready()

        viewport_width, viewport_height = await self._browser.get_viewport_size()
        screenshot = initial_screenshot or await self._browser.screenshot()
        screenshot_b64 = encode_png_base64(screenshot)

        request_payload = self._build_initial_request(
            goal=goal,
            screenshot_b64=screenshot_b64,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            metadata=metadata,
        )

        response = await self._create_response(request_payload)
        response_dict = normalize_response(response)
        result.response_ids.append(response_dict.get("id", ""))

        turn_counter = 0
        previous_response_id = response_dict.get("id")
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: Deque[Tuple[Tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            computer_calls = extract_computer_calls(response_dict)
            assistant_message = extract_assistant_text(response_dict)
            result.final_output = assistant_message or result.final_output
            result.last_response = response_dict

            interaction_mode = metadata.get("interaction_mode") or "execute"
            if (
                interaction_mode != "observe_only"
                and assistant_message
                and not computer_calls
                and self._looks_like_confirmation_request(assistant_message)
            ):
                attempts = metadata.setdefault("_auto_confirmation_attempts", 0)
                if attempts >= 2:
                    logger.debug(
                        "Auto-confirmation limit reached; returning control to caller.",
                        extra={"response_id": response_dict.get("id")},
                    )
                    break
                metadata["_auto_confirmation_attempts"] = attempts + 1
                logger.debug(
                    "Computer Use model requested confirmation; auto-affirming continuation.",
                    extra={"response_id": response_dict.get("id")},
                )
                confirmation_payload = await self._build_confirmation_request(
                    previous_response_id=previous_response_id,
                    metadata=metadata,
                )
                response = await self._create_response(confirmation_payload)
                response_dict = normalize_response(response)
                result.response_ids.append(response_dict.get("id", ""))
                previous_response_id = response_dict.get("id")
                continue

            if not computer_calls:
                break

            turn_counter += 1
            if turn_counter > self._settings.actions_computer_tool_max_turns:
                max_turns = self._settings.actions_computer_tool_max_turns
                message = (
                    f"Computer Use max turns exceeded after {turn_counter} turns (limit: {max_turns})."
                )
                logger.warning(
                    "Computer Use max turns exceeded",
                    extra={
                        "max_turns": max_turns,
                        "turn_count": turn_counter,
                        "step_number": metadata.get("step_number"),
                    },
                )
                failure_turn = ComputerToolTurn(
                    call_id=f"turn-limit-{uuid4()}",
                    action_type="system_notice",
                    parameters={
                        "type": "system_notice",
                        "reason": "max_turns_exceeded",
                        "turn_count": turn_counter,
                        "max_turns": max_turns,
                    },
                    response_id=response_dict.get("id"),
                )
                failure_turn.status = "failed"
                failure_turn.error_message = message
                _inject_context_metadata(failure_turn, metadata)
                failure_turn.metadata.update(
                    {
                        "reason": "max_turns_exceeded",
                        "turn_count": turn_counter,
                        "max_turns": max_turns,
                    }
                )
                result.actions.append(failure_turn)
                result.final_output = message
                break

            call = computer_calls[0]
            turn = ComputerToolTurn(
                call_id=call.get("call_id", ""),
                action_type=call.get("action", {}).get("type", "unknown"),
                parameters=call.get("action", {}),
                response_id=response_dict.get("id"),
                pending_safety_checks=call.get("pending_safety_checks", []) or [],
            )

            _inject_context_metadata(turn, metadata)

            if turn.pending_safety_checks:
                acknowledged_ids = self._maybe_acknowledge_safety(turn, metadata)
                safety_event = SafetyEvent(
                    call_id=turn.call_id,
                    code=turn.pending_safety_checks[0].get("code", "unknown"),
                    message=turn.pending_safety_checks[0].get("message", ""),
                    acknowledged=bool(acknowledged_ids),
                    response_id=response_dict.get("id"),
                    acknowledged_ids=acknowledged_ids,
                )
                result.safety_events.append(safety_event)

                if acknowledged_ids:
                    turn.metadata["acknowledged_safety_checks"] = acknowledged_ids
                    logger.info(
                        "Computer Use safety check acknowledged automatically",
                        extra={
                            "call_id": turn.call_id,
                            "codes": [c.get("code") for c in turn.pending_safety_checks],
                            "acknowledged_ids": acknowledged_ids,
                        },
                    )
                elif self._settings.actions_computer_tool_fail_fast_on_safety:
                    turn.status = "failed"
                    turn.error_message = (
                        "Safety check triggered; action halted (fail-fast enabled)."
                    )
                    result.actions.append(turn)
                    logger.warning(
                        "Computer Use safety check triggered; aborting action execution",
                        extra={
                            "call_id": turn.call_id,
                            "code": safety_event.code,
                            "safety_message": safety_event.message,
                        },
                    )
                    return result

            try:
                await self._execute_tool_action(turn, metadata, turn_counter)
            except Exception as exc:
                turn.status = "failed"
                turn.error_message = str(exc)
                logger.exception("Computer Use action execution failed", extra={"call_id": turn.call_id})

            result.actions.append(turn)
            metadata["_auto_confirmation_attempts"] = 0

            loop_detection = self._update_loop_history(
                turn=turn,
                history=loop_history,
                window=loop_window,
            )
            if loop_detection:
                message = loop_detection["message"]
                logger.warning(
                    "Computer Use loop detected",
                    extra={
                        "step_number": metadata.get("step_number"),
                        "action_type": turn.action_type,
                        "signature": loop_detection.get("signature"),
                        "loop_window": loop_detection.get("loop_window"),
                    },
                )
                failure_turn = ComputerToolTurn(
                    call_id=f"loop-detected-{uuid4()}",
                    action_type="system_notice",
                    parameters={
                        "type": "system_notice",
                        "reason": "loop_detected",
                        "signature": loop_detection.get("signature"),
                        "screenshot_hash": loop_detection.get("screenshot_hash"),
                        "loop_window": loop_detection.get("loop_window"),
                    },
                    response_id=response_dict.get("id"),
                )
                failure_turn.status = "failed"
                failure_turn.error_message = message
                _inject_context_metadata(failure_turn, metadata)
                failure_turn.metadata.update(loop_detection)
                result.actions.append(failure_turn)
                result.final_output = message
                break

            follow_up_payload = await self._build_follow_up_request(
                previous_response_id=previous_response_id,
                call=turn,
                metadata=metadata,
            )

            response = await self._create_response(follow_up_payload)
            response_dict = normalize_response(response)
            result.response_ids.append(response_dict.get("id", ""))
            previous_response_id = response_dict.get("id")

        self._allowed_actions = None
        return result

    async def _execute_tool_action(
        self,
        turn: ComputerToolTurn,
        metadata: Dict[str, Any],
        turn_index: int,
    ) -> None:
        """Execute a single Computer Use tool action via the browser driver."""
        action = turn.parameters
        raw_action_type = action.get("type")
        action_type = self._canonicalize_action_type(raw_action_type)
        if action_type and action_type != raw_action_type:
            turn.metadata["normalized_action_type"] = action_type
            turn.action_type = action_type
        start = time.perf_counter()

        viewport_width, viewport_height = await self._browser.get_viewport_size()
        allow_action, deny_reason = self._is_action_allowed(action_type)

        try:
            if not allow_action:
                turn.status = "failed"
                turn.error_message = deny_reason or "Action blocked by policy."
                turn.metadata["policy"] = "observe_only"
            else:
                await self._enforce_domain_policy(action_type)

                if action_type == "click":
                    await self._execute_click(action, viewport_width, viewport_height)
                elif action_type in {"double_click", "right_click"}:
                    await self._execute_special_click(action, viewport_width, viewport_height)
                elif action_type == "move":
                    x, y = normalize_coordinates(
                        action.get("x"),
                        action.get("y"),
                        viewport_width,
                        viewport_height,
                    )
                    raw_steps = action.get("steps") or action.get("step_count")
                    steps = 1
                    if raw_steps is not None:
                        try:
                            steps = int(float(raw_steps))
                        except (TypeError, ValueError):
                            steps = 1
                    if steps <= 0:
                        steps = 1
                    await asyncio.wait_for(
                        self._browser.move_mouse(x, y, steps=steps),
                        timeout=self._action_timeout_seconds,
                    )
                elif action_type == "drag":
                    def _coerce_float(value: Any) -> Optional[float]:
                        if value is None:
                            return None
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    start_x_raw = _coerce_float(
                        action.get("start_x")
                        or action.get("from_x")
                        or action.get("x")
                    )
                    start_y_raw = _coerce_float(
                        action.get("start_y")
                        or action.get("from_y")
                        or action.get("y")
                    )

                    end_x_raw = _coerce_float(
                        action.get("end_x") or action.get("to_x")
                    )
                    end_y_raw = _coerce_float(
                        action.get("end_y") or action.get("to_y")
                    )

                    path_points = action.get("path") or action.get("points")
                    if isinstance(path_points, list) and len(path_points) >= 2:
                        first_point = path_points[0]
                        last_point = path_points[-1]

                        if start_x_raw is None:
                            start_x_raw = _coerce_float(first_point.get("x"))
                        if start_y_raw is None:
                            start_y_raw = _coerce_float(first_point.get("y"))
                        if end_x_raw is None:
                            end_x_raw = _coerce_float(last_point.get("x"))
                        if end_y_raw is None:
                            end_y_raw = _coerce_float(last_point.get("y"))

                    if start_x_raw is None or start_y_raw is None:
                        raise ComputerUseExecutionError("Drag action missing start coordinates.")

                    delta_x_raw = _coerce_float(action.get("dx") or action.get("delta_x"))
                    delta_y_raw = _coerce_float(action.get("dy") or action.get("delta_y"))

                    if end_x_raw is None and delta_x_raw is not None:
                        end_x_raw = start_x_raw + delta_x_raw
                    if end_y_raw is None and delta_y_raw is not None:
                        end_y_raw = start_y_raw + delta_y_raw

                    if end_x_raw is None or end_y_raw is None:
                        raise ComputerUseExecutionError("Drag action missing destination coordinates.")

                    start_x, start_y = normalize_coordinates(
                        start_x_raw, start_y_raw, viewport_width, viewport_height
                    )
                    end_x, end_y = normalize_coordinates(
                        end_x_raw, end_y_raw, viewport_width, viewport_height
                    )

                    raw_steps = action.get("steps") or action.get("step_count")
                    steps = 1
                    if raw_steps is not None:
                        try:
                            steps = int(float(raw_steps))
                        except (TypeError, ValueError):
                            steps = 1
                    elif isinstance(path_points, list) and len(path_points) >= 2:
                        steps = max(1, len(path_points) - 1)

                    if steps <= 0:
                        steps = 1

                    await asyncio.wait_for(
                        self._browser.drag_mouse(start_x, start_y, end_x, end_y, steps=steps),
                        timeout=self._action_timeout_seconds,
                    )
                elif action_type == "scroll":
                    await self._execute_scroll(action)
                elif action_type == "type":
                    text_payload = action.get("text")
                    if not text_payload:
                        text_payload = action.get("value") or action.get("input") or metadata.get("value")
                        if text_payload:
                            turn.metadata["synthetic_text_payload"] = text_payload
                    if not text_payload:
                        raise ComputerUseExecutionError("Type action missing text payload.")
                    await self._browser.type_text(text_payload)
                elif action_type == "keypress":
                    key_sequence = self._resolve_key_sequence(action, metadata)
                    if not key_sequence:
                        raise ComputerUseExecutionError("Key press action missing key payload.")
                    if not action.get("keys"):
                        turn.metadata["synthetic_key_sequence"] = key_sequence
                    for key in key_sequence:
                        normalized = normalize_key_sequence(key)
                        await self._browser.press_key(normalized)
                elif action_type == "wait":
                    duration = int(
                        action.get("duration_ms")
                        or self._settings.actions_computer_tool_stabilization_wait_ms
                    )
                    await self._browser.wait(duration)
                elif action_type == "screenshot":
                    # No-op; screenshot captured after execution
                    logger.debug("Computer Use requested screenshot action; no browser operation executed.")
                elif action_type == "navigate":
                    raise ComputerUseExecutionError(
                        "Direct navigation actions are not supported by the browser driver."
                    )
                else:
                    raise ComputerUseExecutionError(f"Unsupported computer action type: {action_type}")

                turn.status = "executed"

        except ComputerUseExecutionError as policy_error:
            turn.status = "failed"
            turn.error_message = str(policy_error)
            turn.metadata["policy"] = "rejected"
            logger.warning(
                "Computer Use action rejected",
                extra={
                    "call_id": turn.call_id,
                    "action_type": action_type,
                    "reason": turn.error_message,
                },
            )
        finally:
            turn.latency_ms = (time.perf_counter() - start) * 1000
            if turn.status == "executed":
                await self._post_action_wait()
            await self._record_turn_snapshot(turn, metadata, turn_index)

    async def _record_turn_snapshot(
        self,
        turn: ComputerToolTurn,
        metadata: Dict[str, Any],
        turn_index: int,
    ) -> None:
        """Capture screenshot and update metadata after action execution."""
        screenshot_bytes = await self._browser.screenshot()
        turn.screenshot_path = self._save_turn_screenshot(
            screenshot_bytes,
            suffix=f"{turn.action_type}_{turn_index}",
            step_number=metadata.get("step_number"),
        )

        turn.metadata.update(
            {
                "screenshot_base64": encode_png_base64(screenshot_bytes),
                "current_url": await self._maybe_get_current_url(),
            }
        )

    async def _execute_click(
        self, action: Dict[str, Any], viewport_width: int, viewport_height: int
    ) -> None:
        """Execute a primary click event."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        button = action.get("button", "left")
        click_count = int(action.get("click_count", 1))
        await asyncio.wait_for(
            self._browser.click(x, y, button=button, click_count=click_count),
            timeout=self._action_timeout_seconds,
        )

    async def _execute_special_click(
        self, action: Dict[str, Any], viewport_width: int, viewport_height: int
    ) -> None:
        """Execute double or right click events."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        action_type = action.get("type")
        if action_type == "double_click":
            await asyncio.wait_for(
                self._browser.click(x, y, button="left", click_count=2),
                timeout=self._action_timeout_seconds,
            )
        elif action_type == "right_click":
            await asyncio.wait_for(
                self._browser.click(x, y, button="right", click_count=1),
                timeout=self._action_timeout_seconds,
            )

    async def _execute_scroll(self, action: Dict[str, Any]) -> None:
        """Execute a scroll event via pixel deltas."""
        scroll_x = int(action.get("scroll_x", 0))
        scroll_y = int(action.get("scroll_y", 0))
        await asyncio.wait_for(
            self._browser.scroll_by_pixels(x=scroll_x, y=scroll_y, smooth=False),
            timeout=self._action_timeout_seconds,
        )

    async def _build_follow_up_request(
        self,
        previous_response_id: Optional[str],
        call: ComputerToolTurn,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the payload for a follow-up request after executing an action."""
        screenshot_b64 = call.metadata.get("screenshot_base64")
        if not screenshot_b64:
            screenshot_bytes = await self._browser.screenshot()
            screenshot_b64 = encode_png_base64(screenshot_bytes)

        viewport_width, viewport_height = await self._browser.get_viewport_size()
        acknowledged_safety_checks = call.metadata.get("acknowledged_safety_checks") or []

        payload: Dict[str, Any] = {
            "model": self._model,
            "previous_response_id": previous_response_id,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_width,
                    "display_height": viewport_height,
                    "environment": "browser",
                }
            ],
            "input": [
                {
                    "type": "computer_call_output",
                    "call_id": call.call_id,
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                    "current_url": call.metadata.get("current_url"),
                    "acknowledged_safety_checks": acknowledged_safety_checks,
                }
            ],
            "truncation": "auto",
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        interaction_mode = metadata.get("interaction_mode")
        if interaction_mode:
            reminder = (
                "Reminder: You are in observe-only mode—analyze the UI and report findings without interacting."
                if interaction_mode == "observe_only"
                else "Reminder: You have approval to execute the requested action directly. Do not pause for confirmation; complete the interaction."
            )
            payload["input"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": reminder,
                        }
                    ],
                }
            )

        if call.status != "executed" and call.error_message:
            payload["input"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Execution error: {call.error_message}",
                        }
                    ],
                }
            )

        # Remove base64 payload after use to keep metadata lightweight
        call.metadata.pop("screenshot_base64", None)

        return payload

    def _update_loop_history(
        self,
        turn: ComputerToolTurn,
        history: Deque[Tuple[Tuple[str, ...], str]],
        window: int,
    ) -> Optional[Dict[str, Any]]:
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
        return None

    def _compute_turn_signature(self, turn: ComputerToolTurn) -> Optional[Tuple[str, ...]]:
        """Build a lightweight signature representing the Computer Use turn."""
        action_type_raw = turn.action_type or turn.parameters.get("type")
        if not action_type_raw:
            return None
        action_type = str(action_type_raw).strip().lower()
        if not action_type:
            return None

        params = turn.parameters or {}
        signature: List[str] = [action_type]

        def append_coord(label: str, x_value: Any, y_value: Any) -> None:
            coord = self._format_coordinate(x_value, y_value)
            if coord:
                signature.append(f"{label}:{coord}")

        append_coord("xy", params.get("x"), params.get("y"))
        append_coord(
            "start",
            params.get("start_x") or params.get("from_x"),
            params.get("start_y") or params.get("from_y"),
        )
        append_coord(
            "end",
            params.get("end_x") or params.get("to_x"),
            params.get("end_y") or params.get("to_y"),
        )

        if action_type == "type":
            text_value = params.get("text") or params.get("value") or ""
            signature.append(str(text_value).strip().lower()[:64])
        elif action_type == "key_press":
            key_value = params.get("key") or params.get("keys")
            if isinstance(key_value, list):
                key_value = "+".join(str(part) for part in key_value)
            signature.append(str(key_value).strip().lower())

        if "scroll_x" in params or "scroll_y" in params:
            signature.append(f"scroll:{params.get('scroll_x', 0)}:{params.get('scroll_y', 0)}")

        return tuple(component for component in signature if component)

    @staticmethod
    def _format_coordinate(x_value: Any, y_value: Any) -> Optional[str]:
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
    def _hash_base64(value: Optional[str]) -> Optional[str]:
        """Hash base64 strings for quick comparison without storing raw data."""
        if not value or not isinstance(value, str):
            return None
        try:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        except Exception:  # pragma: no cover - defensive
            return None

    def _build_initial_request(
        self,
        goal: str,
        screenshot_b64: str,
        viewport_width: int,
        viewport_height: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the payload for the initial Computer Use request."""
        context_lines = []
        if metadata.get("test_plan_name"):
            context_lines.append(f"Test plan: {metadata['test_plan_name']}")
        if metadata.get("test_case_name"):
            context_lines.append(f"Test case: {metadata['test_case_name']}")
        if metadata.get("step_number") is not None:
            context_lines.append(f"Step number: {metadata['step_number']}")
        if metadata.get("target"):
            context_lines.append(f"Target description: {metadata['target']}")
        if metadata.get("value"):
            context_lines.append(f"Associated value: {metadata['value']}")
        if metadata.get("current_url"):
            context_lines.append(f"Current URL: {metadata['current_url']}")
        if metadata.get("interaction_mode"):
            context_lines.append(f"Interaction mode: {metadata['interaction_mode']}")

        context_text = goal
        if context_lines:
            context_text = f"{goal}\n\nContext:\n" + "\n".join(f"- {line}" for line in context_lines)

        payload: Dict[str, Any] = {
            "model": self._model,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_width,
                    "display_height": viewport_height,
                    "environment": self._environment,
                }
            ],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": context_text,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    ],
                }
            ],
            "truncation": "auto",
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        return payload

    async def _create_response(self, payload: Dict[str, Any]) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        timeout = float(self._settings.openai_request_timeout_seconds)
        logger.debug("Calling OpenAI Responses API", extra={"model": payload.get("model")})
        return await self._client.responses.create(timeout=timeout, **payload)

    async def _ensure_browser_ready(self) -> None:
        """Ensure the browser session is started before execution."""
        if getattr(self._browser, "page", None) is None:
            await self._browser.start()

    async def _post_action_wait(self) -> None:
        """Wait for the configured stabilization interval."""
        wait_ms = self._settings.actions_computer_tool_stabilization_wait_ms
        if wait_ms > 0:
            await self._browser.wait(wait_ms)

    async def _maybe_get_current_url(self) -> Optional[str]:
        """Attempt to retrieve the current page URL from the browser driver."""
        get_url = getattr(self._browser, "get_page_url", None)
        if callable(get_url):
            try:
                return await get_url()
            except Exception:
                logger.debug("Failed to retrieve current URL from browser driver", exc_info=True)
        return None

    @staticmethod
    def _canonicalize_action_type(action_type: Optional[str]) -> Optional[str]:
        """Normalize OpenAI action types to the browser driver's expectations."""
        if not action_type:
            return action_type

        normalized = action_type.replace("-", "_").lower()
        alias_map = {
            "key_press": "keypress",
            "keypress": "keypress",
            "press_key": "keypress",
            "press": "keypress",
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
            "drag_and_drop": "drag",
            "dragdrop": "drag",
            "mouse_drag": "drag",
        }
        return alias_map.get(normalized, normalized)

    def _resolve_key_sequence(
        self,
        action: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[str]:
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

    async def _build_confirmation_request(
        self,
        previous_response_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct a follow-up request that confirms execution should proceed."""
        confirmation_text = (
            "Yes, proceed. Execute the requested action now without asking for additional confirmation."
        )
        target_text = metadata.get("target")
        if target_text:
            confirmation_text += f" Focus on: {target_text}."

        viewport_width, viewport_height = await self._browser.get_viewport_size()

        payload: Dict[str, Any] = {
            "model": self._model,
            "previous_response_id": previous_response_id,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_width,
                    "display_height": viewport_height,
                    "environment": self._environment,
                }
            ],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": confirmation_text}
                    ],
                }
            ],
            "truncation": "auto",
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        return payload

    def _is_action_allowed(self, action_type: Optional[str]) -> tuple[bool, Optional[str]]:
        """Determine if the requested action type is permitted in the current mode."""
        if not action_type:
            return False, "Computer Use response omitted action type information."
        if self._allowed_actions is None:
            return True, None
        if action_type in self._allowed_actions:
            return True, None

        return False, (
            f"Action '{action_type}' is not permitted in observe-only mode."
        )

    async def _enforce_domain_policy(self, action_type: Optional[str]) -> None:
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
            self._domain_matches(normalized_host, domain) for domain in self._allowed_domains
        ):
            raise ComputerUseExecutionError(
                f"Current domain '{hostname}' is not in the allowlist."
            )

        if self._blocked_domains and any(
            self._domain_matches(normalized_host, domain) for domain in self._blocked_domains
        ):
            raise ComputerUseExecutionError(
                f"Current domain '{hostname}' is blocked by policy."
            )

    def _save_turn_screenshot(
        self, screenshot_bytes: bytes, suffix: str, step_number: Optional[int]
    ) -> Optional[str]:
        """Persist a screenshot for observability if a debug logger is available."""
        if not self._debug_logger:
            return None
        name = f"computer_use_turn_{suffix}"
        return self._debug_logger.save_screenshot(
            screenshot_bytes, name=name, step_number=step_number
        )

    @staticmethod
    def _normalize_domain_set(domains: List[str]) -> Set[str]:
        normalized: Set[str] = set()
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

    def _maybe_acknowledge_safety(
        self, turn: ComputerToolTurn, metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Determine whether to acknowledge pending safety checks and return the IDs to send.

        When enabled, this allows trusted steps (e.g., window focus) to proceed without fail-fast.
        """
        if not turn.pending_safety_checks:
            return []

        if not self._should_auto_acknowledge_safety(metadata):
            return []

        eligible_codes = self._eligible_safety_codes(metadata)
        acknowledged_ids: List[str] = []

        for check in turn.pending_safety_checks:
            code = (check.get("code") or "").strip().lower()
            check_id = check.get("id")
            if eligible_codes is not None and code and code not in eligible_codes:
                continue
            if isinstance(check_id, str) and check_id:
                acknowledged_ids.append(check_id)

        return acknowledged_ids

    def _should_auto_acknowledge_safety(self, metadata: Dict[str, Any]) -> bool:
        """Resolve whether auto-acknowledgement is enabled for this run/step."""
        if "auto_ack_safety" in metadata:
            return bool(metadata.get("auto_ack_safety"))
        return self._auto_ack_safety

    def _eligible_safety_codes(self, metadata: Dict[str, Any]) -> Optional[Set[str]]:
        """
        Resolve the allowed safety codes for acknowledgement.

        Returns None when all codes are allowed (i.e., list is empty).
        """
        override = metadata.get("auto_ack_safety_codes")
        if override is not None:
            return self._normalize_safety_code_set(override)
        return self._auto_ack_codes

    @staticmethod
    def _normalize_safety_code_set(codes: Optional[List[str]]) -> Optional[Set[str]]:
        """Normalize safety check codes to a lowercase set; empty list -> None (all codes)."""
        if codes is None:
            return None
        normalized: Set[str] = set()
        for code in codes or []:
            if not code:
                continue
            value = str(code).strip().lower()
            if value:
                normalized.add(value)
        return normalized or None

    @property
    def _action_timeout_seconds(self) -> float:
        return max(self._settings.actions_computer_tool_action_timeout_ms / 1000.0, 0.5)


def encode_png_base64(data: bytes) -> str:
    """Encode PNG bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def normalize_response(response: Any) -> Dict[str, Any]:
    """Normalize OpenAI response objects into standard dictionaries."""
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise ComputerUseExecutionError(f"Unsupported response type: {type(response)}")


def normalize_key_sequence(key: str) -> str:
    """
    Convert Computer Use key strings to Playwright-compatible sequences.

    The model commonly emits uppercase tokens (e.g., "ENTER") or modifier
    combinations like "CTRL+ENTER". Playwright expects specific casing, so we
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
            return token_upper  # Playwright expects F-keys uppercase.

        if len(token) == 1:
            return token

        return token.capitalize()

    if "+" in key:
        parts = [part.strip() for part in key.split("+") if part.strip()]
        normalized_parts = [normalize_single(part) for part in parts]
        return "+".join(normalized_parts)

    return normalize_single(key.strip())


def extract_computer_calls(response_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract computer_call items from a response."""
    return [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "computer_call"
    ]


def extract_assistant_text(response_dict: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text output from a response."""
    messages = [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "message"
    ]
    texts: List[str] = []
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    combined = "\n".join(texts).strip()
    return combined or None


def _inject_context_metadata(turn: ComputerToolTurn, metadata: Dict[str, Any]) -> None:
    """Copy high-level context into the turn metadata for observability."""
    if not isinstance(turn.metadata, dict):
        return
    for key in ("step_number", "test_plan_name", "test_case_name", "target", "value"):
        if metadata.get(key) is not None:
            turn.metadata[key] = metadata[key]
    if metadata.get("safety_identifier") is not None:
        turn.metadata["safety_identifier"] = metadata["safety_identifier"]
    if metadata.get("interaction_mode") is not None:
        turn.metadata["interaction_mode"] = metadata["interaction_mode"]


def normalize_coordinates(
    x: Optional[float],
    y: Optional[float],
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """Normalize coordinates into the viewport bounds."""
    def _clamp(value: float, maximum: int) -> int:
        return max(0, min(int(round(value)), max(0, maximum - 1)))

    if x is None or y is None:
        return viewport_width // 2, viewport_height // 2
    return _clamp(x, viewport_width), _clamp(y, viewport_height)
