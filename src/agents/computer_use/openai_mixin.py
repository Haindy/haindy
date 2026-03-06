# mypy: disable-error-code=misc
"""OpenAI provider loop for Computer Use sessions."""

from __future__ import annotations

import base64
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from src.core.enhanced_types import ComputerToolTurn

from .common import (
    _inject_context_metadata,
    encode_png_base64,
    extract_assistant_text,
    extract_computer_calls,
    normalize_response,
)
from .types import ComputerUseSessionResult

logger = logging.getLogger("src.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class OpenAIComputerUseMixin:
    """OpenAI-specific request builders and execution loop."""

    async def _run_openai(
        self: _ComputerUseSession,
        *,
        goal: str,
        initial_screenshot: bytes | None,
        metadata: dict[str, Any],
        environment: str,
        cache_label: str | None,
        cache_action: str,
        use_cache: bool,
        model: str,
    ) -> ComputerUseSessionResult:
        result = ComputerUseSessionResult()

        await self._ensure_automation_driver_ready()
        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
        goal = self._wrap_goal_for_mobile(
            goal, environment, viewport_width, viewport_height
        )
        screenshot = initial_screenshot or await self._automation_driver.screenshot()
        screenshot_b64 = encode_png_base64(screenshot)
        tool_environment = self._map_openai_environment(environment)

        request_payload = self._build_initial_request(
            goal=goal,
            screenshot_b64=screenshot_b64,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            metadata=metadata,
            environment=tool_environment,
            model=model,
        )

        response = await self._create_response(request_payload)
        await self._model_logger.log_call(
            agent="computer_use.openai.initial",
            model=model,
            prompt=goal,
            request_payload=self._sanitize_payload_for_log(request_payload),
            response=response,
            screenshots=[("computer_use_initial", screenshot)],
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        result.response_ids.append(response_dict.get("id", ""))

        turn_counter = 0
        previous_response_id = response_dict.get("id")
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=loop_window)

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
                    environment=tool_environment,
                    model=model,
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
                    f"Computer Use max turns exceeded after {turn_counter} turns "
                    f"(limit: {max_turns})."
                )
                logger.warning(
                    "Computer Use max turns exceeded",
                    extra={
                        "max_turns": max_turns,
                        "turn_count": turn_counter,
                        "step_number": metadata.get("step_number"),
                    },
                )
                self._append_terminal_failure_turn(
                    result=result,
                    metadata=metadata,
                    response_id=response_dict.get("id"),
                    reason=message,
                    code="max_turns_exceeded",
                    parameters={
                        "turn_count": turn_counter,
                        "max_turns": max_turns,
                    },
                    metadata_updates={
                        "turn_count": turn_counter,
                        "max_turns": max_turns,
                    },
                    call_id_prefix="turn-limit",
                )
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

            if self._should_abort_on_safety(turn, result):
                return result

            try:
                await self._execute_tool_action(
                    turn=turn,
                    metadata=metadata,
                    turn_index=turn_counter,
                    normalized_coords=False,
                    allow_unknown=False,
                    environment=environment,
                    cache_label=cache_label,
                    cache_action=cache_action,
                    use_cache=use_cache,
                )
            except Exception as exc:
                turn.status = "failed"
                turn.error_message = str(exc)
                logger.exception(
                    "Computer Use action execution failed",
                    extra={"call_id": turn.call_id},
                )

            result.actions.append(turn)
            metadata["_auto_confirmation_attempts"] = 0

            if self._is_observe_only_policy_violation(turn):
                reason = turn.error_message or (
                    "Observe-only policy violation blocked action execution."
                )
                self._mark_terminal_failure(
                    result=result,
                    reason=reason,
                    code="observe_only_policy_violation",
                )
                logger.warning(
                    "Observe-only policy violation detected (openai); aborting action",
                    extra={
                        "step_number": metadata.get("step_number"),
                        "action_type": turn.action_type,
                        "call_id": turn.call_id,
                    },
                )
                return result

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
                self._append_terminal_failure_turn(
                    result=result,
                    metadata=metadata,
                    response_id=response_dict.get("id"),
                    reason=message,
                    code="loop_detected",
                    parameters={
                        "signature": loop_detection.get("signature"),
                        "screenshot_hash": loop_detection.get("screenshot_hash"),
                        "loop_window": loop_detection.get("loop_window"),
                    },
                    metadata_updates=loop_detection,
                    call_id_prefix="loop-detected",
                )
                break

            follow_up_payload = await self._build_follow_up_request(
                previous_response_id=previous_response_id,
                call=turn,
                metadata=metadata,
                environment=tool_environment,
                model=model,
            )

            response = await self._create_response(follow_up_payload)
            follow_up_screenshot: bytes | None = None
            try:
                image_url = (
                    follow_up_payload.get("input", [{}])[0]
                    .get("output", {})
                    .get("image_url", "")
                )
                if isinstance(image_url, str) and "," in image_url:
                    follow_up_screenshot = base64.b64decode(image_url.split(",", 1)[1])
            except Exception:
                follow_up_screenshot = None
            await self._model_logger.log_call(
                agent="computer_use.openai.follow_up",
                model=model,
                prompt=f"{goal} (follow-up)",
                request_payload=self._sanitize_payload_for_log(follow_up_payload),
                response=response,
                screenshots=(
                    [("computer_use_follow_up", follow_up_screenshot)]
                    if follow_up_screenshot
                    else None
                ),
                metadata={"environment": environment, **metadata},
            )
            response_dict = normalize_response(response)
            result.response_ids.append(response_dict.get("id", ""))
            previous_response_id = response_dict.get("id")

        return result

    async def _build_follow_up_request(
        self: _ComputerUseSession,
        previous_response_id: str | None,
        call: ComputerToolTurn,
        metadata: dict[str, Any],
        environment: str = "browser",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build the payload for a follow-up request after executing an action."""
        screenshot_b64 = call.metadata.get("screenshot_base64")
        if not screenshot_b64:
            screenshot_bytes = await self._automation_driver.screenshot()
            screenshot_b64 = encode_png_base64(screenshot_bytes)

        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
        acknowledged_safety_checks = self._build_acknowledged_safety_checks(call)
        if acknowledged_safety_checks:
            call.acknowledged = True
            call.metadata["acknowledged_safety_checks"] = acknowledged_safety_checks

        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "previous_response_id": previous_response_id,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_width,
                    "display_height": viewport_height,
                    "environment": environment,
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
                "Reminder: You are in observe-only mode - analyze the UI and report findings without interacting."
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

        call.metadata.pop("screenshot_base64", None)
        return payload

    def _build_acknowledged_safety_checks(
        self: _ComputerUseSession,
        call: ComputerToolTurn,
    ) -> list[dict[str, str]]:
        """Return safety checks to acknowledge when policy allows auto-approval."""
        if not call.pending_safety_checks:
            return []

        policy = self._settings.cu_safety_policy
        fail_fast = bool(
            getattr(self._settings, "actions_computer_tool_fail_fast_on_safety", True)
        )
        allow_auto_approve_override = bool(
            call.metadata.get("allow_safety_auto_approve")
        )
        should_auto_ack = policy == "auto_approve" and (
            not fail_fast or allow_auto_approve_override
        )
        if not should_auto_ack:
            return []

        acknowledged: list[dict[str, str]] = []
        for check in call.pending_safety_checks:
            if not isinstance(check, dict):
                continue
            safety_id = str(check.get("id") or "").strip()
            if not safety_id:
                continue

            check_payload: dict[str, str] = {"id": safety_id}
            code = check.get("code")
            if isinstance(code, str) and code:
                check_payload["code"] = code
            message = check.get("message")
            if isinstance(message, str) and message:
                check_payload["message"] = message
            acknowledged.append(check_payload)

        return acknowledged

    def _build_initial_request(
        self: _ComputerUseSession,
        goal: str,
        screenshot_b64: str,
        viewport_width: int,
        viewport_height: int,
        metadata: dict[str, Any],
        environment: str = "browser",
        model: str | None = None,
    ) -> dict[str, Any]:
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
            context_text = f"{goal}\n\nContext:\n" + "\n".join(
                f"- {line}" for line in context_lines
            )

        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "tools": [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_width,
                    "display_height": viewport_height,
                    "environment": environment,
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

    @staticmethod
    def _sanitize_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
        def _scrub(value: Any) -> Any:
            if isinstance(value, dict):
                if (
                    value.get("type") == "base64"
                    and "data" in value
                    and isinstance(value.get("data"), str)
                ):
                    copied = dict(value)
                    copied["data"] = "<<attached screenshot>>"
                    return copied
                return {k: _scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_scrub(item) for item in value]
            if isinstance(value, str) and value.startswith("data:image"):
                return "<<attached screenshot>>"
            return value

        return cast(dict[str, Any], _scrub(payload))

    async def _create_response(
        self: _ComputerUseSession, payload: dict[str, Any]
    ) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        logger.debug(
            "Calling OpenAI Responses API", extra={"model": payload.get("model")}
        )
        return await self._client.responses.create(**payload)
