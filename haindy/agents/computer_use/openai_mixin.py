# mypy: disable-error-code=misc
"""OpenAI provider loop for Computer Use sessions."""

from __future__ import annotations

import base64
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from haindy.core.enhanced_types import ComputerToolTurn
from haindy.utils.model_logging import log_model_call_failure

from .common import (
    _inject_context_metadata,
    extract_assistant_text,
    extract_computer_call_actions,
    extract_computer_calls,
    normalize_response,
)
from .transports import ComputerUseTransport, OpenAIResponsesHTTPTransport
from .turn_result import ComputerUseCallResult, ComputerUseFollowUpBatch
from .types import ComputerUseExecutionError, ComputerUseSessionResult

logger = logging.getLogger("haindy.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class OpenAIComputerUseMixin:
    """OpenAI-specific request builders and execution loop."""

    _openai_transport: ComputerUseTransport | None

    @staticmethod
    def _apply_openai_localization_guidance(goal: str, environment: str) -> str:
        """Add OpenAI-specific screenshot targeting guidance."""
        if environment != "mobile_adb":
            return goal

        guidance = (
            "OPENAI CLICK LOCALIZATION RULES:\n"
            "- For a visible button, tap inside the button body itself, not nearby subtitle text, whitespace, or background.\n"
            "- Prefer the geometric center of the button rectangle when the full control is visible.\n"
            "- On bottom action rows, anchor to the actual button pill/rectangle, not the copy above it.\n"
            "- Do not stop after a tap unless the expected destination screen is visible in the latest screenshot.\n"
            "- If a tap does not change the screen, adjust the tap point within the visible control on the next turn instead of repeating the same off-target area.\n"
        )
        if guidance in goal:
            return goal
        return f"{goal}\n\n{guidance}"

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
        previous_response_id: str | None,
        stop_after_actions: bool = False,
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
        goal = self._apply_openai_localization_guidance(goal, environment)
        goal = self._apply_interaction_mode_guidance(goal, metadata)
        goal = self._apply_localization_protocol_guidance(goal, metadata)
        del initial_screenshot

        request_payload = self._build_openai_action_request(
            goal=goal,
            metadata=metadata,
            model=model,
            previous_response_id=previous_response_id,
        )

        response = await self._create_response(
            request_payload,
            agent=(
                "computer_use.openai.initial"
                if previous_response_id is None
                else "computer_use.openai.continuation"
            ),
            prompt=goal,
            request_payload_for_log=self._sanitize_payload_for_log(request_payload),
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        result.response_ids.append(response_dict.get("id", ""))

        turn_counter = 0
        current_response_id = response_dict.get("id")
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            computer_calls = extract_computer_calls(response_dict)
            assistant_message = self._consume_localization_response(
                extract_assistant_text(response_dict),
                metadata=metadata,
                provider="openai",
                model=model,
            )
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
                    previous_response_id=current_response_id,
                    metadata=metadata,
                    model=model,
                )
                response = await self._create_response(
                    confirmation_payload,
                    agent="computer_use.openai.confirmation",
                    prompt=f"{goal} (confirmation)",
                    request_payload_for_log=self._sanitize_payload_for_log(
                        confirmation_payload
                    ),
                    metadata={"environment": environment, **metadata},
                )
                response_dict = normalize_response(response)
                result.response_ids.append(response_dict.get("id", ""))
                current_response_id = response_dict.get("id")
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

            completed_call_turns: list[list[ComputerToolTurn]] = []
            should_stop = False

            for call in computer_calls:
                call_id = str(call.get("call_id") or "")
                pending_safety_checks = call.get("pending_safety_checks", []) or []
                turns = [
                    ComputerToolTurn(
                        call_id=call_id,
                        action_type=action.get("type", "screenshot"),
                        parameters=action,
                        response_id=response_dict.get("id"),
                        pending_safety_checks=(
                            pending_safety_checks if index == 0 else []
                        ),
                    )
                    for index, action in enumerate(extract_computer_call_actions(call))
                ]

                for turn in turns:
                    _inject_context_metadata(turn, metadata)

                if self._should_abort_on_safety(turns[0], result):
                    return result

                acknowledged_safety_checks = self._build_acknowledged_safety_checks(
                    turns[0]
                )
                if acknowledged_safety_checks:
                    turns[0].acknowledged = True
                    turns[0].metadata["acknowledged_safety_checks"] = (
                        acknowledged_safety_checks
                    )

                processed_turns: list[ComputerToolTurn] = []
                for action_index, turn in enumerate(turns, start=1):
                    try:
                        await self._execute_tool_action(
                            turn=turn,
                            metadata=metadata,
                            turn_index=(turn_counter * 100) + action_index,
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

                    processed_turns.append(turn)
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
                                "screenshot_hash": loop_detection.get(
                                    "screenshot_hash"
                                ),
                                "loop_window": loop_detection.get("loop_window"),
                            },
                            metadata_updates=loop_detection,
                            call_id_prefix="loop-detected",
                        )
                        should_stop = True
                        break

                completed_call_turns.append(processed_turns or [turns[0]])
                if should_stop:
                    break

            if should_stop:
                break

            if stop_after_actions:
                # Tool-call act mode: the coding agent handles validation.
                # Skip the follow-up API call entirely -- action_agent will
                # take a fresh screenshot from the driver.
                # However, if the only actions this turn were screenshots,
                # the model hasn't actually performed anything yet -- let it
                # see the screen and respond with a real action first.
                has_real_action = any(
                    turn.action_type != "screenshot"
                    for call_turns in completed_call_turns
                    for turn in call_turns
                )
                if has_real_action:
                    break

            follow_up_payload, follow_up_batch = await self._build_follow_up_request(
                previous_response_id=current_response_id,
                calls=completed_call_turns,
                metadata=metadata,
                model=model,
            )

            response = await self._create_response(
                follow_up_payload,
                agent="computer_use.openai.follow_up",
                prompt=f"{goal} (follow-up)",
                request_payload_for_log=self._sanitize_payload_for_log(
                    follow_up_payload
                ),
                screenshots=(
                    [("computer_use_follow_up", follow_up_batch.screenshot_bytes)]
                    if follow_up_batch.screenshot_bytes
                    else None
                ),
                metadata={"environment": environment, **metadata},
            )
            result.final_visual_frame = follow_up_batch.visual_frame
            result.final_artifact_frame = follow_up_batch.artifact_frame
            response_dict = normalize_response(response)
            result.response_ids.append(response_dict.get("id", ""))
            current_response_id = response_dict.get("id")

        self._openai_previous_response_id = current_response_id
        self._step_last_response = response_dict
        self._step_response_ids.extend(
            response_id for response_id in result.response_ids if response_id
        )

        return result

    async def _build_follow_up_request(
        self: _ComputerUseSession,
        previous_response_id: str | None,
        calls: list[list[ComputerToolTurn]],
        metadata: dict[str, Any],
        model: str | None = None,
    ) -> tuple[dict[str, Any], ComputerUseFollowUpBatch]:
        """Build the payload for a follow-up request after executing a model turn."""
        follow_up_batch = await self._build_follow_up_batch(
            call_groups=calls,
            metadata=metadata,
        )
        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "previous_response_id": previous_response_id,
            "tools": [{"type": "computer"}],
            "input": [],
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        for call_result in follow_up_batch.calls:
            payload["input"].append(
                self._build_follow_up_item(call_result, follow_up_batch)
            )
        extra_text_blocks: list[str] = []
        if (
            follow_up_batch.visual_frame is not None
            and follow_up_batch.visual_frame.kind == "patch"
        ):
            extra_text_blocks = [
                text
                for text in (
                    follow_up_batch.grounding_text,
                    self._build_visual_grounding_text(follow_up_batch),
                    follow_up_batch.reminder_text,
                )
                if text
            ]
        localization_prompt = self._build_follow_up_localization_prompt(
            follow_up_batch,
            metadata,
        )
        if localization_prompt:
            extra_text_blocks.append(localization_prompt)
        if extra_text_blocks:
            payload["input"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "\n\n".join(extra_text_blocks),
                        }
                    ],
                }
            )

        return payload, follow_up_batch

    def _build_follow_up_item(
        self: _ComputerUseSession,
        call_result: ComputerUseCallResult,
        follow_up_batch: ComputerUseFollowUpBatch,
    ) -> dict[str, Any]:
        """Build a single computer_call_output item for an executed computer_call."""
        item: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": call_result.call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": (
                    f"data:image/png;base64,{follow_up_batch.screenshot_base64}"
                ),
                "detail": "original",
            },
        }
        if call_result.acknowledged_safety_checks:
            item["acknowledged_safety_checks"] = list(
                call_result.acknowledged_safety_checks
            )
        return item

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

    @staticmethod
    def _build_visual_grounding_text(
        follow_up_batch: ComputerUseFollowUpBatch,
    ) -> str | None:
        """Describe the current visual frame when using patch mode."""
        frame = follow_up_batch.visual_frame
        if frame is None or frame.kind != "patch":
            return None
        bounds = frame.bounds
        full_width, full_height = frame.screen_size
        details = [
            "Visual frame mode: patch",
            f"Patch bounds in full screenshot coordinates: x={bounds.x}, y={bounds.y}, width={bounds.width}, height={bounds.height}",
            f"Original full screenshot size: width={full_width}, height={full_height}",
        ]
        if frame.parent_keyframe_id:
            details.append(f"Parent keyframe id: {frame.parent_keyframe_id}")
        if frame.target_bounds is not None:
            target = frame.target_bounds
            details.append(
                "Target bounds in full screenshot coordinates: "
                f"x={target.x}, y={target.y}, width={target.width}, height={target.height}"
            )
        if frame.diff_bounds is not None:
            diff = frame.diff_bounds
            details.append(
                "Detected changed-region bounds in full screenshot coordinates: "
                f"x={diff.x}, y={diff.y}, width={diff.width}, height={diff.height}"
            )
        details.append(
            "Interpret all future action coordinates in the full screenshot coordinate space, not the patch-local coordinate space."
        )
        return "\n".join(details)

    async def _reflect_openai_step(
        self: _ComputerUseSession,
        *,
        prompt: str,
        metadata: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Request a structured step verdict in the active OpenAI conversation."""
        if not self._openai_previous_response_id:
            raise RuntimeError(
                "OpenAI step reflection requires an active prior response id."
            )

        payload: dict[str, Any] = {
            "model": model,
            "previous_response_id": self._openai_previous_response_id,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "text": {"format": {"type": "json_object"}},
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        logger.info(
            "OpenAI step reflection provider handoff started",
            extra={
                "provider": "openai",
                "model": model,
                "payload_type": "step_reflection",
                "step_number": metadata.get("step_number"),
                "test_case": metadata.get("test_case_name"),
                "run_id": metadata.get("tool_mode_run_id"),
                "phase": metadata.get("validation_phase") or metadata.get("phase"),
                "configured_step_timeout_seconds": metadata.get(
                    "configured_step_timeout_seconds"
                ),
                "remaining_test_budget_seconds": metadata.get(
                    "remaining_test_budget_seconds"
                ),
                "effective_timeout_seconds": metadata.get("effective_timeout_seconds"),
            },
        )
        response = await self._create_response(
            payload,
            agent="computer_use.openai.step_reflection",
            prompt=prompt,
            request_payload_for_log=self._sanitize_payload_for_log(payload),
            metadata=metadata,
        )

        response_dict = normalize_response(response)
        response_id = response_dict.get("id")
        if response_id:
            self._openai_previous_response_id = response_id
            self._step_response_ids.append(response_id)
        self._step_last_response = response_dict

        return {
            "response_id": response_id,
            "response_ids": [response_id] if response_id else [],
            "response_dict": response_dict,
            "raw_text": extract_assistant_text(response_dict) or "",
        }

    def _build_openai_goal_input(
        self: _ComputerUseSession,
        goal: str,
        metadata: dict[str, Any],
    ) -> str:
        """Build the goal text sent to OpenAI for one action turn."""
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

        return context_text

    def _build_openai_action_request(
        self: _ComputerUseSession,
        goal: str,
        metadata: dict[str, Any],
        model: str | None = None,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the payload for an OpenAI Computer Use action turn."""
        context_text = self._build_openai_goal_input(goal, metadata)

        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "tools": [{"type": "computer"}],
        }
        if previous_response_id is None:
            payload["input"] = context_text
        else:
            payload["previous_response_id"] = previous_response_id
            payload["input"] = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": context_text}],
                }
            ]

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

    @staticmethod
    def _extract_follow_up_screenshot(payload: dict[str, Any]) -> bytes | None:
        """Decode the first logged screenshot from a follow-up payload."""
        for item in payload.get("input", []):
            output = item.get("output") if isinstance(item, dict) else None
            image_url = output.get("image_url", "") if isinstance(output, dict) else ""
            if isinstance(image_url, str) and "," in image_url:
                try:
                    return base64.b64decode(image_url.split(",", 1)[1])
                except Exception:
                    return None
        return None

    async def _create_response(
        self: _ComputerUseSession,
        payload: dict[str, Any],
        *,
        agent: str | None = None,
        prompt: str | None = None,
        request_payload_for_log: Any | None = None,
        screenshots: list[tuple[str, bytes]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        logger.debug(
            "Calling OpenAI Responses API", extra={"model": payload.get("model")}
        )
        client = self._client
        if client is None:
            raise ComputerUseExecutionError(
                "OpenAI client is not configured for this Computer Use session."
            )

        async def _log_success(
            response: Any,
            *,
            attempt_number: int,
            transport_name: str,
        ) -> None:
            if not agent or prompt is None:
                return
            await self._model_logger.log_outcome(
                agent=agent,
                model=str(payload.get("model") or self._model),
                prompt=prompt,
                request_payload=request_payload_for_log,
                response=response,
                screenshots=screenshots,
                metadata={
                    "provider": "openai",
                    "attempt_number": attempt_number,
                    "transport": transport_name,
                    **(metadata or {}),
                },
                outcome="success",
            )

        async def _log_failure(
            exc: BaseException,
            *,
            attempt_number: int,
            transport_name: str,
        ) -> None:
            if not agent or prompt is None:
                return
            await log_model_call_failure(
                self._model_logger,
                agent=agent,
                model=str(payload.get("model") or self._model),
                prompt=prompt,
                request_payload=request_payload_for_log,
                exception=exc,
                screenshots=screenshots,
                metadata={
                    "provider": "openai",
                    "attempt_number": attempt_number,
                    "transport": transport_name,
                    **(metadata or {}),
                },
            )

        transport = getattr(self, "_openai_transport", None)
        if transport is None:
            try:
                response = await client.responses.create(**payload)
            except Exception as exc:
                await _log_failure(exc, attempt_number=1, transport_name="client")
                raise
            await _log_success(response, attempt_number=1, transport_name="client")
            return response
        try:
            response = await transport.request(payload)
        except Exception as exc:
            await _log_failure(
                exc,
                attempt_number=1,
                transport_name=type(transport).__name__,
            )
            transport_mode = str(
                getattr(self._settings, "openai_cu_transport", "responses_websocket")
            ).strip()
            if transport_mode != "responses_websocket":
                raise
            logger.warning(
                "OpenAI CU WebSocket transport failed; falling back to HTTP transport",
                exc_info=True,
            )
            fallback_transport = OpenAIResponsesHTTPTransport(client)
            self._openai_transport = fallback_transport
            try:
                response = await fallback_transport.request(payload)
            except Exception as fallback_exc:
                await _log_failure(
                    fallback_exc,
                    attempt_number=2,
                    transport_name=type(fallback_transport).__name__,
                )
                raise
            await _log_success(
                response,
                attempt_number=2,
                transport_name=type(fallback_transport).__name__,
            )
            return response

        await _log_success(
            response,
            attempt_number=1,
            transport_name=type(transport).__name__,
        )
        return response
