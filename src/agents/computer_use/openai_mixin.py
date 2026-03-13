# mypy: disable-error-code=misc
"""OpenAI provider loop for Computer Use sessions."""

from __future__ import annotations

import base64
import json
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, cast

from src.core.enhanced_types import ComputerToolTurn

from .common import (
    _inject_context_metadata,
    extract_assistant_text,
    extract_computer_call_actions,
    extract_computer_calls,
    normalize_response,
)
from .transports import ComputerUseTransport
from .turn_result import ComputerUseCallResult, ComputerUseFollowUpBatch
from .types import ComputerUseSessionResult
from .visual_state import (
    CartographyMap,
    CartographyTarget,
    VisualBounds,
    VisualFrame,
    build_keyframe,
)

logger = logging.getLogger("src.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class OpenAIComputerUseMixin:
    """OpenAI-specific request builders and execution loop."""

    _openai_transport: ComputerUseTransport | None
    _current_keyframe: VisualFrame | None
    _last_visual_frame: VisualFrame | None

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
        del initial_screenshot

        request_payload = self._build_openai_action_request(
            goal=goal,
            metadata=metadata,
            model=model,
            previous_response_id=previous_response_id,
        )

        response = await self._create_response(request_payload)
        await self._model_logger.log_call(
            agent=(
                "computer_use.openai.initial"
                if previous_response_id is None
                else "computer_use.openai.continuation"
            ),
            model=model,
            prompt=goal,
            request_payload=self._sanitize_payload_for_log(request_payload),
            response=response,
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
                    previous_response_id=current_response_id,
                    metadata=metadata,
                    model=model,
                )
                response = await self._create_response(confirmation_payload)
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
            should_refresh_visual_context = False

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

                    if turn.metadata.get("visual_context_invalidated"):
                        should_refresh_visual_context = True
                        break

                completed_call_turns.append(processed_turns or [turns[0]])
                if should_stop or should_refresh_visual_context:
                    break

            if should_stop:
                break

            follow_up_metadata = dict(metadata)
            follow_up_metadata["_defer_openai_cartography"] = True
            follow_up_batch = await self._build_follow_up_batch(
                call_groups=completed_call_turns,
                metadata=follow_up_metadata,
            )
            cartography_outputs_consumed = False
            if self._should_request_openai_cartography_turn(
                follow_up_batch=follow_up_batch,
                metadata=metadata,
            ):
                try:
                    (
                        cartography_payload,
                        cartography_prompt,
                    ) = self._build_openai_cartography_request(
                        previous_response_id=current_response_id,
                        follow_up_batch=follow_up_batch,
                        metadata=metadata,
                        model=model,
                    )
                    cartography_response = await self._create_response(
                        cartography_payload
                    )
                    cartography_outputs_consumed = True
                    await self._model_logger.log_call(
                        agent="computer_use.openai.cartography",
                        model=str(cartography_payload["model"]),
                        prompt=cartography_prompt,
                        request_payload=self._sanitize_payload_for_log(
                            cartography_payload
                        ),
                        response=cartography_response,
                        screenshots=(
                            [
                                (
                                    "computer_use_cartography",
                                    follow_up_batch.screenshot_bytes,
                                )
                            ]
                            if follow_up_batch.screenshot_bytes
                            else None
                        ),
                        metadata={"environment": environment, **metadata},
                    )
                    cartography_response_dict = normalize_response(cartography_response)
                    result.response_ids.append(cartography_response_dict.get("id", ""))
                    current_response_id = cartography_response_dict.get("id")
                    cartography = self._parse_openai_cartography_response(
                        response_dict=cartography_response_dict,
                        frame=(
                            self._current_keyframe
                            or follow_up_batch.visual_frame
                            or build_keyframe(
                                follow_up_batch.screenshot_bytes,
                                source="follow_up_capture",
                            )
                        ),
                        metadata=metadata,
                        model=str(cartography_payload["model"]),
                    )
                    if cartography is not None:
                        self._apply_openai_cartography_to_follow_up_batch(
                            follow_up_batch=follow_up_batch,
                            cartography=cartography,
                        )
                except Exception:
                    logger.warning(
                        "OpenAI in-thread cartography generation failed",
                        exc_info=True,
                    )

            follow_up_payload = self._build_follow_up_request_from_batch(
                previous_response_id=current_response_id,
                follow_up_batch=follow_up_batch,
                metadata=metadata,
                model=model,
                include_call_outputs=not cartography_outputs_consumed,
                include_cartography_grounding=False,
                continuation_text=(
                    self._build_openai_cartography_continuation_text()
                    if cartography_outputs_consumed
                    else None
                ),
            )

            response = await self._create_response(follow_up_payload)
            await self._model_logger.log_call(
                agent="computer_use.openai.follow_up",
                model=model,
                prompt=f"{goal} (follow-up)",
                request_payload=self._sanitize_payload_for_log(follow_up_payload),
                response=response,
                screenshots=(
                    [("computer_use_follow_up", follow_up_batch.screenshot_bytes)]
                    if follow_up_batch.screenshot_bytes
                    else None
                ),
                metadata={"environment": environment, **metadata},
            )
            result.final_visual_frame = follow_up_batch.visual_frame
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
        payload = self._build_follow_up_request_from_batch(
            previous_response_id=previous_response_id,
            follow_up_batch=follow_up_batch,
            metadata=metadata,
            model=model,
        )

        return payload, follow_up_batch

    def _build_follow_up_request_from_batch(
        self: _ComputerUseSession,
        *,
        previous_response_id: str | None,
        follow_up_batch: ComputerUseFollowUpBatch,
        metadata: dict[str, Any],
        model: str | None = None,
        include_call_outputs: bool = True,
        include_cartography_grounding: bool = True,
        continuation_text: str | None = None,
    ) -> dict[str, Any]:
        """Build a follow-up payload from a precomputed batch."""
        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "previous_response_id": previous_response_id,
            "tools": [{"type": "computer"}],
            "input": [],
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        if include_call_outputs:
            for call_result in follow_up_batch.calls:
                payload["input"].append(
                    self._build_follow_up_item(call_result, follow_up_batch)
                )
        visual_frame = follow_up_batch.visual_frame
        include_patch_grounding = (
            visual_frame is not None and visual_frame.kind == "patch"
        )
        include_cartography_grounding = (
            include_cartography_grounding and follow_up_batch.cartography is not None
        )
        extra_text_blocks: list[str] = []
        if (
            include_patch_grounding
            or include_cartography_grounding
            or follow_up_batch.reminder_text
            or continuation_text
        ):
            extra_text_blocks = [
                text
                for text in (
                    follow_up_batch.grounding_text,
                    (
                        self._build_visual_grounding_text(follow_up_batch)
                        if include_patch_grounding
                        else None
                    ),
                    (
                        self._build_cartography_grounding_text(follow_up_batch)
                        if include_cartography_grounding
                        else None
                    ),
                    follow_up_batch.reminder_text,
                    continuation_text,
                )
                if text
            ]
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

        return payload

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

    @staticmethod
    def _build_cartography_grounding_text(
        follow_up_batch: ComputerUseFollowUpBatch,
    ) -> str | None:
        """Describe the available cartography map for follow-up turns."""
        cartography = follow_up_batch.cartography
        if cartography is None or not cartography.targets:
            return None

        payload = {
            "origin": follow_up_batch.cartography_origin or "unknown",
            "frame_id": cartography.frame_id,
            "provider": cartography.provider,
            "model": cartography.model,
            "targets": [
                {
                    "target_id": target.target_id,
                    "label": target.label,
                    "bbox": {
                        "x": target.bounds.x,
                        "y": target.bounds.y,
                        "width": target.bounds.width,
                        "height": target.bounds.height,
                    },
                    "interaction_point": {
                        "x": target.interaction_point[0],
                        "y": target.interaction_point[1],
                    },
                    "confidence": round(target.confidence, 4),
                }
                for target in cartography.targets
            ],
        }
        return (
            "Visual cartography (full-screen coordinates; use it as target guidance "
            "alongside the latest screenshot):\n"
            + json.dumps(payload, separators=(",", ":"))
        )

    def _should_request_openai_cartography_turn(
        self: _ComputerUseSession,
        *,
        follow_up_batch: ComputerUseFollowUpBatch,
        metadata: dict[str, Any],
    ) -> bool:
        """Return True when the next OpenAI follow-up should begin with cartography."""
        if follow_up_batch.cartography is not None:
            return False
        if follow_up_batch.visual_frame is None:
            return False
        if follow_up_batch.visual_frame.kind != "keyframe":
            return False
        return bool(str(metadata.get("target") or "").strip())

    def _build_openai_cartography_request(
        self: _ComputerUseSession,
        *,
        previous_response_id: str | None,
        follow_up_batch: ComputerUseFollowUpBatch,
        metadata: dict[str, Any],
        model: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Build an in-thread cartography request that consumes the latest screenshot."""
        target_text = str(metadata.get("target") or "").strip()
        prompt = (
            "Generate visual cartography for the current task target using the latest "
            "screenshot already provided in this turn. Do not call tools, do not "
            "propose actions, and return ONLY valid JSON with this shape: "
            '{"targets":[{"target_id":"target_1","label":"visible label or short descriptor",'
            '"bbox":{"x":0,"y":0,"width":0,"height":0},'
            '"interaction_point":{"x":0,"y":0},"confidence":0.0}]}. '
            f"Requested target: {target_text!r}. "
            "Use absolute pixel coordinates in the full screenshot coordinate space. "
            'If the target is not visible, return {"targets":[]}.'
        )
        payload: dict[str, Any] = {
            "model": model or self._openai_model,
            "previous_response_id": previous_response_id,
            "tools": [{"type": "computer"}],
            "input": [],
            "text": {"format": {"type": "json_object"}},
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        for call_result in follow_up_batch.calls:
            payload["input"].append(
                self._build_follow_up_item(call_result, follow_up_batch)
            )

        extra_text = prompt
        if follow_up_batch.grounding_text:
            extra_text = f"{follow_up_batch.grounding_text}\n\n{prompt}"
        payload["input"].append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": extra_text}],
            }
        )
        return payload, prompt

    @staticmethod
    def _build_openai_cartography_continuation_text() -> str:
        """Return the continuation prompt used after an in-thread cartography turn."""
        return (
            "Continue with the current task using the latest screenshot and your "
            "immediately prior cartography analysis. If the requested visible state "
            "is already achieved, report success and stop. Otherwise use the "
            "computer tool now."
        )

    def _parse_openai_cartography_response(
        self: _ComputerUseSession,
        *,
        response_dict: dict[str, Any],
        frame: VisualFrame,
        metadata: dict[str, Any],
        model: str,
    ) -> CartographyMap | None:
        """Parse a cartography response into a structured map."""
        raw_text = extract_assistant_text(response_dict)
        if not raw_text:
            return None

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.debug(
                "OpenAI cartography response was not valid JSON",
                extra={"response_text": raw_text},
            )
            return None

        raw_targets = parsed.get("targets", []) if isinstance(parsed, dict) else []
        target_text = str(metadata.get("target") or "").strip()
        targets: list[CartographyTarget] = []
        for index, item in enumerate(raw_targets, start=1):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or {}
            point = item.get("interaction_point") or {}
            try:
                bounds = VisualBounds(
                    x=int(bbox.get("x", 0)),
                    y=int(bbox.get("y", 0)),
                    width=int(bbox.get("width", 0)),
                    height=int(bbox.get("height", 0)),
                )
                target = CartographyTarget(
                    target_id=str(item.get("target_id") or f"target_{index}"),
                    label=str(item.get("label") or target_text).strip() or target_text,
                    bounds=bounds,
                    interaction_point=(
                        int(point.get("x", bounds.x + (bounds.width // 2))),
                        int(point.get("y", bounds.y + (bounds.height // 2))),
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                )
            except (TypeError, ValueError):
                continue
            if target.bounds.is_empty():
                continue
            targets.append(target)

        return CartographyMap(
            frame_id=frame.frame_id,
            targets=tuple(targets),
            model=model,
            provider="openai",
        )

    def _apply_openai_cartography_to_follow_up_batch(
        self: _ComputerUseSession,
        *,
        follow_up_batch: ComputerUseFollowUpBatch,
        cartography: CartographyMap,
    ) -> None:
        """Persist in-thread cartography onto the active keyframe and batch."""
        current_keyframe = self._current_keyframe
        if current_keyframe is None:
            return
        updated_keyframe = build_keyframe(
            current_keyframe.image_bytes,
            source=current_keyframe.source,
            cartography=cartography,
        )
        self._current_keyframe = updated_keyframe
        if (
            self._last_visual_frame is not None
            and self._last_visual_frame.kind == "keyframe"
            and self._last_visual_frame.frame_id == updated_keyframe.frame_id
        ):
            self._last_visual_frame = updated_keyframe
        if (
            follow_up_batch.visual_frame is not None
            and follow_up_batch.visual_frame.kind == "keyframe"
            and follow_up_batch.visual_frame.frame_id == updated_keyframe.frame_id
        ):
            follow_up_batch.visual_frame = updated_keyframe
        follow_up_batch.cartography = cartography
        follow_up_batch.cartography_origin = "current_keyframe"

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

        response = await self._create_response(payload)
        await self._model_logger.log_call(
            agent="computer_use.openai.step_reflection",
            model=model,
            prompt=prompt,
            request_payload=self._sanitize_payload_for_log(payload),
            response=response,
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

    async def _generate_openai_cartography_map(
        self: _ComputerUseSession,
        frame: VisualFrame,
        metadata: dict[str, Any],
    ) -> CartographyMap | None:
        """Generate a lightweight target map for the current keyframe."""
        target_text = str(metadata.get("target") or "").strip()
        if not target_text:
            return None

        prompt = (
            "You are generating a strictly visual cartography map for a computer-use "
            "agent. Look only at the screenshot. Do not infer DOM or hidden state. "
            "Find the best visible interactable target matching the requested target "
            f"description: {target_text!r}. "
            "Return ONLY valid JSON with this shape: "
            '{"targets":[{"target_id":"target_1","label":"visible label or short descriptor",'
            '"bbox":{"x":0,"y":0,"width":0,"height":0},'
            '"interaction_point":{"x":0,"y":0},"confidence":0.0}]}. '
            "Use absolute pixel coordinates in the screenshot coordinate space. "
            'If the target is not visible, return {"targets":[]}.'
        )
        image_b64 = base64.b64encode(frame.image_bytes).decode("utf-8")
        payload = {
            "model": getattr(self._settings, "cu_cartography_model", "").strip()
            or self._openai_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
        }
        response = await self._create_response(payload)
        return self._parse_openai_cartography_response(
            response_dict=normalize_response(response),
            frame=frame,
            metadata=metadata,
            model=str(payload["model"]),
        )

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
        self: _ComputerUseSession, payload: dict[str, Any]
    ) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        logger.debug(
            "Calling OpenAI Responses API", extra={"model": payload.get("model")}
        )
        transport = getattr(self, "_openai_transport", None)
        if transport is None:
            return await self._client.responses.create(**payload)
        return await transport.request(payload)
