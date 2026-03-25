# mypy: disable-error-code=misc
"""Google provider loop for Computer Use sessions."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import warnings
from collections import deque
from typing import TYPE_CHECKING, Any

try:
    import google.genai as _genai
except Exception:
    genai: Any | None = None
else:
    genai = _genai

from haindy.core.enhanced_types import ComputerToolTurn
from haindy.runtime.environment import runtime_environment_spec

from .common import (
    _inject_context_metadata,
    extract_assistant_text,
    extract_google_function_call_envelopes,
    normalize_response,
)
from .turn_result import ComputerUseActionResult
from .types import (
    ComputerUseExecutionError,
    ComputerUseSessionResult,
    GoogleFunctionCallEnvelope,
    _strip_bytes,
)

logger = logging.getLogger("haindy.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class GoogleComputerUseMixin:
    """Google-specific request builders and execution loop."""

    _GOOGLE_INTERACTIONS_WARNING_MESSAGES: tuple[str, ...] = (
        "Interactions usage is experimental and may change in future versions.",
        "Async interactions client cannot use aiohttp, fallingback to httpx.",
    )

    @staticmethod
    def _sanitize_google_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
        def _scrub(value: Any) -> Any:
            if isinstance(value, dict):
                scrubbed: dict[str, Any] = {}
                for key, item in value.items():
                    if key == "data" and isinstance(item, str):
                        scrubbed[key] = f"<<base64:{len(item)}>>"
                    elif (
                        key == "image_url"
                        and isinstance(item, str)
                        and item.startswith("data:image/")
                    ):
                        scrubbed[key] = "<<data-image-url>>"
                    else:
                        scrubbed[key] = _scrub(item)
                return scrubbed
            if isinstance(value, list):
                return [_scrub(item) for item in value]
            return value

        sanitized = _scrub(payload)
        if isinstance(sanitized, dict):
            return sanitized
        return {}

    @classmethod
    def _summarize_google_payload_for_log(
        cls, payload: dict[str, Any]
    ) -> dict[str, Any]:
        input_items = payload.get("input")
        input_summary: list[dict[str, Any]] = []
        if isinstance(input_items, list):
            for index, item in enumerate(input_items[:8], start=1):
                if not isinstance(item, dict):
                    input_summary.append({"index": index, "type": type(item).__name__})
                    continue
                summary: dict[str, Any] = {
                    "index": index,
                    "type": str(item.get("type") or "").strip() or "unknown",
                }
                text_value = item.get("text")
                if isinstance(text_value, str):
                    summary["text_length"] = len(text_value)
                    summary["text_preview"] = text_value[:200]
                if item.get("type") == "image" and isinstance(item.get("data"), str):
                    summary["image_base64_length"] = len(str(item["data"]))
                result = item.get("result")
                if isinstance(result, dict):
                    summary["result_status"] = result.get("status")
                    result_items = result.get("items")
                    if isinstance(result_items, list):
                        summary["result_item_types"] = [
                            str(result_item.get("type") or "").strip() or "unknown"
                            for result_item in result_items[:6]
                            if isinstance(result_item, dict)
                        ]
                        for result_item in result_items:
                            if not isinstance(result_item, dict):
                                continue
                            result_text = result_item.get("text")
                            if isinstance(result_text, str):
                                summary["result_text_preview"] = result_text[:200]
                                break
                input_summary.append(summary)
        tool_types = [
            str(tool.get("type") or "").strip() or "unknown"
            for tool in (payload.get("tools") or [])
            if isinstance(tool, dict)
        ]
        return {
            "api_surface": str(payload.get("api_surface") or "").strip() or None,
            "model": str(payload.get("model") or "").strip() or None,
            "previous_interaction_id": payload.get("previous_interaction_id"),
            "response_mime_type": payload.get("response_mime_type"),
            "has_response_format": payload.get("response_format") is not None,
            "input_count": len(input_items) if isinstance(input_items, list) else None,
            "input_summary": input_summary or None,
            "tool_types": tool_types or None,
        }

    async def _run_google(
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
        previous_interaction_id: str | None,
        stop_after_actions: bool = False,
    ) -> ComputerUseSessionResult:
        result = ComputerUseSessionResult()

        await self._ensure_automation_driver_ready()
        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
        goal = self._apply_interaction_mode_guidance(goal, metadata)
        goal = self._apply_localization_protocol_guidance(goal, metadata)
        wrapped_goal = (
            self._wrap_goal_for_google(goal, environment)
            if previous_interaction_id is None
            else goal
        )
        include_screenshot = (
            previous_interaction_id is None or self._last_visual_frame is None
        )
        request_screenshot = (
            initial_screenshot if include_screenshot and initial_screenshot else None
        )
        if include_screenshot and request_screenshot is None:
            request_screenshot = await self._automation_driver.screenshot()
        if include_screenshot:
            self._prime_initial_visual_state_for_request(
                request_screenshot,
                source=(
                    "initial_screenshot"
                    if previous_interaction_id is None
                    else "continuation_screenshot"
                ),
            )
        request_payload, logged_screenshot = self._build_google_initial_request(
            goal=wrapped_goal,
            screenshot_bytes=request_screenshot,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            environment=environment,
            model=model,
            previous_interaction_id=previous_interaction_id,
        )
        response = await self._create_google_response(request_payload)
        await self._model_logger.log_call(
            agent=(
                "computer_use.google.initial"
                if previous_interaction_id is None
                else "computer_use.google.continuation"
            ),
            model=model,
            prompt=wrapped_goal,
            request_payload={
                "provider": "google",
                "environment": environment,
                "payload_type": (
                    "initial" if previous_interaction_id is None else "continuation"
                ),
                "api_surface": "interactions",
                "request": self._sanitize_google_payload_for_log(request_payload),
            },
            response=response,
            screenshots=(
                [("computer_use_initial", logged_screenshot)]
                if logged_screenshot
                else None
            ),
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        response_id = response_dict.get("id")
        current_interaction_id = response_id or previous_interaction_id
        if response_id:
            result.response_ids.append(response_id)

        turn_counter = 0
        scroll_turns = 0
        consecutive_ignored = 0
        max_turn_hit = False
        max_turn_reason: str | None = None
        max_turn_code: str | None = None
        last_assistant_text: str | None = None
        last_response_dict: dict[str, Any] | None = response_dict
        ambiguous_reask_attempts = 0
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            call_envelopes = extract_google_function_call_envelopes(response_dict)
            assistant_text = self._consume_localization_response(
                extract_assistant_text(response_dict),
                metadata=metadata,
                provider="google",
                model=model,
            )
            if assistant_text:
                result.final_output = assistant_text
                last_assistant_text = assistant_text
            result.last_response = response_dict
            last_response_dict = response_dict

            block_reason, block_reason_message = (
                self._extract_google_prompt_block_feedback(response_dict)
            )
            if block_reason:
                failure_reason = (
                    "Google prompt was blocked before tool execution "
                    f"(reason: {block_reason})."
                )
                if block_reason_message:
                    failure_reason = f"{failure_reason} {block_reason_message}"
                self._append_terminal_failure_turn(
                    result=result,
                    metadata=metadata,
                    response_id=response_dict.get("id"),
                    reason=failure_reason,
                    code="google_prompt_blocked",
                    parameters={
                        "block_reason": block_reason,
                        "block_reason_message": block_reason_message,
                    },
                    metadata_updates={
                        "google_block_reason": block_reason,
                        "google_block_reason_message": block_reason_message,
                    },
                )
                logger.warning(
                    "Google prompt blocked before tool execution",
                    extra={
                        "response_id": response_dict.get("id"),
                        "block_reason": block_reason,
                        "block_reason_message": block_reason_message,
                    },
                )
                break

            if not call_envelopes:
                break

            self._google_turn_index += 1
            google_turn_index = self._google_turn_index
            ambiguous_names = self._extract_google_ambiguous_function_names(
                call_envelopes
            )
            if ambiguous_names:
                if ambiguous_reask_attempts >= 1:
                    ambiguous_list = ", ".join(ambiguous_names)
                    self._append_terminal_failure_turn(
                        result=result,
                        metadata=metadata,
                        response_id=response_dict.get("id"),
                        reason=(
                            "Google FunctionCall batch is ambiguous: duplicate "
                            "function names without call IDs "
                            f"({ambiguous_list}) remained after controlled re-ask."
                        ),
                        code="google_ambiguous_function_call_batch",
                        parameters={
                            "ambiguous_function_names": ambiguous_names,
                            "reask_attempts": ambiguous_reask_attempts,
                        },
                        metadata_updates={
                            "ambiguous_function_names": ambiguous_names,
                            "reask_attempts": ambiguous_reask_attempts,
                        },
                    )
                    break

                ambiguous_reask_attempts += 1
                reask_payload = self._build_google_single_call_reask_request(
                    previous_interaction_id=current_interaction_id,
                    environment=environment,
                    model=model,
                )
                response = await self._create_google_response(reask_payload)
                await self._model_logger.log_call(
                    agent="computer_use.google.reask",
                    model=model,
                    prompt=f"{goal} (ambiguous-batch-reask)",
                    request_payload={
                        "provider": "google",
                        "environment": environment,
                        "payload_type": "ambiguous_reask",
                        "api_surface": "interactions",
                        "request": self._sanitize_google_payload_for_log(reask_payload),
                    },
                    response=response,
                    screenshots=None,
                    metadata={
                        "environment": environment,
                        "ambiguous_function_names": ambiguous_names,
                        "reask_attempt": ambiguous_reask_attempts,
                        **metadata,
                    },
                )
                response_dict = normalize_response(response)
                current_interaction_id = (
                    response_dict.get("id") or current_interaction_id
                )
                if response_dict.get("id"):
                    result.response_ids.append(response_dict.get("id", ""))
                continue

            ambiguous_reask_attempts = 0
            executed_turns: list[ComputerToolTurn] = []
            for envelope in call_envelopes:
                call = envelope.function_call
                raw_action_type = str(getattr(call, "name", "") or "unknown").strip()
                if not raw_action_type:
                    raw_action_type = "unknown"
                raw_parameters = (
                    getattr(call, "args", None)
                    or getattr(call, "arguments", None)
                    or {}
                )
                if not isinstance(raw_parameters, dict):
                    raw_parameters = {}
                google_call_id = str(getattr(call, "id", None) or "").strip()
                if google_call_id:
                    local_call_id = google_call_id
                    correlation_mode = "provider_id"
                else:
                    local_call_id = self._build_google_fallback_call_id(
                        turn_index=google_turn_index,
                        sequence=envelope.sequence,
                    )
                    correlation_mode = "sequence_fallback"
                pending_safety_checks = self._extract_google_pending_safety_checks(
                    raw_parameters
                )
                turn = ComputerToolTurn(
                    call_id=local_call_id,
                    action_type=raw_action_type,
                    parameters=raw_parameters,
                    response_id=response_dict.get("id"),
                    pending_safety_checks=pending_safety_checks,
                )
                _inject_context_metadata(turn, metadata)
                turn.metadata["google_function_call_name"] = raw_action_type
                turn.metadata["google_function_call_sequence"] = envelope.sequence
                turn.metadata["google_correlation_mode"] = correlation_mode
                if google_call_id:
                    turn.metadata["google_function_call_id"] = google_call_id
                else:
                    turn.metadata["google_function_call_fallback_id"] = local_call_id
                if pending_safety_checks:
                    turn.metadata["google_safety_decision"] = pending_safety_checks[0]

                if self._should_abort_on_safety(turn, result):
                    if any(
                        executed_turn.status == "executed"
                        for executed_turn in executed_turns
                    ):
                        (
                            response,
                            response_dict,
                            follow_up_batch,
                        ) = await self._flush_google_batch_follow_up(
                            goal=goal,
                            previous_interaction_id=current_interaction_id,
                            turns=executed_turns,
                            metadata=metadata,
                            environment=environment,
                            model=model,
                        )
                        result.final_visual_frame = follow_up_batch.visual_frame
                        result.final_artifact_frame = follow_up_batch.artifact_frame
                        current_interaction_id = (
                            response_dict.get("id") or current_interaction_id
                        )
                        if response_dict.get("id"):
                            result.response_ids.append(response_dict.get("id", ""))
                    return self._finalize_google_session_state(
                        result=result,
                        response_dict=response_dict,
                        current_interaction_id=current_interaction_id,
                    )

                try:
                    await self._execute_tool_action(
                        turn=turn,
                        metadata=metadata,
                        turn_index=turn_counter + 1,
                        normalized_coords=True,
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
                        "Computer Use action execution failed (google)",
                        extra={"call_id": turn.call_id},
                    )

                result.actions.append(turn)
                executed_turns.append(turn)

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
                        "Observe-only policy violation detected (google); aborting action",
                        extra={
                            "step_number": metadata.get("step_number"),
                            "action_type": turn.action_type,
                            "call_id": turn.call_id,
                        },
                    )
                    if any(
                        executed_turn.status == "executed"
                        for executed_turn in executed_turns
                    ):
                        (
                            response,
                            response_dict,
                            follow_up_batch,
                        ) = await self._flush_google_batch_follow_up(
                            goal=goal,
                            previous_interaction_id=current_interaction_id,
                            turns=executed_turns,
                            metadata=metadata,
                            environment=environment,
                            model=model,
                        )
                        result.final_visual_frame = follow_up_batch.visual_frame
                        result.final_artifact_frame = follow_up_batch.artifact_frame
                        current_interaction_id = (
                            response_dict.get("id") or current_interaction_id
                        )
                        if response_dict.get("id"):
                            result.response_ids.append(response_dict.get("id", ""))
                    return self._finalize_google_session_state(
                        result=result,
                        response_dict=response_dict,
                        current_interaction_id=current_interaction_id,
                    )

                loop_detection = self._update_loop_history(
                    turn=turn,
                    history=loop_history,
                    window=loop_window,
                )
                if loop_detection:
                    message = loop_detection["message"]
                    logger.warning(
                        "Computer Use loop detected (google)",
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
                    max_turn_hit = True
                    break

                if turn.status == "ignored":
                    consecutive_ignored += 1
                    if consecutive_ignored >= 4:
                        raise ComputerUseExecutionError(
                            "Repeated disallowed actions ignored; refusing to continue tool loop."
                        )
                    continue
                consecutive_ignored = 0

                if self._is_scroll_action(turn.action_type):
                    scroll_turns += 1
                else:
                    turn_counter += 1

            if executed_turns:
                if stop_after_actions:
                    # Tool-call act mode: the coding agent handles validation.
                    # Skip the follow-up API call entirely — action_agent will
                    # take a fresh screenshot from the driver.
                    break
                (
                    response,
                    response_dict,
                    follow_up_batch,
                ) = await self._flush_google_batch_follow_up(
                    goal=goal,
                    previous_interaction_id=current_interaction_id,
                    turns=executed_turns,
                    metadata=metadata,
                    environment=environment,
                    model=model,
                )
                result.final_visual_frame = follow_up_batch.visual_frame
                result.final_artifact_frame = follow_up_batch.artifact_frame
                current_interaction_id = (
                    response_dict.get("id") or current_interaction_id
                )
                if response_dict.get("id"):
                    result.response_ids.append(response_dict.get("id", ""))

            if max_turn_hit:
                break

            if scroll_turns >= self._scroll_turn_limit:
                logger.warning(
                    "Computer Use scroll turn limit reached (google)",
                    extra={
                        "scroll_turns": scroll_turns,
                        "scroll_turn_limit": self._scroll_turn_limit,
                        "max_turns": self._settings.actions_computer_tool_max_turns,
                    },
                )
                max_turn_code = "scroll_turn_limit_reached"
                max_turn_reason = (
                    "Computer Use max turns exceeded: "
                    f"scroll turn limit reached after {scroll_turns} scroll actions."
                )
                max_turn_hit = True
                break

            if turn_counter >= self._settings.actions_computer_tool_max_turns:
                logger.warning(
                    "Computer Use max turns reached (google)",
                    extra={"max_turns": self._settings.actions_computer_tool_max_turns},
                )
                max_turn_code = "max_turns_exceeded"
                max_turn_reason = (
                    "Computer Use max turns exceeded after "
                    f"{turn_counter} turns (limit: "
                    f"{self._settings.actions_computer_tool_max_turns})."
                )
                max_turn_hit = True
                break

        if max_turn_hit:
            if result.terminal_status != "failed":
                self._append_terminal_failure_turn(
                    result=result,
                    metadata=metadata,
                    response_id=(
                        last_response_dict.get("id")
                        if isinstance(last_response_dict, dict)
                        else None
                    ),
                    reason=max_turn_reason
                    or "Computer Use max turn limit reached (google).",
                    code=max_turn_code or "max_turns_exceeded",
                )
            logger.error(
                "Computer Use max turns reached (google)",
                extra={
                    "goal": goal,
                    "max_turns": self._settings.actions_computer_tool_max_turns,
                    "reason": max_turn_reason,
                    "code": max_turn_code,
                    "last_assistant_text": last_assistant_text,
                    "last_response": _strip_bytes(last_response_dict),
                },
            )

        return self._finalize_google_session_state(
            result=result,
            response_dict=response_dict,
            current_interaction_id=current_interaction_id,
        )

    def _finalize_google_session_state(
        self: _ComputerUseSession,
        *,
        result: ComputerUseSessionResult,
        response_dict: dict[str, Any],
        current_interaction_id: str | None,
    ) -> ComputerUseSessionResult:
        """Persist Google interaction state after one action run."""
        self._google_previous_interaction_id = current_interaction_id
        self._step_last_response = response_dict
        self._step_response_ids.extend(
            response_id for response_id in result.response_ids if response_id
        )
        result.last_response = response_dict
        return result

    @staticmethod
    def _build_google_fallback_call_id(turn_index: int, sequence: int) -> str:
        """Create a deterministic local call identifier for id-less Google calls."""
        return f"google_turn_{turn_index}_call_{sequence}"

    @staticmethod
    def _extract_google_ambiguous_function_names(
        call_envelopes: list[GoogleFunctionCallEnvelope],
    ) -> list[str]:
        """Return duplicate function names that lack provider call IDs."""
        name_counts: dict[str, int] = {}
        for envelope in call_envelopes:
            function_call = envelope.function_call
            call_id = str(getattr(function_call, "id", None) or "").strip()
            if call_id:
                continue
            call_name = str(getattr(function_call, "name", "") or "unknown").strip()
            if not call_name:
                call_name = "unknown"
            name_counts[call_name] = name_counts.get(call_name, 0) + 1
        return sorted(name for name, count in name_counts.items() if count > 1)

    def _build_google_single_call_reask_request(
        self: _ComputerUseSession,
        *,
        previous_interaction_id: str | None,
        environment: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Request a single function call when the previous batch was ambiguous."""
        reask_text = (
            "Your previous response returned multiple function calls with duplicate "
            "names and missing call IDs, which is ambiguous. "
            "Return exactly one function call in this turn. "
            "If additional steps are required, emit one function call per turn."
        )
        payload: dict[str, Any] = {
            "api_surface": "interactions",
            "model": model or self._google_model,
            "input": [{"type": "text", "text": reask_text}],
            "tools": self._build_google_interaction_tools(environment),
        }
        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id
        return payload

    async def _flush_google_batch_follow_up(
        self: _ComputerUseSession,
        *,
        goal: str,
        previous_interaction_id: str | None,
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        environment: str,
        model: str,
    ) -> tuple[Any, dict[str, Any], Any]:
        """Send a Google follow-up for all executed turns in a batch."""
        (
            follow_up_payload,
            follow_up_batch,
            follow_up_screenshot,
        ) = await self._build_google_follow_up_request(
            goal=goal,
            previous_interaction_id=previous_interaction_id,
            turns=turns,
            metadata=metadata,
            environment=environment,
            model=model,
        )
        response = await self._create_google_response(follow_up_payload)
        await self._model_logger.log_call(
            agent="computer_use.google.follow_up",
            model=model,
            prompt=f"{goal} (follow-up)",
            request_payload={
                "provider": "google",
                "environment": environment,
                "payload_type": "follow_up",
                "api_surface": "interactions",
                "request": self._sanitize_google_payload_for_log(follow_up_payload),
            },
            response=response,
            screenshots=[("computer_use_follow_up", follow_up_screenshot)],
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        return response, response_dict, follow_up_batch

    def _build_google_initial_request(
        self: _ComputerUseSession,
        goal: str,
        screenshot_bytes: bytes | None,
        viewport_width: int,
        viewport_height: int,
        environment: str = "desktop",
        model: str | None = None,
        previous_interaction_id: str | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        del viewport_width, viewport_height
        input_items: list[dict[str, Any]] = [{"type": "text", "text": goal}]
        if screenshot_bytes:
            input_items.append(
                {
                    "type": "image",
                    "data": base64.b64encode(screenshot_bytes).decode("utf-8"),
                    "mime_type": "image/png",
                    "resolution": "high",
                }
            )

        payload: dict[str, Any] = {
            "api_surface": "interactions",
            "model": model or self._google_model,
            "input": input_items,
            "tools": self._build_google_interaction_tools(environment),
        }
        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id
        return payload, screenshot_bytes

    async def _build_google_follow_up_request(
        self: _ComputerUseSession,
        goal: str,
        previous_interaction_id: str | None,
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        environment: str = "desktop",
        model: str | None = None,
    ) -> tuple[dict[str, Any], Any, bytes]:
        del goal
        follow_up_batch = await self._build_follow_up_batch(
            call_groups=[[turn] for turn in turns],
            metadata=metadata,
        )
        effective_current_url = self._resolve_google_follow_up_current_url(
            current_url=follow_up_batch.current_url,
            environment=environment,
        )
        follow_up_context_text = self._build_google_follow_up_context_text(
            follow_up_batch=follow_up_batch,
            current_url=effective_current_url,
            metadata=metadata,
        )
        input_items: list[dict[str, Any]] = []
        for call_result in follow_up_batch.calls:
            input_items.append(
                self._build_google_follow_up_item(
                    call_result,
                    current_url=effective_current_url,
                )
            )
        input_items.extend(
            self._build_google_follow_up_shared_items(
                follow_up_batch=follow_up_batch,
                current_url=effective_current_url,
                context_text=follow_up_context_text,
            )
        )

        payload: dict[str, Any] = {
            "api_surface": "interactions",
            "model": model or self._google_model,
            "input": input_items,
            "tools": self._build_google_interaction_tools(environment),
        }
        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id
        return payload, follow_up_batch, follow_up_batch.screenshot_bytes

    def _build_google_interaction_tools(
        self: _ComputerUseSession, environment: str
    ) -> list[dict[str, Any]]:
        """Build Interactions API tool declarations for Google CU."""
        spec = runtime_environment_spec(self._normalize_environment_name(environment))
        tool: dict[str, Any] = {"type": "computer_use"}
        interaction_environment = self._map_google_interaction_environment(environment)
        if interaction_environment:
            tool["environment"] = interaction_environment
        if spec.is_mobile:
            tool["excluded_predefined_functions"] = (
                self._google_mobile_excluded_predefined_functions()
            )
            return [tool, *self._google_mobile_function_tools()]

        tools = [tool]
        if not spec.is_mobile:
            tools.extend(self._google_modifier_click_function_tools())
        return tools

    def _build_google_follow_up_item(
        self: _ComputerUseSession,
        call_result: Any,
        *,
        current_url: str,
    ) -> dict[str, Any]:
        """Build a single Interactions API function_result item."""
        action_result = (
            call_result.actions[0]
            if call_result.actions
            else ComputerUseActionResult(action_type="unknown", status="pending")
        )
        response_name = str(
            call_result.provider_metadata.get("google_function_call_name") or ""
        ).strip()
        if not response_name:
            response_name = (
                str(
                    action_result.action_type or call_result.call_id or "action"
                ).strip()
                or "action"
            )

        provider_call_id = str(
            call_result.provider_metadata.get("google_function_call_id")
            or call_result.call_id
            or ""
        ).strip()
        if not provider_call_id:
            provider_call_id = response_name

        result_payload: dict[str, Any] = {"url": current_url}
        if call_result.requires_safety_acknowledgement:
            result_payload["safety_acknowledgement"] = True
        return {
            "type": "function_result",
            "call_id": provider_call_id,
            "name": response_name,
            "is_error": action_result.status != "executed",
            "result": result_payload,
        }

    def _build_google_follow_up_shared_items(
        self: _ComputerUseSession,
        *,
        follow_up_batch: Any,
        current_url: str,
        context_text: str | None,
    ) -> list[dict[str, Any]]:
        """Build shared summary/image items appended after function_result blocks."""
        items: list[dict[str, Any]] = []
        call_texts: list[str] = []
        interaction_mode = str(follow_up_batch.interaction_mode or "").strip().lower()
        for call_result in follow_up_batch.calls:
            if not self._should_include_google_follow_up_call_text(
                call_result=call_result,
                interaction_mode=interaction_mode,
            ):
                continue
            action_result = (
                call_result.actions[0]
                if call_result.actions
                else ComputerUseActionResult(action_type="unknown", status="pending")
            )
            call_text = self._build_google_follow_up_call_text(
                call_result=call_result,
                action_result=action_result,
                current_url=current_url,
            )
            if call_text:
                call_texts.append(call_text)
        if call_texts:
            items.append({"type": "text", "text": "\n\n".join(call_texts)})
        if context_text:
            items.append({"type": "text", "text": context_text})
        items.append(
            {
                "type": "image",
                "data": follow_up_batch.screenshot_base64,
                "mime_type": "image/png",
                "resolution": "high",
            }
        )
        return items

    @staticmethod
    def _should_include_google_follow_up_call_text(
        *,
        call_result: Any,
        interaction_mode: str,
    ) -> bool:
        """Keep rich follow-up summaries only for non-routine Google turns."""
        if interaction_mode == "observe_only":
            return False
        if getattr(call_result, "requires_safety_acknowledgement", False):
            return True
        actions = getattr(call_result, "actions", []) or []
        return any(
            str(getattr(action, "status", "") or "").strip().lower() != "executed"
            for action in actions
        )

    @staticmethod
    def _build_google_follow_up_call_text(
        *,
        call_result: Any,
        action_result: ComputerUseActionResult,
        current_url: str,
    ) -> str:
        """Render a deterministic per-call execution summary for Gemini follow-up turns."""
        parts = [
            f"current_url={json.dumps(current_url)}",
            f"call_id={json.dumps(str(call_result.call_id or ''))}",
            f"action={json.dumps(str(action_result.action_type or 'unknown'))}",
            f"status={json.dumps(str(action_result.status or 'pending'))}",
        ]
        if action_result.x is not None:
            parts.append(f"x={action_result.x}")
        if action_result.y is not None:
            parts.append(f"y={action_result.y}")
        if action_result.start_x is not None:
            parts.append(f"start_x={action_result.start_x}")
        if action_result.start_y is not None:
            parts.append(f"start_y={action_result.start_y}")
        if action_result.end_x is not None:
            parts.append(f"end_x={action_result.end_x}")
        if action_result.end_y is not None:
            parts.append(f"end_y={action_result.end_y}")
        if action_result.clipboard_text is not None:
            parts.append(f"clipboard_text={json.dumps(action_result.clipboard_text)}")
        if action_result.clipboard_truncated is not None:
            parts.append(
                f"clipboard_truncated={str(action_result.clipboard_truncated).lower()}"
            )
        if action_result.clipboard_error is not None:
            parts.append(f"clipboard_error={json.dumps(action_result.clipboard_error)}")
        if action_result.error_message is not None:
            parts.append(f"error={json.dumps(action_result.error_message)}")
        google_sequence = call_result.provider_metadata.get(
            "google_function_call_sequence"
        )
        if google_sequence is not None:
            parts.append(f"google_function_call_sequence={google_sequence}")
        google_correlation_mode = call_result.provider_metadata.get(
            "google_correlation_mode"
        )
        if google_correlation_mode is not None:
            parts.append(
                f"google_correlation_mode={json.dumps(str(google_correlation_mode))}"
            )
        google_fallback_id = call_result.provider_metadata.get(
            "google_function_call_fallback_id"
        )
        if google_fallback_id is not None:
            parts.append(
                f"google_function_call_fallback_id={json.dumps(str(google_fallback_id))}"
            )
        google_function_id = call_result.provider_metadata.get(
            "google_function_call_id"
        )
        if google_function_id is not None:
            parts.append(
                f"google_function_call_id={json.dumps(str(google_function_id))}"
            )
        if call_result.requires_safety_acknowledgement:
            parts.append("safety_acknowledgement=true")
        return "\n".join(parts)

    def _build_google_follow_up_context_text(
        self: _ComputerUseSession,
        *,
        follow_up_batch: Any,
        current_url: str,
        metadata: dict[str, Any],
    ) -> str | None:
        """Render shared follow-up guidance appended after function_result items."""
        blocks: list[str] = []
        localization_prompt = self._build_follow_up_localization_prompt(
            follow_up_batch,
            metadata,
        )
        if localization_prompt:
            blocks.append(localization_prompt)
        visual_grounding = self._build_google_visual_grounding_text(follow_up_batch)
        if visual_grounding:
            blocks.append(visual_grounding)
        error_text = str(follow_up_batch.error_text or "").strip()
        if error_text:
            blocks.append(error_text)
        if follow_up_batch.interaction_mode == "observe_only":
            reminder_text = str(follow_up_batch.reminder_text or "").strip()
            if reminder_text:
                blocks.append(reminder_text)
        if not blocks:
            return None
        return "\n\n".join(blocks)

    def _resolve_google_follow_up_current_url(
        self: _ComputerUseSession,
        *,
        current_url: str,
        environment: str,
    ) -> str:
        """Normalize the URL/state field required by Gemini follow-up tool results."""
        normalized = str(current_url or "").strip()
        if normalized and normalized != "desktop://":
            return normalized
        if runtime_environment_spec(
            self._normalize_environment_name(environment)
        ).is_mobile:
            return "android://screen"
        return normalized or "desktop://"

    @staticmethod
    def _build_google_visual_grounding_text(follow_up_batch: Any) -> str | None:
        """Describe the current visual frame for Google follow-up turns."""
        frame = follow_up_batch.visual_frame
        if frame is None or frame.kind != "patch":
            return None

        bounds = frame.bounds
        details = [
            "Visual frame mode: patch",
            f"Patch bounds in full screenshot coordinates: x={bounds.x}, y={bounds.y}, width={bounds.width}, height={bounds.height}",
            "Interpret future coordinates against this patch image, not the full screenshot. The harness remaps them back to full-screen pixels.",
        ]
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
        return "\n".join(details)

    async def _reflect_google_step(
        self: _ComputerUseSession,
        *,
        prompt: str,
        metadata: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Request a structured step verdict in the active Google interaction."""
        if not self._google_previous_interaction_id:
            raise RuntimeError(
                "Google step reflection requires an active prior interaction id."
            )

        payload: dict[str, Any] = {
            "api_surface": "interactions",
            "model": model,
            "previous_interaction_id": self._google_previous_interaction_id,
            "input": [{"type": "text", "text": prompt}],
            "response_mime_type": "application/json",
            "response_format": {"type": "object"},
        }

        response = await self._create_google_response(payload)
        await self._model_logger.log_call(
            agent="computer_use.google.step_reflection",
            model=model,
            prompt=prompt,
            request_payload={
                "provider": "google",
                "payload_type": "step_reflection",
                "api_surface": "interactions",
                "request": self._sanitize_google_payload_for_log(payload),
            },
            response=response,
            metadata=metadata,
        )

        response_dict = normalize_response(response)
        response_id = response_dict.get("id")
        if response_id:
            self._google_previous_interaction_id = response_id
            self._step_response_ids.append(response_id)
        self._step_last_response = response_dict

        return {
            "response_id": response_id,
            "response_ids": [response_id] if response_id else [],
            "response_dict": response_dict,
            "raw_text": extract_assistant_text(response_dict) or "",
        }

    def _build_google_generate_config(
        self: _ComputerUseSession, environment: str
    ) -> Any:
        from google.genai import types  # type: ignore

        return types.GenerateContentConfig(
            tools=self._build_google_tools(environment),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def _build_google_tools(self: _ComputerUseSession, environment: str) -> list[Any]:
        from google.genai import types  # type: ignore

        spec = runtime_environment_spec(self._normalize_environment_name(environment))
        computer_use_kwargs: dict[str, Any] = {
            "environment": self._map_google_environment(environment),
        }
        if spec.is_mobile:
            computer_use_kwargs["excluded_predefined_functions"] = (
                self._google_mobile_excluded_predefined_functions()
            )

        tools: list[Any] = [
            types.Tool(computer_use=types.ComputerUse(**computer_use_kwargs))
        ]
        if spec.is_mobile:
            tools.append(self._google_mobile_custom_tools())
        else:
            tools.append(self._google_modifier_click_tools())
        return tools

    @staticmethod
    def _extract_google_pending_safety_checks(
        action_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract Google safety decision metadata from call arguments."""
        if not isinstance(action_args, dict):
            return []

        safety_decision = action_args.get("safety_decision")
        if not isinstance(safety_decision, dict):
            return []

        decision = str(safety_decision.get("decision") or "").strip()
        if not decision:
            return []

        explanation = str(safety_decision.get("explanation") or "").strip()
        payload: dict[str, Any] = {
            "decision": decision,
            "code": decision,
            "message": explanation,
        }
        safety_id = str(safety_decision.get("id") or "").strip()
        if safety_id:
            payload["id"] = safety_id

        return [payload]

    async def _create_google_response(
        self: _ComputerUseSession, payload: dict[str, Any]
    ) -> Any:
        client = self._ensure_google_client()
        if not client:
            raise ComputerUseExecutionError(
                "Google computer-use provider requested but google-genai is not installed."
            )

        api_surface = str(payload.get("api_surface") or "").strip().lower()
        request_payload = {
            key: value for key, value in payload.items() if key != "api_surface"
        }

        def _call_generate_content() -> Any:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                contents = request_payload.get("contents") or request_payload.get(
                    "input"
                )
                config = request_payload.get("config")
                return client.models.generate_content(
                    model=request_payload.get("model"), contents=contents, config=config
                )
            if hasattr(client, "responses"):
                responses = client.responses
                if hasattr(responses, "generate"):
                    return responses.generate(**request_payload)
                if hasattr(responses, "create"):
                    return responses.create(**request_payload)
            raise ComputerUseExecutionError(
                "Google GenAI client does not support responses.generate/create calls."
            )

        async def _call_interactions() -> Any:
            with warnings.catch_warnings():
                for message in self._GOOGLE_INTERACTIONS_WARNING_MESSAGES:
                    warnings.filterwarnings(
                        "ignore",
                        message=message,
                        category=UserWarning,
                    )

                if hasattr(client, "aio") and hasattr(client.aio, "interactions"):
                    return await client.aio.interactions.create(**request_payload)
                if hasattr(client, "interactions") and hasattr(
                    client.interactions, "create"
                ):
                    return await self._invoke_google_request(
                        lambda: client.interactions.create(**request_payload)
                    )
            raise ComputerUseExecutionError(
                "Google GenAI client does not support interactions.create calls."
            )

        retry_delays = self._resolve_google_retry_delays()
        safety_retry_delays = self._resolve_google_prompt_safety_retry_delays()
        transport_retry_count = 0
        safety_retry_count = 0

        while True:
            attempt_number = transport_retry_count + 1
            try:
                if api_surface == "interactions":
                    response = await _call_interactions()
                else:
                    response = await self._invoke_google_request(_call_generate_content)
            except Exception as exc:
                payload_summary = self._summarize_google_payload_for_log(payload)
                if transport_retry_count >= len(
                    retry_delays
                ) or not self._is_google_retryable_error(exc):
                    logger.error(
                        "Google Computer Use request failed",
                        extra={
                            "api_surface": api_surface or None,
                            "attempt": attempt_number,
                            "error": str(exc),
                            "payload_summary": payload_summary,
                        },
                    )
                    raise

                delay_seconds = retry_delays[transport_retry_count]
                transport_retry_count += 1
                logger.warning(
                    "Transient Google Computer Use failure; retrying request",
                    extra={
                        "attempt": attempt_number,
                        "max_attempts": 1 + len(retry_delays),
                        "delay_seconds": delay_seconds,
                        "error": str(exc),
                        "payload_summary": payload_summary,
                    },
                )
                await self._sleep_google_retry(delay_seconds)
                continue

            response_dict = normalize_response(response)
            block_reason, _ = self._extract_google_prompt_block_feedback(response_dict)
            if block_reason != "SAFETY":
                return response

            if safety_retry_count >= len(safety_retry_delays):
                return response

            delay_seconds = self._compute_google_prompt_safety_retry_delay(
                safety_retry_delays[safety_retry_count]
            )
            safety_retry_count += 1
            logger.warning(
                "Google prompt blocked by SAFETY; retrying request with jitter",
                extra={
                    "retry": safety_retry_count,
                    "max_retries": len(safety_retry_delays),
                    "delay_seconds": delay_seconds,
                },
            )
            await self._sleep_google_retry(delay_seconds)

    async def _invoke_google_request(self: _ComputerUseSession, call: Any) -> Any:
        """Run a synchronous Google client request off the event loop."""
        return await asyncio.to_thread(call)

    async def _sleep_google_retry(
        self: _ComputerUseSession, delay_seconds: float
    ) -> None:
        """Sleep between Google retry attempts.

        Kept as an instance method so tests can stub retry waits without monkeypatching
        the global asyncio module used by event-loop teardown.
        """
        from . import session as session_module

        await session_module.asyncio.sleep(delay_seconds)

    @staticmethod
    def _normalize_google_block_reason(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if "." in text:
            text = text.split(".")[-1].strip()
        return text.upper() or None

    @classmethod
    def _extract_google_prompt_block_feedback(
        cls, response_dict: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        prompt_feedback = response_dict.get("prompt_feedback")
        if not isinstance(prompt_feedback, dict):
            return None, None

        block_reason = cls._normalize_google_block_reason(
            prompt_feedback.get("block_reason")
        )
        if not block_reason:
            return None, None

        block_reason_message = str(
            prompt_feedback.get("block_reason_message") or ""
        ).strip()
        return block_reason, block_reason_message or None

    @staticmethod
    def _is_google_retryable_error(exc: Exception) -> bool:
        """Detect transient Google provider failures that should be retried."""
        message = str(exc).lower()
        if "resource_exhausted" in message:
            return True
        if "429" in message:
            return True
        if "rate limit" in message:
            return True

        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code == 429:
            return True

        code = getattr(exc, "code", None)
        if isinstance(code, int) and code == 429:
            return True
        if isinstance(code, str) and code.strip().upper() == "RESOURCE_EXHAUSTED":
            return True

        return False

    def _resolve_google_retry_delays(
        self: _ComputerUseSession,
    ) -> tuple[float, ...]:
        """Return retry delays for transient Google provider failures."""
        configured = getattr(self._settings, "google_cu_retry_delays_seconds", None)
        if isinstance(configured, (list, tuple)):
            parsed: list[float] = []
            for raw in configured:
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    continue
                if delay > 0:
                    parsed.append(delay)
            if parsed:
                return tuple(parsed)

        return self._GOOGLE_RETRY_DELAYS_SECONDS

    def _resolve_google_prompt_safety_retry_delays(
        self: _ComputerUseSession,
    ) -> tuple[float, ...]:
        """Return retry delays for prompt-level Google SAFETY blocks."""
        configured = getattr(
            self._settings, "google_cu_prompt_safety_retry_delays_seconds", None
        )
        if isinstance(configured, (list, tuple)):
            parsed: list[float] = []
            for raw in configured:
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    continue
                if delay > 0:
                    parsed.append(delay)
            if parsed:
                return tuple(parsed)

        return self._GOOGLE_PROMPT_SAFETY_RETRY_DELAYS_SECONDS

    def _compute_google_prompt_safety_retry_delay(
        self: _ComputerUseSession, base_delay: float
    ) -> float:
        """Add short random jitter to SAFETY retry delays."""
        from . import session as session_module

        configured = getattr(
            self._settings, "google_cu_prompt_safety_retry_jitter_seconds", None
        )
        low, high = self._GOOGLE_PROMPT_SAFETY_RETRY_JITTER_SECONDS
        if isinstance(configured, (list, tuple)) and len(configured) == 2:
            try:
                low = float(configured[0])
                high = float(configured[1])
            except (TypeError, ValueError):
                low, high = self._GOOGLE_PROMPT_SAFETY_RETRY_JITTER_SECONDS

        low = max(low, 0.0)
        high = max(high, 0.0)
        if high < low:
            low, high = high, low

        jitter = session_module.random.uniform(low, high) if high > 0 else 0.0
        return max(float(base_delay) + jitter, 0.0)

    def _ensure_google_client(
        self: _ComputerUseSession,
    ) -> Any | None:
        from . import session as session_module

        if self._provider != "google":
            return None
        if self._google_client:
            return self._google_client
        if session_module.genai is None:
            return None
        vertex_project = str(
            getattr(self._settings, "vertex_project", "") or ""
        ).strip()
        vertex_location = str(
            getattr(self._settings, "vertex_location", "us-central1") or ""
        ).strip()
        vertex_api_key = str(
            getattr(self._settings, "vertex_api_key", "") or ""
        ).strip()

        if vertex_project:
            if not vertex_location:
                raise ComputerUseExecutionError(
                    "HAINDY_VERTEX_LOCATION is required when "
                    "HAINDY_VERTEX_PROJECT is configured."
                )
            try:
                vertex_kwargs: dict[str, Any] = {
                    "vertexai": True,
                    "project": vertex_project,
                    "location": vertex_location,
                }
                if vertex_api_key:
                    logger.warning(
                        "Ignoring HAINDY_VERTEX_API_KEY because "
                        "HAINDY_VERTEX_PROJECT is configured; using Vertex "
                        "project/location mode."
                    )
                self._google_client = session_module.genai.Client(**vertex_kwargs)
                logger.info(
                    "Initialized Google CU client in Vertex mode",
                    extra={
                        "vertex_project": vertex_project,
                        "vertex_location": vertex_location,
                    },
                )
                return self._google_client
            except Exception as exc:
                raise ComputerUseExecutionError(
                    f"Failed to initialize Vertex Google client: {exc}"
                ) from exc

        if not vertex_api_key:
            raise ComputerUseExecutionError(
                "Google CU provider requires either "
                "HAINDY_VERTEX_PROJECT+HAINDY_VERTEX_LOCATION or "
                "HAINDY_VERTEX_API_KEY."
            )
        self._google_client = session_module.genai.Client(api_key=vertex_api_key)
        logger.debug("Initialized Google CU client in API key mode")
        return self._google_client
