# mypy: disable-error-code=misc
"""Google provider loop for Computer Use sessions."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

try:
    import google.genai as _genai
except Exception:
    genai: Any | None = None
else:
    genai = _genai

from src.core.enhanced_types import ComputerToolTurn
from src.runtime.environment import runtime_environment_spec

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

logger = logging.getLogger("src.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class GoogleComputerUseMixin:
    """Google-specific request builders and execution loop."""

    async def _run_google(
        self: _ComputerUseSession,
        *,
        goal: str,
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
        initial_screenshot = await self._automation_driver.screenshot()
        goal = self._wrap_goal_for_mobile(
            goal, environment, viewport_width, viewport_height
        )
        goal = self._apply_interaction_mode_guidance(goal, metadata)
        wrapped_goal = self._wrap_goal_for_google(goal, environment)
        contents, config = self._build_google_initial_request(
            goal=wrapped_goal,
            screenshot_bytes=initial_screenshot,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            environment=environment,
        )

        history: list[Any] = list(contents)
        initial_request = {"model": model, "contents": contents, "config": config}
        response = await self._create_google_response(initial_request)
        await self._model_logger.log_call(
            agent="computer_use.google.initial",
            model=model,
            prompt=goal,
            request_payload={
                "provider": "google",
                "environment": environment,
                "payload": "initial",
            },
            response=response,
            screenshots=[("computer_use_initial", initial_screenshot)],
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        result.response_ids.append(response_dict.get("id", ""))
        if hasattr(response, "candidates"):
            try:
                history.append(response.candidates[0].content)
            except Exception:
                logger.debug(
                    "Unable to append Google response content to history",
                    exc_info=True,
                )

        turn_counter = 0
        scroll_turns = 0
        consecutive_ignored = 0
        max_turn_hit = False
        max_turn_reason: str | None = None
        max_turn_code: str | None = None
        last_assistant_text: str | None = None
        last_response_dict: dict[str, Any] | None = None
        google_turn_index = 0
        ambiguous_reask_attempts = 0
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            call_envelopes = extract_google_function_call_envelopes(response)
            assistant_text = extract_assistant_text(response_dict)
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

            google_turn_index += 1
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
                (
                    reask_payload,
                    reask_content,
                ) = self._build_google_single_call_reask_request(
                    history=history,
                    environment=environment,
                    model=model,
                )
                history.append(reask_content)
                response = await self._create_google_response(reask_payload)
                await self._model_logger.log_call(
                    agent="computer_use.google.reask",
                    model=model,
                    prompt=f"{goal} (ambiguous-batch-reask)",
                    request_payload={
                        "provider": "google",
                        "environment": environment,
                        "payload": "ambiguous_reask",
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
                result.response_ids.append(response_dict.get("id", ""))
                if hasattr(response, "candidates"):
                    try:
                        history.append(response.candidates[0].content)
                    except Exception:
                        logger.debug(
                            "Unable to append Google re-ask response content to history",
                            exc_info=True,
                        )
                continue

            ambiguous_reask_attempts = 0
            executed_turns: list[ComputerToolTurn] = []
            for envelope in call_envelopes:
                call = envelope.function_call
                raw_action_type = str(getattr(call, "name", "") or "unknown").strip()
                if not raw_action_type:
                    raw_action_type = "unknown"
                raw_parameters = getattr(call, "args", {}) or {}
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
                        ) = await self._flush_google_batch_follow_up(
                            goal=goal,
                            history=history,
                            turns=executed_turns,
                            metadata=metadata,
                            environment=environment,
                            model=model,
                        )
                        result.response_ids.append(response_dict.get("id", ""))
                    return result

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
                        ) = await self._flush_google_batch_follow_up(
                            goal=goal,
                            history=history,
                            turns=executed_turns,
                            metadata=metadata,
                            environment=environment,
                            model=model,
                        )
                        result.response_ids.append(response_dict.get("id", ""))
                    return result

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
                response, response_dict = await self._flush_google_batch_follow_up(
                    goal=goal,
                    history=history,
                    turns=executed_turns,
                    metadata=metadata,
                    environment=environment,
                    model=model,
                )
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
        history: list[Any],
        environment: str,
        model: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Request a single function call when the previous batch was ambiguous."""
        from google.genai import types  # type: ignore

        reask_text = (
            "Your previous response returned multiple function calls with duplicate "
            "names and missing call IDs, which is ambiguous. "
            "Return exactly one function call in this turn. "
            "If additional steps are required, emit one function call per turn."
        )
        reask_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=reask_text)],
        )
        return (
            {
                "model": model or self._google_model,
                "contents": list(history) + [reask_content],
                "config": self._build_google_generate_config(environment),
            },
            reask_content,
        )

    async def _flush_google_batch_follow_up(
        self: _ComputerUseSession,
        *,
        goal: str,
        history: list[Any],
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        environment: str,
        model: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Send a Google follow-up for all executed turns in a batch."""
        (
            follow_up_payload,
            func_response_content,
            follow_up_screenshot,
        ) = await self._build_google_follow_up_request(
            goal=goal,
            history=history,
            turns=turns,
            metadata=metadata,
            environment=environment,
            model=model,
        )
        history.append(func_response_content)
        response = await self._create_google_response(follow_up_payload)
        await self._model_logger.log_call(
            agent="computer_use.google.follow_up",
            model=model,
            prompt=f"{goal} (follow-up)",
            request_payload={
                "provider": "google",
                "environment": environment,
                "payload": "follow_up",
            },
            response=response,
            screenshots=[("computer_use_follow_up", follow_up_screenshot)],
            metadata={"environment": environment, **metadata},
        )
        response_dict = normalize_response(response)
        if hasattr(response, "candidates"):
            try:
                history.append(response.candidates[0].content)
            except Exception:
                logger.debug(
                    "Unable to append Google response content to history",
                    exc_info=True,
                )
        return response, response_dict

    def _build_google_initial_request(
        self: _ComputerUseSession,
        goal: str,
        screenshot_bytes: bytes,
        viewport_width: int,
        viewport_height: int,
        environment: str = "desktop",
    ) -> tuple[list[Any], Any]:
        from google.genai import types  # type: ignore

        del viewport_width, viewport_height
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=goal),
                    types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                ],
            )
        ]
        return contents, self._build_google_generate_config(environment)

    async def _build_google_follow_up_request(
        self: _ComputerUseSession,
        goal: str,
        history: list[Any],
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        environment: str = "desktop",
        model: str | None = None,
    ) -> tuple[dict[str, Any], Any, bytes]:
        from google.genai import types  # type: ignore

        del goal
        follow_up_batch = await self._build_follow_up_batch(
            call_groups=[[turn] for turn in turns],
            metadata=metadata,
        )
        parts: list[Any] = []
        for call_result in follow_up_batch.calls:
            action_result = (
                call_result.actions[0]
                if call_result.actions
                else ComputerUseActionResult(action_type="unknown", status="pending")
            )
            google_call_id = str(
                call_result.provider_metadata.get("google_function_call_id") or ""
            ).strip()
            response_name = str(
                call_result.provider_metadata.get("google_function_call_name") or ""
            ).strip()
            if not response_name:
                response_name = (
                    str(action_result.action_type or "action").strip() or "action"
                )
            response_payload = {
                "status": action_result.status,
                "call_id": call_result.call_id,
                "google_function_call_sequence": call_result.provider_metadata.get(
                    "google_function_call_sequence"
                ),
                "google_correlation_mode": call_result.provider_metadata.get(
                    "google_correlation_mode"
                ),
                "google_function_call_fallback_id": call_result.provider_metadata.get(
                    "google_function_call_fallback_id"
                ),
                "url": follow_up_batch.current_url,
                "x": action_result.x,
                "y": action_result.y,
                "clipboard_text": action_result.clipboard_text,
                "clipboard_truncated": action_result.clipboard_truncated,
                "clipboard_error": action_result.clipboard_error,
                "error": action_result.error_message,
            }
            if call_result.requires_safety_acknowledgement:
                response_payload["safety_acknowledgement"] = "true"
            response_payload = {
                k: v for k, v in response_payload.items() if v is not None
            }
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=google_call_id or None,
                        name=response_name,
                        response=response_payload,
                        parts=[
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=follow_up_batch.screenshot_bytes,
                                )
                            )
                        ],
                    )
                )
            )

        function_response_content = types.Content(role="user", parts=parts)
        contents = list(history) + [function_response_content]
        if follow_up_batch.reminder_text:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=follow_up_batch.reminder_text)],
                )
            )
        return (
            {
                "model": model or self._google_model,
                "contents": contents,
                "config": self._build_google_generate_config(environment),
            },
            function_response_content,
            follow_up_batch.screenshot_bytes,
        )

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

        tools: list[Any] = [
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=self._map_google_environment(environment),
                )
            )
        ]
        if not runtime_environment_spec(
            self._normalize_environment_name(environment)
        ).is_mobile:
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
        from . import session as session_module

        client = self._ensure_google_client()
        if not client:
            raise ComputerUseExecutionError(
                "Google computer-use provider requested but google-genai is not installed."
            )

        def _call() -> Any:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                contents = payload.get("contents") or payload.get("input")
                config = payload.get("config")
                return client.models.generate_content(
                    model=payload.get("model"), contents=contents, config=config
                )
            if hasattr(client, "responses"):
                responses = client.responses
                if hasattr(responses, "generate"):
                    return responses.generate(**payload)
                if hasattr(responses, "create"):
                    return responses.create(**payload)
            raise ComputerUseExecutionError(
                "Google GenAI client does not support responses.generate/create calls."
            )

        retry_delays = self._resolve_google_retry_delays()
        safety_retry_delays = self._resolve_google_prompt_safety_retry_delays()
        transport_retry_count = 0
        safety_retry_count = 0

        while True:
            attempt_number = transport_retry_count + 1
            try:
                response = await self._invoke_google_request(_call)
            except Exception as exc:
                if transport_retry_count >= len(
                    retry_delays
                ) or not self._is_google_retryable_error(exc):
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

    async def _invoke_google_request(
        self: _ComputerUseSession, call: Any
    ) -> Any:
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
                    "VERTEX_LOCATION is required when VERTEX_PROJECT is configured."
                )
            try:
                vertex_kwargs: dict[str, Any] = {
                    "vertexai": True,
                    "project": vertex_project,
                    "location": vertex_location,
                }
                if vertex_api_key:
                    logger.warning(
                        "Ignoring VERTEX_API_KEY because VERTEX_PROJECT is configured; using Vertex project/location mode."
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
                "Google CU provider requires either VERTEX_PROJECT+VERTEX_LOCATION or VERTEX_API_KEY."
            )
        self._google_client = session_module.genai.Client(api_key=vertex_api_key)
        logger.debug("Initialized Google CU client in API key mode")
        return self._google_client
