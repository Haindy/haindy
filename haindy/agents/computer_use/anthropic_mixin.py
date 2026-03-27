# mypy: disable-error-code=misc
"""Anthropic provider loop for Computer Use sessions."""

from __future__ import annotations

import io
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

try:
    from anthropic import AsyncAnthropic as _AsyncAnthropicType
except Exception:
    _AsyncAnthropic: type[Any] | None = None
else:
    _AsyncAnthropic = _AsyncAnthropicType

from haindy.core.enhanced_types import ComputerToolTurn
from haindy.utils.model_logging import log_model_call_failure

from .common import (
    _inject_context_metadata,
    encode_png_base64,
    extract_anthropic_computer_calls,
    extract_assistant_text,
    normalize_response,
)
from .turn_result import ComputerUseActionResult
from .types import (
    ComputerUseExecutionError,
    ComputerUseSessionResult,
    _strip_bytes,
)

logger = logging.getLogger("haindy.agents.computer_use.session")

if TYPE_CHECKING:
    from .session import ComputerUseSession as _ComputerUseSession


class AnthropicComputerUseMixin:
    """Anthropic-specific request builders and execution loop."""

    _last_pointer_position: tuple[int, int] | None

    async def _run_anthropic(
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
        # Cartography/localization is not used for Anthropic CU: the model
        # navigates from the full-screen screenshot directly without needing
        # to emit a JSON coordinate map.
        screenshot = initial_screenshot or await self._automation_driver.screenshot()
        self._prime_initial_visual_state_for_request(
            screenshot,
            source="initial_screenshot",
        )
        # Use actual screenshot dimensions rather than wm_size so the model's
        # coordinate space matches what _map_point_to_device expects.
        if (
            self._current_keyframe is not None
            and self._current_keyframe.screen_size[0] > 0
        ):
            viewport_width, viewport_height = self._current_keyframe.screen_size
        # Downscale the screenshot for the API to reduce coordinate-mapping
        # errors caused by the model's internal image rendering.  The original
        # viewport size is preserved so coordinates can be scaled back.
        api_screenshot, original_size, api_size = (
            self._downscale_screenshot_for_anthropic(
                screenshot, self._ANTHROPIC_MAX_SCREENSHOT_LONG_EDGE
            )
        )
        # Fall back to the raw viewport when PIL cannot decode the screenshot.
        if api_size[0] <= 0:
            original_size = (viewport_width, viewport_height)
            api_size = original_size
            api_screenshot = screenshot
        api_width, api_height = api_size
        # Wrap the goal text AFTER determining the API dimensions so the
        # resolution mentioned in the prompt matches the tool definition.
        goal = self._wrap_goal_for_mobile(goal, environment, api_width, api_height)
        goal = self._apply_interaction_mode_guidance(goal, metadata)

        request_payload = self._build_anthropic_initial_request(
            goal=goal,
            screenshot_bytes=api_screenshot,
            viewport_width=api_width,
            viewport_height=api_height,
            metadata=metadata,
            model=model,
        )

        response = await self._create_anthropic_response(
            request_payload,
            agent="computer_use.anthropic.initial",
            prompt=goal,
            request_payload_for_log=self._sanitize_payload_for_log(request_payload),
            screenshots=[("computer_use_initial", screenshot)],
            metadata={"environment": environment, **metadata},
        )

        response_dict = normalize_response(response)
        result.response_ids.append(response_dict.get("id", ""))

        history_messages = list(request_payload.get("messages", []))
        turn_counter = 0
        scroll_turns = 0
        consecutive_ignored = 0
        max_turn_hit = False
        max_turn_reason: str | None = None
        max_turn_code: str | None = None
        last_assistant_text: str | None = None
        last_response_dict: dict[str, Any] | None = None
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            calls = extract_anthropic_computer_calls(response_dict)
            assistant_text = extract_assistant_text(response_dict)
            if assistant_text:
                result.final_output = assistant_text
                last_assistant_text = assistant_text
            result.last_response = response_dict
            last_response_dict = response_dict

            if not calls:
                break

            executed_turns: list[ComputerToolTurn] = []
            for call in calls:
                translated_action = self._translate_anthropic_action(
                    call.get("action") or {}
                )
                # Scale coordinates from the API image space back to the
                # original device viewport so the driver taps land correctly.
                if api_size != original_size:
                    for key_x, key_y in (
                        ("x", "y"),
                        ("start_x", "start_y"),
                        ("end_x", "end_y"),
                    ):
                        if key_x in translated_action and key_y in translated_action:
                            translated_action[key_x], translated_action[key_y] = (
                                self._rescale_coordinates(
                                    int(translated_action[key_x]),
                                    int(translated_action[key_y]),
                                    from_size=api_size,
                                    to_size=original_size,
                                )
                            )
                turn = ComputerToolTurn(
                    call_id=str(call.get("id") or ""),
                    action_type=str(translated_action.get("type") or "unknown"),
                    parameters=translated_action,
                    response_id=response_dict.get("id"),
                    pending_safety_checks=[],
                )
                _inject_context_metadata(turn, metadata)

                if self._should_abort_on_safety(turn, result):
                    return result

                try:
                    await self._execute_tool_action(
                        turn=turn,
                        metadata=metadata,
                        turn_index=turn_counter + 1,
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
                        "Computer Use action execution failed (anthropic)",
                        extra={"call_id": turn.call_id},
                    )

                self._update_last_pointer_position(turn)
                # _last_pointer_position is now in device-space but
                # _translate_anthropic_action uses it as a fallback for the
                # *next* model turn.  Convert back to API-space so the
                # rescaling step above doesn't double-scale.
                if (
                    api_size != original_size
                    and self._last_pointer_position is not None
                ):
                    self._last_pointer_position = self._rescale_coordinates(
                        self._last_pointer_position[0],
                        self._last_pointer_position[1],
                        from_size=original_size,
                        to_size=api_size,
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
                        "Observe-only policy violation detected (anthropic); aborting action",
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
                        "Computer Use loop detected (anthropic)",
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
                    if scroll_turns >= self._scroll_turn_limit:
                        logger.warning(
                            "Computer Use scroll turn limit reached (anthropic)",
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
                else:
                    turn_counter += 1
                    if turn_counter >= self._settings.actions_computer_tool_max_turns:
                        logger.warning(
                            "Computer Use max turns reached (anthropic)",
                            extra={
                                "max_turns": self._settings.actions_computer_tool_max_turns
                            },
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
                break

            (
                follow_up_payload,
                follow_up_screenshot,
                original_size,
                api_size,
            ) = await self._build_anthropic_follow_up_request(
                history_messages=history_messages,
                previous_response=response_dict,
                turns=executed_turns,
                metadata=metadata,
                model=model,
            )
            history_messages = list(follow_up_payload.get("messages", []))
            response = await self._create_anthropic_response(
                follow_up_payload,
                agent="computer_use.anthropic.follow_up",
                prompt=f"{goal} (follow-up)",
                request_payload_for_log=self._sanitize_payload_for_log(
                    follow_up_payload
                ),
                screenshots=(
                    [("computer_use_follow_up", follow_up_screenshot)]
                    if follow_up_screenshot
                    else None
                ),
                metadata={"environment": environment, **metadata},
            )
            response_dict = normalize_response(response)
            result.response_ids.append(response_dict.get("id", ""))

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
                    or "Computer Use max turn limit reached (anthropic).",
                    code=max_turn_code or "max_turns_exceeded",
                )
            logger.error(
                "Computer Use max turns reached (anthropic)",
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

    def _build_anthropic_initial_request(
        self: _ComputerUseSession,
        goal: str,
        screenshot_bytes: bytes,
        viewport_width: int,
        viewport_height: int,
        metadata: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
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

        screenshot_b64 = encode_png_base64(screenshot_bytes)
        return {
            "model": model or self._anthropic_model,
            "max_tokens": self._anthropic_max_tokens,
            "betas": list(self._anthropic_betas),
            "tools": [
                {
                    "type": self._anthropic_tool_type,
                    "name": self._anthropic_tool_name,
                    "display_width_px": viewport_width,
                    "display_height_px": viewport_height,
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                    ],
                }
            ],
        }

    async def _build_anthropic_follow_up_request(
        self: _ComputerUseSession,
        history_messages: list[dict[str, Any]],
        previous_response: dict[str, Any],
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        model: str | None = None,
    ) -> tuple[dict[str, Any], bytes | None, tuple[int, int], tuple[int, int]]:
        """Build follow-up and return *(payload, screenshot, original_size, api_size)*."""
        follow_up_batch = await self._build_follow_up_batch(
            call_groups=[[turn] for turn in turns],
            metadata=metadata,
            skip_localization=True,
        )
        # Anthropic always expects full-screen screenshots. When the visual
        # pipeline chose a patch frame, fall back to the stored full keyframe so
        # the model's coordinate space matches the actual display.
        display_frame = (
            follow_up_batch.artifact_frame
            if (
                follow_up_batch.visual_frame is not None
                and follow_up_batch.visual_frame.kind == "patch"
                and follow_up_batch.artifact_frame is not None
            )
            else follow_up_batch.visual_frame
        )
        # Use actual screenshot dimensions so display_width_px/height_px
        # matches the coordinate space the model is seeing.
        if display_frame is not None and display_frame.screen_size[0] > 0:
            viewport_width, viewport_height = display_frame.screen_size
        else:
            (
                viewport_width,
                viewport_height,
            ) = await self._automation_driver.get_viewport_size()
        display_bytes = (
            display_frame.image_bytes
            if display_frame is not None
            else follow_up_batch.screenshot_bytes
        )
        # Downscale for the API to improve coordinate accuracy.
        api_screenshot, fu_original_size, fu_api_size = (
            self._downscale_screenshot_for_anthropic(
                display_bytes, self._ANTHROPIC_MAX_SCREENSHOT_LONG_EDGE
            )
        )
        if fu_api_size[0] <= 0:
            fu_original_size = (viewport_width, viewport_height)
            fu_api_size = fu_original_size
            api_screenshot = display_bytes
        api_width, api_height = fu_api_size
        display_b64 = encode_png_base64(api_screenshot)

        tool_results: list[dict[str, Any]] = []
        extra_content: list[dict[str, Any]] = []

        for call_result in follow_up_batch.calls:
            action_result = (
                call_result.actions[0]
                if call_result.actions
                else ComputerUseActionResult(action_type="unknown", status="pending")
            )
            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call_result.call_id,
            }

            if action_result.status != "executed":
                error_text = action_result.error_message or "Action execution failed."
                # Anthropic API requires text-only content when is_error is true.
                tool_result_block["content"] = [
                    {"type": "text", "text": f"Execution error: {error_text}"}
                ]
                tool_result_block["is_error"] = True
            else:
                tool_result_block["content"] = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": display_b64,
                        },
                    }
                ]
            tool_results.append(tool_result_block)

        if follow_up_batch.grounding_text:
            extra_content.append(
                {"type": "text", "text": follow_up_batch.grounding_text}
            )
        if follow_up_batch.reminder_text:
            extra_content.append(
                {"type": "text", "text": follow_up_batch.reminder_text}
            )
        # Cartography/localization is not used for Anthropic CU: the model
        # navigates from the full-screen screenshot directly.

        messages: list[dict[str, Any]] = list(history_messages)
        assistant_content = previous_response.get("content") or []
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results + extra_content})

        payload: dict[str, Any] = {
            "model": model or self._anthropic_model,
            "max_tokens": self._anthropic_max_tokens,
            "betas": list(self._anthropic_betas),
            "tools": [
                {
                    "type": self._anthropic_tool_type,
                    "name": self._anthropic_tool_name,
                    "display_width_px": api_width,
                    "display_height_px": api_height,
                }
            ],
            "messages": messages,
        }
        return payload, display_bytes, fu_original_size, fu_api_size

    async def _create_anthropic_response(
        self: _ComputerUseSession,
        payload: dict[str, Any],
        *,
        agent: str | None = None,
        prompt: str | None = None,
        request_payload_for_log: Any | None = None,
        screenshots: list[tuple[str, bytes]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        client = self._ensure_anthropic_client()
        if not client:
            raise ComputerUseExecutionError(
                "Anthropic computer-use provider requested but anthropic SDK is not installed."
            )

        async def _log_success(response: Any) -> None:
            if not agent or prompt is None:
                return
            await self._model_logger.log_outcome(
                agent=agent,
                model=str(payload.get("model") or self._model),
                prompt=prompt,
                request_payload=request_payload_for_log,
                response=response,
                screenshots=screenshots,
                metadata={"provider": "anthropic", **(metadata or {})},
                outcome="success",
            )

        async def _log_failure(exc: BaseException) -> None:
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
                metadata={"provider": "anthropic", **(metadata or {})},
            )

        if hasattr(client, "beta") and hasattr(client.beta, "messages"):
            create_call = getattr(client.beta.messages, "create", None)
            if callable(create_call):
                try:
                    response = await create_call(**payload)
                except Exception as exc:
                    await _log_failure(exc)
                    raise
                await _log_success(response)
                return response

        if hasattr(client, "messages"):
            create_call = getattr(client.messages, "create", None)
            if callable(create_call):
                fallback_payload = dict(payload)
                fallback_payload.pop("betas", None)
                try:
                    response = await create_call(**fallback_payload)
                except Exception as exc:
                    await _log_failure(exc)
                    raise
                await _log_success(response)
                return response

        raise ComputerUseExecutionError(
            "Anthropic client does not support messages.create calls."
        )

    def _ensure_anthropic_client(
        self: _ComputerUseSession,
    ) -> Any | None:
        from . import session as session_module

        if self._provider != "anthropic":
            return None
        if self._anthropic_client:
            return self._anthropic_client
        if session_module._AsyncAnthropic is None:
            return None

        anthropic_api_key = str(
            getattr(self._settings, "anthropic_api_key", "") or ""
        ).strip()
        if not anthropic_api_key:
            raise ComputerUseExecutionError(
                "Anthropic CU provider requires HAINDY_ANTHROPIC_API_KEY."
            )

        self._anthropic_client = session_module._AsyncAnthropic(
            api_key=anthropic_api_key
        )
        logger.info("Initialized Anthropic CU client in API key mode")
        return self._anthropic_client

    def _translate_anthropic_action(
        self: _ComputerUseSession, raw_action: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate Anthropic computer tool payloads into internal action payloads."""
        if not isinstance(raw_action, dict):
            return {"type": "unknown"}

        translated = dict(raw_action)
        action_name = str(
            raw_action.get("action") or raw_action.get("type") or ""
        ).strip()
        if not action_name:
            return {"type": "unknown"}
        translated["type"] = action_name

        coordinate = raw_action.get("coordinate") or raw_action.get("coordinates")
        coord_pair = self._extract_anthropic_coordinate_pair(coordinate)
        if coord_pair is not None:
            translated["x"], translated["y"] = coord_pair

        normalized_name = action_name.replace("-", "_").strip().lower()
        if normalized_name in {
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
        }:
            if coord_pair is None and self._last_pointer_position is not None:
                translated["x"], translated["y"] = self._last_pointer_position
            if normalized_name == "left_click":
                translated["type"] = "click"
            elif normalized_name == "middle_click":
                translated["type"] = "click"
                translated["button"] = "middle"
            elif normalized_name == "triple_click":
                translated["type"] = "click"
                translated["click_count"] = 3
        elif normalized_name == "mouse_move":
            if coord_pair is None and self._last_pointer_position is not None:
                translated["x"], translated["y"] = self._last_pointer_position
        elif normalized_name == "left_click_drag":
            translated["type"] = "drag"
            start_coordinate = raw_action.get("start_coordinate")
            start_pair = self._extract_anthropic_coordinate_pair(start_coordinate)
            if start_pair is not None:
                translated["start_x"], translated["start_y"] = start_pair
            elif self._last_pointer_position is not None:
                translated["start_x"], translated["start_y"] = (
                    self._last_pointer_position
                )
            if coord_pair is not None:
                translated["end_x"], translated["end_y"] = coord_pair
        elif normalized_name == "key":
            key_value = (
                raw_action.get("text")
                or raw_action.get("key")
                or raw_action.get("keys")
                or raw_action.get("value")
            )
            translated = {"type": "keypress", "key": str(key_value or "")}
        elif normalized_name == "type":
            translated = {
                "type": "type",
                "text": str(
                    raw_action.get("text")
                    or raw_action.get("value")
                    or raw_action.get("input")
                    or ""
                ),
            }
        elif normalized_name == "scroll":
            direction = str(
                raw_action.get("scroll_direction")
                or raw_action.get("direction")
                or "down"
            ).strip()
            amount = raw_action.get("scroll_amount")
            if amount is None:
                amount = raw_action.get("amount")
            if amount is None:
                amount = self._settings.scroll_default_magnitude
            try:
                magnitude = abs(int(float(amount)))
            except (TypeError, ValueError):
                magnitude = int(self._settings.scroll_default_magnitude)
            translated = {
                "type": "scroll_document",
                "direction": direction.lower() or "down",
                "magnitude": magnitude,
            }
        elif normalized_name == "wait":
            translated = {
                "type": "wait",
                "duration_ms": raw_action.get("duration_ms")
                or raw_action.get("duration")
                or 1000,
            }
        elif normalized_name == "screenshot":
            translated = {"type": "screenshot"}
        elif normalized_name == "left_mouse_down":
            translated["type"] = "click"
            if coord_pair is None and self._last_pointer_position is not None:
                translated["x"], translated["y"] = self._last_pointer_position
        elif normalized_name == "left_mouse_up":
            translated = {"type": "screenshot"}
        elif normalized_name == "hold_key":
            key_value = raw_action.get("text") or raw_action.get("key")
            translated = {"type": "keypress", "key": str(key_value or "")}

        return translated

    @staticmethod
    def _extract_anthropic_coordinate_pair(
        raw_coordinate: Any,
    ) -> tuple[int, int] | None:
        if isinstance(raw_coordinate, (list, tuple)) and len(raw_coordinate) == 2:
            try:
                return int(float(raw_coordinate[0])), int(float(raw_coordinate[1]))
            except (TypeError, ValueError):
                return None
        if isinstance(raw_coordinate, dict):
            if "x" in raw_coordinate and "y" in raw_coordinate:
                try:
                    return int(float(raw_coordinate["x"])), int(
                        float(raw_coordinate["y"])
                    )
                except (TypeError, ValueError):
                    return None
        return None

    # Maximum long-edge pixels for screenshots sent to Anthropic.  The model
    # internally downscales images to fit its context window and then must map
    # coordinates back to the declared ``display_width_px x display_height_px``
    # space.  On high-resolution mobile devices the internal rendering can
    # differ significantly from the declared display size, causing systematic
    # coordinate errors.  Sending a pre-scaled screenshot whose dimensions
    # match what the model will actually render eliminates this bias.
    _ANTHROPIC_MAX_SCREENSHOT_LONG_EDGE: int = 1280

    @staticmethod
    def _downscale_screenshot_for_anthropic(
        screenshot_bytes: bytes,
        max_long_edge: int,
    ) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
        """Return *(resized_png, original_size, resized_size)*.

        If neither dimension exceeds *max_long_edge* the original bytes are
        returned unchanged.  Falls back gracefully if PIL cannot decode the
        image (e.g. minimal stubs used in tests).
        """
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(screenshot_bytes))
            original_size: tuple[int, int] = (img.width, img.height)
        except Exception:
            # Cannot decode -- pass through unchanged.
            return screenshot_bytes, (0, 0), (0, 0)

        long_edge = max(img.width, img.height)
        if long_edge <= max_long_edge:
            return screenshot_bytes, original_size, original_size

        scale = max_long_edge / long_edge
        new_w = max(int(round(img.width * scale)), 1)
        new_h = max(int(round(img.height * scale)), 1)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue(), original_size, (new_w, new_h)

    @staticmethod
    def _rescale_coordinates(
        x: int,
        y: int,
        from_size: tuple[int, int],
        to_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Scale *(x, y)* from one coordinate space to another."""
        if from_size == to_size or from_size[0] <= 0 or from_size[1] <= 0:
            return x, y
        return (
            round(x * to_size[0] / from_size[0]),
            round(y * to_size[1] / from_size[1]),
        )

    def _update_last_pointer_position(
        self: _ComputerUseSession, turn: ComputerToolTurn
    ) -> None:
        if turn.status != "executed":
            return

        end_x = turn.metadata.get("end_x")
        end_y = turn.metadata.get("end_y")
        if end_x is not None and end_y is not None:
            try:
                self._last_pointer_position = (int(end_x), int(end_y))
                return
            except (TypeError, ValueError):
                pass

        x = turn.metadata.get("x")
        y = turn.metadata.get("y")
        if x is None or y is None:
            return
        try:
            self._last_pointer_position = (int(x), int(y))
        except (TypeError, ValueError):
            return
