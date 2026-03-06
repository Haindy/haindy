# mypy: disable-error-code=misc
"""Anthropic provider loop for Computer Use sessions."""

from __future__ import annotations

import base64
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

try:
    from anthropic import AsyncAnthropic as _AsyncAnthropicType
except Exception:
    _AsyncAnthropic: type[Any] | None = None
else:
    _AsyncAnthropic = _AsyncAnthropicType

from src.core.enhanced_types import ComputerToolTurn

from .common import (
    _inject_context_metadata,
    encode_png_base64,
    extract_anthropic_computer_calls,
    extract_assistant_text,
    normalize_response,
)
from .types import (
    ComputerUseExecutionError,
    ComputerUseSessionResult,
    _strip_bytes,
)

logger = logging.getLogger("src.agents.computer_use.session")

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
        goal = self._wrap_goal_for_mobile(
            goal, environment, viewport_width, viewport_height
        )
        screenshot = initial_screenshot or await self._automation_driver.screenshot()

        request_payload = self._build_anthropic_initial_request(
            goal=goal,
            screenshot_bytes=screenshot,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            metadata=metadata,
            model=model,
        )

        response = await self._create_anthropic_response(request_payload)
        await self._model_logger.log_call(
            agent="computer_use.anthropic.initial",
            model=model,
            prompt=goal,
            request_payload=self._sanitize_payload_for_log(request_payload),
            response=response,
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
            ) = await self._build_anthropic_follow_up_request(
                history_messages=history_messages,
                previous_response=response_dict,
                turns=executed_turns,
                model=model,
            )
            history_messages = list(follow_up_payload.get("messages", []))
            response = await self._create_anthropic_response(follow_up_payload)
            await self._model_logger.log_call(
                agent="computer_use.anthropic.follow_up",
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
        model: str | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
        tool_results: list[dict[str, Any]] = []
        primary_screenshot: bytes | None = None

        for turn in turns:
            screenshot_b64 = turn.metadata.get("screenshot_base64")
            if not screenshot_b64:
                screenshot_bytes = await self._automation_driver.screenshot()
                screenshot_b64 = encode_png_base64(screenshot_bytes)
                turn.metadata["screenshot_base64"] = screenshot_b64
            elif primary_screenshot is None:
                try:
                    primary_screenshot = base64.b64decode(str(screenshot_b64))
                except Exception:
                    primary_screenshot = None

            if primary_screenshot is None:
                try:
                    primary_screenshot = base64.b64decode(str(screenshot_b64))
                except Exception:
                    primary_screenshot = None

            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": turn.call_id,
            }

            if turn.status != "executed":
                error_text = turn.error_message or "Action execution failed."
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
                            "data": screenshot_b64,
                        },
                    }
                ]
            tool_results.append(tool_result_block)
            turn.metadata.pop("screenshot_base64", None)

        messages: list[dict[str, Any]] = list(history_messages)
        assistant_content = previous_response.get("content") or []
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

        payload: dict[str, Any] = {
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
            "messages": messages,
        }
        return payload, primary_screenshot

    async def _create_anthropic_response(
        self: _ComputerUseSession, payload: dict[str, Any]
    ) -> Any:
        client = self._ensure_anthropic_client()
        if not client:
            raise ComputerUseExecutionError(
                "Anthropic computer-use provider requested but anthropic SDK is not installed."
            )

        if hasattr(client, "beta") and hasattr(client.beta, "messages"):
            create_call = getattr(client.beta.messages, "create", None)
            if callable(create_call):
                return await create_call(**payload)

        if hasattr(client, "messages"):
            create_call = getattr(client.messages, "create", None)
            if callable(create_call):
                fallback_payload = dict(payload)
                fallback_payload.pop("betas", None)
                return await create_call(**fallback_payload)

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
                "Anthropic CU provider requires ANTHROPIC_API_KEY."
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
            if normalized_name == "middle_click":
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
            if self._last_pointer_position is not None:
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
