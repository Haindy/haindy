"""Computer Use tool orchestration for the Action Agent."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import urlparse
from uuid import uuid4

from openai import AsyncOpenAI

try:  # Optional dependency for Gemini computer-use
    import google.genai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

from src.config.settings import Settings
from src.desktop.cache import CoordinateCache
from src.core.enhanced_types import ComputerToolTurn, SafetyEvent
from src.core.interfaces import BrowserDriver
from src.monitoring.debug_logger import DebugLogger
from src.utils.model_logging import ModelCallLogger, get_model_logger


logger = logging.getLogger(__name__)


class ComputerUseExecutionError(RuntimeError):
    """Raised when the Computer Use orchestration fails irrecoverably."""


@dataclass(frozen=True)
class InteractionConstraints:
    """Coarse-grained interaction constraints derived from step/action text."""

    disallow_scroll: bool = False

    def has_any(self) -> bool:
        return self.disallow_scroll

    def to_prompt(self) -> str:
        lines: list[str] = []
        if self.disallow_scroll:
            lines.append("- Do NOT scroll (no scroll actions; no mouse wheel).")
        return "\n".join(lines)

    def apply_overrides(self, metadata: Optional[Dict[str, Any]]) -> "InteractionConstraints":
        """Apply optional constraint overrides supplied via runtime metadata."""
        if not metadata:
            return self

        disallow_scroll_override = metadata.get("disallow_scroll")
        if isinstance(disallow_scroll_override, bool):
            return InteractionConstraints(disallow_scroll=disallow_scroll_override)

        policy = str(metadata.get("scroll_policy") or "").strip().lower()
        if policy in {"auto", ""}:
            return self
        if policy in {"allow", "allow_scroll"}:
            return InteractionConstraints(disallow_scroll=False)
        if policy in {"disallow", "disallow_scroll"}:
            return InteractionConstraints(disallow_scroll=True)
        return self

    @staticmethod
    def from_text(text: str) -> "InteractionConstraints":
        lowered = (text or "").lower()
        strict_no_scroll = any(
            phrase in lowered
            for phrase in (
                "without scrolling",
                "no scrolling",
            )
        )
        if strict_no_scroll:
            return InteractionConstraints(disallow_scroll=True)

        soft_no_scroll = any(
            phrase in lowered
            for phrase in (
                "do not scroll",
                "don't scroll",
                "avoid scrolling",
            )
        )
        if not soft_no_scroll:
            return InteractionConstraints(disallow_scroll=False)

        allows_scroll = any(
            phrase in lowered
            for phrase in (
                "scroll down",
                "scroll up",
                "scroll left",
                "scroll right",
                "scroll to",
                "scroll until",
                "scroll just",
                "scroll by",
                "scroll a bit",
                "scroll a little",
                "scroll slightly",
                "scroll enough",
                "scroll more",
                "scroll further down",
                "scroll further up",
                "scroll the page",
            )
        )
        return InteractionConstraints(disallow_scroll=not allows_scroll)

@dataclass
class ComputerUseSessionResult:
    """Result of executing a Computer Use session."""

    actions: List[ComputerToolTurn] = field(default_factory=list)
    safety_events: List[SafetyEvent] = field(default_factory=list)
    final_output: Optional[str] = None
    response_ids: List[str] = field(default_factory=list)
    last_response: Optional[Dict[str, Any]] = None
    terminal_status: Literal["success", "failed"] = "success"
    terminal_failure_reason: Optional[str] = None
    terminal_failure_code: Optional[str] = None


class ComputerUseSession:
    """Wraps computer-use providers and orchestrates action execution."""

    _DISALLOWED_NAVIGATION_ACTIONS = frozenset(
        {
            "open_web_browser",
            "navigate",
            "search",
            "go_back",
            "go_forward",
        }
    )

    def __init__(
        self,
        client: AsyncOpenAI,
        browser: BrowserDriver,
        settings: Settings,
        debug_logger: Optional[DebugLogger] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        google_client: Optional[Any] = None,
        environment: str = "browser",
        coordinate_cache: Optional[CoordinateCache] = None,
        model_logger: Optional[ModelCallLogger] = None,
    ) -> None:
        self._client = client
        self._browser = browser
        self._settings = settings
        self._debug_logger = debug_logger
        self._provider = (provider or settings.cu_provider or "openai").lower()
        self._openai_model = (
            model
            if model and self._provider == "openai"
            else settings.computer_use_model
        )
        self._google_model = (
            model
            if model and self._provider == "google"
            else settings.google_cu_model
        )
        self._model = (
            self._google_model if self._provider == "google" else self._openai_model
        )
        self._google_client = google_client
        self._default_environment = self._normalize_environment_name(environment)
        self._coordinate_cache = coordinate_cache or CoordinateCache(
            self._settings.desktop_coordinate_cache_path
        )
        self._model_logger = model_logger or get_model_logger(
            self._settings.model_log_path,
            max_screenshots=getattr(self._settings, "max_screenshots", None),
        )
        self._allowed_actions: Optional[Set[str]] = None
        self._allowed_domains: Set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_allowed_domains
        )
        self._blocked_domains: Set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_blocked_domains
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
            "click_at",
            "type_text_at",
            "key_combination",
            "hover_at",
            "drag_and_drop",
            "scroll_at",
            "scroll_document",
        }
        self._scroll_turn_limit = max(
            int(
                round(
                    self._settings.actions_computer_tool_max_turns
                    * self._settings.scroll_turn_multiplier
                )
            ),
            self._settings.actions_computer_tool_max_turns,
        )
        self._pending_context_menu_selection = False
        self._interaction_constraints = InteractionConstraints()

    async def run(
        self,
        goal: str,
        initial_screenshot: Optional[bytes],
        metadata: Optional[Dict[str, Any]] = None,
        allowed_actions: Optional[Set[str]] = None,
        environment: Optional[str] = None,
        cache_label: Optional[str] = None,
        cache_action: str = "click",
        use_cache: bool = True,
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
        self._pending_context_menu_selection = False

        step_goal = str(metadata.get("step_goal") or "").strip()
        constraint_source = " ".join([step_goal, goal]).strip()
        self._interaction_constraints = (
            InteractionConstraints.from_text(constraint_source)
            .apply_overrides(metadata)
        )
        if self._interaction_constraints.has_any():
            goal = goal + "\n\nCONSTRAINTS:\n" + self._interaction_constraints.to_prompt()

        env_mode = self._normalize_environment_name(
            environment or metadata.get("environment") or self._default_environment
        )
        try:
            if self._provider == "google":
                try:
                    return await self._run_google(
                        goal=goal,
                        metadata=metadata,
                        environment=env_mode,
                        cache_label=cache_label,
                        cache_action=cache_action,
                        use_cache=use_cache,
                        model=self._google_model,
                    )
                except Exception as exc:
                    logger.warning(
                        "Google Computer Use provider failed; falling back to OpenAI",
                        extra={"error": str(exc)},
                    )
            return await self._run_openai(
                goal=goal,
                initial_screenshot=initial_screenshot,
                metadata=metadata,
                environment=env_mode,
                cache_label=cache_label,
                cache_action=cache_action,
                use_cache=use_cache,
                model=self._openai_model,
            )
        finally:
            self._allowed_actions = None

    async def _run_openai(
        self,
        *,
        goal: str,
        initial_screenshot: Optional[bytes],
        metadata: Dict[str, Any],
        environment: str,
        cache_label: Optional[str],
        cache_action: str,
        use_cache: bool,
        model: str,
    ) -> ComputerUseSessionResult:
        result = ComputerUseSessionResult()

        await self._ensure_browser_ready()

        viewport_width, viewport_height = await self._browser.get_viewport_size()
        screenshot = initial_screenshot or await self._browser.screenshot()
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
                    "Computer Use action execution failed", extra={"call_id": turn.call_id}
                )

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
            follow_up_screenshot: Optional[bytes] = None
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

    async def _run_google(
        self,
        *,
        goal: str,
        metadata: Dict[str, Any],
        environment: str,
        cache_label: Optional[str],
        cache_action: str,
        use_cache: bool,
        model: str,
    ) -> ComputerUseSessionResult:
        result = ComputerUseSessionResult()

        await self._ensure_browser_ready()
        viewport_width, viewport_height = await self._browser.get_viewport_size()
        initial_screenshot = await self._browser.screenshot()
        wrapped_goal = self._wrap_goal_for_google(goal, environment)
        contents, config = self._build_google_initial_request(
            goal=wrapped_goal,
            screenshot_bytes=initial_screenshot,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            environment=environment,
        )

        history: List[Any] = list(contents)
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
        max_turn_reason: Optional[str] = None
        max_turn_code: Optional[str] = None
        last_assistant_text: Optional[str] = None
        last_response_dict: Optional[Dict[str, Any]] = None
        loop_window = max(2, self._settings.actions_computer_tool_loop_detection_window)
        loop_history: Deque[Tuple[Tuple[str, ...], str]] = deque(maxlen=loop_window)

        while True:
            calls = extract_google_function_calls(response)
            assistant_text = extract_assistant_text(response_dict)
            if assistant_text:
                result.final_output = assistant_text
                last_assistant_text = assistant_text
            result.last_response = response_dict
            last_response_dict = response_dict

            if not calls:
                break

            executed_turns: List[ComputerToolTurn] = []
            for call in calls:
                turn = ComputerToolTurn(
                    call_id=str(getattr(call, "name", "") or ""),
                    action_type=str(getattr(call, "name", "") or "unknown"),
                    parameters=getattr(call, "args", {}) or {},
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
                        normalized_coords=True,
                        allow_unknown=True,
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
                else:
                    turn_counter += 1
                    if turn_counter >= self._settings.actions_computer_tool_max_turns:
                        logger.warning(
                            "Computer Use max turns reached (google)",
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

            follow_up_payload, func_response_content, follow_up_screenshot = (
                await self._build_google_follow_up_request(
                    goal=goal,
                    history=history,
                    turns=executed_turns,
                    environment=environment,
                    model=model,
                )
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
            result.response_ids.append(response_dict.get("id", ""))
            if hasattr(response, "candidates"):
                try:
                    history.append(response.candidates[0].content)
                except Exception:
                    logger.debug(
                        "Unable to append Google response content to history",
                        exc_info=True,
                    )

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
                    "last_response": last_response_dict,
                },
            )

        return result

    async def _execute_tool_action(
        self,
        turn: ComputerToolTurn,
        metadata: Dict[str, Any],
        turn_index: int,
        normalized_coords: bool = False,
        allow_unknown: bool = False,
        environment: str = "browser",
        cache_label: Optional[str] = None,
        cache_action: str = "click",
        use_cache: bool = True,
    ) -> None:
        """Execute a single Computer Use tool action via the browser driver."""
        action = turn.parameters or {}
        raw_action_type = action.get("type") or turn.action_type
        action_type = self._canonicalize_action_type(raw_action_type)
        if action_type and action_type != raw_action_type:
            turn.metadata["normalized_action_type"] = action_type
            turn.action_type = action_type
        start = time.perf_counter()

        viewport_width, viewport_height = await self._browser.get_viewport_size()
        turn.metadata["resolution"] = (viewport_width, viewport_height)
        turn.metadata["normalized_coords"] = bool(normalized_coords)
        allow_action, deny_reason = self._is_action_allowed(action_type)
        disallowed_reason = self._disallowed_action_reason(action_type, environment)
        capture_clipboard_after_action = False
        cached = None
        cache_allowed = False
        cache_hit = False

        if cache_label and self._action_matches_cache(action_type, cache_action):
            cache_allowed = True
            if use_cache:
                cached = self._coordinate_cache.lookup(
                    cache_label, cache_action, (viewport_width, viewport_height)
                )
            cache_hit = cached is not None if use_cache else False
            turn.metadata.update(
                {
                    "cache_label": cache_label,
                    "cache_action": cache_action,
                    "cache_hit": cache_hit,
                    "cache_lookup_allowed": use_cache,
                }
            )

        try:
            if not allow_action:
                turn.status = "failed"
                turn.error_message = deny_reason or "Action blocked by policy."
                turn.metadata["policy"] = "observe_only"
            elif disallowed_reason:
                turn.status = "ignored"
                turn.error_message = disallowed_reason
                turn.metadata["ignored"] = True
            else:
                await self._enforce_domain_policy(action_type)

                if self._interaction_constraints.disallow_scroll and self._is_scroll_action(action_type):
                    raise ComputerUseExecutionError(
                        "Step context prohibits scrolling for this action."
                    )

                if self._pending_context_menu_selection and action_type not in {
                    "click",
                    "click_at",
                    "double_click",
                    "right_click",
                }:
                    self._pending_context_menu_selection = False

                if action_type in {"click", "click_at", "move_mouse_and_click"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                        )
                    button = (action.get("button", "left") or "left").lower()
                    click_count = int(action.get("click_count", 1))
                    await asyncio.wait_for(
                        self._browser.click(x, y, button=button, click_count=click_count),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                    if button == "right":
                        self._pending_context_menu_selection = True
                    elif self._pending_context_menu_selection and click_count == 1:
                        capture_clipboard_after_action = True
                        self._pending_context_menu_selection = False
                    else:
                        self._pending_context_menu_selection = False
                elif action_type in {"double_click", "right_click"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                        )
                    button = "right" if action_type == "right_click" else "left"
                    click_count = 2 if action_type == "double_click" else 1
                    await asyncio.wait_for(
                        self._browser.click(x, y, button=button, click_count=click_count),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                    if button == "right":
                        self._pending_context_menu_selection = True
                elif action_type in {"move", "hover_at"}:
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
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
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type == "drag":
                    def _coerce_float(value: Any) -> Optional[float]:
                        if value is None:
                            return None
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    start_x_raw = _coerce_float(
                        action.get("start_x") or action.get("from_x") or action.get("x")
                    )
                    start_y_raw = _coerce_float(
                        action.get("start_y") or action.get("from_y") or action.get("y")
                    )
                    end_x_raw = _coerce_float(action.get("end_x") or action.get("to_x"))
                    end_y_raw = _coerce_float(action.get("end_y") or action.get("to_y"))

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

                    if normalized_coords:
                        start_x, start_y = denormalize_coordinates(
                            start_x_raw, start_y_raw, viewport_width, viewport_height
                        )
                        end_x, end_y = denormalize_coordinates(
                            end_x_raw, end_y_raw, viewport_width, viewport_height
                        )
                    else:
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
                    turn.metadata.update(
                        {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}
                    )
                    turn.status = "executed"
                elif action_type == "drag_and_drop":
                    start_x, start_y = self._resolve_coordinates(
                        action,
                        viewport_width,
                        viewport_height,
                        normalized=normalized_coords,
                    )
                    end_x, end_y = self._resolve_coordinates(
                        action,
                        viewport_width,
                        viewport_height,
                        prefix="destination_",
                        normalized=normalized_coords,
                    )
                    await asyncio.wait_for(
                        self._browser.drag_mouse(start_x, start_y, end_x, end_y, steps=1),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update(
                        {
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                        }
                    )
                    turn.status = "executed"
                elif action_type == "scroll":
                    scroll_x = int(action.get("scroll_x", 0))
                    scroll_y = int(action.get("scroll_y", 0))
                    max_pixels = int(self._settings.scroll_max_magnitude)
                    scroll_x = max(-max_pixels, min(scroll_x, max_pixels))
                    scroll_y = max(-max_pixels, min(scroll_y, max_pixels))
                    direction = self._extract_scroll_direction(action_type, action)
                    if direction:
                        turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_x"] = scroll_x
                    turn.metadata["scroll_y"] = scroll_y
                    await asyncio.wait_for(
                        self._browser.scroll_by_pixels(x=scroll_x, y=scroll_y, smooth=False),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.status = "executed"
                elif action_type == "scroll_document":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError("scroll_document action missing direction.")
                    magnitude_raw = action.get("magnitude")
                    magnitude = (
                        abs(int(magnitude_raw))
                        if magnitude_raw is not None
                        else int(self._settings.scroll_default_magnitude)
                    )
                    magnitude = min(magnitude, int(self._settings.scroll_max_magnitude))
                    turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_magnitude"] = magnitude
                    await self._browser.scroll(direction, magnitude)
                    turn.status = "executed"
                elif action_type == "scroll_at":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError("scroll_at action missing direction.")
                    magnitude_raw = action.get("magnitude")
                    magnitude = (
                        abs(int(magnitude_raw))
                        if magnitude_raw is not None
                        else int(self._settings.scroll_default_magnitude)
                    )
                    magnitude = min(magnitude, int(self._settings.scroll_max_magnitude))
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                        )
                    turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_magnitude"] = magnitude
                    await self._browser.move_mouse(x, y, steps=1)
                    await self._browser.scroll(direction, magnitude)
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type == "type":
                    text_payload = action.get("text")
                    if not text_payload:
                        text_payload = (
                            action.get("value")
                            or action.get("input")
                            or metadata.get("value")
                        )
                        if text_payload:
                            turn.metadata["synthetic_text_payload"] = text_payload
                    if not text_payload:
                        raise ComputerUseExecutionError("Type action missing text payload.")
                    await self._browser.type_text(text_payload)
                    turn.status = "executed"
                elif action_type == "type_text_at":
                    text_payload = action.get("text")
                    if text_payload is None:
                        raise ComputerUseExecutionError("type_text_at action missing text.")
                    press_enter_default = False if normalized_coords else True
                    press_enter = bool(action.get("press_enter", press_enter_default))
                    clear_before = bool(action.get("clear_before_typing", True))
                    if cached:
                        x, y = int(cached.x), int(cached.y)
                    else:
                        x, y = self._resolve_coordinates(
                            action,
                            viewport_width,
                            viewport_height,
                            normalized=normalized_coords,
                        )
                    await self._browser.click(x, y, button="left", click_count=1)
                    if clear_before:
                        await self._browser.press_key("ctrl+a")
                        await self._browser.press_key("backspace")
                    await self._browser.type_text(str(text_payload))
                    if press_enter:
                        await self._browser.press_key("enter")
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type in {"keypress", "key_combination"}:
                    if action_type == "keypress":
                        key_sequence = self._resolve_key_sequence(action, metadata)
                        if not key_sequence:
                            raise ComputerUseExecutionError(
                                "Key press action missing key payload."
                            )
                        if not action.get("keys"):
                            turn.metadata["synthetic_key_sequence"] = key_sequence
                        for key in key_sequence:
                            normalized = normalize_key_sequence(key)
                            await self._browser.press_key(normalized)
                            if self._should_capture_clipboard_after_key_combo(normalized):
                                capture_clipboard_after_action = True
                    else:
                        keys = action.get("keys")
                        if not keys:
                            raise ComputerUseExecutionError(
                                "Key combination action missing keys."
                            )
                        if isinstance(keys, (list, tuple)):
                            for key in keys:
                                normalized = normalize_key_sequence(str(key))
                                await self._browser.press_key(normalized)
                                if self._should_capture_clipboard_after_key_combo(
                                    normalized
                                ):
                                    capture_clipboard_after_action = True
                        else:
                            normalized = normalize_key_sequence(str(keys))
                            await self._browser.press_key(normalized)
                            if self._should_capture_clipboard_after_key_combo(
                                normalized
                            ):
                                capture_clipboard_after_action = True
                    turn.status = "executed"
                elif action_type == "read_clipboard":
                    clipboard_text = await self._maybe_read_clipboard(turn)
                    if clipboard_text is not None:
                        turn.metadata["clipboard_text"] = clipboard_text
                    turn.status = "executed"
                elif action_type == "wait":
                    duration = int(
                        action.get("duration_ms")
                        or self._settings.actions_computer_tool_stabilization_wait_ms
                    )
                    await self._browser.wait(duration)
                    turn.metadata["duration_ms"] = duration
                    turn.status = "executed"
                elif action_type == "wait_5_seconds":
                    await self._browser.wait(5000)
                    turn.metadata["duration_ms"] = 5000
                    turn.status = "executed"
                elif action_type == "go_back":
                    await self._browser.press_key("alt+left")
                    turn.status = "executed"
                elif action_type == "go_forward":
                    await self._browser.press_key("alt+right")
                    turn.status = "executed"
                elif action_type == "search":
                    await self._navigate_via_address_bar("https://www.google.com/")
                    turn.status = "executed"
                elif action_type == "navigate":
                    url = action.get("url")
                    if not url:
                        raise ComputerUseExecutionError("Navigate action missing url.")
                    await self._navigate_via_address_bar(str(url))
                    turn.status = "executed"
                elif action_type == "screenshot":
                    logger.debug(
                        "Computer Use requested screenshot action; no browser operation executed."
                    )
                    turn.status = "executed"
                else:
                    if allow_unknown:
                        turn.status = "ignored"
                        turn.error_message = f"unsupported action: {action_type}"
                        turn.metadata["ignored"] = True
                    else:
                        raise ComputerUseExecutionError(
                            f"Unsupported computer action type: {action_type}"
                        )

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
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "", cache_action, (viewport_width, viewport_height)
                    )
                except Exception:
                    logger.debug("Failed to invalidate coordinate cache", exc_info=True)
        except Exception:
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "", cache_action, (viewport_width, viewport_height)
                    )
                except Exception:
                    logger.debug("Failed to invalidate coordinate cache", exc_info=True)
            raise
        finally:
            turn.latency_ms = (time.perf_counter() - start) * 1000
            if turn.status == "executed":
                await self._post_action_wait()
            if capture_clipboard_after_action and turn.status == "executed":
                clipboard_text = await self._maybe_read_clipboard(turn)
                if clipboard_text is not None:
                    turn.metadata["clipboard_text"] = clipboard_text
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
        environment: str = "browser",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the payload for a follow-up request after executing an action."""
        screenshot_b64 = call.metadata.get("screenshot_base64")
        if not screenshot_b64:
            screenshot_bytes = await self._browser.screenshot()
            screenshot_b64 = encode_png_base64(screenshot_bytes)

        viewport_width, viewport_height = await self._browser.get_viewport_size()

        payload: Dict[str, Any] = {
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
                    "acknowledged_safety_checks": [],
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
        elif action_type in {"key_press", "keypress", "key_combination"}:
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
        environment: str = "browser",
        model: Optional[str] = None,
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
    def _sanitize_payload_for_log(payload: Dict[str, Any]) -> Dict[str, Any]:
        def _scrub(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_scrub(item) for item in value]
            if isinstance(value, str) and value.startswith("data:image"):
                return "<<attached screenshot>>"
            return value

        return _scrub(payload)

    def _build_google_initial_request(
        self,
        goal: str,
        screenshot_bytes: bytes,
        viewport_width: int,
        viewport_height: int,
        environment: str = "desktop",
    ) -> Tuple[List[Any], Any]:
        from google.genai import types  # type: ignore

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=goal),
                    types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                ],
            )
        ]
        tools = [
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=self._map_google_environment(environment),
                )
            ),
        ]
        return contents, types.GenerateContentConfig(tools=tools)

    async def _build_google_follow_up_request(
        self,
        goal: str,
        history: List[Any],
        turns: List[ComputerToolTurn],
        environment: str = "desktop",
        model: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Any, bytes]:
        from google.genai import types  # type: ignore

        screenshot_bytes = await self._browser.screenshot()
        page_url = ""
        try:
            page_url = await self._browser.get_page_url()
        except Exception:
            page_url = ""
        if not page_url:
            page_url = "desktop://"
        parts: List[Any] = []
        for turn in turns:
            response_payload = {
                "status": turn.status,
                "call_id": turn.call_id,
                "url": page_url,
                "x": turn.metadata.get("x"),
                "y": turn.metadata.get("y"),
                "clipboard_text": turn.metadata.get("clipboard_text"),
                "clipboard_truncated": turn.metadata.get("clipboard_truncated"),
                "clipboard_error": turn.metadata.get("clipboard_error"),
                "error": turn.error_message,
            }
            response_payload = {
                k: v for k, v in response_payload.items() if v is not None
            }
            parts.append(
                types.Part.from_function_response(
                    name=turn.action_type or "action",
                    response=response_payload,
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_bytes
                            )
                        )
                    ],
                )
            )

        function_response_content = types.Content(role="user", parts=parts)
        contents = list(history) + [function_response_content]
        tools = [
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=self._map_google_environment(environment),
                )
            ),
        ]
        return (
            {
                "model": model or self._google_model,
                "contents": contents,
                "config": types.GenerateContentConfig(tools=tools),
            },
            function_response_content,
            screenshot_bytes,
        )

    async def _create_response(self, payload: Dict[str, Any]) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        timeout = float(self._settings.openai_request_timeout_seconds)
        logger.debug("Calling OpenAI Responses API", extra={"model": payload.get("model")})
        return await self._client.responses.create(timeout=timeout, **payload)

    async def _create_google_response(self, payload: Dict[str, Any]) -> Any:
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
                responses = getattr(client, "responses")
                if hasattr(responses, "generate"):
                    return responses.generate(**payload)
                if hasattr(responses, "create"):
                    return responses.create(**payload)
            raise ComputerUseExecutionError(
                "Google GenAI client does not support responses.generate/create calls."
            )

        return await asyncio.to_thread(_call)

    def _ensure_google_client(self) -> Optional[Any]:
        if self._provider != "google":
            return None
        if self._google_client:
            return self._google_client
        if genai is None:
            return None
        vertex_project = str(getattr(self._settings, "vertex_project", "") or "").strip()
        vertex_location = str(
            getattr(self._settings, "vertex_location", "us-central1") or ""
        ).strip()
        vertex_api_key = str(getattr(self._settings, "vertex_api_key", "") or "").strip()

        if vertex_project:
            if not vertex_location:
                raise ComputerUseExecutionError(
                    "VERTEX_LOCATION is required when VERTEX_PROJECT is configured."
                )
            try:
                vertex_kwargs: Dict[str, Any] = {
                    "vertexai": True,
                    "project": vertex_project,
                    "location": vertex_location,
                }
                if vertex_api_key:
                    vertex_kwargs["api_key"] = vertex_api_key
                self._google_client = genai.Client(
                    **vertex_kwargs,
                )
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
        self._google_client = genai.Client(api_key=vertex_api_key)
        logger.info("Initialized Google CU client in API key mode")
        return self._google_client

    async def _ensure_browser_ready(self) -> None:
        """Ensure the browser session is started before execution."""
        await self._browser.start()

    async def invalidate_cache(self, cache_label: str, cache_action: str) -> None:
        """Invalidate a cached coordinate for the current resolution."""
        logger.info("ComputerUseSession: invalidating coordinate cache after failure")
        try:
            resolution = await self._browser.get_viewport_size()
            self._coordinate_cache.invalidate(cache_label, cache_action, resolution)
        except Exception:
            logger.debug(
                "Failed to invalidate coordinate cache entry",
                exc_info=True,
                extra={"cache_label": cache_label, "cache_action": cache_action},
            )

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
        environment: str = "browser",
        model: Optional[str] = None,
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

    def _should_abort_on_safety(
        self,
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
        if not fail_fast and policy == "auto_approve":
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
        self,
        *,
        result: ComputerUseSessionResult,
        metadata: Dict[str, Any],
        reason: str,
        code: str,
        response_id: Optional[str],
        parameters: Optional[Dict[str, Any]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        call_id_prefix: str = "terminal",
    ) -> ComputerToolTurn:
        """Append a terminal failed system notice turn and mark the result as failed."""
        params = {
            "type": "system_notice",
            "reason": code,
        }
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

    @classmethod
    def _disallowed_action_reason(
        cls, action_type: Optional[str], environment: str
    ) -> Optional[str]:
        if not action_type:
            return None
        env_mode = cls._normalize_environment_name(environment)
        normalized = action_type.lower().strip()
        if normalized in cls._DISALLOWED_NAVIGATION_ACTIONS and env_mode in {
            "desktop",
            "browser",
        }:
            return (
                f"{normalized} is disabled; use direct UI interactions in the existing "
                "window instead."
            )
        return None

    def _resolve_coordinates(
        self,
        action: Dict[str, Any],
        viewport_width: int,
        viewport_height: int,
        *,
        prefix: str = "",
        normalized: bool = False,
    ) -> Tuple[int, int]:
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
    def _action_matches_cache(action_type: Optional[str], cache_action: str) -> bool:
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
    def _extract_scroll_direction(action_type: str, action: Dict[str, Any]) -> Optional[str]:
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
        """Return True when the key combo likely performed a copy-to-clipboard action."""
        try:
            if isinstance(keys, str):
                keys_str = keys
            else:
                keys_str = "+".join([str(k) for k in keys])  # type: ignore[arg-type]
        except Exception:
            keys_str = str(keys)
        normalized = keys_str.lower().replace(" ", "")
        if normalized in {"ctrl+c", "control+c"}:
            return True
        if (("ctrl+" in normalized) or ("control+" in normalized)) and normalized.endswith(
            "+c"
        ):
            return True
        return False

    async def _maybe_read_clipboard(self, turn: ComputerToolTurn) -> Optional[str]:
        reader = getattr(self._browser, "read_clipboard", None)
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
        return clipboard_text[:max_chars]

    async def _navigate_via_address_bar(self, url: str) -> None:
        await self._browser.press_key("ctrl+l")
        await self._browser.type_text(url)
        await self._browser.press_key("enter")

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

    def _wrap_goal_for_google(self, goal: str, env_mode: str) -> str:
        """Wrap the goal with context for Google CU per environment."""
        requires_json = self._goal_requires_json(goal)
        completion_instruction = (
            "When the task is complete, STOP issuing function calls and reply ONLY with the requested JSON (no prose).\n\n"
            if requires_json
            else "When the task is complete, stop issuing function calls and reply with a short confirmation of what you accomplished.\n\n"
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
        """Heuristic: treat goals that request JSON output as strict-JSON completion tasks."""
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
    def _normalize_environment_name(environment: Optional[str]) -> str:
        value = (environment or "desktop").strip().lower()
        if value in {"unspecified", "env_unspecified", "default"}:
            return "unspecified"
        if value in {"browser", "web"}:
            return "browser"
        if value in {"linux", "desktop", "os", "system"}:
            return "desktop"
        return "desktop"

    @staticmethod
    def _map_openai_environment(env_mode: str) -> str:
        return "browser" if env_mode == "browser" else "linux"

    @staticmethod
    def _map_google_environment(env_mode: str) -> Any:
        from google.genai import types  # type: ignore

        if env_mode == "browser":
            return getattr(
                types.Environment,
                "ENVIRONMENT_BROWSER",
                types.Environment.ENVIRONMENT_UNSPECIFIED,
            )
        return types.Environment.ENVIRONMENT_UNSPECIFIED

    @staticmethod
    def _is_scroll_action(action_type: Optional[str]) -> bool:
        if not action_type:
            return False
        normalized = action_type.lower()
        return normalized in {"scroll", "scroll_at", "scroll_document", "scroll_window"}

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
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:
            pass
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


def extract_google_function_calls(response_obj: Any) -> List[Any]:
    """Extract function_call parts from a Google response object."""
    calls: List[Any] = []
    candidates = getattr(response_obj, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        for part in parts:
            func_call = getattr(part, "function_call", None)
            if func_call:
                calls.append(func_call)
    return calls


def extract_google_computer_calls(response_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function/call items from a Google response."""
    calls: List[Dict[str, Any]] = []
    output_items = response_dict.get("output", [])
    for item in output_items:
        if item.get("type") in {"function_call", "computer_call"}:
            calls.append(item)
    if calls:
        return calls

    candidates = response_dict.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            fn_call = part.get("functionCall") or part.get("function_call")
            if not fn_call:
                continue
            calls.append(
                {
                    "id": fn_call.get("id") or fn_call.get("name"),
                    "call_id": fn_call.get("id") or fn_call.get("name"),
                    "action": {
                        "type": fn_call.get("name"),
                        **(fn_call.get("args") or {}),
                    },
                }
            )
    return calls


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
    if not texts:
        candidates = response_dict.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                text = part.get("text") or part.get("output_text")
                if text:
                    texts.append(text)
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


def denormalize_coordinates(
    x: Optional[float],
    y: Optional[float],
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int]:
    """Convert normalized 0-999 coordinates to absolute pixels."""

    def _clamp(value: float, maximum: int) -> int:
        return max(0, min(int(round(value)), max(0, maximum - 1)))

    if x is None or y is None:
        return viewport_width // 2, viewport_height // 2
    return _clamp(float(x) * viewport_width / 999.0, viewport_width), _clamp(
        float(y) * viewport_height / 999.0, viewport_height
    )
