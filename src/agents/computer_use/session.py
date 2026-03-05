"""Computer Use tool orchestration for the Action Agent."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import urlparse
from uuid import uuid4

from openai import AsyncOpenAI

try:  # Optional dependency for Gemini computer-use
    import google.genai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

try:  # Optional dependency for Claude computer-use
    from anthropic import AsyncAnthropic as _AsyncAnthropic
except Exception:  # pragma: no cover - optional dependency
    _AsyncAnthropic = None

from src.config.settings import Settings
from src.core.enhanced_types import ComputerToolTurn, SafetyEvent
from src.core.interfaces import AutomationDriver
from src.desktop.cache import CoordinateCache
from src.monitoring.debug_logger import DebugLogger
from src.runtime.environment import (
    RuntimeEnvironmentName,
    coordinate_cache_path_for_environment,
    normalize_runtime_environment_name,
    runtime_environment_spec,
)
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

    def apply_overrides(
        self, metadata: dict[str, Any] | None
    ) -> InteractionConstraints:
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
    def from_text(text: str) -> InteractionConstraints:
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


def _strip_bytes(obj: Any) -> Any:
    """Recursively replace bytes values with a placeholder string for logging."""
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, dict):
        return {k: _strip_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_bytes(v) for v in obj]
    return obj


@dataclass
class ComputerUseSessionResult:
    """Result of executing a Computer Use session."""

    actions: list[ComputerToolTurn] = field(default_factory=list)
    safety_events: list[SafetyEvent] = field(default_factory=list)
    final_output: str | None = None
    response_ids: list[str] = field(default_factory=list)
    last_response: dict[str, Any] | None = None
    terminal_status: Literal["success", "failed"] = "success"
    terminal_failure_reason: str | None = None
    terminal_failure_code: str | None = None


@dataclass(frozen=True)
class GoogleFunctionCallEnvelope:
    """Function call plus deterministic ordering metadata for a single turn."""

    function_call: Any
    sequence: int
    candidate_index: int
    part_index: int


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
    _GOOGLE_RETRY_DELAYS_SECONDS = (1.0, 5.0, 10.0)
    _GOOGLE_PROMPT_SAFETY_RETRY_DELAYS_SECONDS = (0.25, 0.75)
    _GOOGLE_PROMPT_SAFETY_RETRY_JITTER_SECONDS = (0.05, 0.2)

    def __init__(
        self,
        client: AsyncOpenAI,
        automation_driver: AutomationDriver,
        settings: Settings,
        debug_logger: DebugLogger | None = None,
        model: str | None = None,
        provider: str | None = None,
        google_client: Any | None = None,
        anthropic_client: Any | None = None,
        environment: str = "browser",
        coordinate_cache: CoordinateCache | None = None,
        model_logger: ModelCallLogger | None = None,
    ) -> None:
        self._client = client
        self._automation_driver = automation_driver
        self._settings = settings
        self._debug_logger = debug_logger
        raw_provider = provider if provider is not None else settings.cu_provider
        self._provider = str(raw_provider or "").strip().lower()
        if self._provider not in {"openai", "google", "anthropic"}:
            raise ValueError(
                f"Unsupported computer-use provider '{raw_provider}'. "
                "Supported providers are 'openai', 'google', and 'anthropic'."
            )
        self._openai_model = (
            model
            if model and self._provider == "openai"
            else settings.computer_use_model
        )
        self._google_model = (
            model
            if model and self._provider == "google"
            else getattr(
                settings, "google_cu_model", "gemini-2.5-computer-use-preview-10-2025"
            )
        )
        self._anthropic_model = (
            model
            if model and self._provider == "anthropic"
            else getattr(settings, "anthropic_cu_model", "claude-sonnet-4-6")
        )
        self._model = {
            "google": self._google_model,
            "anthropic": self._anthropic_model,
        }.get(self._provider, self._openai_model)
        self._google_client = google_client
        self._anthropic_client = anthropic_client
        self._anthropic_tool_type = "computer_20251124"
        self._anthropic_tool_name = "computer"
        self._anthropic_betas = self._parse_betas(
            str(getattr(self._settings, "anthropic_cu_beta", "") or "")
        )
        self._anthropic_max_tokens = int(
            getattr(self._settings, "anthropic_cu_max_tokens", 16384)
        )
        self._default_environment = self._normalize_environment_name(environment)
        coordinate_cache_path = coordinate_cache_path_for_environment(
            self._settings,
            self._default_environment,
        )
        self._coordinate_cache = coordinate_cache or CoordinateCache(
            coordinate_cache_path
        )
        self._model_logger = model_logger or get_model_logger(
            self._settings.model_log_path,
            max_screenshots=getattr(self._settings, "max_screenshots", None),
        )
        self._allowed_actions: set[str] | None = None
        self._allowed_domains: set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_allowed_domains
        )
        self._blocked_domains: set[str] = self._normalize_domain_set(
            settings.actions_computer_tool_blocked_domains
        )
        self._stateful_actions: set[str] = {
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
        self._last_pointer_position: tuple[int, int] | None = None

    async def run(
        self,
        goal: str,
        initial_screenshot: bytes | None,
        metadata: dict[str, Any] | None = None,
        allowed_actions: set[str] | None = None,
        environment: str | None = None,
        cache_label: str | None = None,
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
        self._last_pointer_position = None

        step_goal = str(metadata.get("step_goal") or "").strip()
        constraint_source = " ".join([step_goal, goal]).strip()
        self._interaction_constraints = InteractionConstraints.from_text(
            constraint_source
        ).apply_overrides(metadata)
        if self._interaction_constraints.has_any():
            goal = (
                goal + "\n\nCONSTRAINTS:\n" + self._interaction_constraints.to_prompt()
            )

        env_mode = self._normalize_environment_name(
            environment or metadata.get("environment") or self._default_environment
        )
        try:
            if self._provider == "google":
                return await self._run_google(
                    goal=goal,
                    metadata=metadata,
                    environment=env_mode,
                    cache_label=cache_label,
                    cache_action=cache_action,
                    use_cache=use_cache,
                    model=self._google_model,
                )
            if self._provider == "openai":
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
            if self._provider == "anthropic":
                return await self._run_anthropic(
                    goal=goal,
                    initial_screenshot=initial_screenshot,
                    metadata=metadata,
                    environment=env_mode,
                    cache_label=cache_label,
                    cache_action=cache_action,
                    use_cache=use_cache,
                    model=self._anthropic_model,
                )
            raise ComputerUseExecutionError(
                f"Unsupported computer-use provider '{self._provider}'. "
                "Supported providers are 'openai', 'google', and 'anthropic'."
            )
        finally:
            self._allowed_actions = None

    async def _run_openai(
        self,
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
                message = f"Computer Use max turns exceeded after {turn_counter} turns (limit: {max_turns})."
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

    async def _run_google(
        self,
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
        self,
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
        self,
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

    async def _run_anthropic(
        self,
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

    async def _execute_tool_action(
        self,
        turn: ComputerToolTurn,
        metadata: dict[str, Any],
        turn_index: int,
        normalized_coords: bool = False,
        allow_unknown: bool = False,
        environment: str = "browser",
        cache_label: str | None = None,
        cache_action: str = "click",
        use_cache: bool = True,
    ) -> None:
        """Execute a single Computer Use tool action via the automation driver."""
        action = turn.parameters or {}
        raw_action_type = action.get("type") or turn.action_type
        action_type = self._canonicalize_action_type(raw_action_type)
        if action_type and action_type != raw_action_type:
            turn.metadata["normalized_action_type"] = action_type
            turn.action_type = action_type
        start = time.perf_counter()

        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()
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
                # Normalize custom modifier-click actions before all policy checks.
                implicit_modifier: str | None = None
                if action_type == "ctrl_click":
                    action_type = "click_at"
                    implicit_modifier = "ctrl"
                elif action_type == "shift_click":
                    action_type = "click_at"
                    implicit_modifier = "shift"

                await self._enforce_domain_policy(action_type)

                if (
                    self._interaction_constraints.disallow_scroll
                    and self._is_scroll_action(action_type)
                ):
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
                    raw_modifiers = action.get("modifiers") or []
                    if isinstance(raw_modifiers, str):
                        raw_modifiers = [raw_modifiers]
                    modifiers = [str(m).lower() for m in raw_modifiers if m]
                    if implicit_modifier and implicit_modifier not in modifiers:
                        modifiers = [implicit_modifier] + modifiers
                    await asyncio.wait_for(
                        self._automation_driver.click(
                            x,
                            y,
                            button=button,
                            click_count=click_count,
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    if modifiers:
                        turn.metadata["modifiers"] = modifiers
                        turn.metadata["modifiers_applied"] = False
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
                        self._automation_driver.click(
                            x, y, button=button, click_count=click_count
                        ),
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
                        self._automation_driver.move_mouse(x, y, steps=steps),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.metadata.update({"x": x, "y": y})
                    turn.status = "executed"
                elif action_type == "drag":

                    def _coerce_float(value: Any) -> float | None:
                        if value is None:
                            return None
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    def _first_numeric(*keys: str) -> float | None:
                        for key in keys:
                            if key not in action:
                                continue
                            numeric = _coerce_float(action.get(key))
                            if numeric is not None:
                                return numeric
                        return None

                    start_x_raw = _first_numeric("start_x", "from_x", "x")
                    start_y_raw = _first_numeric("start_y", "from_y", "y")
                    # Accept legacy drag-and-drop destination aliases from provider payloads.
                    end_x_raw = _first_numeric(
                        "end_x",
                        "to_x",
                        "destination_x",
                        "target_x",
                    )
                    end_y_raw = _first_numeric(
                        "end_y",
                        "to_y",
                        "destination_y",
                        "target_y",
                    )

                    path_points = action.get("path") or action.get("points")
                    if isinstance(path_points, list) and len(path_points) >= 2:
                        first_point = path_points[0]
                        last_point = path_points[-1]
                        if isinstance(first_point, dict):
                            if start_x_raw is None:
                                start_x_raw = _coerce_float(first_point.get("x"))
                            if start_y_raw is None:
                                start_y_raw = _coerce_float(first_point.get("y"))
                        if isinstance(last_point, dict):
                            if end_x_raw is None:
                                end_x_raw = _coerce_float(last_point.get("x"))
                            if end_y_raw is None:
                                end_y_raw = _coerce_float(last_point.get("y"))

                    if start_x_raw is None or start_y_raw is None:
                        raise ComputerUseExecutionError(
                            "Drag action missing start coordinates."
                        )

                    delta_x_raw = _first_numeric("dx", "delta_x")
                    delta_y_raw = _first_numeric("dy", "delta_y")
                    if end_x_raw is None and delta_x_raw is not None:
                        end_x_raw = start_x_raw + delta_x_raw
                    if end_y_raw is None and delta_y_raw is not None:
                        end_y_raw = start_y_raw + delta_y_raw
                    if end_x_raw is None or end_y_raw is None:
                        raise ComputerUseExecutionError(
                            "Drag action missing destination coordinates."
                        )

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
                        self._automation_driver.drag_mouse(
                            start_x, start_y, end_x, end_y, steps=steps
                        ),
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
                        self._automation_driver.scroll_by_pixels(
                            x=scroll_x, y=scroll_y, smooth=False
                        ),
                        timeout=self._action_timeout_seconds,
                    )
                    turn.status = "executed"
                elif action_type == "scroll_document":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError(
                            "scroll_document action missing direction."
                        )
                    magnitude_raw = action.get("magnitude")
                    magnitude = (
                        abs(int(magnitude_raw))
                        if magnitude_raw is not None
                        else int(self._settings.scroll_default_magnitude)
                    )
                    magnitude = min(magnitude, int(self._settings.scroll_max_magnitude))
                    turn.metadata["scroll_direction"] = direction
                    turn.metadata["scroll_magnitude"] = magnitude
                    await self._automation_driver.scroll(direction, magnitude)
                    turn.status = "executed"
                elif action_type == "scroll_at":
                    direction = (action.get("direction") or "").lower()
                    if direction not in {"up", "down", "left", "right"}:
                        raise ComputerUseExecutionError(
                            "scroll_at action missing direction."
                        )
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
                    await self._automation_driver.move_mouse(x, y, steps=1)
                    await self._automation_driver.scroll(direction, magnitude)
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
                        raise ComputerUseExecutionError(
                            "Type action missing text payload."
                        )
                    await self._automation_driver.type_text(text_payload)
                    turn.status = "executed"
                elif action_type == "type_text_at":
                    text_payload = action.get("text")
                    if text_payload is None:
                        raise ComputerUseExecutionError(
                            "type_text_at action missing text."
                        )
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
                    tap_count = (
                        3 if (clear_before and environment == "mobile_adb") else 1
                    )
                    await self._automation_driver.click(
                        x, y, button="left", click_count=tap_count
                    )
                    if clear_before:
                        if environment == "mobile_adb":
                            pass  # triple-tap selected all; type_text replaces the selection
                        else:
                            await self._automation_driver.press_key("ctrl+a")
                            await self._automation_driver.press_key("backspace")
                    await self._automation_driver.type_text(str(text_payload))
                    if press_enter:
                        await self._automation_driver.press_key("enter")
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
                            await self._automation_driver.press_key(normalized)
                            if self._should_capture_clipboard_after_key_combo(
                                normalized
                            ):
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
                                await self._automation_driver.press_key(normalized)
                                if self._should_capture_clipboard_after_key_combo(
                                    normalized
                                ):
                                    capture_clipboard_after_action = True
                        else:
                            normalized = normalize_key_sequence(str(keys))
                            await self._automation_driver.press_key(normalized)
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
                    await self._automation_driver.wait(duration)
                    turn.metadata["duration_ms"] = duration
                    turn.status = "executed"
                elif action_type == "wait_5_seconds":
                    await self._automation_driver.wait(5000)
                    turn.metadata["duration_ms"] = 5000
                    turn.status = "executed"
                elif action_type == "go_back":
                    await self._automation_driver.press_key("alt+left")
                    turn.status = "executed"
                elif action_type == "go_forward":
                    await self._automation_driver.press_key("alt+right")
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
                        "Computer Use requested screenshot action; no automation_driver operation executed."
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
                    "raw_action_type": raw_action_type,
                    "action_keys": sorted(action.keys()),
                    "reason": turn.error_message,
                },
            )
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "",
                        cache_action,
                        (viewport_width, viewport_height),
                    )
                except Exception:
                    logger.debug("Failed to invalidate coordinate cache", exc_info=True)
        except Exception:
            if cache_allowed and cache_hit:
                try:
                    self._coordinate_cache.invalidate(
                        cache_label or "",
                        cache_action,
                        (viewport_width, viewport_height),
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
        metadata: dict[str, Any],
        turn_index: int,
    ) -> None:
        """Capture screenshot and update metadata after action execution."""
        screenshot_bytes = await self._automation_driver.screenshot()
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
        self, action: dict[str, Any], viewport_width: int, viewport_height: int
    ) -> None:
        """Execute a primary click event."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        button = action.get("button", "left")
        click_count = int(action.get("click_count", 1))
        await asyncio.wait_for(
            self._automation_driver.click(x, y, button=button, click_count=click_count),
            timeout=self._action_timeout_seconds,
        )

    async def _execute_special_click(
        self, action: dict[str, Any], viewport_width: int, viewport_height: int
    ) -> None:
        """Execute double or right click events."""
        x, y = normalize_coordinates(
            action.get("x"), action.get("y"), viewport_width, viewport_height
        )
        action_type = action.get("type")
        if action_type == "double_click":
            await asyncio.wait_for(
                self._automation_driver.click(x, y, button="left", click_count=2),
                timeout=self._action_timeout_seconds,
            )
        elif action_type == "right_click":
            await asyncio.wait_for(
                self._automation_driver.click(x, y, button="right", click_count=1),
                timeout=self._action_timeout_seconds,
            )

    async def _execute_scroll(self, action: dict[str, Any]) -> None:
        """Execute a scroll event via pixel deltas."""
        scroll_x = int(action.get("scroll_x", 0))
        scroll_y = int(action.get("scroll_y", 0))
        await asyncio.wait_for(
            self._automation_driver.scroll_by_pixels(
                x=scroll_x, y=scroll_y, smooth=False
            ),
            timeout=self._action_timeout_seconds,
        )

    async def _build_follow_up_request(
        self,
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

    def _build_acknowledged_safety_checks(
        self,
        call: ComputerToolTurn,
    ) -> list[dict[str, str]]:
        """Return safety checks to acknowledge when the policy allows auto-approval."""
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
    def _extract_google_pending_safety_checks(
        action_args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract Google Computer Use safety decision metadata from call arguments."""
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

    @staticmethod
    def _google_safety_acknowledgement(turn: ComputerToolTurn) -> str | None:
        """Return the acknowledgement marker required for confirm-gated actions."""
        for check in turn.pending_safety_checks:
            if not isinstance(check, dict):
                continue
            decision = str(check.get("decision") or check.get("code") or "").strip()
            if decision.lower() == "require_confirmation":
                return "true"
        return None

    def _update_loop_history(
        self,
        turn: ComputerToolTurn,
        history: deque[tuple[tuple[str, ...], str]],
        window: int,
    ) -> dict[str, Any] | None:
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

        # Secondary check: screen unchanged across the full window regardless of action.
        # Catches cases where the model varies coordinates slightly but nothing changes.
        if all(hash_ == screenshot_hash for _, hash_ in history):
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

    def _compute_turn_signature(self, turn: ComputerToolTurn) -> tuple[str, ...] | None:
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

        # Handle both flat {"x": x, "y": y} and Google CU {"coordinate": [x, y]} formats.
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
        except Exception:  # pragma: no cover - defensive
            return None

    def _build_initial_request(
        self,
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

    def _build_google_initial_request(
        self,
        goal: str,
        screenshot_bytes: bytes,
        viewport_width: int,
        viewport_height: int,
        environment: str = "desktop",
    ) -> tuple[list[Any], Any]:
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
        return contents, self._build_google_generate_config(environment)

    async def _build_google_follow_up_request(
        self,
        goal: str,
        history: list[Any],
        turns: list[ComputerToolTurn],
        metadata: dict[str, Any],
        environment: str = "desktop",
        model: str | None = None,
    ) -> tuple[dict[str, Any], Any, bytes]:
        from google.genai import types  # type: ignore

        screenshot_bytes = await self._automation_driver.screenshot()
        page_url = ""
        try:
            page_url = await self._automation_driver.get_page_url()
        except Exception:
            page_url = ""
        if not page_url:
            page_url = "desktop://"
        parts: list[Any] = []
        for turn in turns:
            google_call_id = str(
                turn.metadata.get("google_function_call_id") or ""
            ).strip()
            response_name = str(
                turn.metadata.get("google_function_call_name") or ""
            ).strip()
            if not response_name:
                response_name = str(turn.action_type or "action").strip() or "action"
            response_payload = {
                "status": turn.status,
                "call_id": turn.call_id,
                "google_function_call_sequence": turn.metadata.get(
                    "google_function_call_sequence"
                ),
                "google_correlation_mode": turn.metadata.get("google_correlation_mode"),
                "google_function_call_fallback_id": turn.metadata.get(
                    "google_function_call_fallback_id"
                ),
                "url": page_url,
                "x": turn.metadata.get("x"),
                "y": turn.metadata.get("y"),
                "clipboard_text": turn.metadata.get("clipboard_text"),
                "clipboard_truncated": turn.metadata.get("clipboard_truncated"),
                "clipboard_error": turn.metadata.get("clipboard_error"),
                "error": turn.error_message,
            }
            safety_acknowledgement = self._google_safety_acknowledgement(turn)
            if safety_acknowledgement:
                response_payload["safety_acknowledgement"] = safety_acknowledgement
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
                                    mime_type="image/png", data=screenshot_bytes
                                )
                            )
                        ],
                    )
                )
            )

        function_response_content = types.Content(role="user", parts=parts)
        contents = list(history) + [function_response_content]
        interaction_mode = str(metadata.get("interaction_mode") or "").strip().lower()
        if interaction_mode:
            reminder = (
                "Reminder: Observe-only mode is active. Do not interact with the UI. "
                "Do not call click_at, type_text_at, key_combination, or drag actions. "
                "Only inspect and report findings."
                if interaction_mode == "observe_only"
                else "Reminder: Execute mode is active. Complete the requested interaction directly without asking for confirmation."
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=reminder)],
                )
            )
        return (
            {
                "model": model or self._google_model,
                "contents": contents,
                "config": self._build_google_generate_config(environment),
            },
            function_response_content,
            screenshot_bytes,
        )

    def _build_google_generate_config(self, environment: str) -> Any:
        from google.genai import types  # type: ignore

        return types.GenerateContentConfig(
            tools=self._build_google_tools(environment),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def _build_google_tools(self, environment: str) -> list[Any]:
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

    def _build_anthropic_initial_request(
        self,
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
        return payload

    async def _build_anthropic_follow_up_request(
        self,
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
                # Anthropic requires text-only content when is_error is true.
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

            # Remove base64 payload after use to keep metadata lightweight
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

    async def _create_response(self, payload: dict[str, Any]) -> Any:
        """Call the OpenAI Responses API with the provided payload."""
        logger.debug(
            "Calling OpenAI Responses API", extra={"model": payload.get("model")}
        )
        return await self._client.responses.create(**payload)

    async def _create_anthropic_response(self, payload: dict[str, Any]) -> Any:
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

    async def _create_google_response(self, payload: dict[str, Any]) -> Any:
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
                response = await asyncio.to_thread(_call)
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
                await asyncio.sleep(delay_seconds)
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
            await asyncio.sleep(delay_seconds)

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

    def _resolve_google_retry_delays(self) -> tuple[float, ...]:
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

    def _resolve_google_prompt_safety_retry_delays(self) -> tuple[float, ...]:
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

    def _compute_google_prompt_safety_retry_delay(self, base_delay: float) -> float:
        """Add short random jitter to SAFETY retry delays."""
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

        jitter = random.uniform(low, high) if high > 0 else 0.0
        return max(float(base_delay) + jitter, 0.0)

    def _ensure_anthropic_client(self) -> Any | None:
        if self._provider != "anthropic":
            return None
        if self._anthropic_client:
            return self._anthropic_client
        if _AsyncAnthropic is None:
            return None

        anthropic_api_key = str(
            getattr(self._settings, "anthropic_api_key", "") or ""
        ).strip()
        if not anthropic_api_key:
            raise ComputerUseExecutionError(
                "Anthropic CU provider requires ANTHROPIC_API_KEY."
            )

        self._anthropic_client = _AsyncAnthropic(api_key=anthropic_api_key)
        logger.info("Initialized Anthropic CU client in API key mode")
        return self._anthropic_client

    def _ensure_google_client(self) -> Any | None:
        if self._provider != "google":
            return None
        if self._google_client:
            return self._google_client
        if genai is None:
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
        logger.debug("Initialized Google CU client in API key mode")
        return self._google_client

    async def _ensure_automation_driver_ready(self) -> None:
        """Ensure the automation_driver session is started before execution."""
        await self._automation_driver.start()

    async def invalidate_cache(self, cache_label: str, cache_action: str) -> None:
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

    async def _post_action_wait(self) -> None:
        """Wait for the configured stabilization interval."""
        wait_ms = self._settings.actions_computer_tool_stabilization_wait_ms
        if wait_ms > 0:
            await self._automation_driver.wait(wait_ms)

    async def _maybe_get_current_url(self) -> str | None:
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
        }
        return alias_map.get(normalized, normalized)

    def _translate_anthropic_action(self, raw_action: dict[str, Any]) -> dict[str, Any]:
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

    def _update_last_pointer_position(self, turn: ComputerToolTurn) -> None:
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

    def _resolve_key_sequence(
        self,
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
            return goal

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

    async def _build_confirmation_request(
        self,
        previous_response_id: str | None,
        metadata: dict[str, Any],
        environment: str = "browser",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Construct a follow-up request that confirms execution should proceed."""
        confirmation_text = "Yes, proceed. Execute the requested action now without asking for additional confirmation."
        target_text = metadata.get("target")
        if target_text:
            confirmation_text += f" Focus on: {target_text}."

        (
            viewport_width,
            viewport_height,
        ) = await self._automation_driver.get_viewport_size()

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
                    "role": "user",
                    "content": [{"type": "input_text", "text": confirmation_text}],
                }
            ],
            "truncation": "auto",
        }

        safety_identifier = metadata.get("safety_identifier")
        if safety_identifier:
            payload["safety_identifier"] = safety_identifier

        return payload

    def _is_action_allowed(self, action_type: str | None) -> tuple[bool, str | None]:
        """Determine if the requested action type is permitted in the current mode."""
        if not action_type:
            return False, "Computer Use response omitted action type information."
        if self._allowed_actions is None:
            return True, None
        if action_type in self._allowed_actions:
            return True, None

        return False, (f"Action '{action_type}' is not permitted in observe-only mode.")

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
        self,
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

    @staticmethod
    def _mark_terminal_failure(
        *,
        result: ComputerUseSessionResult,
        reason: str,
        code: str,
    ) -> None:
        """Mark a session result as terminally failed without appending a synthetic turn."""
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

    async def _enforce_domain_policy(self, action_type: str | None) -> None:
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
        cls, action_type: str | None, environment: str
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
                f"{normalized} is disabled; use direct UI interactions in the existing "
                "window instead."
            )
        return None

    def _resolve_coordinates(
        self,
        action: dict[str, Any],
        viewport_width: int,
        viewport_height: int,
        *,
        prefix: str = "",
        normalized: bool = False,
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
            return denormalize_coordinates(x, y, viewport_width, viewport_height)
        return normalize_coordinates(x, y, viewport_width, viewport_height)

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
        if (
            ("ctrl+" in normalized) or ("control+" in normalized)
        ) and normalized.endswith("+c"):
            return True
        return False

    async def _maybe_read_clipboard(self, turn: ComputerToolTurn) -> str | None:
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

    async def _navigate_via_address_bar(self, url: str) -> None:
        await self._automation_driver.press_key("ctrl+l")
        await self._automation_driver.type_text(url)
        await self._automation_driver.press_key("enter")

    def _save_turn_screenshot(
        self, screenshot_bytes: bytes, suffix: str, step_number: int | None
    ) -> str | None:
        """Persist a screenshot for observability if a debug logger is available."""
        if not self._debug_logger:
            return None
        name = f"computer_use_turn_{suffix}"
        screenshot_path = self._debug_logger.save_screenshot(
            screenshot_bytes, name=name, step_number=step_number
        )
        return cast(str | None, screenshot_path)

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

        if env_mode == "mobile_adb":
            mobile_context = (
                "IMPORTANT: You are controlling an Android mobile app through ADB-backed screenshots. "
                "Treat coordinates as mobile screen positions. Avoid desktop assumptions (dock, alt-tab, windows).\n\n"
                "SCREEN DETAILS:\n"
                "- Screenshot orientation and resolution may vary per device; always rely on the provided screenshot.\n"
                "- Prefer tap and type interactions; use scroll_document/scroll_at for vertical movement.\n"
                "- Do not use open_web_browser/search/go_back/go_forward desktop-browser actions unless explicitly required.\n\n"
                "AVAILABLE ACTIONS:\n"
                "- click_at: Tap at screen coordinates\n"
                "- type_text_at: Tap and type text into focused field\n"
                "- key_combination: Android key events — valid values include: 'home' (go to home screen), 'back' (navigate back), 'app_switch' (open recent apps switcher), 'enter', 'delete'\n"
                "- scroll_at / scroll_document: Scroll app content\n"
                "- drag_and_drop: Swipe/drag gestures when necessary\n\n"
                "APP SWITCHING ON ANDROID:\n"
                "- To switch to a recently used app: use key_combination: 'app_switch' to open the recents overlay, then tap the app card you want.\n"
                "- Do NOT use swipe/drag gestures from the bottom edge to open recents — use key_combination: 'app_switch' instead.\n"
                "- To go to the home screen: use key_combination: 'home'.\n\n"
                "TEXT INPUT ON ANDROID:\n"
                "- To replace existing text in a field: use type_text_at directly — it automatically selects all existing content before typing.\n"
                "- Do NOT use key_combination with ctrl+a or any other desktop shortcut to select or clear text; these do not work on Android.\n"
                "- For type_text_at, set press_enter=true ONLY when the task explicitly says to submit the form or press Enter. Do NOT set press_enter=true just because you are typing into a field — doing so submits the form prematurely.\n"
                "- Do NOT use key_combination or press_key with 'enter' or 'return' after typing into a field unless the task explicitly says to submit the form. Pressing Enter submits the whole form — if there are more fields to fill, use tap/type_text_at for the next field instead. Only tap the on-screen submit button (e.g. 'Reset Password', 'Sign In') when all fields are filled.\n"
                "- For password fields that mask input with dots or asterisks: a single type_text_at call is sufficient. Do NOT retry just because you cannot read the entered text. Seeing masked characters (dots) after typing confirms the field is populated — stop immediately.\n\n"
                + completion_instruction
                + "YOUR TASK: "
            )
            return mobile_context + goal

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
        """Return a Tool with custom modifier-click function declarations.

        Google's built-in click_at has no modifier support, so the model cannot
        perform Ctrl+click or Shift+click natively. These custom functions fill
        that gap and are handled by _execute_tool_action.
        """
        from google.genai import types  # type: ignore

        coord_props = {
            "x": types.Schema(type="NUMBER", description="X coordinate (0-999 scale)"),
            "y": types.Schema(type="NUMBER", description="Y coordinate (0-999 scale)"),
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
                        type="OBJECT",
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
                        type="OBJECT",
                        properties=coord_props,
                        required=["x", "y"],
                    ),
                ),
            ]
        )

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
    def _action_timeout_seconds(self) -> float:
        return max(
            float(self._settings.actions_computer_tool_action_timeout_ms) / 1000.0,
            0.5,
        )


def encode_png_base64(data: bytes) -> str:
    """Encode PNG bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def normalize_response(response: Any) -> dict[str, Any]:
    """Normalize OpenAI response objects into standard dictionaries."""
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return cast(dict[str, Any], response.model_dump())
    if hasattr(response, "to_dict"):
        try:
            return cast(dict[str, Any], response.to_dict())
        except Exception:
            pass
    if isinstance(response, dict):
        return response
    raise ComputerUseExecutionError(f"Unsupported response type: {type(response)}")


def normalize_key_sequence(key: str) -> str:
    """
    Convert Computer Use key strings to Automation-compatible sequences.

    The model commonly emits uppercase tokens (e.g., "ENTER") or modifier
    combinations like "CTRL+ENTER". Automation expects specific casing, so we
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
            return token_upper  # Automation expects F-keys uppercase.

        if len(token) == 1:
            return token

        return token.capitalize()

    if "+" in key:
        parts = [part.strip() for part in key.split("+") if part.strip()]
        normalized_parts = [normalize_single(part) for part in parts]
        return "+".join(normalized_parts)

    return normalize_single(key.strip())


def extract_computer_calls(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract computer_call items from a response."""
    return [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "computer_call"
    ]


def extract_google_function_calls(response_obj: Any) -> list[Any]:
    """Extract function_call parts from a Google response object."""
    return [
        envelope.function_call
        for envelope in extract_google_function_call_envelopes(response_obj)
    ]


def extract_google_function_call_envelopes(
    response_obj: Any,
    *,
    candidate_index: int = 0,
) -> list[GoogleFunctionCallEnvelope]:
    """Extract ordered function_call parts from a selected Google candidate."""
    envelopes: list[GoogleFunctionCallEnvelope] = []
    candidates = getattr(response_obj, "candidates", []) or []
    if not candidates:
        return envelopes

    selected_index = candidate_index
    if selected_index < 0 or selected_index >= len(candidates):
        selected_index = 0

    candidate = candidates[selected_index]
    content = getattr(candidate, "content", None)
    if not content:
        return envelopes

    sequence = 0
    parts = getattr(content, "parts", []) or []
    for part_index, part in enumerate(parts):
        func_call = getattr(part, "function_call", None)
        if not func_call:
            continue
        sequence += 1
        envelopes.append(
            GoogleFunctionCallEnvelope(
                function_call=func_call,
                sequence=sequence,
                candidate_index=selected_index,
                part_index=part_index,
            )
        )
    return envelopes


def extract_google_computer_calls(
    response_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract function/call items from a Google response."""
    calls: list[dict[str, Any]] = []
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


def extract_anthropic_computer_calls(
    response_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract Anthropic computer tool-use calls from a response."""
    calls: list[dict[str, Any]] = []
    for item in response_dict.get("content", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use":
            continue
        tool_name = str(item.get("name") or "").strip().lower()
        if tool_name and tool_name != "computer":
            continue
        action_payload = item.get("input")
        if not isinstance(action_payload, dict):
            action_payload = {}
        calls.append(
            {
                "id": item.get("id") or "",
                "name": item.get("name") or "computer",
                "action": action_payload,
            }
        )
    return calls


def extract_assistant_text(response_dict: dict[str, Any]) -> str | None:
    """Extract assistant text output from a response."""
    messages = [
        item
        for item in response_dict.get("output", [])
        if item.get("type") == "message"
    ]
    texts: list[str] = []
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
    if not texts:
        for content in response_dict.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") != "text":
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    combined = "\n".join(texts).strip()
    return combined or None


def _inject_context_metadata(turn: ComputerToolTurn, metadata: dict[str, Any]) -> None:
    """Copy high-level context into the turn metadata for observability."""
    if not isinstance(turn.metadata, dict):
        return
    for key in ("step_number", "test_plan_name", "test_case_name", "target", "value"):
        if metadata.get(key) is not None:
            turn.metadata[key] = metadata[key]
    if metadata.get("allow_safety_auto_approve") is not None:
        turn.metadata["allow_safety_auto_approve"] = bool(
            metadata["allow_safety_auto_approve"]
        )
    if metadata.get("safety_identifier") is not None:
        turn.metadata["safety_identifier"] = metadata["safety_identifier"]
    if metadata.get("interaction_mode") is not None:
        turn.metadata["interaction_mode"] = metadata["interaction_mode"]


def normalize_coordinates(
    x: float | None,
    y: float | None,
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
    x: float | None,
    y: float | None,
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
