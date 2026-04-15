"""Computer Use tool orchestration for the Action Agent."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from openai import AsyncOpenAI

from haindy.config.settings import Settings
from haindy.core.interfaces import AutomationDriver
from haindy.desktop.cache import CoordinateCache
from haindy.monitoring.debug_logger import DebugLogger
from haindy.runtime.environment import coordinate_cache_path_for_environment
from haindy.utils.model_logging import ModelCallLogger, get_model_logger

from .action_mixin import ComputerUseActionMixin
from .anthropic_mixin import AnthropicComputerUseMixin, _AsyncAnthropic
from .common import (
    _inject_context_metadata,
    denormalize_coordinates,
    encode_png_base64,
    extract_anthropic_computer_calls,
    extract_assistant_text,
    extract_computer_calls,
    extract_google_computer_calls,
    extract_google_function_call_envelopes,
    extract_google_function_calls,
    normalize_coordinates,
    normalize_key_sequence,
    normalize_response,
)
from .google_mixin import GoogleComputerUseMixin, genai
from .openai_mixin import OpenAIComputerUseMixin
from .support_mixin import ComputerUseSupportMixin
from .transports import (
    ComputerUseTransport,
    OpenAIResponsesHTTPTransport,
    OpenAIResponsesWebSocketTransport,
)
from .types import (
    ComputerUseExecutionError,
    ComputerUseSessionResult,
    GoogleFunctionCallEnvelope,
    InteractionConstraints,
)
from .visual_pipeline import VisualStatePlanner
from .visual_state import VisualFrame

logger = logging.getLogger(__name__)


class ComputerUseSession(
    OpenAIComputerUseMixin,
    GoogleComputerUseMixin,
    AnthropicComputerUseMixin,
    ComputerUseActionMixin,
    ComputerUseSupportMixin,
):
    """Wraps computer-use providers and orchestrates action execution."""

    _LEGACY_OPENAI_COMPUTER_MODEL = "computer-use-preview"

    _DISALLOWED_NAVIGATION_ACTIONS: frozenset[str] = frozenset(
        {
            "open_web_browser",
            "navigate",
            "search",
            "go_back",
            "go_forward",
        }
    )
    _GOOGLE_RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 5.0, 10.0)
    _GOOGLE_PROMPT_SAFETY_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.25, 0.75)
    _GOOGLE_PROMPT_SAFETY_RETRY_JITTER_SECONDS: tuple[float, float] = (0.05, 0.2)

    _client: AsyncOpenAI | None
    _automation_driver: AutomationDriver
    _settings: Settings
    _debug_logger: DebugLogger | None
    _provider: str
    _openai_model: str
    _google_model: str
    _anthropic_model: str
    _model: str
    _google_client: Any | None
    _anthropic_client: Any | None
    _anthropic_tool_type: str
    _anthropic_tool_name: str
    _anthropic_betas: list[str]
    _anthropic_max_tokens: int
    _coordinate_cache: CoordinateCache
    _model_logger: ModelCallLogger
    _allowed_actions: set[str] | None
    _allowed_domains: set[str]
    _blocked_domains: set[str]
    _stateful_actions: set[str]
    _scroll_turn_limit: int
    _pending_context_menu_selection: bool
    _interaction_constraints: InteractionConstraints
    _last_pointer_position: tuple[int, int] | None
    _openai_transport: ComputerUseTransport | None
    _visual_state_planner: VisualStatePlanner
    _current_keyframe: VisualFrame | None
    _last_visual_frame: VisualFrame | None
    _turns_since_keyframe: int
    _turns_since_cartography_refresh: int
    _openai_previous_response_id: str | None
    _google_previous_interaction_id: str | None
    _google_turn_index: int
    _step_response_ids: list[str]
    _step_last_response: dict[str, Any]

    def __init__(
        self,
        client: AsyncOpenAI | None,
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
        if (
            self._provider == "openai"
            and str(self._openai_model or "").strip().lower()
            == self._LEGACY_OPENAI_COMPUTER_MODEL
        ):
            raise ValueError(
                "OpenAI computer-use model 'computer-use-preview' is no longer "
                "supported. Set HAINDY_COMPUTER_USE_MODEL=gpt-5.4."
            )
        self._google_model = (
            model
            if model and self._provider == "google"
            else getattr(settings, "google_cu_model", "gemini-3-flash-preview")
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
            "append_text_at",
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
        self._openai_transport = None
        if self._provider == "openai":
            self._openai_transport = self._build_openai_transport()
        self._visual_state_planner = VisualStatePlanner(
            visual_mode=getattr(self._settings, "cu_visual_mode", "keyframe_patch"),
            keyframe_max_turns=getattr(self._settings, "cu_keyframe_max_turns", 3),
            patch_max_area_ratio=getattr(
                self._settings, "cu_patch_max_area_ratio", 0.35
            ),
            patch_margin_ratio=getattr(self._settings, "cu_patch_margin_ratio", 0.12),
        )
        self._current_keyframe = None
        self._last_visual_frame = None
        self._turns_since_keyframe = 0
        self._turns_since_cartography_refresh = 0
        self._openai_previous_response_id = None
        self._google_previous_interaction_id = None
        self._google_turn_index = 0
        self._step_response_ids = []
        self._step_last_response = {}

    def _build_openai_transport(self) -> ComputerUseTransport:
        """Build the transport used for OpenAI computer-use requests."""
        client = self._client
        if client is None:
            raise ValueError(
                "OpenAI client is required for OpenAI Computer Use sessions."
            )
        transport_mode = str(
            getattr(self._settings, "openai_cu_transport", "responses_websocket")
        ).strip()
        if transport_mode == "responses_http":
            return OpenAIResponsesHTTPTransport(client)
        return OpenAIResponsesWebSocketTransport(
            client=client,
            timeout_seconds=float(self._settings.openai_request_timeout_seconds),
        )

    @property
    def provider(self) -> str:
        """Return the configured Computer Use provider."""
        return self._provider

    @property
    def step_response_ids(self) -> list[str]:
        """Return the response ids accumulated across the current step scope."""
        return list(self._step_response_ids)

    def begin_step_scope(self) -> None:
        """Reset state so the session can be reused across one test step."""
        self._allowed_actions = None
        self._pending_context_menu_selection = False
        self._last_pointer_position = None
        self._current_keyframe = None
        self._last_visual_frame = None
        self._turns_since_keyframe = 0
        self._turns_since_cartography_refresh = 0
        self._openai_previous_response_id = None
        self._google_previous_interaction_id = None
        self._google_turn_index = 0
        self._step_response_ids = []
        self._step_last_response = {}

    async def close(self) -> None:
        """Close provider transport state for the current session."""
        if self._openai_transport is not None:
            await self._openai_transport.close()
        self._allowed_actions = None
        self._pending_context_menu_selection = False
        self._last_pointer_position = None
        self._current_keyframe = None
        self._last_visual_frame = None
        self._turns_since_keyframe = 0
        self._turns_since_cartography_refresh = 0
        self._openai_previous_response_id = None
        self._google_previous_interaction_id = None
        self._google_turn_index = 0
        self._step_response_ids = []
        self._step_last_response = {}

    async def execute_step_action(
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
        """Execute one action inside an existing step-scoped session."""
        if self._provider not in {"openai", "google"}:
            return await self.run(
                goal=goal,
                initial_screenshot=initial_screenshot,
                metadata=metadata,
                allowed_actions=allowed_actions,
                environment=environment,
                cache_label=cache_label,
                cache_action=cache_action,
                use_cache=use_cache,
            )

        metadata = metadata or {}
        step_goal = str(metadata.get("step_goal") or "").strip()
        constraint_source = " ".join([step_goal, goal]).strip()
        self._interaction_constraints = InteractionConstraints.from_text(
            constraint_source
        ).apply_overrides(metadata)
        if self._interaction_constraints.has_any():
            goal = (
                goal + "\n\nCONSTRAINTS:\n" + self._interaction_constraints.to_prompt()
            )
        self._allowed_actions = allowed_actions
        env_mode = self._normalize_environment_name(
            environment or metadata.get("environment") or self._default_environment
        )
        self._maybe_seed_initial_keyframe(initial_screenshot)
        try:
            if self._provider == "google":
                return await self._run_google(
                    goal=goal,
                    initial_screenshot=initial_screenshot,
                    metadata=metadata,
                    environment=env_mode,
                    cache_label=cache_label,
                    cache_action=cache_action,
                    use_cache=use_cache,
                    model=self._google_model,
                    previous_interaction_id=self._google_previous_interaction_id,
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
                previous_response_id=self._openai_previous_response_id,
            )
        finally:
            self._allowed_actions = None

    async def reflect_step(
        self,
        *,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a final structured verdict from the active step-scoped session."""
        metadata = metadata or {}
        model_name = (
            self._google_model
            if self._provider == "google"
            else self._openai_model if self._provider == "openai" else None
        )
        configured_timeout_seconds = self._action_timeout_seconds
        remaining_budget_seconds: float | None = None
        remaining_raw = metadata.get("remaining_test_budget_seconds")
        if remaining_raw is not None:
            try:
                remaining_budget_seconds = max(float(remaining_raw), 0.0)
            except (TypeError, ValueError):
                remaining_budget_seconds = None
        effective_timeout_seconds = configured_timeout_seconds
        if remaining_budget_seconds is not None:
            effective_timeout_seconds = min(
                effective_timeout_seconds,
                remaining_budget_seconds,
            )
        effective_timeout_seconds = max(effective_timeout_seconds, 0.0)

        async def _perform_reflection() -> dict[str, Any]:
            if self._provider == "google":
                return await self._reflect_google_step(
                    prompt=prompt,
                    metadata=metadata,
                    model=self._google_model,
                )
            if self._provider != "openai":
                raise ComputerUseExecutionError(
                    "Step-scoped reflection is only supported for the OpenAI and Google providers."
                )
            return await self._reflect_openai_step(
                prompt=prompt,
                metadata=metadata,
                model=self._openai_model,
            )

        logger.info(
            "Computer Use step reflection request started",
            extra={
                "provider": self._provider,
                "model": model_name,
                "payload_type": "step_reflection",
                "step_number": metadata.get("step_number"),
                "test_case": metadata.get("test_case_name"),
                "run_id": metadata.get("tool_mode_run_id"),
                "phase": metadata.get("validation_phase") or metadata.get("phase"),
                "configured_step_timeout_seconds": configured_timeout_seconds,
                "remaining_test_budget_seconds": remaining_budget_seconds,
                "effective_timeout_seconds": effective_timeout_seconds,
            },
        )
        try:
            result = await asyncio.wait_for(
                _perform_reflection(),
                timeout=effective_timeout_seconds,
            )
            logger.info(
                "Computer Use step reflection request completed",
                extra={
                    "provider": self._provider,
                    "model": model_name,
                    "payload_type": "step_reflection",
                    "step_number": metadata.get("step_number"),
                    "test_case": metadata.get("test_case_name"),
                    "run_id": metadata.get("tool_mode_run_id"),
                    "phase": metadata.get("validation_phase") or metadata.get("phase"),
                    "response_ids": result.get("response_ids", []),
                },
            )
            return result
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Computer Use step reflection request timed out",
                extra={
                    "provider": self._provider,
                    "model": model_name,
                    "payload_type": "step_reflection",
                    "step_number": metadata.get("step_number"),
                    "test_case": metadata.get("test_case_name"),
                    "run_id": metadata.get("tool_mode_run_id"),
                    "phase": metadata.get("validation_phase") or metadata.get("phase"),
                    "configured_step_timeout_seconds": configured_timeout_seconds,
                    "remaining_test_budget_seconds": remaining_budget_seconds,
                    "effective_timeout_seconds": effective_timeout_seconds,
                },
            )
            raise ComputerUseExecutionError(
                "Computer Use step reflection timed out after "
                f"{effective_timeout_seconds:.1f} seconds."
            ) from exc

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
        stop_after_actions: bool = False,
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
        self.begin_step_scope()
        self._allowed_actions = allowed_actions

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
        self._maybe_seed_initial_keyframe(initial_screenshot)
        try:
            if self._provider == "google":
                return await self._run_google(
                    goal=goal,
                    initial_screenshot=initial_screenshot,
                    metadata=metadata,
                    environment=env_mode,
                    cache_label=cache_label,
                    cache_action=cache_action,
                    use_cache=use_cache,
                    model=self._google_model,
                    previous_interaction_id=None,
                    stop_after_actions=stop_after_actions,
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
                    previous_response_id=None,
                    stop_after_actions=stop_after_actions,
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
            await self.close()


__all__ = [
    "ComputerUseExecutionError",
    "ComputerUseSession",
    "ComputerUseSessionResult",
    "GoogleFunctionCallEnvelope",
    "_AsyncAnthropic",
    "_inject_context_metadata",
    "asyncio",
    "denormalize_coordinates",
    "encode_png_base64",
    "extract_anthropic_computer_calls",
    "extract_assistant_text",
    "extract_computer_calls",
    "extract_google_computer_calls",
    "extract_google_function_call_envelopes",
    "extract_google_function_calls",
    "genai",
    "normalize_coordinates",
    "normalize_key_sequence",
    "normalize_response",
    "random",
]
