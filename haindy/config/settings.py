"""Configuration management for the HAINDY framework."""

import json
import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from haindy.core.interfaces import ConfigProvider
from haindy.runtime.environment import (
    normalize_automation_backend as normalize_automation_backend_value,
)

AGENT_ENV_PREFIX: dict[str, str] = {
    "scope_triage": "HAINDY_SCOPE_TRIAGE",
    "test_planner": "HAINDY_TEST_PLANNER",
    "test_runner": "HAINDY_TEST_RUNNER",
    "situational_agent": "HAINDY_SITUATIONAL_AGENT",
}
SUPPORTED_AGENT_PROVIDERS: tuple[str, ...] = (
    "openai",
    "openai-codex",
    "google",
    "anthropic",
)
SUPPORTED_CU_PROVIDERS: tuple[str, ...] = ("openai", "google", "anthropic")
SUPPORTED_OPENAI_MODEL = "gpt-5.4"
SUPPORTED_OPENAI_COMPUTER_USE_MODEL = "gpt-5.4"
LEGACY_OPENAI_COMPUTER_USE_MODEL = "computer-use-preview"
DEFAULT_NON_CU_PROVIDER_MODELS: dict[str, str] = {
    "openai": SUPPORTED_OPENAI_MODEL,
    "openai-codex": SUPPORTED_OPENAI_MODEL,
    "google": "gemini-3-flash-preview",
    "anthropic": "claude-sonnet-4-6",
}
DEFAULT_CU_PROVIDER_MODELS: dict[str, str] = {
    "openai": SUPPORTED_OPENAI_COMPUTER_USE_MODEL,
    "google": "gemini-3-flash-preview",
    "anthropic": "claude-sonnet-4-6",
}

ALLOWED_REASONING_LEVELS: set[str] = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}
ALLOWED_OPENAI_CU_TRANSPORTS: set[str] = {"responses_websocket", "responses_http"}
ALLOWED_CU_VISUAL_MODES: set[str] = {"keyframe_patch", "legacy_full_frame"}

SETTINGS_ENV_VARS: dict[str, str] = {
    "openai_api_key": "HAINDY_OPENAI_API_KEY",
    "openai_model": "HAINDY_OPENAI_MODEL",
    "openai_codex_model": "HAINDY_OPENAI_CODEX_MODEL",
    "agent_provider": "HAINDY_AGENT_PROVIDER",
    "anthropic_model": "HAINDY_ANTHROPIC_MODEL",
    "google_model": "HAINDY_GOOGLE_MODEL",
    "openai_max_retries": "HAINDY_OPENAI_MAX_RETRIES",
    "openai_request_timeout_seconds": "HAINDY_OPENAI_REQUEST_TIMEOUT_SECONDS",
    "automation_backend": "HAINDY_AUTOMATION_BACKEND",
    "desktop_prefer_resolution": "HAINDY_DESKTOP_RESOLUTION",
    "desktop_keyboard_layout": "HAINDY_DESKTOP_KEYBOARD_LAYOUT",
    "desktop_enable_keyboard_scancodes": "HAINDY_DESKTOP_KEYBOARD_SCANCODES",
    "desktop_keyboard_key_delay_ms": "HAINDY_DESKTOP_KEY_DELAY_MS",
    "desktop_enable_resolution_switch": "HAINDY_DESKTOP_ENABLE_RESOLUTION_SWITCH",
    "desktop_screenshot_dir": "HAINDY_DESKTOP_SCREENSHOT_DIR",
    "desktop_coordinate_cache_path": "HAINDY_DESKTOP_COORDINATE_CACHE_PATH",
    "task_plan_cache_path": "HAINDY_TASK_PLAN_CACHE_PATH",
    "enable_planning_cache": "HAINDY_ENABLE_PLANNING_CACHE",
    "planning_cache_path": "HAINDY_PLANNING_CACHE_PATH",
    "enable_situational_cache": "HAINDY_ENABLE_SITUATIONAL_CACHE",
    "situational_cache_path": "HAINDY_SITUATIONAL_CACHE_PATH",
    "enable_execution_replay_cache": "HAINDY_ENABLE_EXECUTION_REPLAY_CACHE",
    "execution_replay_cache_path": "HAINDY_EXECUTION_REPLAY_CACHE_PATH",
    "desktop_display": "HAINDY_DESKTOP_DISPLAY",
    "desktop_clipboard_timeout_seconds": "HAINDY_DESKTOP_CLIPBOARD_TIMEOUT_SECONDS",
    "desktop_clipboard_hold_seconds": "HAINDY_DESKTOP_CLIPBOARD_HOLD_SECONDS",
    "mobile_screenshot_dir": "HAINDY_MOBILE_SCREENSHOT_DIR",
    "mobile_coordinate_cache_path": "HAINDY_MOBILE_COORDINATE_CACHE_PATH",
    "mobile_default_adb_serial": "HAINDY_MOBILE_DEFAULT_ADB_SERIAL",
    "mobile_adb_timeout_seconds": "HAINDY_MOBILE_ADB_TIMEOUT_SECONDS",
    "ios_screenshot_dir": "HAINDY_IOS_SCREENSHOT_DIR",
    "ios_coordinate_cache_path": "HAINDY_IOS_COORDINATE_CACHE_PATH",
    "ios_default_device_udid": "HAINDY_IOS_DEFAULT_DEVICE_UDID",
    "ios_idb_timeout_seconds": "HAINDY_IOS_IDB_TIMEOUT_SECONDS",
    "macos_screenshot_dir": "HAINDY_MACOS_SCREENSHOT_DIR",
    "macos_coordinate_cache_path": "HAINDY_MACOS_COORDINATE_CACHE_PATH",
    "macos_keyboard_layout": "HAINDY_MACOS_KEYBOARD_LAYOUT",
    "macos_keyboard_key_delay_ms": "HAINDY_MACOS_KEY_DELAY_MS",
    "macos_clipboard_timeout_seconds": "HAINDY_MACOS_CLIPBOARD_TIMEOUT_SECONDS",
    "macos_clipboard_hold_seconds": "HAINDY_MACOS_CLIPBOARD_HOLD_SECONDS",
    "enable_screen_recording": "HAINDY_ENABLE_SCREEN_RECORDING",
    "screen_recording_output_dir": "HAINDY_SCREEN_RECORDING_OUTPUT_DIR",
    "screen_recording_framerate": "HAINDY_SCREEN_RECORDING_FRAMERATE",
    "screen_recording_draw_cursor": "HAINDY_SCREEN_RECORDING_DRAW_CURSOR",
    "screen_recording_prefix": "HAINDY_SCREEN_RECORDING_PREFIX",
    "max_test_steps": "HAINDY_MAX_TEST_STEPS",
    "step_timeout": "HAINDY_STEP_TIMEOUT",
    "max_retries_per_step": "HAINDY_MAX_RETRIES_PER_STEP",
    "screenshot_quality": "HAINDY_SCREENSHOT_QUALITY",
    "computer_use_model": "HAINDY_COMPUTER_USE_MODEL",
    "cu_provider": "HAINDY_CU_PROVIDER",
    "google_cu_model": "HAINDY_GOOGLE_CU_MODEL",
    "anthropic_api_key": "HAINDY_ANTHROPIC_API_KEY",
    "anthropic_cu_model": "HAINDY_ANTHROPIC_CU_MODEL",
    "anthropic_cu_beta": "HAINDY_ANTHROPIC_CU_BETA",
    "anthropic_cu_max_tokens": "HAINDY_ANTHROPIC_CU_MAX_TOKENS",
    "vertex_api_key": "HAINDY_VERTEX_API_KEY",
    "vertex_project": "HAINDY_VERTEX_PROJECT",
    "vertex_location": "HAINDY_VERTEX_LOCATION",
    "cu_safety_policy": "HAINDY_CU_SAFETY_POLICY",
    "openai_cu_transport": "HAINDY_OPENAI_CU_TRANSPORT",
    "cu_visual_mode": "HAINDY_CU_VISUAL_MODE",
    "cu_cartography_model": "HAINDY_CU_CARTOGRAPHY_MODEL",
    "cu_keyframe_max_turns": "HAINDY_CU_KEYFRAME_MAX_TURNS",
    "cu_patch_max_area_ratio": "HAINDY_CU_PATCH_MAX_AREA_RATIO",
    "cu_patch_margin_ratio": "HAINDY_CU_PATCH_MARGIN_RATIO",
    "actions_computer_tool_max_turns": "HAINDY_ACTIONS_COMPUTER_TOOL_MAX_TURNS",
    "actions_computer_tool_loop_detection_window": "HAINDY_ACTIONS_COMPUTER_TOOL_LOOP_WINDOW",
    "actions_computer_tool_action_timeout_seconds": "HAINDY_ACTIONS_COMPUTER_TOOL_ACTION_TIMEOUT_SECONDS",
    "actions_computer_tool_stabilization_wait_ms": "HAINDY_ACTIONS_COMPUTER_TOOL_STABILIZATION_WAIT_MS",
    "actions_computer_tool_fail_fast_on_safety": "HAINDY_ACTIONS_COMPUTER_TOOL_FAIL_FAST",
    "actions_computer_tool_allowed_domains": "HAINDY_ACTIONS_COMPUTER_TOOL_ALLOWED_DOMAINS",
    "actions_computer_tool_blocked_domains": "HAINDY_ACTIONS_COMPUTER_TOOL_BLOCKED_DOMAINS",
    "scroll_turn_multiplier": "HAINDY_SCROLL_TURN_MULTIPLIER",
    "scroll_default_magnitude": "HAINDY_SCROLL_DEFAULT_MAGNITUDE",
    "scroll_max_magnitude": "HAINDY_SCROLL_MAX_MAGNITUDE",
    "log_level": "HAINDY_LOG_LEVEL",
    "log_format": "HAINDY_LOG_FORMAT",
    "log_file": "HAINDY_LOG_FILE",
    "model_log_path": "HAINDY_MODEL_LOG_PATH",
    "max_screenshots": "HAINDY_MAX_SCREENSHOTS",
    "data_dir": "HAINDY_DATA_DIR",
    "reports_dir": "HAINDY_REPORTS_DIR",
    "screenshots_dir": "HAINDY_SCREENSHOTS_DIR",
    "cache_dir": "HAINDY_CACHE_DIR",
    "rate_limit_enabled": "HAINDY_RATE_LIMIT_ENABLED",
    "rate_limit_requests_per_minute": "HAINDY_RATE_LIMIT_REQUESTS_PER_MINUTE",
    "sanitize_screenshots": "HAINDY_SANITIZE_SCREENSHOTS",
    "debug_mode": "HAINDY_DEBUG_MODE",
    "save_agent_conversations": "HAINDY_SAVE_AGENT_CONVERSATIONS",
    "haindy_home": "HAINDY_HOME",
}

OPTIONAL_STRING_FIELDS = {"desktop_display", "log_file"}

_SECRET_FIELD_TO_PROVIDER: dict[str, str] = {
    "openai_api_key": "openai",
    "anthropic_api_key": "anthropic",
    "vertex_api_key": "vertex",
}
LIST_FIELDS = {
    "actions_computer_tool_allowed_domains",
    "actions_computer_tool_blocked_domains",
}

_NON_CU_PROVIDER_MODEL_FIELDS: dict[str, str] = {
    "openai": "openai_model",
    "openai-codex": "openai_codex_model",
    "google": "google_model",
    "anthropic": "anthropic_model",
}
_CU_PROVIDER_MODEL_FIELDS: dict[str, str] = {
    "openai": "computer_use_model",
    "google": "google_cu_model",
    "anthropic": "anthropic_cu_model",
}


def get_default_provider_model(provider: str, *, computer_use: bool = False) -> str:
    """Return the built-in default model for a provider."""
    normalized = str(provider or "").strip().lower()
    defaults = (
        DEFAULT_CU_PROVIDER_MODELS if computer_use else DEFAULT_NON_CU_PROVIDER_MODELS
    )
    if normalized not in defaults:
        scope = "computer-use" if computer_use else "non-CU"
        raise ValueError(
            f"Unsupported {scope} provider '{provider}'. "
            f"Supported providers are {sorted(defaults)}."
        )
    return defaults[normalized]


def get_provider_model_field_name(provider: str, *, computer_use: bool = False) -> str:
    """Return the Settings field name that stores a provider's configured model."""
    normalized = str(provider or "").strip().lower()
    fields = (
        _CU_PROVIDER_MODEL_FIELDS if computer_use else _NON_CU_PROVIDER_MODEL_FIELDS
    )
    if normalized not in fields:
        scope = "computer-use" if computer_use else "non-CU"
        raise ValueError(
            f"Unsupported {scope} provider '{provider}'. "
            f"Supported providers are {sorted(fields)}."
        )
    return fields[normalized]


class AgentModelConfig(BaseModel):
    """Per-agent model configuration."""

    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    reasoning_level: str = Field(default="medium")
    modalities: set[str] = Field(default_factory=lambda: {"text"})

    @field_validator("reasoning_level")
    @classmethod
    def validate_reasoning_level(cls, value: str) -> str:
        """Ensure reasoning level is valid."""
        if value not in ALLOWED_REASONING_LEVELS:
            raise ValueError(
                f"Invalid reasoning level: {value}. Allowed values: {sorted(ALLOWED_REASONING_LEVELS)}"
            )
        return value

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: set[str]) -> set[str]:
        if not value:
            raise ValueError("Agent modalities cannot be empty")
        return value


DEFAULT_AGENT_MODELS: dict[str, AgentModelConfig] = {
    "scope_triage": AgentModelConfig(
        temperature=0.15,
        reasoning_level="high",
    ),
    "test_planner": AgentModelConfig(
        temperature=0.35,
        reasoning_level="high",
    ),
    "test_runner": AgentModelConfig(
        temperature=0.55,
        reasoning_level="medium",
    ),
    "situational_agent": AgentModelConfig(
        temperature=0.1,
        reasoning_level="high",
    ),
}


def _parse_resolution_env(raw: str) -> tuple[int, int]:
    text = raw.strip()
    if not text:
        raise ValueError("Resolution value cannot be empty")
    if text[0] in "[(" and text[-1] in "])":
        try:
            decoded = json.loads(text.replace("(", "[").replace(")", "]"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid resolution value: {raw}") from exc
        if isinstance(decoded, list) and len(decoded) == 2:
            return int(decoded[0]), int(decoded[1])
        raise ValueError(f"Invalid resolution value: {raw}")

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid resolution value: {raw}")


def _parse_string_list_env(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if text[0] == "[" and text[-1] == "]":
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid list value: {raw}") from exc
        if not isinstance(decoded, list):
            raise ValueError(f"Invalid list value: {raw}")
        return [str(item).strip() for item in decoded if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _optional_string_value(raw: str) -> str | None:
    value = raw.strip()
    return value or None


def _parse_env_field(field_name: str, raw: str) -> Any:
    parser: Callable[[str], Any] | None = None
    if field_name == "desktop_prefer_resolution":
        parser = _parse_resolution_env
    elif field_name in LIST_FIELDS:
        parser = _parse_string_list_env
    elif field_name in OPTIONAL_STRING_FIELDS:
        parser = _optional_string_value

    if parser is not None:
        return parser(raw)

    annotation = Settings.model_fields[field_name].annotation
    return TypeAdapter(annotation).validate_python(raw)


def _merge_runtime_env() -> dict[str, str]:
    env_values = {
        key: value for key, value in dotenv_values(".env").items() if value is not None
    }
    env_values.update(os.environ)
    return env_values


def _build_agent_models(env: Mapping[str, str]) -> dict[str, AgentModelConfig]:
    configured_models: dict[str, AgentModelConfig] = {}

    for agent_name, prefix in AGENT_ENV_PREFIX.items():
        config_payload = DEFAULT_AGENT_MODELS[agent_name].model_dump()

        model_override = env.get(f"{prefix}_MODEL")
        if model_override:
            config_payload["model"] = model_override

        reasoning_override = env.get(f"{prefix}_REASONING_LEVEL")
        if reasoning_override:
            config_payload["reasoning_level"] = reasoning_override.lower()

        modalities_override = env.get(f"{prefix}_MODALITIES")
        if modalities_override is not None:
            config_payload["modalities"] = set(
                _parse_string_list_env(modalities_override)
            )

        configured_models[agent_name] = AgentModelConfig(**config_payload)

    return configured_models


def load_settings(env: Mapping[str, str] | None = None) -> "Settings":
    """Load settings from all sources, low-to-high priority.

    Priority (lowest to highest):
      1. Pydantic field defaults
      2. ~/.haindy/settings.json
      3. API keys from system keychain / encrypted file fallback
      4. HAINDY_* environment variables + .env file
    """
    from haindy.auth.credentials import get_api_key
    from haindy.config.settings_file import flatten_settings_dict, load_settings_file

    payload: dict[str, Any] = {}

    # Layer 1: settings file
    user_path = Path("~/.haindy/settings.json").expanduser()
    payload.update(flatten_settings_dict(load_settings_file(user_path)))

    # Layer 3: env vars (build raw_env first so we can check which secret keys
    # are already covered by an env var before hitting the keychain)
    raw_env = _merge_runtime_env() if env is None else dict(env)

    # Layer 3: keychain / encrypted file for secret fields not already set by env
    for field_name, provider in _SECRET_FIELD_TO_PROVIDER.items():
        env_name = SETTINGS_ENV_VARS[field_name]
        if env_name not in raw_env:
            key = get_api_key(provider)
            if key:
                payload[field_name] = key

    # Layer 4: env vars (highest priority)
    for field_name, env_name in SETTINGS_ENV_VARS.items():
        raw_value = raw_env.get(env_name)
        if raw_value is None:
            continue
        payload[field_name] = _parse_env_field(field_name, raw_value)

    payload["agent_models"] = _build_agent_models(raw_env)
    return Settings(**payload)


class Settings(BaseModel):
    """Application settings with environment variable support."""

    model_config = ConfigDict(extra="forbid")

    # Provider Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default=DEFAULT_NON_CU_PROVIDER_MODELS["openai"],
        description="Default OpenAI model for non-CU agent calls",
    )
    openai_codex_model: str = Field(
        default=DEFAULT_NON_CU_PROVIDER_MODELS["openai-codex"],
        description="Default OpenAI model when using openai-codex auth for non-CU agent calls",
    )
    openai_max_retries: int = Field(
        default=3, ge=1, description="Maximum API retry attempts"
    )
    openai_request_timeout_seconds: int = Field(
        default=900,
        ge=60,
        description="Request timeout for OpenAI API calls in seconds",
    )
    agent_provider: str = Field(
        default="openai",
        description="AI provider for non-CU agent calls (openai, openai-codex, anthropic, or google)",
    )
    anthropic_model: str = Field(
        default=DEFAULT_NON_CU_PROVIDER_MODELS["anthropic"],
        description="Anthropic model for non-CU agent calls",
    )
    google_model: str = Field(
        default=DEFAULT_NON_CU_PROVIDER_MODELS["google"],
        description="Google model for non-CU agent calls",
    )
    agent_models: dict[str, AgentModelConfig] = Field(
        default_factory=dict,
        description="Per-agent model configuration",
    )

    # Desktop Configuration
    automation_backend: str = Field(
        default="desktop",
        description="Automation backend to use (desktop or mobile_adb)",
    )
    desktop_prefer_resolution: tuple[int, int] = Field(
        default=(1920, 1080),
        description="Preferred resolution for desktop sessions",
    )
    desktop_keyboard_layout: str = Field(
        default="us",
        description="Keyboard layout for desktop automation (us, es)",
    )
    desktop_enable_keyboard_scancodes: bool = Field(
        default=True,
        description="Emit MSC_SCAN scancodes for key events",
    )
    desktop_keyboard_key_delay_ms: int = Field(
        default=12,
        ge=0,
        description="Delay between key events when sending combos",
    )
    desktop_enable_resolution_switch: bool = Field(
        default=True,
        description="Allow resolution downshift for desktop runs",
    )
    desktop_screenshot_dir: Path = Field(
        default=Path("data/screenshots/desktop"),
        description="Directory for desktop screenshots",
    )
    desktop_coordinate_cache_path: Path = Field(
        default=Path("data/desktop_cache/coordinates.json"),
        description="Coordinate cache path for desktop actions",
    )
    task_plan_cache_path: Path = Field(
        default=Path("data/task_plan_cache.json"),
        description="Task planning cache path",
    )
    enable_planning_cache: bool = Field(
        default=True,
        description="Enable caching for scope triage and test plan generation",
    )
    planning_cache_path: Path = Field(
        default=Path("data/planning_cache.json"),
        description="Scope triage and test planning cache path",
    )
    enable_situational_cache: bool = Field(
        default=True,
        description="Enable caching for situational execution-context validation",
    )
    situational_cache_path: Path = Field(
        default=Path("data/situational_cache.json"),
        description="Situational execution-context cache path",
    )
    enable_execution_replay_cache: bool = Field(
        default=True,
        description="Enable execution replay cache (record/replay driver actions per step)",
    )
    execution_replay_cache_path: Path = Field(
        default=Path("data/execution_replay_cache.json"),
        description="Execution replay cache path",
    )
    desktop_display: str | None = Field(
        default=None,
        description="X11 display override for desktop capture",
    )
    desktop_clipboard_timeout_seconds: float = Field(
        default=3.0,
        ge=0.5,
        description="Timeout for desktop clipboard reads",
    )
    desktop_clipboard_hold_seconds: float = Field(
        default=15.0,
        ge=0.5,
        description="Max time to hold clipboard owner process",
    )
    mobile_screenshot_dir: Path = Field(
        default=Path("data/screenshots/mobile"),
        description="Directory for mobile screenshots",
    )
    mobile_coordinate_cache_path: Path = Field(
        default=Path("data/mobile_cache/coordinates.json"),
        description="Coordinate cache path for mobile actions",
    )
    mobile_default_adb_serial: str = Field(
        default="",
        description="Default Android device serial for mobile ADB runs",
    )
    mobile_adb_timeout_seconds: float = Field(
        default=15.0,
        ge=0.5,
        description="Timeout in seconds for individual ADB commands",
    )
    ios_screenshot_dir: Path = Field(
        default=Path("data/screenshots/ios"),
        description="Directory for iOS screenshots",
    )
    ios_coordinate_cache_path: Path = Field(
        default=Path("data/ios_cache/coordinates.json"),
        description="Coordinate cache path for iOS actions",
    )
    ios_default_device_udid: str = Field(
        default="",
        description="Default iOS device UDID for idb runs",
    )
    ios_idb_timeout_seconds: float = Field(
        default=15.0,
        ge=0.5,
        description="Timeout in seconds for individual idb commands",
    )
    macos_screenshot_dir: Path = Field(
        default=Path("data/screenshots/macos"),
        description="Directory for macOS desktop screenshots",
    )
    macos_coordinate_cache_path: Path = Field(
        default=Path("data/macos_cache/coordinates.json"),
        description="Coordinate cache path for macOS desktop actions",
    )
    macos_keyboard_layout: str = Field(
        default="us",
        description="Keyboard layout for macOS automation",
    )
    macos_keyboard_key_delay_ms: int = Field(
        default=12,
        ge=0,
        description="Delay between key events for macOS automation",
    )
    macos_clipboard_timeout_seconds: float = Field(
        default=3.0,
        ge=0.5,
        description="Timeout for macOS clipboard operations",
    )
    macos_clipboard_hold_seconds: float = Field(
        default=15.0,
        ge=0.5,
        description="Max time to hold macOS clipboard owner process",
    )
    enable_screen_recording: bool = Field(
        default=False,
        description="Enable GNOME desktop screen recording during test execution",
    )
    screen_recording_output_dir: Path = Field(
        default=Path("reports/recordings"),
        description="Directory for optional desktop screen recordings",
    )
    screen_recording_framerate: int = Field(
        default=30,
        ge=1,
        description="Framerate for GNOME desktop screen recordings",
    )
    screen_recording_draw_cursor: bool = Field(
        default=True,
        description="Draw cursor in GNOME desktop screen recordings",
    )
    screen_recording_prefix: str = Field(
        default="haindy-agent",
        description="Filename prefix for desktop screen recordings",
    )

    # Execution Configuration
    max_test_steps: int = Field(default=100, ge=1, description="Maximum steps per test")
    step_timeout: int = Field(
        default=30000, ge=1000, description="Timeout per step (ms)"
    )
    max_retries_per_step: int = Field(
        default=3, ge=1, description="Maximum retries per step"
    )
    screenshot_quality: int = Field(
        default=80, ge=1, le=100, description="Screenshot JPEG quality"
    )

    @field_validator("openai_model")
    @classmethod
    def validate_openai_model(cls, value: str) -> str:
        if value != SUPPORTED_OPENAI_MODEL:
            raise ValueError(
                f"Unsupported OpenAI model '{value}' for openai_model. "
                f"Supported model is '{SUPPORTED_OPENAI_MODEL}'."
            )
        return value

    @field_validator("openai_codex_model")
    @classmethod
    def validate_openai_codex_model(cls, value: str) -> str:
        if value != SUPPORTED_OPENAI_MODEL:
            raise ValueError(
                f"Unsupported OpenAI model '{value}' for openai_codex_model. "
                f"Supported model is '{SUPPORTED_OPENAI_MODEL}'."
            )
        return value

    # Computer Use Configuration
    computer_use_model: str = Field(
        default=DEFAULT_CU_PROVIDER_MODELS["openai"],
        description="OpenAI model for computer-use execution",
    )
    cu_provider: str = Field(
        default="google",
        description="Computer-use provider to run actions (openai, google, or anthropic)",
    )
    google_cu_model: str = Field(
        default=DEFAULT_CU_PROVIDER_MODELS["google"],
        description="Google Gemini computer-use model name",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude computer-use",
    )
    anthropic_cu_model: str = Field(
        default=DEFAULT_CU_PROVIDER_MODELS["anthropic"],
        description="Anthropic Claude computer-use model name",
    )
    anthropic_cu_beta: str = Field(
        default="computer-use-2025-11-24",
        description="Anthropic beta flag for computer-use tool availability",
    )
    anthropic_cu_max_tokens: int = Field(
        default=16384,
        ge=256,
        description="Max output tokens for Anthropic computer-use requests",
    )
    vertex_api_key: str = Field(
        default="",
        description="API key for Google Vertex computer-use",
    )
    vertex_project: str = Field(
        default="",
        description="Vertex project for Google computer-use runs",
    )
    vertex_location: str = Field(
        default="us-central1",
        description="Vertex location for Google computer-use runs",
    )
    cu_safety_policy: str = Field(
        default="auto_approve",
        description="Handling for provider safety confirmations (auto_approve, auto_deny, error)",
    )
    openai_cu_transport: str = Field(
        default="responses_websocket",
        description="Transport for OpenAI computer-use requests",
    )
    cu_visual_mode: str = Field(
        default="keyframe_patch",
        description="Visual-state strategy for computer-use follow-up screenshots",
    )
    cu_cartography_model: str = Field(
        default="",
        description="Optional override model for provider-owned cartography generation",
    )
    cu_keyframe_max_turns: int = Field(
        default=3,
        ge=1,
        description="Maximum turns to reuse a keyframe before forcing a refresh",
    )
    cu_patch_max_area_ratio: float = Field(
        default=0.35,
        ge=0.01,
        le=1.0,
        description="Maximum full-frame area ratio allowed for patch-mode follow-ups",
    )
    cu_patch_margin_ratio: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="Margin ratio to expand patch crops around the target/delta union",
    )
    actions_computer_tool_max_turns: int = Field(
        default=12,
        ge=1,
        description="Maximum tool turns per action when using Computer Use",
    )
    actions_computer_tool_loop_detection_window: int = Field(
        default=4,
        ge=2,
        description="Repeated identical turns (with identical screenshots) before flagging a loop",
    )
    actions_computer_tool_action_timeout_seconds: float = Field(
        default=600.0,
        ge=0.5,
        description="Timeout in seconds for executing a single computer action",
    )
    actions_computer_tool_stabilization_wait_ms: int = Field(
        default=2000,
        ge=0,
        le=10000,
        description="Delay after executing an action before capturing the next screenshot",
    )
    actions_computer_tool_fail_fast_on_safety: bool = Field(
        default=True,
        description=(
            "Abort immediately on pending safety checks before cu_safety_policy is applied"
        ),
    )
    actions_computer_tool_allowed_domains: list[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool is permitted to interact with",
    )
    actions_computer_tool_blocked_domains: list[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool must never interact with",
    )
    scroll_turn_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        description="Multiplier for extra scroll turns before max-turn enforcement",
    )
    scroll_default_magnitude: int = Field(
        default=450,
        ge=1,
        description="Default scroll magnitude (pixels) when not specified",
    )
    scroll_max_magnitude: int = Field(
        default=600,
        ge=1,
        description="Maximum per-scroll magnitude/pixel delta",
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: str = Field(
        default="text",
        description="Log format (json or text)",
    )
    log_file: str | None = Field(default=None, description="Log file path")
    model_log_path: Path = Field(
        default=Path("data/model_logs/model_calls.jsonl"),
        description="Log file for raw model prompts/responses",
    )
    max_screenshots: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on retained screenshots per run; unset preserves all evidence",
    )

    # Storage Configuration
    data_dir: Path = Field(default=Path("data"), description="Data storage directory")
    reports_dir: Path = Field(
        default=Path("reports"), description="Reports output directory"
    )
    screenshots_dir: Path = Field(
        default=Path("data/screenshots"), description="Screenshots directory"
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")

    # Security Configuration
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(
        default=60, ge=1, description="API requests per minute"
    )
    sanitize_screenshots: bool = Field(
        default=True, description="Sanitize PII from screenshots"
    )

    # Development Configuration
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    save_agent_conversations: bool = Field(
        default=True, description="Save agent conversation logs"
    )
    haindy_home: Path = Field(
        default=Path("~/.haindy"),
        description="Home directory for tool-call mode session state",
    )

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @field_validator("log_format")
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        if v not in ["json", "text"]:
            raise ValueError(f"Invalid log format: {v}")
        return v

    @field_validator("desktop_prefer_resolution", mode="before")
    @classmethod
    def parse_desktop_resolution(cls, value: Any) -> tuple[int, int]:
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        if isinstance(value, list) and len(value) == 2:
            return int(value[0]), int(value[1])
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                return int(parts[0]), int(parts[1])
        return (1920, 1080)

    @field_validator("desktop_keyboard_layout")
    @classmethod
    def normalize_keyboard_layout(cls, value: str) -> str:
        normalized = (value or "us").strip().lower()
        if normalized in {"us", "es"}:
            return normalized
        return "us"

    @field_validator("automation_backend")
    @classmethod
    def normalize_automation_backend(cls, value: str) -> str:
        return normalize_automation_backend_value(value)

    @field_validator("haindy_home", mode="before")
    @classmethod
    def normalize_haindy_home(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value or "~/.haindy")).expanduser()

    @field_validator("agent_provider")
    @classmethod
    def normalize_agent_provider(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in SUPPORTED_AGENT_PROVIDERS:
            raise ValueError(
                f"Unsupported agent_provider '{value}'. "
                "Supported providers are 'openai', 'openai-codex', "
                "'anthropic', and 'google'."
            )
        return normalized

    @field_validator("cu_provider")
    @classmethod
    def normalize_cu_provider(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in SUPPORTED_CU_PROVIDERS:
            raise ValueError(
                "Unsupported cu_provider "
                f"'{value}'. Supported providers are 'openai', 'google', and "
                "'anthropic'."
            )
        return normalized

    @field_validator("cu_safety_policy")
    @classmethod
    def normalize_cu_safety_policy(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"auto_approve", "auto_deny", "error"}:
            return "auto_approve"
        return normalized

    @field_validator("openai_cu_transport")
    @classmethod
    def normalize_openai_cu_transport(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in ALLOWED_OPENAI_CU_TRANSPORTS:
            return "responses_websocket"
        return normalized

    @field_validator("cu_visual_mode")
    @classmethod
    def normalize_cu_visual_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in ALLOWED_CU_VISUAL_MODES:
            return "keyframe_patch"
        return normalized

    @model_validator(mode="after")
    def normalize_domain_settings(self) -> "Settings":
        """Coerce allow/block domain lists into normalized string lists."""
        self.actions_computer_tool_allowed_domains = self._coerce_domain_list(
            self.actions_computer_tool_allowed_domains
        )
        self.actions_computer_tool_blocked_domains = self._coerce_domain_list(
            self.actions_computer_tool_blocked_domains
        )
        return self

    @model_validator(mode="after")
    def apply_computer_use_defaults(self) -> "Settings":
        """Apply provider-specific defaults for computer-use sessions."""
        if self.cu_provider == "google":
            if self.desktop_prefer_resolution == (1920, 1080):
                self.desktop_prefer_resolution = (1440, 900)
        return self

    @model_validator(mode="after")
    def validate_openai_computer_use_model(self) -> "Settings":
        """Reject the legacy preview model on the OpenAI computer-use path."""
        if (
            self.cu_provider == "openai"
            and str(self.computer_use_model or "").strip().lower()
            == LEGACY_OPENAI_COMPUTER_USE_MODEL
        ):
            raise ValueError(
                "OpenAI computer-use model 'computer-use-preview' is no longer "
                "supported. Set HAINDY_COMPUTER_USE_MODEL=gpt-5.4."
            )
        return self

    @staticmethod
    def _coerce_domain_list(raw: Any) -> list[str]:
        """Normalize various list inputs (list, tuple, comma-separated string)."""
        if raw is None:
            return []
        if isinstance(raw, str):
            if not raw.strip():
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, (list, tuple, set)):
            coerced: list[str] = []
            for item in raw:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    coerced.append(text)
            return coerced
        return []

    def create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.reports_dir,
            self.screenshots_dir,
            self.desktop_screenshot_dir,
            self.mobile_screenshot_dir,
            self.ios_screenshot_dir,
            self.cache_dir,
            self.haindy_home,
            self.desktop_coordinate_cache_path.parent,
            self.mobile_coordinate_cache_path.parent,
            self.ios_coordinate_cache_path.parent,
            self.task_plan_cache_path.parent,
            self.planning_cache_path.parent,
            self.situational_cache_path.parent,
            self.execution_replay_cache_path.parent,
            self.model_log_path.parent,
            self.screen_recording_output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def populate_agent_models(self) -> "Settings":
        """Populate agent model configurations from defaults and explicit input."""
        configured_models: dict[str, AgentModelConfig] = {}

        existing_models = self.agent_models.copy()

        for agent_name, default_config in DEFAULT_AGENT_MODELS.items():
            if agent_name in existing_models:
                configured_models[agent_name] = existing_models[agent_name]
            else:
                configured_models[agent_name] = default_config.model_copy(deep=True)

        for agent_name, config in existing_models.items():
            if agent_name not in configured_models:
                configured_models[agent_name] = config

        self.agent_models = configured_models
        return self

    def get_agent_model_config(self, agent_name: str) -> AgentModelConfig:
        """Return agent-specific model configuration."""
        if agent_name in self.agent_models:
            return self.agent_models[agent_name]

        return AgentModelConfig(
            temperature=0.7,
            reasoning_level="medium",
        )

    def get_provider_model(self, provider: str, *, computer_use: bool = False) -> str:
        """Return the configured model for a provider."""
        field_name = get_provider_model_field_name(provider, computer_use=computer_use)
        return str(getattr(self, field_name))


class ConfigManager(ConfigProvider):
    """Configuration manager implementation."""

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize config manager.

        Args:
            settings: Optional settings instance
        """
        self.settings = settings or get_settings()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return getattr(self.settings, key)
        except AttributeError:
            return default

    def get_required(self, key: str) -> Any:
        """Get required configuration value."""
        try:
            return getattr(self.settings, key)
        except AttributeError:
            raise KeyError(f"Required configuration key not found: {key}") from None

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values."""
        return dict(self.settings.model_dump())


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = load_settings()
    settings.create_directories()
    return settings


def get_config() -> ConfigManager:
    """Get configuration manager instance."""
    return ConfigManager(get_settings())
