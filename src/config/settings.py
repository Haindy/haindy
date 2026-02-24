"""Configuration management for the HAINDY framework."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.interfaces import ConfigProvider

AGENT_ENV_PREFIX: dict[str, str] = {
    "scope_triage": "HAINDY_SCOPE_TRIAGE",
    "test_planner": "HAINDY_TEST_PLANNER",
    "test_runner": "HAINDY_TEST_RUNNER",
    "action_agent": "HAINDY_ACTION_AGENT",
    "situational_agent": "HAINDY_SITUATIONAL_AGENT",
}

ALLOWED_REASONING_LEVELS: set[str] = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}


class AgentModelConfig(BaseModel):
    """Per-agent model configuration."""

    model: str
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
        model="gpt-5.2",
        temperature=0.15,
        reasoning_level="high",
    ),
    "test_planner": AgentModelConfig(
        model="gpt-5.2",
        temperature=0.35,
        reasoning_level="high",
    ),
    "test_runner": AgentModelConfig(
        model="gpt-5.2",
        temperature=0.55,
        reasoning_level="medium",
    ),
    "action_agent": AgentModelConfig(
        model="gpt-5.2",
        temperature=0.25,
        reasoning_level="low",
        modalities={"text", "vision"},
    ),
    "situational_agent": AgentModelConfig(
        model="gpt-5.2",
        temperature=0.1,
        reasoning_level="high",
    ),
}


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-5.2", description="Default OpenAI model"
    )
    openai_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default temperature"
    )
    openai_max_retries: int = Field(
        default=3, ge=1, description="Maximum API retry attempts"
    )
    openai_request_timeout_seconds: int = Field(
        default=900,
        ge=60,
        description="Request timeout for OpenAI API calls in seconds",
    )
    agent_models: dict[str, AgentModelConfig] = Field(
        default_factory=dict,
        description="Per-agent OpenAI model configuration",
    )

    # Grid System Configuration
    grid_size: int = Field(
        default=60, ge=10, le=100, description="Grid size (NxN)"
    )
    grid_refinement_enabled: bool = Field(
        default=True, description="Enable adaptive refinement"
    )
    grid_confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence threshold for refinement"
    )

    # Desktop Configuration
    desktop_prefer_resolution: tuple[int, int] = Field(
        default=(1920, 1080),
        description="Preferred resolution for desktop sessions",
        validation_alias=AliasChoices("HAINDY_DESKTOP_RESOLUTION"),
    )
    desktop_keyboard_layout: str = Field(
        default="us",
        description="Keyboard layout for desktop automation (us, es)",
        validation_alias=AliasChoices("HAINDY_DESKTOP_KEYBOARD_LAYOUT"),
    )
    desktop_enable_keyboard_scancodes: bool = Field(
        default=True,
        description="Emit MSC_SCAN scancodes for key events",
        validation_alias=AliasChoices("HAINDY_DESKTOP_KEYBOARD_SCANCODES"),
    )
    desktop_keyboard_key_delay_ms: int = Field(
        default=12,
        ge=0,
        description="Delay between key events when sending combos",
        validation_alias=AliasChoices("HAINDY_DESKTOP_KEY_DELAY_MS"),
    )
    desktop_enable_resolution_switch: bool = Field(
        default=True,
        description="Allow resolution downshift for desktop runs",
        validation_alias=AliasChoices("HAINDY_DESKTOP_ENABLE_RESOLUTION_SWITCH"),
    )
    desktop_screenshot_dir: Path = Field(
        default=Path("data/screenshots/desktop"),
        description="Directory for desktop screenshots",
        validation_alias=AliasChoices("HAINDY_DESKTOP_SCREENSHOT_DIR"),
    )
    desktop_coordinate_cache_path: Path = Field(
        default=Path("data/desktop_cache/coordinates.json"),
        description="Coordinate cache path for desktop actions",
        validation_alias=AliasChoices("HAINDY_DESKTOP_COORDINATE_CACHE_PATH"),
    )
    task_plan_cache_path: Path = Field(
        default=Path("data/task_plan_cache.json"),
        description="Task planning cache path",
        validation_alias=AliasChoices("HAINDY_TASK_PLAN_CACHE_PATH"),
    )
    enable_execution_replay_cache: bool = Field(
        default=True,
        description="Enable execution replay cache (record/replay driver actions per step)",
        validation_alias=AliasChoices("HAINDY_ENABLE_EXECUTION_REPLAY_CACHE"),
    )
    execution_replay_cache_path: Path = Field(
        default=Path("data/execution_replay_cache.json"),
        description="Execution replay cache path",
        validation_alias=AliasChoices("HAINDY_EXECUTION_REPLAY_CACHE_PATH"),
    )
    desktop_display: str | None = Field(
        default=None,
        description="X11 display override for desktop capture",
        validation_alias=AliasChoices("HAINDY_DESKTOP_DISPLAY"),
    )
    desktop_clipboard_timeout_seconds: float = Field(
        default=3.0,
        ge=0.5,
        description="Timeout for desktop clipboard reads",
        validation_alias=AliasChoices("HAINDY_DESKTOP_CLIPBOARD_TIMEOUT_SECONDS"),
    )
    desktop_clipboard_hold_seconds: float = Field(
        default=15.0,
        ge=0.5,
        description="Max time to hold clipboard owner process",
        validation_alias=AliasChoices("HAINDY_DESKTOP_CLIPBOARD_HOLD_SECONDS"),
    )
    enable_screen_recording: bool = Field(
        default=False,
        description="Enable GNOME desktop screen recording during test execution",
        validation_alias=AliasChoices("HAINDY_ENABLE_SCREEN_RECORDING"),
    )
    screen_recording_output_dir: Path = Field(
        default=Path("reports/recordings"),
        description="Directory for optional desktop screen recordings",
        validation_alias=AliasChoices("HAINDY_SCREEN_RECORDING_OUTPUT_DIR"),
    )
    screen_recording_framerate: int = Field(
        default=30,
        ge=1,
        description="Framerate for GNOME desktop screen recordings",
        validation_alias=AliasChoices("HAINDY_SCREEN_RECORDING_FRAMERATE"),
    )
    screen_recording_draw_cursor: bool = Field(
        default=True,
        description="Draw cursor in GNOME desktop screen recordings",
        validation_alias=AliasChoices("HAINDY_SCREEN_RECORDING_DRAW_CURSOR"),
    )
    screen_recording_prefix: str = Field(
        default="haindy-agent",
        description="Filename prefix for desktop screen recordings",
        validation_alias=AliasChoices("HAINDY_SCREEN_RECORDING_PREFIX"),
    )

    # Execution Configuration
    max_test_steps: int = Field(
        default=100, ge=1, description="Maximum steps per test"
    )
    step_timeout: int = Field(
        default=30000, ge=1000, description="Timeout per step (ms)"
    )
    max_retries_per_step: int = Field(
        default=3, ge=1, description="Maximum retries per step"
    )
    screenshot_quality: int = Field(
        default=80, ge=1, le=100, description="Screenshot JPEG quality"
    )

    # Computer Use Configuration
    actions_use_computer_tool: bool = Field(
        default=False,
        description="Enable OpenAI Computer Use tool for action execution",
        validation_alias=AliasChoices("HAINDY_ACTIONS_USE_COMPUTER_TOOL"),
    )
    computer_use_model: str = Field(
        default="computer-use-preview",
        description="OpenAI model for computer-use execution",
        validation_alias=AliasChoices("HAINDY_COMPUTER_USE_MODEL", "COMPUTER_USE_MODEL"),
    )
    cu_provider: str = Field(
        default="google",
        description="Computer-use provider to run actions (openai or google)",
        validation_alias=AliasChoices("CU_PROVIDER", "HAINDY_CU_PROVIDER"),
    )
    google_cu_model: str = Field(
        default="gemini-2.5-computer-use-preview-10-2025",
        description="Google Gemini computer-use model name",
        validation_alias=AliasChoices("GOOGLE_CU_MODEL", "HAINDY_GOOGLE_CU_MODEL"),
    )
    vertex_api_key: str = Field(
        default="",
        description="API key for Google Vertex computer-use",
        validation_alias=AliasChoices("VERTEX_API_KEY", "HAINDY_VERTEX_API_KEY"),
    )
    vertex_project: str = Field(
        default="",
        description="Vertex project for Google computer-use runs",
        validation_alias=AliasChoices("VERTEX_PROJECT", "HAINDY_VERTEX_PROJECT"),
    )
    vertex_location: str = Field(
        default="us-central1",
        description="Vertex location for Google computer-use runs",
        validation_alias=AliasChoices("VERTEX_LOCATION", "HAINDY_VERTEX_LOCATION"),
    )
    cu_safety_policy: str = Field(
        default="auto_approve",
        description="Handling for provider safety confirmations (auto_approve, auto_deny, error)",
        validation_alias=AliasChoices("CU_SAFETY_POLICY", "HAINDY_CU_SAFETY_POLICY"),
    )
    actions_computer_tool_max_turns: int = Field(
        default=12,
        ge=1,
        description="Maximum tool turns per action when using Computer Use",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_MAX_TURNS"),
    )
    actions_computer_tool_loop_detection_window: int = Field(
        default=3,
        ge=2,
        description="Repeated identical turns (with identical screenshots) before flagging a loop",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_LOOP_WINDOW"),
    )
    actions_computer_tool_action_timeout_ms: int = Field(
        default=7000,
        ge=500,
        description="Timeout in milliseconds for executing a single computer action",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_ACTION_TIMEOUT_MS"),
    )
    actions_computer_tool_stabilization_wait_ms: int = Field(
        default=1000,
        ge=0,
        le=1000,
        description="Delay after executing an action before capturing the next screenshot",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_STABILIZATION_WAIT_MS"),
    )
    actions_computer_tool_fail_fast_on_safety: bool = Field(
        default=True,
        description=(
            "Abort immediately on pending safety checks before cu_safety_policy is applied"
        ),
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_FAIL_FAST"),
    )
    actions_computer_tool_allowed_domains: list[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool is permitted to interact with",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_ALLOWED_DOMAINS"),
    )
    actions_computer_tool_blocked_domains: list[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool must never interact with",
        validation_alias=AliasChoices("HAINDY_ACTIONS_COMPUTER_TOOL_BLOCKED_DOMAINS"),
    )
    scroll_turn_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        description="Multiplier for extra scroll turns before max-turn enforcement",
        validation_alias=AliasChoices("HAINDY_SCROLL_TURN_MULTIPLIER"),
    )
    scroll_default_magnitude: int = Field(
        default=450,
        ge=1,
        description="Default scroll magnitude (pixels) when not specified",
        validation_alias=AliasChoices("HAINDY_SCROLL_DEFAULT_MAGNITUDE"),
    )
    scroll_max_magnitude: int = Field(
        default=600,
        ge=1,
        description="Maximum per-scroll magnitude/pixel delta",
        validation_alias=AliasChoices("HAINDY_SCROLL_MAX_MAGNITUDE"),
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )
    log_file: str | None = Field(
        default=None, description="Log file path"
    )
    model_log_path: Path = Field(
        default=Path("data/model_logs/model_calls.jsonl"),
        description="Log file for raw model prompts/responses",
        validation_alias=AliasChoices("MODEL_LOG_PATH", "HAINDY_MODEL_LOG_PATH"),
    )
    max_screenshots: int = Field(
        default=12,
        ge=1,
        description="Cap on retained screenshots per run",
        validation_alias=AliasChoices("HAINDY_MAX_SCREENSHOTS"),
    )

    # Storage Configuration
    data_dir: Path = Field(
        default=Path("data"), description="Data storage directory"
    )
    reports_dir: Path = Field(
        default=Path("reports"), description="Reports output directory"
    )
    screenshots_dir: Path = Field(
        default=Path("data/screenshots"), description="Screenshots directory"
    )
    cache_dir: Path = Field(
        default=Path(".cache"), description="Cache directory"
    )

    # Security Configuration
    rate_limit_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60, ge=1, description="API requests per minute"
    )
    sanitize_screenshots: bool = Field(
        default=True, description="Sanitize PII from screenshots"
    )

    # Development Configuration
    debug_mode: bool = Field(
        default=False, description="Enable debug mode"
    )
    save_agent_conversations: bool = Field(
        default=True, description="Save agent conversation logs"
    )
    enable_grid_overlay: bool = Field(
        default=False, description="Show grid overlay in screenshots"
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

    @field_validator("cu_provider")
    @classmethod
    def normalize_cu_provider(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"openai", "google"}:
            return "openai"
        return normalized

    @field_validator("cu_safety_policy")
    @classmethod
    def normalize_cu_safety_policy(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"auto_approve", "auto_deny", "error"}:
            return "auto_approve"
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
            self.cache_dir,
            self.desktop_coordinate_cache_path.parent,
            self.task_plan_cache_path.parent,
            self.execution_replay_cache_path.parent,
            self.model_log_path.parent,
            self.screen_recording_output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def populate_agent_models(self) -> "Settings":
        """Populate agent model configurations from defaults and environment."""
        env = os.environ

        configured_models: dict[str, AgentModelConfig] = {}
        openai_model_env_set = "OPENAI_MODEL" in env
        openai_temperature_env_set = "OPENAI_TEMPERATURE" in env

        # Preserve user supplied mapping while ensuring defaults exist
        existing_models = self.agent_models.copy()

        for agent_name, prefix in AGENT_ENV_PREFIX.items():
            base_config = existing_models.get(agent_name, DEFAULT_AGENT_MODELS[agent_name])
            config_payload = base_config.model_dump()

            model_override = env.get(f"{prefix}_MODEL")
            if model_override:
                config_payload["model"] = model_override
            elif openai_model_env_set:
                config_payload["model"] = self.openai_model

            temperature_override = env.get(f"{prefix}_TEMPERATURE")
            if temperature_override:
                try:
                    config_payload["temperature"] = float(temperature_override)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid temperature for {agent_name}: {temperature_override}"
                    ) from exc
            elif openai_temperature_env_set:
                config_payload["temperature"] = self.openai_temperature

            reasoning_override = env.get(f"{prefix}_REASONING_LEVEL")
            if reasoning_override:
                config_payload["reasoning_level"] = reasoning_override.lower()

            modalities_override = env.get(f"{prefix}_MODALITIES")
            if modalities_override:
                config_payload["modalities"] = {
                    modality.strip().lower()
                    for modality in modalities_override.split(",")
                    if modality.strip()
                }

            configured_models[agent_name] = AgentModelConfig(**config_payload)

        # Include any extra agent configs supplied directly
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
            model=self.openai_model,
            temperature=self.openai_temperature,
            reasoning_level="medium",
        )


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
        return self.settings.model_dump()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    settings = Settings()
    settings.create_directories()
    return settings


def get_config() -> ConfigManager:
    """Get configuration manager instance."""
    return ConfigManager(get_settings())
