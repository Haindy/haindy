"""Configuration management for the HAINDY framework."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.interfaces import ConfigProvider


AGENT_ENV_PREFIX: Dict[str, str] = {
    "scope_triage": "HAINDY_SCOPE_TRIAGE",
    "test_planner": "HAINDY_TEST_PLANNER",
    "test_runner": "HAINDY_TEST_RUNNER",
    "action_agent": "HAINDY_ACTION_AGENT",
}


class AgentModelConfig(BaseModel):
    """Per-agent model configuration."""

    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    reasoning_level: str = Field(default="medium")
    modalities: Set[str] = Field(default_factory=lambda: {"text"})

    @field_validator("reasoning_level")
    @classmethod
    def validate_reasoning_level(cls, value: str) -> str:
        """Ensure reasoning level is valid."""
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            raise ValueError(
                f"Invalid reasoning level: {value}. Allowed values: {sorted(allowed)}"
            )
        return value

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: Set[str]) -> Set[str]:
        if not value:
            raise ValueError("Agent modalities cannot be empty")
        return value


DEFAULT_AGENT_MODELS: Dict[str, AgentModelConfig] = {
    "scope_triage": AgentModelConfig(
        model="gpt-5.1-mini",
        temperature=0.1,
        reasoning_level="medium",
    ),
    "test_planner": AgentModelConfig(
        model="gpt-5.1",
        temperature=0.35,
        reasoning_level="high",
    ),
    "test_runner": AgentModelConfig(
        model="gpt-5.1",
        temperature=0.5,
        reasoning_level="medium",
    ),
    "action_agent": AgentModelConfig(
        model="gpt-5.1",
        temperature=0.25,
        reasoning_level="low",
        modalities={"text", "vision"},
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
        default="gpt-5.1", description="Default OpenAI model"
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
    agent_models: Dict[str, AgentModelConfig] = Field(
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

    # Browser Configuration
    browser_headless: bool = Field(
        default=True, description="Run browser in headless mode"
    )
    browser_timeout: int = Field(
        default=30000, ge=1000, description="Default browser timeout (ms)"
    )
    browser_viewport_width: int = Field(
        default=1920, ge=800, description="Browser viewport width"
    )
    browser_viewport_height: int = Field(
        default=1080, ge=600, description="Browser viewport height"
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
        env="HAINDY_ACTIONS_USE_COMPUTER_TOOL",
    )
    actions_computer_tool_max_turns: int = Field(
        default=12,
        ge=1,
        description="Maximum tool turns per action when using Computer Use",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_MAX_TURNS",
    )
    actions_computer_tool_loop_detection_window: int = Field(
        default=3,
        ge=2,
        description="Repeated identical turns (with identical screenshots) before flagging a loop",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_LOOP_WINDOW",
    )
    actions_computer_tool_action_timeout_ms: int = Field(
        default=7000,
        ge=500,
        description="Timeout in milliseconds for executing a single computer action",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_ACTION_TIMEOUT_MS",
    )
    actions_computer_tool_stabilization_wait_ms: int = Field(
        default=1000,
        ge=0,
        le=1000,
        description="Delay after executing an action before capturing the next screenshot",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_STABILIZATION_WAIT_MS",
    )
    actions_computer_tool_fail_fast_on_safety: bool = Field(
        default=True,
        description="Abort immediately when Computer Use returns pending safety checks",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_FAIL_FAST",
    )
    actions_computer_tool_auto_ack_safety: bool = Field(
        default=True,
        description="Automatically acknowledge Computer Use safety checks instead of failing fast",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_AUTO_ACK_SAFETY",
    )
    actions_computer_tool_auto_ack_codes: List[str] = Field(
        default_factory=list,
        description=(
            "Safety check codes eligible for auto-ack when enabled; empty list means acknowledge all codes"
        ),
        env="HAINDY_ACTIONS_COMPUTER_TOOL_AUTO_ACK_CODES",
    )
    actions_computer_tool_allowed_domains: List[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool is permitted to interact with",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_ALLOWED_DOMAINS",
    )
    actions_computer_tool_blocked_domains: List[str] = Field(
        default_factory=list,
        description="Domains the Computer Use tool must never interact with",
        env="HAINDY_ACTIONS_COMPUTER_TOOL_BLOCKED_DOMAINS",
    )

    # Desktop Configuration
    desktop_mode_enabled: bool = Field(
        default=True,
        description="Enable desktop computer-use mode (OS-level automation)",
        env="HAINDY_DESKTOP_MODE",
    )
    desktop_preferred_width: int = Field(
        default=1920,
        description="Preferred desktop width when switching resolution",
        env="HAINDY_DESKTOP_WIDTH",
    )
    desktop_preferred_height: int = Field(
        default=1080,
        description="Preferred desktop height when switching resolution",
        env="HAINDY_DESKTOP_HEIGHT",
    )
    desktop_enable_resolution_switch: bool = Field(
        default=False,
        description="Allow temporary resolution downshift for desktop runs",
        validation_alias=AliasChoices("HAINDY_DESKTOP_RES_SWITCH"),
    )
    desktop_screenshot_dir: Path = Field(
        default=Path("debug_screenshots/desktop"),
        description="Directory for desktop screenshots",
    )
    desktop_cache_path: Path = Field(
        default=Path("data/desktop_cache/linkedin.json"),
        description="Coordinate cache path for desktop mode",
    )
    desktop_window_hint: str = Field(
        default="Firefox window with LinkedIn",
        description="Hint used to surface the correct window during acquisition",
    )
    desktop_display: Optional[str] = Field(
        default=None,
        description="Override DISPLAY for desktop capture/input (e.g., ':1')",
        env="HAINDY_DESKTOP_DISPLAY",
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
    log_file: Optional[str] = Field(
        default=None, description="Log file path"
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

    @staticmethod
    def _coerce_domain_list(raw: Any) -> List[str]:
        """Normalize various list inputs (list, tuple, comma-separated string)."""
        if raw is None:
            return []
        if isinstance(raw, str):
            if not raw.strip():
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, (list, tuple, set)):
            coerced: List[str] = []
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
            self.cache_dir,
            self.desktop_screenshot_dir,
            self.desktop_cache_path.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="after")
    def populate_agent_models(self) -> "Settings":
        """Populate agent model configurations from defaults and environment."""
        env = os.environ

        configured_models: Dict[str, AgentModelConfig] = {}
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

    def __init__(self, settings: Optional[Settings] = None) -> None:
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
            raise KeyError(f"Required configuration key not found: {key}")

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.settings.model_dump()


@lru_cache()
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
