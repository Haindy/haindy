"""Configuration management for the HAINDY framework."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.interfaces import ConfigProvider


AGENT_ENV_PREFIX: Dict[str, str] = {
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

    @model_validator(mode="after")
    def validate_reasoning_level(cls, values: "AgentModelConfig") -> "AgentModelConfig":
        """Ensure reasoning level is valid."""
        allowed = {"low", "medium", "high"}
        if values.reasoning_level not in allowed:
            raise ValueError(
                f"Invalid reasoning level: {values.reasoning_level}. "
                f"Allowed values: {sorted(allowed)}"
            )
        if not values.modalities:
            raise ValueError("Agent modalities cannot be empty")
        return values


DEFAULT_AGENT_MODELS: Dict[str, AgentModelConfig] = {
    "test_planner": AgentModelConfig(
        model="gpt-5",
        temperature=0.35,
        reasoning_level="high",
    ),
    "test_runner": AgentModelConfig(
        model="gpt-5",
        temperature=0.55,
        reasoning_level="medium",
    ),
    "action_agent": AgentModelConfig(
        model="gpt-5",
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
        default="gpt-5", description="Default OpenAI model"
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

    def create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.reports_dir,
            self.screenshots_dir,
            self.cache_dir,
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
