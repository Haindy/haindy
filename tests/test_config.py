"""Tests for application settings."""

from pathlib import Path

import pytest

from src.config.settings import AgentModelConfig, ConfigManager, Settings


class TestSettings:
    def test_default_desktop_configuration(self):
        settings = Settings()
        assert settings.desktop_prefer_resolution[0] >= 800
        assert settings.desktop_prefer_resolution[1] >= 600
        assert settings.desktop_screenshot_dir == Path("data/screenshots/desktop")

    def test_agent_models_include_situational_agent(self):
        settings = Settings()
        assert "situational_agent" in settings.agent_models

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError):
            Settings(log_level="invalid")

    def test_default_openai_model(self):
        settings = Settings()
        assert settings.openai_model == "gpt-5.2"

    def test_cu_provider_accepts_anthropic(self, monkeypatch):
        monkeypatch.setenv("CU_PROVIDER", "anthropic")
        settings = Settings(cu_provider="anthropic")
        assert settings.cu_provider == "anthropic"

    def test_default_anthropic_computer_use_model(self):
        settings = Settings(_env_file=None)
        assert settings.anthropic_cu_model == "claude-sonnet-4-6"

    def test_default_anthropic_computer_use_max_tokens(self):
        settings = Settings(_env_file=None)
        assert settings.anthropic_cu_max_tokens == 16384

    def test_default_computer_action_timeout(self):
        settings = Settings(_env_file=None)
        assert settings.actions_computer_tool_action_timeout_ms == 600000

    def test_planning_cache_defaults(self):
        settings = Settings(_env_file=None)
        assert settings.enable_planning_cache is True
        assert settings.planning_cache_path == Path("data/planning_cache.json")

    def test_planning_cache_env_alias(self, monkeypatch):
        monkeypatch.setenv("HAINDY_ENABLE_PLANNING_CACHE", "false")
        monkeypatch.setenv(
            "HAINDY_PLANNING_CACHE_PATH",
            "tmp/planning_cache_override.json",
        )
        settings = Settings(_env_file=None)
        assert settings.enable_planning_cache is False
        assert settings.planning_cache_path == Path("tmp/planning_cache_override.json")

    @pytest.mark.parametrize(
        "level",
        ["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    def test_reasoning_level_accepts_supported_values(self, level):
        config = AgentModelConfig(model="gpt-5.2", reasoning_level=level)
        assert config.reasoning_level == level

    def test_reasoning_level_rejects_unsupported_value(self):
        with pytest.raises(ValueError):
            AgentModelConfig(model="gpt-5.2", reasoning_level="ultra")


class TestConfigManager:
    def test_get_existing_key(self):
        settings = Settings(log_level="DEBUG")
        config = ConfigManager(settings)
        assert config.get("log_level") == "DEBUG"

    def test_get_all_includes_desktop_fields(self):
        config = ConfigManager(Settings())
        all_config = config.get_all()
        assert "desktop_prefer_resolution" in all_config
        assert "desktop_screenshot_dir" in all_config
