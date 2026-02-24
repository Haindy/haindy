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
