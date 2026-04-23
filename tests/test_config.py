"""Tests for application settings."""

from pathlib import Path

import pytest

from haindy.config.settings import (
    DEFAULT_AGENT_MODELS,
    SETTINGS_ENV_VARS,
    AgentModelConfig,
    ConfigManager,
    Settings,
    load_settings,
)


class TestSettings:
    def test_default_desktop_configuration(self):
        settings = Settings()
        assert settings.desktop_prefer_resolution[0] >= 800
        assert settings.desktop_prefer_resolution[1] >= 600
        assert settings.desktop_screenshot_dir == Path("data/screenshots/desktop")
        assert settings.mobile_screenshot_dir == Path("data/screenshots/mobile")
        assert settings.desktop_keyboard_layout == "auto"
        assert settings.automation_backend == "desktop"

    def test_agent_models_include_situational_agent(self):
        settings = Settings()
        assert "situational_agent" in settings.agent_models

    def test_action_agent_model_overrides_are_ignored(self):
        settings = load_settings(
            {
                "HAINDY_ACTION_AGENT_MODEL": "gpt-5.4",
                "HAINDY_ACTION_AGENT_REASONING_LEVEL": "high",
                "HAINDY_ACTION_AGENT_MODALITIES": "text,vision",
            }
        )
        assert "action_agent" not in settings.agent_models

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError):
            Settings(log_level="invalid")

    def test_default_log_format_is_text(self):
        settings = Settings()
        assert settings.log_format == "text"

    def test_default_openai_model(self):
        settings = Settings()
        assert settings.openai_model == "gpt-5.4"

    def test_default_openai_codex_model(self):
        settings = Settings()
        assert settings.openai_codex_model == "gpt-5.4"

    def test_default_openai_computer_use_model(self):
        settings = load_settings({})
        assert settings.computer_use_model == "gpt-5.4"

    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            ("desktop", "desktop"),
            ("browser", "desktop"),
            ("web", "desktop"),
            ("mobile", "mobile_adb"),
            ("android", "mobile_adb"),
            ("invalid", "desktop"),
        ],
    )
    def test_automation_backend_loader(self, backend: str, expected: str):
        settings = load_settings({"HAINDY_AUTOMATION_BACKEND": backend})
        assert settings.automation_backend == expected

    @pytest.mark.parametrize(
        ("layout", "expected"),
        [
            ("auto", "auto"),
            ("us", "us"),
            ("es", "es"),
            ("ES", "es"),
            ("unknown", "us"),
        ],
    )
    def test_desktop_keyboard_layout_loader(self, layout: str, expected: str):
        settings = load_settings({"HAINDY_DESKTOP_KEYBOARD_LAYOUT": layout})
        assert settings.desktop_keyboard_layout == expected

    def test_cu_provider_accepts_exact_env_name(self):
        settings = load_settings({"HAINDY_CU_PROVIDER": "anthropic"})
        assert settings.cu_provider == "anthropic"

    def test_legacy_cu_provider_env_name_is_ignored(self):
        settings = load_settings({"CU_PROVIDER": "anthropic"})
        # The non-HAINDY-prefixed env var should be ignored; the value must not be "anthropic"
        assert settings.cu_provider != "anthropic"

    def test_default_anthropic_computer_use_model(self):
        settings = load_settings({})
        assert settings.anthropic_cu_model == "claude-sonnet-4-6"

    def test_default_anthropic_computer_use_max_tokens(self):
        settings = load_settings({})
        assert settings.anthropic_cu_max_tokens == 16384

    def test_openai_computer_use_rejects_legacy_preview_model(self):
        with pytest.raises(ValueError, match="computer-use-preview"):
            load_settings(
                {
                    "HAINDY_CU_PROVIDER": "openai",
                    "HAINDY_COMPUTER_USE_MODEL": "computer-use-preview",
                }
            )

    def test_default_computer_action_timeout(self):
        settings = Settings()
        assert settings.actions_computer_tool_action_timeout_seconds == 600.0

    def test_planning_cache_defaults(self):
        settings = load_settings({})
        assert settings.enable_planning_cache is True
        assert settings.planning_cache_path == Path("data/planning_cache.json")

    def test_planning_cache_env_loader(self):
        settings = load_settings(
            {
                "HAINDY_ENABLE_PLANNING_CACHE": "false",
                "HAINDY_PLANNING_CACHE_PATH": "tmp/planning_cache_override.json",
            }
        )
        assert settings.enable_planning_cache is False
        assert settings.planning_cache_path == Path("tmp/planning_cache_override.json")

    def test_situational_cache_defaults(self):
        settings = load_settings({})
        assert settings.enable_situational_cache is True
        assert settings.situational_cache_path == Path("data/situational_cache.json")

    def test_situational_cache_env_loader(self):
        settings = load_settings(
            {
                "HAINDY_ENABLE_SITUATIONAL_CACHE": "false",
                "HAINDY_SITUATIONAL_CACHE_PATH": "tmp/situational_cache_override.json",
            }
        )
        assert settings.enable_situational_cache is False
        assert settings.situational_cache_path == Path(
            "tmp/situational_cache_override.json"
        )

    def test_load_settings_reads_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "HAINDY_CU_PROVIDER=anthropic\nHAINDY_OPENAI_MODEL=gpt-5.4\n",
            encoding="utf-8",
        )
        settings = load_settings()
        assert settings.cu_provider == "anthropic"
        assert settings.openai_model == "gpt-5.4"

    def test_openai_model_rejects_legacy_non_cu_value(self):
        with pytest.raises(ValueError, match="Unsupported OpenAI model 'gpt-5.2'"):
            load_settings({"HAINDY_OPENAI_MODEL": "gpt-5.2"})

    def test_openai_codex_model_rejects_unsupported_value(self):
        with pytest.raises(ValueError, match="Unsupported OpenAI model 'gpt-5.2'"):
            load_settings({"HAINDY_OPENAI_CODEX_MODEL": "gpt-5.2"})

    @pytest.mark.parametrize(
        "level",
        ["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    def test_reasoning_level_accepts_supported_values(self, level):
        config = AgentModelConfig(model="gpt-5.4", reasoning_level=level)
        assert config.reasoning_level == level

    def test_reasoning_level_rejects_unsupported_value(self):
        with pytest.raises(ValueError):
            AgentModelConfig(model="gpt-5.4", reasoning_level="ultra")

    def test_settings_has_agent_provider_defaulting_to_openai(self):
        settings = Settings()
        assert settings.agent_provider == "openai"

    def test_settings_accepts_openai_codex_agent_provider(self):
        settings = load_settings({"HAINDY_AGENT_PROVIDER": "openai-codex"})
        assert settings.agent_provider == "openai-codex"

    def test_settings_has_anthropic_model_default(self):
        settings = Settings()
        assert settings.anthropic_model == "claude-sonnet-4-6"

    def test_settings_has_google_model_default(self):
        settings = Settings()
        assert settings.google_model == "gemini-3-flash-preview"

    def test_settings_env_vars_contains_agent_provider(self):
        assert "agent_provider" in SETTINGS_ENV_VARS
        assert SETTINGS_ENV_VARS["agent_provider"] == "HAINDY_AGENT_PROVIDER"

    def test_settings_env_vars_contains_openai_codex_model(self):
        assert "openai_codex_model" in SETTINGS_ENV_VARS
        assert SETTINGS_ENV_VARS["openai_codex_model"] == "HAINDY_OPENAI_CODEX_MODEL"

    def test_settings_env_vars_contains_anthropic_model(self):
        assert "anthropic_model" in SETTINGS_ENV_VARS
        assert SETTINGS_ENV_VARS["anthropic_model"] == "HAINDY_ANTHROPIC_MODEL"

    def test_settings_env_vars_contains_google_model(self):
        assert "google_model" in SETTINGS_ENV_VARS
        assert SETTINGS_ENV_VARS["google_model"] == "HAINDY_GOOGLE_MODEL"

    def test_agent_model_config_accepts_none_model(self):
        config = AgentModelConfig(model=None)
        assert config.model is None

    def test_agent_model_config_accepts_arbitrary_model_string(self):
        config = AgentModelConfig(model="claude-sonnet-4-6")
        assert config.model == "claude-sonnet-4-6"

    def test_default_agent_models_have_no_explicit_model(self):
        for agent_name, config in DEFAULT_AGENT_MODELS.items():
            assert config.model is None, (
                f"DEFAULT_AGENT_MODELS[{agent_name!r}].model should be None, "
                f"got {config.model!r}"
            )


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
        assert "mobile_screenshot_dir" in all_config
        assert "automation_backend" in all_config
