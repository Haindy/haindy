"""Tests for application settings."""

import json
from pathlib import Path

import pytest

from haindy.config.settings import (
    DEFAULT_AGENT_MODELS,
    SETTINGS_ENV_VARS,
    AgentModelConfig,
    ConfigManager,
    Settings,
    build_project_data_dir,
    load_settings,
)


class TestSettings:
    def test_default_desktop_configuration(self):
        settings = Settings()
        assert settings.desktop_prefer_resolution[0] >= 800
        assert settings.desktop_prefer_resolution[1] >= 600
        assert settings.desktop_screenshot_dir == (
            settings.data_dir / "screenshots" / "desktop"
        )
        assert settings.mobile_screenshot_dir == (
            settings.data_dir / "screenshots" / "mobile"
        )
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
        settings = Settings()
        assert settings.enable_planning_cache is True
        assert settings.planning_cache_path == settings.data_dir / "planning_cache.json"

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
        settings = Settings()
        assert settings.enable_situational_cache is True
        assert settings.situational_cache_path == (
            settings.data_dir / "situational_cache.json"
        )

    def test_default_data_paths_use_project_scoped_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        cwd = tmp_path / "My Project!"
        home = tmp_path / "haindy-home"
        cwd.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
        monkeypatch.chdir(cwd)

        settings = load_settings({"HAINDY_HOME": str(home)})
        expected_data_dir = build_project_data_dir(home, cwd)

        assert settings.data_dir == expected_data_dir
        assert settings.data_dir.parent == home / "data" / "projects"
        assert settings.data_dir.name.startswith("my-project-")
        assert len(settings.data_dir.name.rsplit("-", 1)[-1]) == 12
        assert settings.screenshots_dir == expected_data_dir / "screenshots"
        assert settings.desktop_screenshot_dir == (
            expected_data_dir / "screenshots" / "desktop"
        )
        assert settings.windows_screenshot_dir == (
            expected_data_dir / "screenshots" / "windows"
        )
        assert settings.mobile_screenshot_dir == (
            expected_data_dir / "screenshots" / "mobile"
        )
        assert settings.ios_screenshot_dir == (
            expected_data_dir / "screenshots" / "ios"
        )
        assert settings.macos_screenshot_dir == (
            expected_data_dir / "screenshots" / "macos"
        )
        assert settings.model_log_path == (
            expected_data_dir / "model_logs" / "model_calls.jsonl"
        )
        assert settings.planning_cache_path == (
            expected_data_dir / "planning_cache.json"
        )
        assert settings.situational_cache_path == (
            expected_data_dir / "situational_cache.json"
        )
        assert settings.task_plan_cache_path == (
            expected_data_dir / "task_plan_cache.json"
        )
        assert settings.execution_replay_cache_path == (
            expected_data_dir / "execution_replay_cache.json"
        )
        assert settings.linux_coordinate_cache_path == (
            expected_data_dir / "linux_cache" / "coordinates.json"
        )
        assert settings.macos_coordinate_cache_path == (
            expected_data_dir / "macos_cache" / "coordinates.json"
        )

    def test_data_dir_override_is_exact_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
        data_dir = tmp_path / "explicit-data"

        settings = load_settings(
            {
                "HAINDY_HOME": str(tmp_path / "haindy-home"),
                "HAINDY_DATA_DIR": str(data_dir),
            }
        )

        assert settings.data_dir == data_dir
        assert settings.screenshots_dir == data_dir / "screenshots"
        assert settings.model_log_path == (
            data_dir / "model_logs" / "model_calls.jsonl"
        )
        assert settings.desktop_screenshot_dir == (data_dir / "screenshots" / "desktop")
        assert settings.planning_cache_path == data_dir / "planning_cache.json"

    def test_settings_file_data_dir_override_is_exact_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        user_home = tmp_path / "user-home"
        data_dir = tmp_path / "settings-data"
        settings_path = user_home / ".haindy" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"storage": {"data_dir": str(data_dir)}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(user_home))

        settings = load_settings({})

        assert settings.data_dir == data_dir
        assert settings.screenshots_dir == data_dir / "screenshots"
        assert settings.model_log_path == (
            data_dir / "model_logs" / "model_calls.jsonl"
        )

    def test_specific_path_overrides_win_over_data_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
        data_dir = tmp_path / "explicit-data"

        settings = load_settings(
            {
                "HAINDY_HOME": str(tmp_path / "haindy-home"),
                "HAINDY_DATA_DIR": str(data_dir),
                "HAINDY_MODEL_LOG_PATH": "legacy/model_calls.jsonl",
                "HAINDY_DESKTOP_SCREENSHOT_DIR": "legacy/screenshots/desktop",
                "HAINDY_PLANNING_CACHE_PATH": "legacy/planning_cache.json",
            }
        )

        assert settings.data_dir == data_dir
        assert settings.model_log_path == Path("legacy/model_calls.jsonl")
        assert settings.desktop_screenshot_dir == Path("legacy/screenshots/desktop")
        assert settings.planning_cache_path == Path("legacy/planning_cache.json")
        assert settings.mobile_screenshot_dir == data_dir / "screenshots" / "mobile"

    def test_create_directories_includes_all_data_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data-root"
        settings = Settings(
            data_dir=data_dir,
            haindy_home=tmp_path / "haindy-home",
            cache_dir=tmp_path / "cache",
        )

        settings.create_directories()

        assert (data_dir / "screenshots" / "desktop").is_dir()
        assert (data_dir / "screenshots" / "mobile").is_dir()
        assert (data_dir / "screenshots" / "ios").is_dir()
        assert (data_dir / "screenshots" / "macos").is_dir()
        assert (data_dir / "screenshots" / "windows").is_dir()
        assert (data_dir / "model_logs").is_dir()
        assert (data_dir / "journals").is_dir()

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
