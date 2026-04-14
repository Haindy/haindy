"""Tests for hierarchical settings file loading and flattening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from haindy.config.settings_file import (
    _JSON_TO_FIELD,
    _SETTINGS_SKELETON,
    flat_to_nested,
    flatten_settings_dict,
    load_settings_file,
    write_settings_file,
)


class TestLoadSettingsFile:
    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        result = load_settings_file(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_file_returns_nested_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        data = {"desktop": {"keyboard_layout": "es"}, "logging": {"level": "DEBUG"}}
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_settings_file(path)

        assert result == data

    def test_malformed_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid JSON"):
            load_settings_file(path)

    def test_non_object_root_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON object"):
            load_settings_file(path)

    def test_secret_field_in_section_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"openai": {"openai_api_key": "sk-secret"}}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="secret field"):
            load_settings_file(path)

    def test_secret_field_at_top_level_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"anthropic_api_key": "ant-secret"}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="secret field"):
            load_settings_file(path)


class TestFlattenSettingsDict:
    def test_desktop_keyboard_layout(self) -> None:
        result = flatten_settings_dict({"desktop": {"keyboard_layout": "es"}})
        assert result == {"desktop_keyboard_layout": "es"}

    def test_computer_use_provider(self) -> None:
        result = flatten_settings_dict({"computer_use": {"provider": "anthropic"}})
        assert result == {"cu_provider": "anthropic"}

    def test_provider_specific_openai_model(self) -> None:
        result = flatten_settings_dict({"openai": {"model": "gpt-5.4"}})
        assert result == {"openai_model": "gpt-5.4"}

    def test_provider_specific_openai_codex_model(self) -> None:
        result = flatten_settings_dict({"openai-codex": {"model": "gpt-5.4"}})
        assert result == {"openai_codex_model": "gpt-5.4"}

    def test_provider_specific_google_cu_model(self) -> None:
        result = flatten_settings_dict(
            {"google": {"computer_use_model": "gemini-3-flash-preview"}}
        )
        assert result == {"google_cu_model": "gemini-3-flash-preview"}

    def test_execution_actions_max_turns(self) -> None:
        result = flatten_settings_dict({"execution": {"actions_max_turns": 20}})
        assert result == {"actions_computer_tool_max_turns": 20}

    def test_legacy_execution_action_timeout_ms_migrates_to_seconds(self) -> None:
        result = flatten_settings_dict(
            {"execution": {"actions_action_timeout_ms": 1500}}
        )
        assert result == {"actions_computer_tool_action_timeout_seconds": 1.5}

    def test_legacy_execution_action_timeout_ms_requires_numeric_value(self) -> None:
        with pytest.raises(ValueError, match="actions_action_timeout_ms"):
            flatten_settings_dict(
                {"execution": {"actions_action_timeout_ms": "not-a-number"}}
            )

    def test_logging_level(self) -> None:
        result = flatten_settings_dict({"logging": {"level": "DEBUG"}})
        assert result == {"log_level": "DEBUG"}

    def test_cache_enable_planning(self) -> None:
        result = flatten_settings_dict({"cache": {"enable_planning": False}})
        assert result == {"enable_planning_cache": False}

    def test_agent_models_pass_through(self) -> None:
        agent_data = {"scope_triage": {"model": "gpt-5.4", "temperature": 0.1}}
        result = flatten_settings_dict({"agent_models": agent_data})
        assert result == {"agent_models": agent_data}

    def test_unknown_section_ignored(self) -> None:
        result = flatten_settings_dict({"future_section": {"some_key": "value"}})
        assert result == {}

    def test_unknown_key_within_known_section_ignored(self) -> None:
        result = flatten_settings_dict(
            {"desktop": {"nonexistent_key": "value", "keyboard_layout": "us"}}
        )
        assert result == {"desktop_keyboard_layout": "us"}

    def test_multiple_sections(self) -> None:
        result = flatten_settings_dict(
            {
                "desktop": {"keyboard_layout": "es"},
                "logging": {"level": "WARNING"},
                "execution": {"max_test_steps": 50},
            }
        )
        assert result == {
            "desktop_keyboard_layout": "es",
            "log_level": "WARNING",
            "max_test_steps": 50,
        }

    def test_non_dict_section_value_ignored(self) -> None:
        result = flatten_settings_dict({"desktop": "not-a-dict"})
        assert result == {}

    def test_all_json_keys_map_to_known_fields(self) -> None:
        from haindy.config.settings import Settings

        settings_fields = set(Settings.model_fields.keys())
        for json_path, field_name in _JSON_TO_FIELD.items():
            assert field_name in settings_fields, (
                f"_JSON_TO_FIELD maps '{json_path}' -> '{field_name}' "
                f"but '{field_name}' is not a field in Settings"
            )

    def test_agent_provider_maps_to_agent_provider_field(self) -> None:
        result = flatten_settings_dict({"agent": {"provider": "anthropic"}})
        assert result == {"agent_provider": "anthropic"}

    def test_agent_anthropic_model_maps_to_anthropic_model_field(self) -> None:
        result = flatten_settings_dict({"agent": {"anthropic_model": "claude-opus"}})
        assert result == {"anthropic_model": "claude-opus"}

    def test_agent_google_model_maps_to_google_model_field(self) -> None:
        result = flatten_settings_dict({"agent": {"google_model": "gemini-3.1-pro"}})
        assert result == {"google_model": "gemini-3.1-pro"}

    def test_legacy_computer_use_model_routes_to_provider_specific_field(self) -> None:
        result = flatten_settings_dict(
            {
                "computer_use": {
                    "provider": "google",
                    "model": "gemini-legacy-cu",
                }
            }
        )
        assert result == {
            "cu_provider": "google",
            "google_cu_model": "gemini-legacy-cu",
        }


class TestSettingsSkeleton:
    def test_skeleton_includes_agent_provider(self) -> None:
        assert "agent" in _SETTINGS_SKELETON
        assert _SETTINGS_SKELETON["agent"].get("provider") == "openai"

    def test_skeleton_includes_provider_specific_models(self) -> None:
        assert _SETTINGS_SKELETON["openai"]["model"] == "gpt-5.4"
        assert _SETTINGS_SKELETON["openai"]["computer_use_model"] == "gpt-5.4"
        assert _SETTINGS_SKELETON["openai-codex"]["model"] == "gpt-5.4"
        assert _SETTINGS_SKELETON["google"]["model"] == "gemini-3-flash-preview"
        assert (
            _SETTINGS_SKELETON["google"]["computer_use_model"]
            == "gemini-3-flash-preview"
        )


class TestWriteSettingsFile:
    def test_creates_file_and_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "settings.json"
        data = {"desktop": {"keyboard_layout": "es"}}

        write_settings_file(path, data)

        assert path.exists()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written == data

    def test_merges_with_existing_content(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        existing = {"desktop": {"keyboard_layout": "es"}, "logging": {"level": "INFO"}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        write_settings_file(
            path, {"logging": {"level": "DEBUG"}, "cache": {"enable_planning": False}}
        )

        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["desktop"]["keyboard_layout"] == "es"
        assert result["logging"]["level"] == "DEBUG"
        assert result["cache"]["enable_planning"] is False

    def test_deep_merge_preserves_sibling_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "desktop": {
                        "keyboard_layout": "es",
                        "prefer_resolution": [1920, 1080],
                    }
                }
            ),
            encoding="utf-8",
        )

        write_settings_file(path, {"desktop": {"keyboard_layout": "us"}})

        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["desktop"]["keyboard_layout"] == "us"
        assert result["desktop"]["prefer_resolution"] == [1920, 1080]


class TestFlatToNested:
    def test_basic_conversion(self) -> None:
        result = flat_to_nested(
            {
                "desktop_keyboard_layout": "es",
                "log_level": "DEBUG",
            }
        )
        assert result == {
            "desktop": {"keyboard_layout": "es"},
            "logging": {"level": "DEBUG"},
        }

    def test_provider_models_convert_to_provider_sections(self) -> None:
        result = flat_to_nested(
            {
                "openai_model": "gpt-5.4",
                "openai_codex_model": "gpt-5.4",
                "google_model": "gemini-3-flash-preview",
                "google_cu_model": "gemini-3-flash-preview",
                "computer_use_model": "gpt-5.4",
            }
        )
        assert result == {
            "openai": {
                "model": "gpt-5.4",
                "computer_use_model": "gpt-5.4",
            },
            "openai-codex": {"model": "gpt-5.4"},
            "google": {
                "model": "gemini-3-flash-preview",
                "computer_use_model": "gemini-3-flash-preview",
            },
        }

    def test_unknown_field_dropped(self) -> None:
        result = flat_to_nested({"unknown_field": "value"})
        assert result == {}

    def test_agent_models_pass_through(self) -> None:
        agent_data = {"scope_triage": {"model": "gpt-5.4"}}
        result = flat_to_nested({"agent_models": agent_data})
        assert result == {"agent_models": agent_data}


class TestIntegration:
    def test_flatten_output_is_valid_settings_input(self) -> None:
        """Flat dict from flatten_settings_dict contains only known Settings fields."""
        from haindy.config.settings import Settings

        nested = {
            "desktop": {"keyboard_layout": "es", "enable_resolution_switch": False},
            "logging": {"level": "WARNING"},
            "execution": {"max_test_steps": 50, "automation_backend": "desktop"},
        }
        flat = flatten_settings_dict(nested)

        settings = Settings(**flat)

        assert settings.desktop_keyboard_layout == "es"
        assert settings.desktop_enable_resolution_switch is False
        assert settings.log_level == "WARNING"
        assert settings.max_test_steps == 50
