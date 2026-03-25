"""Read and write hierarchical settings files for HAINDY.

Settings are stored as JSON with nested sections that map to the flat
``Settings`` Pydantic model. API keys/secrets are intentionally excluded
from this schema -- they belong in the system keychain or env vars.

Priority (low to high):
  built-in defaults < ~/.haindy/settings.json < env vars
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SECRET_FIELD_NAMES = frozenset(
    {"openai_api_key", "anthropic_api_key", "vertex_api_key"}
)

# Explicit mapping from "section.json_key" -> Settings field name.
# Covers every non-secret, non-agent_models field in Settings.
_JSON_TO_FIELD: dict[str, str] = {
    # agent section
    "agent.provider": "agent_provider",
    "agent.anthropic_model": "anthropic_model",
    "agent.google_model": "google_model",
    # openai section
    "openai.model": "openai_model",
    "openai.max_retries": "openai_max_retries",
    "openai.request_timeout_seconds": "openai_request_timeout_seconds",
    # computer_use section
    "computer_use.provider": "cu_provider",
    "computer_use.model": "computer_use_model",
    "computer_use.anthropic_beta": "anthropic_cu_beta",
    "computer_use.anthropic_max_tokens": "anthropic_cu_max_tokens",
    "computer_use.vertex_project": "vertex_project",
    "computer_use.vertex_location": "vertex_location",
    "computer_use.safety_policy": "cu_safety_policy",
    "computer_use.openai_transport": "openai_cu_transport",
    "computer_use.visual_mode": "cu_visual_mode",
    "computer_use.cartography_model": "cu_cartography_model",
    "computer_use.keyframe_max_turns": "cu_keyframe_max_turns",
    "computer_use.patch_max_area_ratio": "cu_patch_max_area_ratio",
    "computer_use.patch_margin_ratio": "cu_patch_margin_ratio",
    # desktop section
    "desktop.prefer_resolution": "desktop_prefer_resolution",
    "desktop.keyboard_layout": "desktop_keyboard_layout",
    "desktop.enable_keyboard_scancodes": "desktop_enable_keyboard_scancodes",
    "desktop.keyboard_key_delay_ms": "desktop_keyboard_key_delay_ms",
    "desktop.enable_resolution_switch": "desktop_enable_resolution_switch",
    "desktop.screenshot_dir": "desktop_screenshot_dir",
    "desktop.coordinate_cache_path": "desktop_coordinate_cache_path",
    "desktop.display": "desktop_display",
    "desktop.clipboard_timeout_seconds": "desktop_clipboard_timeout_seconds",
    "desktop.clipboard_hold_seconds": "desktop_clipboard_hold_seconds",
    # mobile section
    "mobile.screenshot_dir": "mobile_screenshot_dir",
    "mobile.coordinate_cache_path": "mobile_coordinate_cache_path",
    "mobile.default_adb_serial": "mobile_default_adb_serial",
    "mobile.adb_timeout_seconds": "mobile_adb_timeout_seconds",
    # ios section
    "ios.screenshot_dir": "ios_screenshot_dir",
    "ios.coordinate_cache_path": "ios_coordinate_cache_path",
    "ios.default_device_udid": "ios_default_device_udid",
    "ios.idb_timeout_seconds": "ios_idb_timeout_seconds",
    # macos section
    "macos.screenshot_dir": "macos_screenshot_dir",
    "macos.coordinate_cache_path": "macos_coordinate_cache_path",
    "macos.keyboard_layout": "macos_keyboard_layout",
    "macos.key_delay_ms": "macos_keyboard_key_delay_ms",
    "macos.clipboard_timeout_seconds": "macos_clipboard_timeout_seconds",
    "macos.clipboard_hold_seconds": "macos_clipboard_hold_seconds",
    # screen_recording section
    "screen_recording.enable": "enable_screen_recording",
    "screen_recording.output_dir": "screen_recording_output_dir",
    "screen_recording.framerate": "screen_recording_framerate",
    "screen_recording.draw_cursor": "screen_recording_draw_cursor",
    "screen_recording.prefix": "screen_recording_prefix",
    # execution section
    "execution.automation_backend": "automation_backend",
    "execution.max_test_steps": "max_test_steps",
    "execution.step_timeout": "step_timeout",
    "execution.max_retries_per_step": "max_retries_per_step",
    "execution.screenshot_quality": "screenshot_quality",
    "execution.actions_max_turns": "actions_computer_tool_max_turns",
    "execution.actions_loop_detection_window": "actions_computer_tool_loop_detection_window",
    "execution.actions_action_timeout_ms": "actions_computer_tool_action_timeout_ms",
    "execution.actions_stabilization_wait_ms": "actions_computer_tool_stabilization_wait_ms",
    "execution.actions_fail_fast_on_safety": "actions_computer_tool_fail_fast_on_safety",
    "execution.actions_allowed_domains": "actions_computer_tool_allowed_domains",
    "execution.actions_blocked_domains": "actions_computer_tool_blocked_domains",
    "execution.scroll_turn_multiplier": "scroll_turn_multiplier",
    "execution.scroll_default_magnitude": "scroll_default_magnitude",
    "execution.scroll_max_magnitude": "scroll_max_magnitude",
    # logging section
    "logging.level": "log_level",
    "logging.format": "log_format",
    "logging.file": "log_file",
    "logging.model_log_path": "model_log_path",
    "logging.max_screenshots": "max_screenshots",
    # storage section
    "storage.data_dir": "data_dir",
    "storage.reports_dir": "reports_dir",
    "storage.screenshots_dir": "screenshots_dir",
    "storage.cache_dir": "cache_dir",
    "storage.task_plan_cache_path": "task_plan_cache_path",
    "storage.planning_cache_path": "planning_cache_path",
    "storage.situational_cache_path": "situational_cache_path",
    "storage.execution_replay_cache_path": "execution_replay_cache_path",
    # cache section
    "cache.enable_planning": "enable_planning_cache",
    "cache.enable_situational": "enable_situational_cache",
    "cache.enable_execution_replay": "enable_execution_replay_cache",
    # security section
    "security.rate_limit_enabled": "rate_limit_enabled",
    "security.rate_limit_requests_per_minute": "rate_limit_requests_per_minute",
    "security.sanitize_screenshots": "sanitize_screenshots",
    # dev section
    "dev.debug_mode": "debug_mode",
    "dev.save_agent_conversations": "save_agent_conversations",
    "dev.haindy_home": "haindy_home",
}

# Reverse mapping: Settings field name -> "section.json_key"
_FIELD_TO_JSON: dict[str, str] = {v: k for k, v in _JSON_TO_FIELD.items()}


_SETTINGS_SKELETON: dict[str, Any] = {
    "agent": {
        "provider": "openai",
    },
    "computer_use": {
        "model": "gpt-5.4",
    },
    "execution": {
        "automation_backend": "desktop",
    },
    "logging": {
        "level": "INFO",
    },
}


def ensure_settings_skeleton(path: Path) -> bool:
    """Create a minimal settings file at *path* if one does not already exist.

    Returns True if a new file was created, False if it already existed.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_SETTINGS_SKELETON, indent=2), encoding="utf-8")
    return True


def load_settings_file(path: Path) -> dict[str, Any]:
    """Load a settings JSON file and return the raw nested dict.

    Returns an empty dict if the file does not exist.
    Raises ``ValueError`` if the file is malformed or contains secret fields.
    """
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Settings file {path} contains invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Settings file {path} must contain a JSON object at the top level"
        )

    _reject_secret_fields(data, path)
    return data


_PROVIDER_MODEL_FIELD: dict[str, str] = {
    "openai": "computer_use_model",
    "google": "google_cu_model",
    "anthropic": "anthropic_cu_model",
}


def flatten_settings_dict(nested: dict[str, Any]) -> dict[str, Any]:
    """Convert the nested settings JSON structure to the flat dict ``Settings`` expects.

    ``computer_use.model`` is routed to the provider-specific model field based on
    ``computer_use.provider`` in the same dict. The ``agent_models`` section passes
    through as-is. Unknown section keys are silently ignored (forward-compatibility).
    """
    flat: dict[str, Any] = {}

    for section, contents in nested.items():
        if section == "agent_models":
            flat["agent_models"] = contents
            continue

        if not isinstance(contents, dict):
            continue

        for json_key, value in contents.items():
            mapping_key = f"{section}.{json_key}"
            field_name = _JSON_TO_FIELD.get(mapping_key)
            if field_name is not None:
                flat[field_name] = value

    # Route computer_use.model to the correct provider-specific field.
    # This must run after the full dict is built so provider is resolved first.
    if "computer_use_model" in flat:
        provider = str(flat.get("cu_provider", "openai")).strip().lower()
        target_field = _PROVIDER_MODEL_FIELD.get(provider, "computer_use_model")
        if target_field != "computer_use_model":
            flat[target_field] = flat.pop("computer_use_model")

    return flat


def write_settings_file(path: Path, data: dict[str, Any]) -> None:
    """Write (or merge into) a settings file at *path*.

    Merges *data* into any existing content rather than overwriting it.
    Creates the parent directory if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    _deep_merge(existing, data)
    path.write_text(
        json.dumps(existing, indent=2, default=_json_default), encoding="utf-8"
    )


def flat_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat ``{field_name: value}`` dict back to the nested JSON structure.

    Useful for building a settings file from parsed env vars during migration.
    Unknown field names (not in ``_FIELD_TO_JSON``) are silently dropped.
    """
    nested: dict[str, Any] = {}

    for field_name, value in flat.items():
        json_path = _FIELD_TO_JSON.get(field_name)
        if json_path is None:
            if field_name == "agent_models":
                nested["agent_models"] = value
            continue

        section, key = json_path.split(".", 1)
        if section not in nested:
            nested[section] = {}
        nested[section][key] = value

    return nested


def _reject_secret_fields(data: dict[str, Any], path: Path) -> None:
    """Raise ValueError if any secret key names appear anywhere in *data*."""
    for section_value in data.values():
        if isinstance(section_value, dict):
            for key in section_value:
                if key in _SECRET_FIELD_NAMES:
                    raise ValueError(
                        f"Settings file {path} contains a secret field '{key}'. "
                        "API keys must not be stored in settings files. "
                        "Use 'haindy auth login <provider>' instead."
                    )
    # Also check top-level keys (in case someone puts them there directly)
    for key in data:
        if key in _SECRET_FIELD_NAMES:
            raise ValueError(
                f"Settings file {path} contains a secret field '{key}'. "
                "API keys must not be stored in settings files. "
                "Use 'haindy auth login <provider>' instead."
            )


def _json_default(obj: Any) -> Any:
    """Fallback serializer for types that json.dumps cannot handle natively."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
