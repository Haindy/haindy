"""Canonical runtime environment semantics for execution flows."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RuntimeEnvironmentName = Literal["desktop", "browser", "mobile_adb", "mobile_ios"]
AutomationBackendName = Literal["desktop", "mobile_adb", "mobile_ios"]
TargetTypeName = Literal["desktop_app", "web", "mobile_adb", "mobile_ios"]

_RUNTIME_ENV_ALIASES: dict[str, RuntimeEnvironmentName] = {
    "desktop": "desktop",
    "linux": "desktop",
    "macos": "desktop",
    "mac": "desktop",
    "windows": "desktop",
    "win": "desktop",
    "win32": "desktop",
    "os": "desktop",
    "system": "desktop",
    "browser": "browser",
    "web": "browser",
    "mobile": "mobile_adb",
    "mobile_adb": "mobile_adb",
    "android": "mobile_adb",
    "phone": "mobile_adb",
    "tablet": "mobile_adb",
    "mobile_ios": "mobile_ios",
    "ios": "mobile_ios",
    "iphone": "mobile_ios",
    "ipad": "mobile_ios",
    "apple": "mobile_ios",
}
_TARGET_TYPE_ALIASES: dict[str, TargetTypeName] = {
    "desktop_app": "desktop_app",
    "desktop": "desktop_app",
    "app": "desktop_app",
    "native": "desktop_app",
    "browser": "web",
    "web": "web",
    "website": "web",
    "mobile": "mobile_adb",
    "mobile_adb": "mobile_adb",
    "android": "mobile_adb",
    "phone": "mobile_adb",
    "tablet": "mobile_adb",
    "mobile_ios": "mobile_ios",
    "ios": "mobile_ios",
    "iphone": "mobile_ios",
    "ipad": "mobile_ios",
}


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _match_runtime_environment(value: Any) -> RuntimeEnvironmentName | None:
    return _RUNTIME_ENV_ALIASES.get(_normalize_token(value))


def _match_target_type(value: Any) -> TargetTypeName | None:
    return _TARGET_TYPE_ALIASES.get(_normalize_token(value))


@dataclass(frozen=True)
class RuntimeEnvironmentSpec:
    """Resolved runtime environment with derived backend and target type."""

    name: RuntimeEnvironmentName
    automation_backend: AutomationBackendName
    target_type: TargetTypeName

    @property
    def is_mobile(self) -> bool:
        return self.name in {"mobile_adb", "mobile_ios"}

    @property
    def is_android(self) -> bool:
        return self.name == "mobile_adb"

    @property
    def is_ios(self) -> bool:
        return self.name == "mobile_ios"

    @property
    def is_browser(self) -> bool:
        return self.name == "browser"

    @property
    def coordinate_cache_attribute(self) -> str:
        if self.name == "mobile_ios":
            return "ios_coordinate_cache_path"
        if self.name == "mobile_adb":
            return "mobile_coordinate_cache_path"
        if self.name == "desktop" and sys.platform == "darwin":
            return "macos_coordinate_cache_path"
        if self.name == "desktop" and sys.platform == "win32":
            return "windows_coordinate_cache_path"
        return "linux_coordinate_cache_path"

    @property
    def openai_computer_environment(self) -> str:
        if self.is_browser:
            return "browser"
        if self.is_ios:
            return "mac"
        if self.name == "desktop" and sys.platform == "darwin":
            return "mac"
        if self.name == "desktop" and sys.platform == "win32":
            # TODO(windows-support): verify OpenAI CUA accepts environment="windows"
            # before merging M1; if rejected, fall back to "linux".
            return "windows"
        return "linux"

    @property
    def google_computer_environment_name(self) -> str:
        return (
            "ENVIRONMENT_BROWSER"
            if (self.is_browser or self.is_mobile)
            else "ENVIRONMENT_UNSPECIFIED"
        )

    def coordinate_cache_path(self, settings: Any) -> Path:
        path = getattr(settings, self.coordinate_cache_attribute)
        return Path(path)


def runtime_environment_spec(name: RuntimeEnvironmentName) -> RuntimeEnvironmentSpec:
    if name == "mobile_adb":
        return RuntimeEnvironmentSpec(
            name="mobile_adb",
            automation_backend="mobile_adb",
            target_type="mobile_adb",
        )
    if name == "mobile_ios":
        return RuntimeEnvironmentSpec(
            name="mobile_ios",
            automation_backend="mobile_ios",
            target_type="mobile_ios",
        )
    if name == "browser":
        return RuntimeEnvironmentSpec(
            name="browser",
            automation_backend="desktop",
            target_type="web",
        )
    return RuntimeEnvironmentSpec(
        name="desktop",
        automation_backend="desktop",
        target_type="desktop_app",
    )


def normalize_runtime_environment_name(
    value: Any,
    *,
    default: RuntimeEnvironmentName = "desktop",
) -> RuntimeEnvironmentName:
    return (
        _match_runtime_environment(value)
        or _match_runtime_environment(default)
        or "desktop"
    )


def normalize_automation_backend(
    value: Any,
    *,
    default: AutomationBackendName = "desktop",
) -> AutomationBackendName:
    matched = _match_runtime_environment(value)
    if matched == "mobile_adb":
        return "mobile_adb"
    if matched == "mobile_ios":
        return "mobile_ios"
    if matched in {"desktop", "browser"}:
        return "desktop"
    return default


def normalize_target_type(
    value: Any,
    *,
    default: TargetTypeName = "desktop_app",
) -> TargetTypeName:
    return _match_target_type(value) or _match_target_type(default) or "desktop_app"


def resolve_runtime_environment(
    *,
    environment: Any = None,
    automation_backend: Any = None,
    target_type: Any = None,
    default: RuntimeEnvironmentName = "desktop",
) -> RuntimeEnvironmentSpec:
    explicit_environment = _match_runtime_environment(environment)
    if explicit_environment is not None:
        return runtime_environment_spec(explicit_environment)

    matched_target_type = _match_target_type(target_type)
    matched_backend_environment = _match_runtime_environment(automation_backend)

    if (
        matched_target_type == "mobile_ios"
        or matched_backend_environment == "mobile_ios"
    ):
        return runtime_environment_spec("mobile_ios")
    if (
        matched_target_type == "mobile_adb"
        or matched_backend_environment == "mobile_adb"
    ):
        return runtime_environment_spec("mobile_adb")
    if matched_target_type == "web" or matched_backend_environment == "browser":
        return runtime_environment_spec("browser")
    if matched_backend_environment == "desktop":
        return runtime_environment_spec("desktop")

    return runtime_environment_spec(
        normalize_runtime_environment_name(default, default="desktop")
    )


def resolve_runtime_environment_from_context(
    context: Mapping[str, Any] | None,
    *,
    default: RuntimeEnvironmentName = "desktop",
) -> RuntimeEnvironmentSpec:
    if not isinstance(context, Mapping):
        return runtime_environment_spec(default)
    return resolve_runtime_environment(
        environment=context.get("environment"),
        automation_backend=context.get("automation_backend"),
        target_type=context.get("target_type"),
        default=default,
    )


def coordinate_cache_path_for_environment(settings: Any, environment: Any) -> Path:
    spec = resolve_runtime_environment(environment=environment)
    return spec.coordinate_cache_path(settings)
