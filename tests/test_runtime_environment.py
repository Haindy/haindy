"""Tests for canonical runtime environment helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from haindy.runtime.environment import (
    coordinate_cache_path_for_environment,
    normalize_automation_backend,
    normalize_runtime_environment_name,
    normalize_target_type,
    resolve_runtime_environment,
    resolve_runtime_environment_from_context,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("desktop", "desktop"),
        ("linux", "desktop"),
        ("browser", "browser"),
        ("web", "browser"),
        ("mobile", "mobile_adb"),
        ("android", "mobile_adb"),
        ("unknown", "desktop"),
        ("", "desktop"),
    ],
)
def test_normalize_runtime_environment_name(raw_value: str, expected: str) -> None:
    assert normalize_runtime_environment_name(raw_value) == expected


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("desktop", "desktop"),
        ("browser", "desktop"),
        ("web", "desktop"),
        ("mobile", "mobile_adb"),
        ("android", "mobile_adb"),
        ("unknown", "desktop"),
    ],
)
def test_normalize_automation_backend_aliases(
    raw_value: str,
    expected: str,
) -> None:
    assert normalize_automation_backend(raw_value) == expected


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("desktop_app", "desktop_app"),
        ("desktop", "desktop_app"),
        ("browser", "web"),
        ("web", "web"),
        ("mobile", "mobile_adb"),
        ("android", "mobile_adb"),
        ("unknown", "desktop_app"),
    ],
)
def test_normalize_target_type_aliases(raw_value: str, expected: str) -> None:
    assert normalize_target_type(raw_value) == expected


def test_resolve_runtime_environment_prefers_mobile_when_backend_and_target_disagree() -> (
    None
):
    resolved = resolve_runtime_environment(
        automation_backend="mobile_adb",
        target_type="web",
    )
    assert resolved.name == "mobile_adb"
    assert resolved.automation_backend == "mobile_adb"
    assert resolved.target_type == "mobile_adb"


def test_resolve_runtime_environment_from_context_uses_target_type_for_browser() -> (
    None
):
    resolved = resolve_runtime_environment_from_context(
        {
            "automation_backend": "desktop",
            "target_type": "browser",
        }
    )
    assert resolved.name == "browser"
    assert resolved.automation_backend == "desktop"
    assert resolved.target_type == "web"


def test_coordinate_cache_path_for_environment_uses_mobile_path_for_mobile() -> None:
    settings = SimpleNamespace(
        linux_coordinate_cache_path=Path("data/linux_cache/coordinates.json"),
        mobile_coordinate_cache_path=Path("data/mobile_cache/coordinates.json"),
    )

    assert coordinate_cache_path_for_environment(settings, "browser") == Path(
        "data/linux_cache/coordinates.json"
    )
    assert coordinate_cache_path_for_environment(settings, "android") == Path(
        "data/mobile_cache/coordinates.json"
    )
