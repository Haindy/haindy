"""Tests for HAINDY's system dependency checker."""

from __future__ import annotations

import io
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text

from haindy.cli import doctor
from haindy.cli.doctor import _check_idb_companion, _check_python_version

_VersionInfo = namedtuple(
    "_VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"]
)


def test_python_311_satisfies_supported_version() -> None:
    version_info = _VersionInfo(3, 11, 0, "final", 0)

    with patch("haindy.cli.doctor.sys.version_info", version_info):
        status, notes = _check_python_version()

    assert status.plain == "OK"
    assert notes == "3.11.0"


def test_python_310_is_reported_as_unsupported() -> None:
    version_info = _VersionInfo(3, 10, 14, "final", 0)

    with patch("haindy.cli.doctor.sys.version_info", version_info):
        status, notes = _check_python_version()

    assert status.plain == "MISSING"
    assert notes == "3.10.14 (need >= 3.11)"


def _make_xcode(root: Path, simulator_kit_parent: str) -> str:
    contents = root / "Xcode.app" / "Contents"
    developer = contents / "Developer"
    developer.mkdir(parents=True)
    (contents / simulator_kit_parent / "SimulatorKit.framework").mkdir(parents=True)
    return str(developer)


def _make_legacy_companion(root: Path) -> str:
    cellar = root / "Cellar" / "idb-companion" / "1.1.8"
    (cellar / "bin").mkdir(parents=True)
    (cellar / "bin" / "idb_companion").touch()
    (cellar / "Frameworks" / "FBControlCore.framework").mkdir(parents=True)
    link = root / "bin" / "idb_companion"
    link.parent.mkdir()
    link.symlink_to(cellar / "bin" / "idb_companion")
    return str(link)


def _make_current_companion(root: Path) -> str:
    libexec = root / "Cellar" / "idb-companion" / "1.6.2" / "libexec"
    (libexec / "Resources").mkdir(parents=True)
    (libexec / "idb_companion").touch()
    return str(libexec / "idb_companion")


def test_idb_companion_missing() -> None:
    status, notes = _check_idb_companion(None, None)

    assert status.plain == "MISSING"
    assert notes == "brew install facebook/fb/idb-companion"


def test_legacy_idb_companion_is_outdated_on_xcode_27(tmp_path: Path) -> None:
    developer_dir = _make_xcode(tmp_path, "SharedFrameworks")
    companion = _make_legacy_companion(tmp_path)

    status, notes = _check_idb_companion(companion, developer_dir)

    assert status.plain == "OUTDATED"
    assert "brew upgrade facebook/fb/idb-companion" in notes


def test_legacy_idb_companion_is_ok_on_xcode_26(tmp_path: Path) -> None:
    developer_dir = _make_xcode(tmp_path, "Developer/Library/PrivateFrameworks")
    companion = _make_legacy_companion(tmp_path)

    status, _ = _check_idb_companion(companion, developer_dir)

    assert status.plain == "OK"


def test_current_idb_companion_is_ok_on_xcode_27(tmp_path: Path) -> None:
    developer_dir = _make_xcode(tmp_path, "SharedFrameworks")
    companion = _make_current_companion(tmp_path)

    status, _ = _check_idb_companion(companion, developer_dir)

    assert status.plain == "OK"


def _status(label: str) -> tuple[Text, str]:
    return Text(label), ""


def _run_doctor_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    *,
    desktop: bool,
    adb: bool,
    companion: str,
    fb_idb: bool,
) -> tuple[int, list[str]]:
    """Run doctor as a macOS host and return the exit code and backend row."""
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    monkeypatch.setattr(doctor, "_check_haindy_installed", lambda: _status("OK"))
    monkeypatch.setattr(doctor, "_check_api_key", lambda _provider: _status("OK"))
    monkeypatch.setattr(doctor, "_check_codex_oauth", lambda: _status("OK"))
    desktop_status = "OK" if desktop else "MISSING"
    monkeypatch.setattr(doctor, "_check_macos_pynput", lambda: _status(desktop_status))
    monkeypatch.setattr(doctor, "_check_macos_mss", lambda: _status("OK"))
    monkeypatch.setattr(doctor, "_check_macos_accessibility", lambda: _status("OK"))
    monkeypatch.setattr(doctor, "_check_macos_screen_recording", lambda: _status("OK"))
    monkeypatch.setattr(
        doctor.shutil,
        "which",
        lambda tool: f"/usr/local/bin/{tool}" if tool == "adb" and adb else None,
    )
    monkeypatch.setattr(doctor, "_xcode_developer_dir", lambda: None)
    monkeypatch.setattr(
        doctor, "_check_idb_companion", lambda _path, _dev: _status(companion)
    )
    fb_idb_status = "OK" if fb_idb else "MISSING"
    monkeypatch.setattr(doctor, "_check_fb_idb_package", lambda: _status(fb_idb_status))
    output = io.StringIO()
    monkeypatch.setattr(
        doctor, "_console", Console(file=output, width=200, color_system=None)
    )

    exit_code = doctor.run_doctor()

    row = next(
        line for line in output.getvalue().splitlines() if "Automation backend" in line
    )
    return exit_code, [cell.strip() for cell in row.split("│")[1:-1]]


def test_ready_idb_counts_as_ios_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code, row = _run_doctor_on_macos(
        monkeypatch, desktop=True, adb=True, companion="OK", fb_idb=True
    )

    assert row == ["Automation backend", "OK", "desktop, android, ios"]
    assert exit_code == 0


def test_ios_only_satisfies_automation_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code, row = _run_doctor_on_macos(
        monkeypatch, desktop=False, adb=False, companion="OK", fb_idb=True
    )

    assert row == ["Automation backend", "OK", "ios"]
    assert exit_code == 0


def test_outdated_idb_companion_is_not_a_ready_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code, row = _run_doctor_on_macos(
        monkeypatch, desktop=False, adb=False, companion="OUTDATED", fb_idb=True
    )

    assert row == [
        "Automation backend",
        "MISSING",
        "Fix desktop deps above, install adb, or install idb",
    ]
    assert exit_code == 1


def test_missing_fb_idb_package_is_not_a_ready_ios_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_code, row = _run_doctor_on_macos(
        monkeypatch, desktop=True, adb=False, companion="OK", fb_idb=False
    )

    assert row == ["Automation backend", "OK", "desktop"]
    assert exit_code == 0


def test_fb_idb_package_detected_via_find_spec() -> None:
    with patch("haindy.cli.doctor.importlib.util.find_spec", return_value=object()):
        status, _ = doctor._check_fb_idb_package()
    assert status.plain == "OK"

    with patch("haindy.cli.doctor.importlib.util.find_spec", return_value=None):
        status, notes = doctor._check_fb_idb_package()
    assert status.plain == "MISSING"
    assert notes == "pip install fb-idb"
