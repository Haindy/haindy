"""Tests for HAINDY's system dependency checker."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

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
