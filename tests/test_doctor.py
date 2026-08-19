"""Tests for HAINDY's system dependency checker."""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import patch

from haindy.cli.doctor import _check_python_version

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
