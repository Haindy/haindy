"""Artifact retention tests for TestRunner screenshots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from haindy.agents.test_runner_artifacts import TestRunnerArtifacts
from haindy.monitoring.debug_logger import DebugLogger


def test_test_runner_artifacts_preserve_all_run_screenshots_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    import haindy.monitoring.debug_logger as debug_logger_module

    debug_logger = DebugLogger("test-run")
    monkeypatch.setattr(debug_logger_module, "_debug_logger", debug_logger)

    artifacts = TestRunnerArtifacts(
        SimpleNamespace(max_screenshots=None),
        automation_driver=None,
    )

    path_one = artifacts.save_screenshot(b"shot-one", "first")
    path_two = artifacts.save_screenshot(b"shot-two", "second")
    path_three = artifacts.save_screenshot(b"shot-three", "third")

    assert path_one.exists()
    assert path_two.exists()
    assert path_three.exists()
    assert len(list((tmp_path / "debug_screenshots" / "test-run").glob("*.png"))) == 3
