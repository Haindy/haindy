"""JournalManager tests for provider-neutral action journaling."""

from pathlib import Path

import pytest

from haindy.config.settings import build_project_data_dir, get_settings
from haindy.core.types import TestPlan, TestStep
from haindy.journal.manager import JournalManager
from haindy.journal.models import ExecutionMode, JournalActionResult


def _plan() -> TestPlan:
    step = TestStep(
        step_number=1,
        description="Click submit",
        action="Click submit",
        expected_result="Submitted",
    )
    return TestPlan(
        name="Plan",
        description="Desc",
        requirements_source="REQ",
        test_cases=[],
        steps=[step],
    )


def test_default_journal_dir_uses_settings_data_root(
    monkeypatch, tmp_path: Path
) -> None:
    cwd = tmp_path / "repo"
    home = tmp_path / "haindy-home"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        "haindy.config.settings._merge_runtime_env",
        lambda: {"HAINDY_HOME": str(home)},
    )
    get_settings.cache_clear()

    try:
        manager = JournalManager()

        assert manager.journal_dir == build_project_data_dir(home, cwd) / "journals"
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_record_action_uses_coordinate_metadata_in_action_description(
    tmp_path,
) -> None:
    manager = JournalManager(journal_dir=tmp_path)
    plan = _plan()
    journal = await manager.create_journal(plan)
    step = plan.steps[0]

    entry = await manager.record_action(
        journal_id=journal.journal_id,
        test_scenario="Scenario",
        step=step,
        action_result=JournalActionResult(
            success=True,
            confidence=0.92,
            coordinate_metadata={
                "target_reference": "submit_button",
                "pixel_coordinates": (840, 610),
                "adjusted": True,
            },
            actual_outcome="Clicked",
        ),
        execution_mode=ExecutionMode.VISUAL,
        execution_time_ms=42,
    )

    assert "submit_button" in entry.action_taken
    assert "(840, 610)" in entry.action_taken
    assert "(adjusted)" in entry.action_taken


@pytest.mark.asyncio
async def test_finalize_journal_removes_active_journal(tmp_path) -> None:
    manager = JournalManager(journal_dir=tmp_path)
    plan = _plan()
    journal = await manager.create_journal(plan)

    finalized = await manager.finalize_journal(journal.journal_id)
    assert finalized.end_time is not None

    loaded = await manager.get_journal(journal.journal_id)
    assert loaded is not None
