"""Journal model tests for provider-neutral coordinate metadata."""

from uuid import uuid4

from haindy.journal.models import ExecutionJournal, ExecutionMode, JournalEntry


def test_journal_entry_supports_coordinate_metadata() -> None:
    entry = JournalEntry(
        test_scenario="Search flow",
        step_reference="Step 1",
        action_taken="Click search",
        coordinate_metadata={
            "target_reference": "search_input",
            "pixel_coordinates": (500, 220),
            "adjusted": False,
        },
        expected_result="Input focused",
        actual_result="Input focused",
        success=True,
    )

    data = entry.model_dump()
    assert data["coordinate_metadata"]["target_reference"] == "search_input"
    assert data["coordinate_metadata"]["pixel_coordinates"] == (500, 220)


def test_execution_journal_summary_updates_from_entries() -> None:
    journal = ExecutionJournal(test_plan_id=uuid4(), test_name="Smoke")

    journal.add_entry(
        JournalEntry(
            test_scenario="A",
            step_reference="1",
            action_taken="Click",
            expected_result="Ok",
            actual_result="Ok",
            success=True,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=120,
        )
    )
    journal.add_entry(
        JournalEntry(
            test_scenario="A",
            step_reference="2",
            action_taken="Type",
            expected_result="Ok",
            actual_result="Fail",
            success=False,
            execution_mode=ExecutionMode.SCRIPTED,
            execution_time_ms=80,
        )
    )

    summary = journal.get_summary()
    assert journal.total_steps == 2
    assert journal.successful_steps == 1
    assert journal.failed_steps == 1
    assert summary["execution_modes"]["visual"] == 1
    assert summary["execution_modes"]["scripted"] == 1
    assert summary["success_rate"] == 0.5
