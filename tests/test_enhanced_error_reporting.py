"""Simple HTML reporter tests for enhanced bug context rendering."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.core.types import TestStatus
from src.monitoring.simple_html_reporter import SimpleHTMLReporter


def test_prepare_template_data_includes_coordinate_context_and_screenshots() -> None:
    reporter = SimpleHTMLReporter()

    bug = SimpleNamespace(
        step_number=2,
        error_message="Click failed",
        attempted_action="Click submit",
        expected_outcome="Submitted",
        actual_outcome="No change",
        severity="high",
        failure_type="execution",
        detailed_error="Element not clickable",
        confidence_scores={"validation": 0.8, "execution": 0.2},
        target_reference="submit_button",
        coordinates_used=SimpleNamespace(pixel_coordinates=(800, 500), relative_x=0.5, relative_y=0.5),
        screenshot_before=b"before_png",
        screenshot_after=b"after_png",
        ui_anomalies=["Button disabled"],
        suggested_fixes=["Wait for button enablement"],
        url_before="https://example.com/form",
        url_after="https://example.com/form",
        page_title_before="Form",
        page_title_after="Form",
    )

    step_result = SimpleNamespace(
        status="failed",
        step_number=2,
        step_description="Submit form",
        action_taken="Click submit",
        actual_result="No change",
        execution_mode="enhanced",
    )

    test_case = SimpleNamespace(steps_total=1, steps_completed=0, steps_failed=1, step_results=[step_result], name="Case")
    test_report = SimpleNamespace(test_cases=[test_case], bugs=[bug], test_plan_name="Plan")
    test_state = SimpleNamespace(
        test_report=test_report,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        status=TestStatus.FAILED,
    )

    data = reporter._prepare_template_data(test_state, [])
    assert data["bug_reports"]
    first_bug = data["bug_reports"][0]
    assert first_bug["target_reference"] == "submit_button"
    assert len(first_bug["screenshots"]) == 2



def test_generate_report_writes_html_file(tmp_path: Path) -> None:
    reporter = SimpleHTMLReporter()

    test_state = SimpleNamespace(
        test_report=None,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        status=TestStatus.PASSED,
        completed_steps=[],
        failed_steps=[],
        test_plan=SimpleNamespace(name="Plan", steps=[]),
    )

    output = reporter.generate_report(test_state, execution_history=[], output_path=tmp_path / "report.html")
    assert output.exists()
    content = output.read_text()
    assert "Test Execution Report" in content
