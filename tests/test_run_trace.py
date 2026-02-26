import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.types import StepResult, TestStatus, TestStep
from src.runtime.trace import RunTraceWriter, load_model_calls_for_run


def test_run_trace_writer_writes_trace(tmp_path: Path) -> None:
    trace = RunTraceWriter("test_run", trace_dir=tmp_path / "traces")
    trace.set_run_metadata({"provider": "openai"})

    step = TestStep(
        step_number=1,
        description="Do the thing",
        action="Click the button",
        expected_result="Thing is done",
        environment="desktop",
    )
    now = datetime.now(timezone.utc)
    step_result = StepResult(
        step_id=step.step_id,
        step_number=1,
        status=TestStatus.PASSED,
        started_at=now,
        completed_at=now,
        action=step.action,
        expected_result=step.expected_result,
        actual_result="ok",
        screenshot_before="data/screenshots/step_before.png",
        screenshot_after="data/screenshots/step_after.png",
        actions_performed=[{"action_type": "click", "x": 10, "y": 20}],
    )
    trace.record_step(
        scenario_name="test_scenario",
        step=step,
        step_result=step_result,
        attempt=1,
        plan_cache_hit=True,
    )
    trace.record_cache_event({"type": "task_plan_cache_hit", "step": "step_one"})
    trace.finalize(success=True, ended_at="2025-01-01T00:00:00Z")
    trace.write()

    payload = json.loads(Path(trace.path).read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"
    assert payload["trace_version"] == 1
    assert payload["run_metadata"]["provider"] == "openai"
    assert payload["success"] is True
    assert payload["ended_at"] == "2025-01-01T00:00:00Z"
    assert len(payload["steps"]) == 1

    stored_step = payload["steps"][0]
    assert stored_step["scenario"] == "test_scenario"
    assert stored_step["plan_cache_hit"] is True
    assert (
        stored_step["step_result"]["screenshot_before"]
        == "data/screenshots/step_before.png"
    )
    assert (
        stored_step["step_result"]["screenshot_after"]
        == "data/screenshots/step_after.png"
    )
    assert stored_step["step_result"]["actions_performed"][0]["action_type"] == "click"


def test_load_model_calls_for_run_filters_by_run_id(tmp_path: Path) -> None:
    log_path = tmp_path / "model_calls.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"run_id": "A", "agent": "x"}),
                json.dumps({"run_id": "B", "agent": "y"}),
                json.dumps({"run_id": "A", "agent": "z"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_model_calls_for_run(log_path, run_id="A")
    assert [entry["agent"] for entry in entries] == ["x", "z"]
