import json
from pathlib import Path

import pytest

from src.utils.model_logging import ModelCallLogger


@pytest.mark.asyncio
async def test_model_call_logger_sanitizes_bytes(tmp_path: Path) -> None:
    log_path = tmp_path / "model_calls.jsonl"
    logger = ModelCallLogger(log_path)

    await logger.log_call(
        agent="test-agent",
        model="test-model",
        prompt="hello",
        request_payload={"payload": {"blob": b"abc"}},
        response={"raw": b"xyz"},
        screenshots=[("shot", b"not-a-real-png")],
        metadata={"meta": b"123"},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    assert isinstance(entry["run_id"], str)
    assert entry["run_id"]
    assert entry["request_payload"]["payload"]["blob"] == "<<bytes:3>>"
    assert entry["response"]["raw"] == "<<bytes:3>>"
    assert entry["metadata"]["meta"] == "<<bytes:3>>"
    assert entry["prompt_has_screenshot"] is True
    assert entry["attached_screenshots"][0]["label"] == "shot"

    screenshot_path = Path(entry["attached_screenshots"][0]["path"])
    assert screenshot_path.exists()


@pytest.mark.asyncio
async def test_model_call_logger_prunes_screenshots(tmp_path: Path) -> None:
    log_path = tmp_path / "model_calls.jsonl"
    logger = ModelCallLogger(log_path, max_screenshots=2)

    for index in range(3):
        await logger.log_call(
            agent="test-agent",
            model="test-model",
            prompt=f"hello-{index}",
            request_payload={},
            response={"ok": True},
            screenshots=[(f"shot-{index}", b"png-bytes")],
            metadata={},
        )

    screenshot_dir = log_path.parent / "screenshots"
    pngs = sorted(screenshot_dir.glob("*.png"))
    assert len(pngs) == 2
