import json
from pathlib import Path

import pytest

from haindy.utils.model_logging import ModelCallLogger, log_model_call_failure


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


@pytest.mark.asyncio
async def test_model_call_logger_suppresses_model_dump_warnings(
    tmp_path: Path,
) -> None:
    class DummyResponse:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def model_dump(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"ok": True}

    log_path = tmp_path / "model_calls.jsonl"
    logger = ModelCallLogger(log_path)
    response = DummyResponse()

    await logger.log_call(
        agent="test-agent",
        model="test-model",
        prompt="hello",
        request_payload={},
        response=response,
        screenshots=None,
        metadata={},
    )

    assert response.calls == [{"warnings": "none"}]


@pytest.mark.asyncio
async def test_model_call_logger_records_failure_outcome(tmp_path: Path) -> None:
    class ProviderError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("provider rejected request")
            self.status_code = 500
            self.code = "internal_error"
            self.body = {"error": {"message": "boom"}}

    log_path = tmp_path / "model_calls.jsonl"
    logger = ModelCallLogger(log_path)

    await log_model_call_failure(
        logger,
        agent="test-agent",
        model="test-model",
        prompt="hello",
        request_payload={"api_key": "hunter2"},
        exception=ProviderError(),
        response={"error": {"message": "boom"}},
        metadata={"note": "failed"},
    )

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["outcome"] == "failure"
    assert entry["failure_kind"] == "provider_http_error"
    assert entry["error"]["type"] == "ProviderError"
    assert entry["error"]["status_code"] == 500
    assert entry["error"]["provider_code"] == "internal_error"
    assert entry["response"]["error"]["message"] == "boom"
    assert entry["metadata"]["note"] == "failed"


@pytest.mark.asyncio
async def test_model_call_logger_suppresses_retryable_rate_limit_failures(
    tmp_path: Path,
) -> None:
    class RateLimitError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("429 RESOURCE_EXHAUSTED")
            self.status_code = 429
            self.code = "RESOURCE_EXHAUSTED"
            self.body = {"error": {"status": "RESOURCE_EXHAUSTED"}}

    log_path = tmp_path / "model_calls.jsonl"
    logger = ModelCallLogger(log_path)

    logged = await log_model_call_failure(
        logger,
        agent="test-agent",
        model="test-model",
        prompt="hello",
        request_payload={},
        exception=RateLimitError(),
        metadata={},
    )

    assert logged is False
    assert not log_path.exists() or log_path.read_text(encoding="utf-8") == ""
