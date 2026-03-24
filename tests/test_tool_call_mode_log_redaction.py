"""Tests for tool-call secret redaction in session-local logs."""

from __future__ import annotations

import json

import pytest

from haindy.monitoring.debug_logger import DebugLogger
from haindy.security.sanitizer import set_literal_redactions
from haindy.utils.model_logging import ModelCallLogger


def test_debug_logger_redacts_runtime_literal_secrets(tmp_path) -> None:
    ai_log_path = tmp_path / "logs" / "ai_interactions.jsonl"
    logger = DebugLogger(
        "tool-call-test",
        debug_dir=tmp_path / "debug",
        reports_dir=tmp_path / "reports",
        ai_log_path=ai_log_path,
    )

    set_literal_redactions(["hunter2"])
    try:
        logger.log_ai_interaction(
            agent_name="ActionAgent",
            action_type="act",
            prompt="Type hunter2 into the password field.",
            response="Typed hunter2 into the password field.",
            additional_context={"note": "hunter2"},
        )
    finally:
        set_literal_redactions([])

    serialized = ai_log_path.read_text(encoding="utf-8")
    assert "hunter2" not in serialized
    assert "[redacted]" in serialized


@pytest.mark.asyncio
async def test_model_call_logger_redacts_runtime_literal_secrets(tmp_path) -> None:
    log_path = tmp_path / "logs" / "model_calls.jsonl"
    logger = ModelCallLogger(log_path)

    set_literal_redactions(["hunter2"])
    try:
        await logger.log_call(
            agent="ActionAgent",
            model="gpt-test",
            prompt="Enter hunter2 in the password field.",
            request_payload={"password": "hunter2"},
            response={"message": "Used hunter2"},
            metadata={"note": "hunter2"},
        )
    finally:
        set_literal_redactions([])

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    serialized = json.dumps(entry)
    assert "hunter2" not in serialized
    assert "[redacted]" in serialized
