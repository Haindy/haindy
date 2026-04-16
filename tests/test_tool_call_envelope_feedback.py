"""Tests for feedback URL injection on tool-call envelopes."""

from __future__ import annotations

import json

import pytest

from haindy.feedback import OPT_OUT_ENV_VAR
from haindy.tool_call_mode.cli import _attach_feedback_url, run_tool_call_cli
from haindy.tool_call_mode.models import (
    CommandStatus,
    ExitReason,
    make_envelope,
)


def _envelope(*, status: CommandStatus, exit_reason: ExitReason):
    return make_envelope(
        session_id="sess-1",
        run_id="run-42",
        command="act",
        status=status,
        response="Click intended element but nothing happened",
        screenshot_path=None,
        exit_reason=exit_reason,
        duration_ms=10,
        actions_taken=1,
    )


def test_success_envelope_has_no_feedback_url() -> None:
    env = _envelope(status=CommandStatus.SUCCESS, exit_reason=ExitReason.COMPLETED)
    _attach_feedback_url(env)
    assert env.feedback_url is None


def test_failure_envelope_populates_feedback_url() -> None:
    env = _envelope(
        status=CommandStatus.FAILURE,
        exit_reason=ExitReason.ELEMENT_NOT_FOUND,
    )
    _attach_feedback_url(env)
    assert env.feedback_url is not None
    assert "github.com/Haindy/haindy/issues/new" in env.feedback_url
    assert "element_not_found" in env.feedback_url


def test_error_envelope_populates_feedback_url() -> None:
    env = _envelope(status=CommandStatus.ERROR, exit_reason=ExitReason.AGENT_ERROR)
    _attach_feedback_url(env)
    assert env.feedback_url is not None


def test_opt_out_suppresses_feedback_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPT_OUT_ENV_VAR, "1")
    env = _envelope(
        status=CommandStatus.FAILURE,
        exit_reason=ExitReason.ELEMENT_NOT_FOUND,
    )
    _attach_feedback_url(env)
    assert env.feedback_url is None


@pytest.mark.asyncio
async def test_missing_session_envelope_reports_actual_command(capsys) -> None:
    exit_code = await run_tool_call_cli(
        ["--session", "does-not-exist-xyz", "act", "click submit"]
    )
    assert exit_code == 3

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "act"
    assert payload["status"] == "error"
    assert payload["feedback_url"] is not None


@pytest.mark.asyncio
async def test_usage_error_envelope_reports_actual_command(capsys) -> None:
    exit_code = await run_tool_call_cli(["test"])
    assert exit_code == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "test"
