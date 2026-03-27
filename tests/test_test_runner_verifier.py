"""Prompt-shaping tests for TestRunnerVerifier."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haindy.agents.test_runner_verifier import TestRunnerVerifier
from haindy.core.types import TestCase, TestStep


@pytest.mark.asyncio
async def test_verify_expected_outcome_scopes_prompt_to_current_step() -> None:
    runner = SimpleNamespace(
        call_model=AsyncMock(
            return_value={
                "content": {
                    "verdict": "PASS",
                    "reasoning": "ok",
                    "actual_result": "ok",
                    "confidence": 1.0,
                    "is_blocker": False,
                    "blocker_reasoning": "",
                }
            }
        ),
        _model_logger=SimpleNamespace(log_call=AsyncMock()),
        model="gpt-5.4",
    )
    verifier = TestRunnerVerifier(runner)
    test_case = TestCase(
        test_id="TC002",
        name="Case-insensitive sign-in",
        description=(
            "Verify sign-in succeeds and later confirm the displayed account email "
            "is normalized."
        ),
        steps=[],
    )
    step = TestStep(
        step_number=7,
        description='Tap "Account Settings".',
        action='Tap "Account Settings".',
        expected_result=(
            "The Account Settings screen is displayed and shows the account email "
            "address."
        ),
    )

    await verifier.verify_expected_outcome(
        test_case=test_case,
        step=step,
        action_results=[],
        screenshot_before=None,
        screenshot_after=None,
        execution_history=[],
        next_test_case=None,
    )

    prompt = runner.call_model.await_args.kwargs["messages"][0]["content"][0]["text"]

    assert "Test case description:" not in prompt
    assert (
        "Evaluate ONLY whether this current step achieved its stated expected result."
        in prompt
    )
    assert "Do NOT fail this step based on broader test-case goals" in prompt
