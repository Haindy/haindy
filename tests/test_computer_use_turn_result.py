"""Shared Computer Use follow-up batch builder tests."""

from __future__ import annotations

from haindy.agents.computer_use.turn_result import build_follow_up_batch
from haindy.core.enhanced_types import ComputerToolTurn


def test_build_follow_up_batch_groups_actions_and_renders_grounding() -> None:
    click_turn = ComputerToolTurn(
        call_id="call_batch",
        action_type="click",
        parameters={"type": "click"},
        status="executed",
        pending_safety_checks=[
            {
                "decision": "require_confirmation",
                "code": "require_confirmation",
                "message": "Confirm action.",
            }
        ],
        metadata={
            "x": 100,
            "y": 200,
            "acknowledged_safety_checks": [
                {
                    "id": "sc1",
                    "code": "require_confirmation",
                    "message": "Confirm action.",
                }
            ],
        },
    )
    type_turn = ComputerToolTurn(
        call_id="call_batch",
        action_type="type",
        parameters={"type": "type"},
        status="executed",
    )

    batch = build_follow_up_batch(
        [[click_turn, type_turn]],
        screenshot_bytes=b"fresh_png",
        current_url="https://example.com",
        interaction_mode="observe_only",
    )

    assert len(batch.calls) == 1
    assert len(batch.calls[0].actions) == 2
    assert batch.calls[0].requires_safety_acknowledgement is True
    assert batch.calls[0].acknowledged_safety_checks == [
        {
            "id": "sc1",
            "code": "require_confirmation",
            "message": "Confirm action.",
        }
    ]
    assert batch.current_url == "https://example.com"
    assert batch.interaction_mode == "observe_only"
    assert batch.screenshot_base64 == "ZnJlc2hfcG5n"
    assert batch.reminder_text is not None
    assert "Observe-only mode is active" in batch.reminder_text
    assert batch.grounding_text is not None
    assert batch.grounding_text.startswith('current_url="https://example.com"\n')
    assert (
        'call_id="call_batch" action_index=1 action="click" status="executed" x=100 y=200 safety_acknowledgement=true'
        in (batch.grounding_text)
    )
    assert 'call_id="call_batch" action_index=2 action="type" status="executed"' in (
        batch.grounding_text
    )


def test_build_follow_up_batch_preserves_google_metadata_and_extracts_error() -> None:
    failed_turn = ComputerToolTurn(
        call_id="call_google",
        action_type="click_at",
        parameters={"type": "click_at"},
        status="failed",
        error_message="click failed",
        metadata={
            "x": 12,
            "y": 34,
            "clipboard_text": "copied",
            "clipboard_truncated": False,
            "google_function_call_name": "click_at",
            "google_function_call_sequence": 5,
            "google_correlation_mode": "provider_id",
            "google_function_call_id": "google_call_5",
        },
    )

    batch = build_follow_up_batch(
        [[failed_turn]],
        screenshot_bytes=b"fresh_png",
        current_url="",
        interaction_mode="",
    )

    call_result = batch.calls[0]
    assert batch.current_url == "desktop://"
    assert batch.interaction_mode == ""
    assert batch.error_text == "Execution error: click failed"
    assert call_result.provider_metadata == {
        "google_function_call_id": "google_call_5",
        "google_function_call_name": "click_at",
        "google_function_call_sequence": 5,
        "google_correlation_mode": "provider_id",
    }
    assert batch.grounding_text is not None
    assert batch.grounding_text.startswith('current_url="desktop://"\n')
    assert (
        'call_id="call_google" action_index=1 action="click_at" status="failed" x=12 y=34 clipboard_text="copied" clipboard_truncated=false error="click failed"'
        in (batch.grounding_text)
    )
