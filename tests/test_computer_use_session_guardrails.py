"""Guardrail and policy tests for Computer Use sessions."""

from __future__ import annotations

from collections import deque

import pytest

from src.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import (
    make_session,
    openai_computer_call,
    openai_message,
    openai_response,
)

pytest_plugins = ("tests.computer_use_session_support",)


@pytest.mark.asyncio
async def test_computer_use_session_blocks_actions_in_observe_mode(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_obs_1",
            [openai_computer_call("call_obs", {"type": "click", "x": 10, "y": 20})],
        )
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Verify state without interaction.",
        initial_screenshot=b"initial_bytes",
        metadata={
            "step_number": 2,
            "test_plan_name": "Plan Observe",
            "test_case_name": "Case Observe",
            "safety_identifier": "test-observe-mode",
        },
        allowed_actions={"screenshot"},
    )

    assert len(result.actions) == 1
    turn = result.actions[0]
    assert turn.status == "failed"
    assert "observe-only" in (turn.error_message or "")
    assert turn.metadata.get("policy") == "observe_only"
    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "observe_only_policy_violation"
    mock_browser.click.assert_not_called()
    mock_client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_computer_use_session_fail_fast_on_safety(
    mock_client, mock_browser, session_settings
):
    session_settings.actions_computer_tool_fail_fast_on_safety = True
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_safe",
            [
                openai_computer_call(
                    "call_safe",
                    {"type": "click", "x": 10, "y": 20},
                    pending_safety_checks=[
                        {
                            "id": "sc1",
                            "code": "malicious_instructions",
                            "message": "Potential malicious action detected.",
                        }
                    ],
                )
            ],
        )
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click the dangerous button.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 5},
    )

    assert len(result.actions) == 1
    assert result.actions[0].status == "failed"
    assert result.safety_events
    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "safety_fail_fast"
    assert mock_browser.click.await_count == 0
    mock_client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_computer_use_session_safety_auto_approve_when_fail_fast_disabled(
    mock_client, mock_browser, session_settings
):
    session_settings.actions_computer_tool_fail_fast_on_safety = False
    session_settings.cu_safety_policy = "auto_approve"
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_safe_continue_1",
            [
                openai_computer_call(
                    "call_safe_continue",
                    {"type": "click", "x": 11, "y": 22},
                    pending_safety_checks=[
                        {
                            "id": "sc1",
                            "code": "review_required",
                            "message": "Safety review requested.",
                        }
                    ],
                )
            ],
        ),
        openai_response("resp_safe_continue_2", [openai_message("Done.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click continue.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 6},
    )

    assert result.safety_events == []
    assert result.terminal_status == "success"
    mock_browser.click.assert_awaited_once_with(11, 22, button="left", click_count=1)
    assert result.actions[0].acknowledged is True
    assert result.actions[0].metadata.get("acknowledged_safety_checks") == [
        {
            "id": "sc1",
            "code": "review_required",
            "message": "Safety review requested.",
        }
    ]
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert follow_up_payload["tools"] == [{"type": "computer"}]
    assert len(follow_up_payload["input"]) == 1
    assert follow_up_payload["input"][0]["acknowledged_safety_checks"] == [
        {
            "id": "sc1",
            "code": "review_required",
            "message": "Safety review requested.",
        }
    ]
    assert "current_url" not in follow_up_payload["input"][0]
    assert follow_up_payload["input"][0]["output"]["detail"] == "original"


@pytest.mark.asyncio
async def test_computer_use_session_safety_auto_approve_with_override_when_fail_fast_enabled(
    mock_client, mock_browser, session_settings
):
    session_settings.actions_computer_tool_fail_fast_on_safety = True
    session_settings.cu_safety_policy = "auto_approve"
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_safe_override_1",
            [
                openai_computer_call(
                    "call_safe_override",
                    {"type": "click", "x": 13, "y": 24},
                    pending_safety_checks=[
                        {
                            "id": "sc_override",
                            "code": "review_required",
                            "message": "Approval override required.",
                        }
                    ],
                )
            ],
        ),
        openai_response(
            "resp_safe_override_2", [openai_message("Done with override.")]
        ),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click continue with override.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 7, "allow_safety_auto_approve": True},
    )

    assert result.terminal_status == "success"
    mock_browser.click.assert_awaited_once_with(13, 24, button="left", click_count=1)
    assert result.actions[0].acknowledged is True
    assert result.actions[0].metadata.get("acknowledged_safety_checks") == [
        {
            "id": "sc_override",
            "code": "review_required",
            "message": "Approval override required.",
        }
    ]
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert follow_up_payload["tools"] == [{"type": "computer"}]
    assert len(follow_up_payload["input"]) == 1
    assert follow_up_payload["input"][0]["acknowledged_safety_checks"] == [
        {
            "id": "sc_override",
            "code": "review_required",
            "message": "Approval override required.",
        }
    ]
    assert "current_url" not in follow_up_payload["input"][0]
    assert follow_up_payload["input"][0]["output"]["detail"] == "original"


@pytest.mark.asyncio
async def test_computer_use_session_marks_terminal_failure_on_max_turns(
    mock_client, mock_browser, session_settings
):
    session_settings.actions_computer_tool_max_turns = 1
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_turn_1",
            [openai_computer_call("call_turn_1", {"type": "click", "x": 1, "y": 1})],
        ),
        openai_response(
            "resp_turn_2",
            [openai_computer_call("call_turn_2", {"type": "click", "x": 2, "y": 2})],
        ),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click repeatedly.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 10},
    )

    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "max_turns_exceeded"
    assert any(
        action.action_type == "system_notice" and action.status == "failed"
        for action in result.actions
    )


@pytest.mark.asyncio
async def test_computer_use_session_enforces_domain_allowlist(
    mock_client, mock_browser, session_settings
):
    session_settings.actions_computer_tool_allowed_domains = ["example.com"]
    mock_browser.get_page_url.return_value = "https://unauthorized.org/page"
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_domain",
            [openai_computer_call("call_domain", {"type": "click", "x": 0, "y": 0})],
        ),
        openai_response(
            "resp_domain_2", [openai_message("Domain policy acknowledged.")]
        ),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Attempt action on unauthorized domain.",
        initial_screenshot=b"initial",
        metadata={"safety_identifier": "domain-test"},
    )

    turn = result.actions[0]
    assert turn.status == "failed"
    assert "allowlist" in (turn.error_message or "")
    assert turn.metadata.get("policy") == "rejected"
    mock_browser.click.assert_not_called()


def test_loop_detection_ignores_mixed_actions_with_unchanged_screen_hash(
    mock_client, mock_browser, session_settings
):
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=3)

    screenshot_turn = ComputerToolTurn(
        call_id="call_1",
        action_type="screenshot",
        parameters={"type": "screenshot"},
        status="executed",
        metadata={"screenshot_base64": "same"},
    )
    click_turn = ComputerToolTurn(
        call_id="call_2",
        action_type="click",
        parameters={"type": "click", "x": 10, "y": 20},
        status="executed",
        metadata={"screenshot_base64": "same"},
    )
    wait_turn = ComputerToolTurn(
        call_id="call_3",
        action_type="wait",
        parameters={"type": "wait"},
        status="executed",
        metadata={"screenshot_base64": "same"},
    )

    assert session._update_loop_history(screenshot_turn, history, 3) is None
    assert session._update_loop_history(click_turn, history, 3) is None
    assert session._update_loop_history(wait_turn, history, 3) is None


def test_loop_detection_still_detects_repeated_wait_actions(
    mock_client, mock_browser, session_settings
):
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    history: deque[tuple[tuple[str, ...], str]] = deque(maxlen=3)

    turns = [
        ComputerToolTurn(
            call_id=f"call_wait_{index}",
            action_type="wait",
            parameters={"type": "wait"},
            status="executed",
            metadata={"screenshot_base64": "same"},
        )
        for index in range(3)
    ]

    assert session._update_loop_history(turns[0], history, 3) is None
    assert session._update_loop_history(turns[1], history, 3) is None
    loop_detection = session._update_loop_history(turns[2], history, 3)
    assert loop_detection is not None
    assert "wait" in loop_detection["message"]
