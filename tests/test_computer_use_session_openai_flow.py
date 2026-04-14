"""OpenAI Computer Use regression tests."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

import haindy.agents.computer_use.session as cu_session_module
from haindy.agents.computer_use import ComputerUseExecutionError
from haindy.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import (
    make_session,
    openai_computer_call,
    openai_message,
    openai_response,
)

pytest_plugins = ("tests.computer_use_session_support",)


@pytest.mark.asyncio
async def test_computer_use_session_uses_task_only_initial_request_before_screenshot_follow_up(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_1",
            [openai_computer_call("call_1", {"type": "screenshot"})],
        ),
        openai_response("resp_2", [openai_message("Screen inspected.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Check whether the filters panel is open.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 1},
    )

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "screenshot"
    assert result.actions[0].status == "executed"
    initial_payload = mock_client.responses.create.await_args_list[0].kwargs
    assert initial_payload["model"] == "gpt-5.4"
    assert initial_payload["tools"] == [{"type": "computer"}]
    assert isinstance(initial_payload["input"], str)
    assert "Check whether the filters panel is open." in initial_payload["input"]

    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert follow_up_payload["previous_response_id"] == "resp_1"
    assert len(follow_up_payload["input"]) == 1
    assert follow_up_payload["input"][0]["type"] == "computer_call_output"
    assert follow_up_payload["input"][0]["output"] == {
        "type": "computer_screenshot",
        "image_url": "data:image/png;base64,ZmFrZV9wbmdfYnl0ZXM=",
        "detail": "original",
    }


@pytest.mark.asyncio
async def test_computer_use_session_executes_actions_successfully(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_1",
            [openai_computer_call("call_1", {"type": "click", "x": 250, "y": 180})],
        ),
        openai_response("resp_2", [openai_message("Action completed successfully.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click the primary action button.",
        initial_screenshot=b"initial_png_bytes",
        metadata={
            "step_number": 1,
            "test_plan_name": "Plan A",
            "test_case_name": "Case 1",
        },
    )

    assert len(result.actions) == 1
    turn = result.actions[0]
    assert turn.status == "executed"
    assert turn.action_type == "click"
    mock_browser.click.assert_awaited_once_with(250, 180, button="left", click_count=1)
    assert result.final_output == "Action completed successfully."
    assert not result.safety_events
    assert result.terminal_status == "success"
    assert mock_client.responses.create.await_count == 2
    initial_payload = mock_client.responses.create.await_args_list[0].kwargs
    assert initial_payload["model"] == "gpt-5.4"
    assert initial_payload["tools"] == [{"type": "computer"}]
    assert isinstance(initial_payload["input"], str)
    assert "Click the primary action button." in initial_payload["input"]

    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert follow_up_payload["tools"] == [{"type": "computer"}]
    assert len(follow_up_payload["input"]) == 1
    assert follow_up_payload["input"][0]["type"] == "computer_call_output"
    assert "actions" not in follow_up_payload["input"][0]
    assert "current_url" not in follow_up_payload["input"][0]
    assert follow_up_payload["input"][0]["output"] == {
        "type": "computer_screenshot",
        "image_url": "data:image/png;base64,ZmFrZV9wbmdfYnl0ZXM=",
        "detail": "original",
    }


@pytest.mark.asyncio
async def test_computer_use_session_executes_drag_and_drop_with_destination_aliases(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_drag_1",
            [
                openai_computer_call(
                    "drag_and_drop",
                    {
                        "type": "drag_and_drop",
                        "x": 0,
                        "y": 0,
                        "destination_x": 300,
                        "destination_y": 200,
                    },
                )
            ],
        ),
        openai_response("resp_drag_2", [openai_message("Dragged successfully.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Drag the item to the target.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 3},
    )

    assert len(result.actions) == 1
    turn = result.actions[0]
    assert turn.status == "executed"
    assert turn.action_type == "drag"
    assert turn.metadata.get("normalized_action_type") == "drag"
    mock_browser.drag_mouse.assert_awaited_once_with(0, 0, 300, 200, steps=1)


@pytest.mark.asyncio
async def test_computer_use_session_logs_rejected_drag_with_raw_action_metadata(
    mock_client, mock_browser, session_settings, monkeypatch
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_drag_fail_1",
            [
                openai_computer_call(
                    "drag_and_drop",
                    {"type": "drag_and_drop", "x": 10, "y": 20},
                )
            ],
        ),
        openai_response(
            "resp_drag_fail_2", [openai_message("Could not drag the item.")]
        ),
    ]
    warning_mock = MagicMock()
    monkeypatch.setattr(cu_session_module.logger, "warning", warning_mock)

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Drag the item.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 4},
    )

    assert len(result.actions) == 1
    turn = result.actions[0]
    assert turn.status == "failed"
    assert turn.error_message == "Drag action missing destination coordinates."

    rejection_logs = [
        call
        for call in warning_mock.call_args_list
        if call.args and call.args[0] == "Computer Use action rejected"
    ]
    assert rejection_logs
    extra = rejection_logs[0].kwargs["extra"]
    assert extra["action_type"] == "drag"
    assert extra["raw_action_type"] == "drag_and_drop"
    assert extra["action_keys"] == ["type", "x", "y"]


@pytest.mark.asyncio
async def test_computer_use_session_records_execution_failure(
    mock_client, mock_browser, session_settings
):
    mock_browser.click.side_effect = RuntimeError("click failed")
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_err",
            [openai_computer_call("call_err", {"type": "click", "x": 40, "y": 60})],
        ),
        openai_response("resp_final", [openai_message("Could not click the button.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click the retry button.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 2},
    )

    assert len(result.actions) == 1
    turn = result.actions[0]
    assert turn.status == "failed"
    assert turn.error_message == "click failed"
    assert result.final_output == "Could not click the button."
    assert mock_client.responses.create.await_count == 2
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert len(follow_up_payload["input"]) == 1
    assert follow_up_payload["input"][0]["output"] == {
        "type": "computer_screenshot",
        "image_url": "data:image/png;base64,ZmFrZV9wbmdfYnl0ZXM=",
        "detail": "original",
    }


@pytest.mark.asyncio
async def test_computer_use_session_runs_all_actions_in_a_returned_batch(
    mock_client, mock_browser, session_settings
):
    mock_browser.click.side_effect = RuntimeError("click failed")
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_batch_fail_1",
            [
                openai_computer_call(
                    "call_batch",
                    [
                        {"type": "click", "x": 10, "y": 20},
                        {"type": "type", "text": "openai"},
                    ],
                )
            ],
        ),
        openai_response("resp_batch_fail_2", [openai_message("Batch attempted.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click the field and type the query.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 8},
    )

    assert [turn.action_type for turn in result.actions] == ["click", "type"]
    assert [turn.status for turn in result.actions] == ["failed", "executed"]
    mock_browser.type_text.assert_awaited_once_with("openai")
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert len(follow_up_payload["input"]) == 1


@pytest.mark.asyncio
async def test_computer_use_session_executes_batched_actions_in_single_call(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_batch_1",
            [
                openai_computer_call(
                    "call_batch",
                    [
                        {"type": "click", "x": 10, "y": 20},
                        {"type": "type", "text": "openai"},
                    ],
                )
            ],
        ),
        openai_response("resp_batch_2", [openai_message("Batch completed.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Click the field and type the query.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 8},
    )

    assert result.terminal_status == "success"
    assert [turn.action_type for turn in result.actions] == ["click", "type"]
    mock_browser.click.assert_awaited_once_with(10, 20, button="left", click_count=1)
    mock_browser.type_text.assert_awaited_once_with("openai")
    assert mock_client.responses.create.await_count == 2
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    assert len(follow_up_payload["input"]) == 1
    assert (
        len(
            [
                item
                for item in follow_up_payload["input"]
                if item.get("type") == "computer_call_output"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_computer_use_session_processes_multiple_computer_calls_in_one_turn(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_multi_1",
            [
                openai_computer_call("call_click", {"type": "click", "x": 5, "y": 6}),
                openai_computer_call("call_type", {"type": "type", "text": "done"}),
            ],
        ),
        openai_response("resp_multi_2", [openai_message("All actions completed.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    result = await session.run(
        goal="Execute both computer calls.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 9},
    )

    assert result.terminal_status == "success"
    assert [turn.call_id for turn in result.actions] == ["call_click", "call_type"]
    follow_up_payload = mock_client.responses.create.await_args_list[1].kwargs
    outputs = [
        item
        for item in follow_up_payload["input"]
        if item.get("type") == "computer_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["call_click", "call_type"]


@pytest.mark.asyncio
async def test_openai_computer_use_calls_do_not_pass_request_timeout(
    mock_client, mock_browser, session_settings
):
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="openai",
    )

    payload = {"model": "gpt-5.4", "input": "hello"}
    await session._create_response(payload)

    mock_client.responses.create.assert_awaited_once_with(**payload)


def test_openai_session_rejects_legacy_preview_model(
    mock_client, mock_browser, session_settings
):
    session_settings.computer_use_model = "computer-use-preview"

    with pytest.raises(ValueError, match="computer-use-preview"):
        make_session(
            mock_client=mock_client,
            mock_browser=mock_browser,
            session_settings=session_settings,
            provider="openai",
        )


@pytest.mark.asyncio
async def test_openai_follow_up_uses_fresh_batch_capture_and_preserves_turn_snapshot_metadata(
    mock_client, mock_browser, session_settings
):
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="openai",
    )
    turn = ComputerToolTurn(
        call_id="call_snapshot",
        action_type="click",
        parameters={"type": "click"},
        status="executed",
        metadata={
            "screenshot_base64": "stored_snapshot",
            "current_url": "https://stale.example.com",
            "x": 42,
            "y": 24,
        },
    )

    payload, _ = await session._build_follow_up_request(
        previous_response_id="resp_prev",
        calls=[[turn]],
        metadata={},
        model="gpt-5.4",
    )

    assert (
        payload["input"][0]["output"]["image_url"]
        == "data:image/png;base64,ZmFrZV9wbmdfYnl0ZXM="
    )
    assert payload["input"][0]["output"]["detail"] == "original"
    assert turn.metadata["screenshot_base64"] == "stored_snapshot"
    assert turn.metadata["current_url"] == "https://stale.example.com"
    assert len(payload["input"]) == 1


@pytest.mark.asyncio
async def test_openai_follow_up_requests_in_loop_localization_on_full_keyframe(
    mock_client, mock_browser, session_settings
):
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="openai",
    )
    turn = ComputerToolTurn(
        call_id="call_localize",
        action_type="click",
        parameters={"type": "click"},
        status="executed",
    )

    payload, batch = await session._build_follow_up_request(
        previous_response_id="resp_prev",
        calls=[[turn]],
        metadata={"target": "Email"},
        model="gpt-5.4",
    )

    assert batch.request_localization is True
    assert batch.localization_reason == "missing_session_cartography"
    assert len(payload["input"]) == 2
    assert payload["input"][1]["role"] == "user"
    assert (
        "Refresh the session cartography now"
        in payload["input"][1]["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_openai_step_actions_reuse_previous_response_chain(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_1",
            [openai_computer_call("call_1", {"type": "click", "x": 12, "y": 24})],
        ),
        openai_response("resp_2", [openai_message("Clicked.")]),
        openai_response(
            "resp_3",
            [openai_computer_call("call_2", {"type": "type", "text": "done"})],
        ),
        openai_response("resp_4", [openai_message("Typed.")]),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    session.begin_step_scope()

    first = await session.execute_step_action(
        goal="Click the first control.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 1},
    )
    second = await session.execute_step_action(
        goal="Type done into the field.",
        initial_screenshot=None,
        metadata={"step_number": 1},
    )

    assert first.response_ids == ["resp_1", "resp_2"]
    assert second.response_ids == ["resp_3", "resp_4"]
    assert session.step_response_ids == ["resp_1", "resp_2", "resp_3", "resp_4"]

    continuation_payload = mock_client.responses.create.await_args_list[2].kwargs
    assert continuation_payload["previous_response_id"] == "resp_2"
    assert continuation_payload["tools"] == [{"type": "computer"}]
    assert continuation_payload["input"][0]["role"] == "user"
    assert (
        continuation_payload["input"][0]["content"][0]["text"]
        == "Type done into the field.\n\nContext:\n- Step number: 1"
    )


@pytest.mark.asyncio
async def test_openai_step_reflection_uses_json_output_without_tools(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = [
        openai_response(
            "resp_1",
            [openai_computer_call("call_1", {"type": "click", "x": 5, "y": 6})],
        ),
        openai_response("resp_2", [openai_message("Clicked.")]),
        openai_response(
            "resp_3",
            [
                openai_message(
                    '{"verdict":"PASS","reasoning":"ok","actual_result":"done","confidence":0.9,"is_blocker":false,"blocker_reasoning":""}'
                )
            ],
        ),
    ]

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )
    session.begin_step_scope()
    await session.execute_step_action(
        goal="Click the primary action.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 2},
    )

    reflection = await session.reflect_step(
        prompt="Respond with valid JSON for the step verdict.",
        metadata={"step_number": 2},
    )

    payload = mock_client.responses.create.await_args_list[2].kwargs
    assert payload["previous_response_id"] == "resp_2"
    assert "tools" not in payload
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert payload["input"][0]["content"][0]["text"].startswith(
        "Respond with valid JSON"
    )
    assert reflection["response_ids"] == ["resp_3"]
    assert '"verdict":"PASS"' in reflection["raw_text"]


@pytest.mark.asyncio
async def test_openai_step_reflection_uses_action_timeout_budget(
    mock_client,
    mock_browser,
    session_settings,
    monkeypatch: pytest.MonkeyPatch,
):
    session_settings.actions_computer_tool_action_timeout_seconds = 0.5
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )

    async def _hang(**kwargs: object) -> dict[str, object]:
        del kwargs
        await asyncio.sleep(1)
        return {}

    monkeypatch.setattr(session, "_reflect_openai_step", _hang)

    with pytest.raises(
        ComputerUseExecutionError,
        match="Computer Use step reflection timed out after",
    ):
        await session.reflect_step(
            prompt="Respond with valid JSON for the step verdict.",
            metadata={"step_number": 2},
        )


@pytest.mark.asyncio
async def test_openai_step_reflection_honors_remaining_test_budget(
    mock_client,
    mock_browser,
    session_settings,
    monkeypatch: pytest.MonkeyPatch,
):
    session_settings.actions_computer_tool_action_timeout_seconds = 5.0
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )

    async def _hang(**kwargs: object) -> dict[str, object]:
        del kwargs
        await asyncio.sleep(1)
        return {}

    monkeypatch.setattr(session, "_reflect_openai_step", _hang)

    with pytest.raises(
        ComputerUseExecutionError,
        match="Computer Use step reflection timed out after 0.1 seconds.",
    ):
        await session.reflect_step(
            prompt="Respond with valid JSON for the step verdict.",
            metadata={
                "step_number": 2,
                "remaining_test_budget_seconds": 0.1,
            },
        )


@pytest.mark.asyncio
async def test_openai_computer_use_logs_failed_request_attempt(
    mock_client, mock_browser, session_settings
):
    mock_client.responses.create.side_effect = RuntimeError("openai transport failed")

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
    )

    with pytest.raises(RuntimeError, match="openai transport failed"):
        await session._create_response(
            {"model": "gpt-5.4", "input": "hello", "tools": [{"type": "computer"}]},
            agent="computer_use.openai.initial",
            prompt="Click the button",
            request_payload_for_log={"request": "sanitized"},
            metadata={"environment": "desktop"},
        )

    entry = json.loads(
        session_settings.model_log_path.read_text(encoding="utf-8").strip()
    )
    assert entry["outcome"] == "failure"
    assert entry["agent"] == "computer_use.openai.initial"
    assert entry["metadata"]["provider"] == "openai"
    assert entry["metadata"]["environment"] == "desktop"
