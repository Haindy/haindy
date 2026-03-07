"""Anthropic Computer Use regression tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import src.agents.computer_use.session as cu_session_module
from src.agents.computer_use import ComputerUseExecutionError
from src.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import (
    DummyResponse,
    make_anthropic_client,
    make_session,
)

pytest_plugins = ("tests.computer_use_session_support",)


@pytest.mark.asyncio
async def test_computer_use_session_anthropic_provider_executes_action(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"

    initial_response = DummyResponse(
        {
            "id": "msg_1",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "computer",
                    "input": {"action": "left_click", "coordinate": [250, 180]},
                }
            ],
        }
    )
    final_response = DummyResponse(
        {
            "id": "msg_2",
            "content": [{"type": "text", "text": "Action completed successfully."}],
        }
    )
    create = AsyncMock(side_effect=[initial_response, final_response])
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
        anthropic_client=make_anthropic_client(create),
    )
    result = await session.run(
        goal="Click the primary action button.",
        initial_screenshot=b"initial_png_bytes",
        metadata={"step_number": 1},
    )

    assert len(result.actions) == 1
    assert result.actions[0].action_type == "click"
    assert result.actions[0].status == "executed"
    assert result.final_output == "Action completed successfully."
    mock_browser.click.assert_awaited_once_with(250, 180, button="left", click_count=1)
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_computer_use_session_anthropic_failure_does_not_fallback_to_openai(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"
    create = AsyncMock(side_effect=RuntimeError("anthropic failed"))

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
        anthropic_client=make_anthropic_client(create),
    )
    session._run_openai = AsyncMock()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="anthropic failed"):
        await session.run(
            goal="Do something",
            initial_screenshot=b"initial",
            metadata={"step_number": 1},
        )

    session._run_openai.assert_not_awaited()  # type: ignore[attr-defined]


def test_computer_use_session_anthropic_client_uses_api_key(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "anthropic-key"

    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(cu_session_module, "_AsyncAnthropic", client_factory)

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
    )
    session._ensure_anthropic_client()

    client_factory.assert_called_once_with(api_key="anthropic-key")


@pytest.mark.asyncio
async def test_computer_use_session_anthropic_requires_api_key(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = ""
    monkeypatch.setattr(cu_session_module, "_AsyncAnthropic", MagicMock())

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
    )

    with pytest.raises(
        ComputerUseExecutionError,
        match="Anthropic CU provider requires ANTHROPIC_API_KEY",
    ):
        await session.run(
            goal="Observe",
            initial_screenshot=b"initial_png_bytes",
            metadata={"step_number": 1},
        )


@pytest.mark.asyncio
async def test_anthropic_error_tool_result_uses_text_only_content(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
        anthropic_client=object(),
    )

    failed_turn = ComputerToolTurn(
        call_id="toolu_error_1",
        action_type="click",
        parameters={"type": "click"},
        response_id="msg_1",
        pending_safety_checks=[],
        status="failed",
        error_message="click failed",
    )

    payload, _ = await session._build_anthropic_follow_up_request(
        history_messages=[
            {"role": "user", "content": [{"type": "text", "text": "go"}]}
        ],
        previous_response={"content": []},
        turns=[failed_turn],
        metadata={},
        model="claude-sonnet-4-6",
    )

    tool_result = payload["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("is_error") is True
    assert all(part.get("type") == "text" for part in tool_result["content"])
    assert payload["max_tokens"] == 16384


@pytest.mark.asyncio
async def test_anthropic_follow_up_adds_shared_grounding_text_and_preserves_turn_snapshot(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
        anthropic_client=object(),
    )

    successful_turn = ComputerToolTurn(
        call_id="toolu_success_1",
        action_type="click",
        parameters={"type": "click"},
        response_id="msg_1",
        pending_safety_checks=[],
        status="executed",
        metadata={
            "screenshot_base64": "stored_snapshot",
            "current_url": "https://stale.example.com",
            "x": 11,
            "y": 22,
        },
    )

    payload, screenshot_bytes = await session._build_anthropic_follow_up_request(
        history_messages=[
            {"role": "user", "content": [{"type": "text", "text": "go"}]}
        ],
        previous_response={"content": []},
        turns=[successful_turn],
        metadata={"interaction_mode": "observe_only"},
        model="claude-sonnet-4-6",
    )

    content = payload["messages"][-1]["content"]
    assert content[0]["type"] == "tool_result"
    assert content[0]["content"][0]["type"] == "image"
    assert content[0]["content"][0]["source"]["data"] == "ZmFrZV9wbmdfYnl0ZXM="
    assert content[1]["text"].startswith('current_url="https://example.com"\n')
    assert (
        'call_id="toolu_success_1" action_index=1 action="click" status="executed" x=11 y=22'
        in (content[1]["text"])
    )
    assert "Observe-only mode is active" in content[2]["text"]
    assert successful_turn.metadata["screenshot_base64"] == "stored_snapshot"
    assert successful_turn.metadata["current_url"] == "https://stale.example.com"
    assert screenshot_bytes == b"fake_png_bytes"


@pytest.mark.asyncio
async def test_anthropic_computer_use_calls_do_not_pass_request_timeout(
    mock_client, mock_browser, session_settings
):
    create = AsyncMock(return_value=DummyResponse({"id": "msg_1", "content": []}))
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="anthropic",
        anthropic_client=make_anthropic_client(create),
    )

    payload = {"model": "claude-sonnet-4-6", "messages": []}
    await session._create_anthropic_response(payload)

    create.assert_awaited_once_with(**payload)
