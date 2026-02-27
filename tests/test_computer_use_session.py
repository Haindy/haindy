"""
Tests for the Computer Use session orchestration.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.agents.computer_use.session as cu_session_module
from src.agents.computer_use import ComputerUseExecutionError, ComputerUseSession
from src.core.enhanced_types import ComputerToolTurn


class DummyResponse:
    """Simple stand-in for OpenAI Responses objects."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


@pytest.fixture
def session_settings(tmp_path):
    """Provide minimal settings required by the session."""
    return SimpleNamespace(
        openai_request_timeout_seconds=900,
        actions_computer_tool_action_timeout_ms=5000,
        actions_computer_tool_stabilization_wait_ms=0,
        actions_computer_tool_max_turns=5,
        actions_computer_tool_loop_detection_window=3,
        actions_computer_tool_fail_fast_on_safety=True,
        actions_computer_tool_allowed_domains=[],
        actions_computer_tool_blocked_domains=[],
        scroll_turn_multiplier=3.0,
        scroll_default_magnitude=450,
        scroll_max_magnitude=600,
        cu_provider="openai",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-2.5-computer-use-preview-10-2025",
        anthropic_api_key="",
        anthropic_cu_model="claude-sonnet-4-6",
        anthropic_cu_beta="computer-use-2025-11-24",
        anthropic_cu_max_tokens=16384,
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        desktop_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
        max_screenshots=12,
    )


@pytest.fixture
def mock_browser():
    """Create a browser driver mock that satisfies the session contract."""
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (1024, 768)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.click.return_value = None
    browser.type_text.return_value = None
    browser.press_key.return_value = None
    browser.scroll_by_pixels.return_value = None
    browser.wait.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"
    return browser


@pytest.fixture
def mock_client():
    """Mock AsyncOpenAI client."""
    client = MagicMock()
    client.responses.create = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_computer_use_session_executes_actions_successfully(
    mock_client, mock_browser, session_settings
):
    """Verify that the session executes a simple click action and captures results."""
    initial_response = DummyResponse(
        {
            "id": "resp_1",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "action": {"type": "click", "x": 250, "y": 180},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    final_response = DummyResponse(
        {
            "id": "resp_2",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Action completed successfully.",
                        }
                    ],
                }
            ],
        }
    )

    mock_client.responses.create.side_effect = [initial_response, final_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
    )

    metadata = {
        "step_number": 1,
        "test_plan_name": "Plan A",
        "test_case_name": "Case 1",
    }

    result = await session.run(
        goal="Click the primary action button.",
        initial_screenshot=b"initial_png_bytes",
        metadata=metadata,
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


@pytest.mark.asyncio
async def test_computer_use_session_blocks_actions_in_observe_mode(
    mock_client, mock_browser, session_settings
):
    """Ensure observe-only mode prevents the Computer Use tool from mutating state."""
    initial_response = DummyResponse(
        {
            "id": "resp_obs_1",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_obs",
                    "action": {"type": "click", "x": 10, "y": 20},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    mock_client.responses.create.side_effect = [initial_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
    )

    metadata = {
        "step_number": 2,
        "test_plan_name": "Plan Observe",
        "test_case_name": "Case Observe",
        "safety_identifier": "test-observe-mode",
    }

    result = await session.run(
        goal="Verify state without interaction.",
        initial_screenshot=b"initial_bytes",
        metadata=metadata,
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
    """Ensure the session aborts when pending safety checks are returned."""
    session_settings.actions_computer_tool_fail_fast_on_safety = True

    safety_response = DummyResponse(
        {
            "id": "resp_safe",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_safe",
                    "action": {"type": "click", "x": 10, "y": 20},
                    "pending_safety_checks": [
                        {
                            "id": "sc1",
                            "code": "malicious_instructions",
                            "message": "Potential malicious action detected.",
                        }
                    ],
                    "status": "completed",
                }
            ],
        }
    )

    mock_client.responses.create.side_effect = [safety_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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
    """Pending safety checks should continue when fail-fast is disabled and policy is auto_approve."""
    session_settings.actions_computer_tool_fail_fast_on_safety = False
    session_settings.cu_safety_policy = "auto_approve"

    first_response = DummyResponse(
        {
            "id": "resp_safe_continue_1",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_safe_continue",
                    "action": {"type": "click", "x": 11, "y": 22},
                    "pending_safety_checks": [
                        {
                            "id": "sc1",
                            "code": "review_required",
                            "message": "Safety review requested.",
                        }
                    ],
                    "status": "completed",
                }
            ],
        }
    )
    second_response = DummyResponse(
        {
            "id": "resp_safe_continue_2",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }
    )
    mock_client.responses.create.side_effect = [first_response, second_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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
    assert follow_up_payload["input"][0]["acknowledged_safety_checks"] == [
        {
            "id": "sc1",
            "code": "review_required",
            "message": "Safety review requested.",
        }
    ]


@pytest.mark.asyncio
async def test_computer_use_session_safety_auto_approve_with_override_when_fail_fast_enabled(
    mock_client, mock_browser, session_settings
):
    """Safety auto-approve override should acknowledge checks even with fail-fast enabled."""
    session_settings.actions_computer_tool_fail_fast_on_safety = True
    session_settings.cu_safety_policy = "auto_approve"

    first_response = DummyResponse(
        {
            "id": "resp_safe_override_1",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_safe_override",
                    "action": {"type": "click", "x": 13, "y": 24},
                    "pending_safety_checks": [
                        {
                            "id": "sc_override",
                            "code": "review_required",
                            "message": "Approval override required.",
                        }
                    ],
                    "status": "completed",
                }
            ],
        }
    )
    second_response = DummyResponse(
        {
            "id": "resp_safe_override_2",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done with override."}],
                }
            ],
        }
    )
    mock_client.responses.create.side_effect = [first_response, second_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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
    assert follow_up_payload["input"][0]["acknowledged_safety_checks"] == [
        {
            "id": "sc_override",
            "code": "review_required",
            "message": "Approval override required.",
        }
    ]


@pytest.mark.asyncio
async def test_computer_use_session_google_failure_does_not_fallback_to_openai(
    mock_client, mock_browser, session_settings
):
    """Google provider failures should be explicit and never fallback to OpenAI."""
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
        provider="google",
    )

    session._run_google = AsyncMock(side_effect=RuntimeError("google failed"))  # type: ignore[assignment]
    session._run_openai = AsyncMock()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="google failed"):
        await session.run(
            goal="Run once",
            initial_screenshot=b"initial",
            metadata={"step_number": 1},
        )

    assert session._provider == "google"
    assert session._run_google.await_count == 1  # type: ignore[attr-defined]
    session._run_openai.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_computer_use_session_marks_terminal_failure_on_max_turns(
    mock_client, mock_browser, session_settings
):
    """OpenAI max-turn exits should emit terminal failed state."""
    session_settings.actions_computer_tool_max_turns = 1

    first_response = DummyResponse(
        {
            "id": "resp_turn_1",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_turn_1",
                    "action": {"type": "click", "x": 1, "y": 1},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    second_response = DummyResponse(
        {
            "id": "resp_turn_2",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_turn_2",
                    "action": {"type": "click", "x": 2, "y": 2},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    mock_client.responses.create.side_effect = [first_response, second_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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
    """Validate that stateful actions are rejected outside the configured allowlist."""
    session_settings.actions_computer_tool_allowed_domains = ["example.com"]
    mock_browser.get_page_url.return_value = "https://unauthorized.org/page"

    initial_response = DummyResponse(
        {
            "id": "resp_domain",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_domain",
                    "action": {"type": "click", "x": 0, "y": 0},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    final_response = DummyResponse(
        {
            "id": "resp_domain_2",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Domain policy acknowledged."}
                    ],
                }
            ],
        }
    )

    mock_client.responses.create.side_effect = [initial_response, final_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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


@pytest.mark.asyncio
async def test_computer_use_session_records_execution_failure(
    mock_client, mock_browser, session_settings
):
    """If browser execution fails, the session should capture the error and continue."""
    mock_browser.click.side_effect = RuntimeError("click failed")

    initial_response = DummyResponse(
        {
            "id": "resp_err",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_err",
                    "action": {"type": "click", "x": 40, "y": 60},
                    "pending_safety_checks": [],
                    "status": "completed",
                }
            ],
        }
    )
    recovery_response = DummyResponse(
        {
            "id": "resp_final",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Could not click the button."}
                    ],
                }
            ],
        }
    )

    mock_client.responses.create.side_effect = [initial_response, recovery_response]

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
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


@pytest.mark.asyncio
async def test_computer_use_session_anthropic_provider_executes_action(
    mock_client, mock_browser, session_settings
):
    """Anthropic provider should execute translated computer actions via the driver."""
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
    anthropic_client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        anthropic_client=anthropic_client,
        debug_logger=None,
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
    """Anthropic provider failures should be explicit and never fallback to OpenAI."""
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"
    create = AsyncMock(side_effect=RuntimeError("anthropic failed"))
    anthropic_client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        anthropic_client=anthropic_client,
        debug_logger=None,
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
    """Anthropic client should initialize with ANTHROPIC_API_KEY."""
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "anthropic-key"

    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(cu_session_module, "_AsyncAnthropic", client_factory)

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        debug_logger=None,
    )
    session._ensure_anthropic_client()

    client_factory.assert_called_once_with(api_key="anthropic-key")


@pytest.mark.asyncio
async def test_computer_use_session_anthropic_requires_api_key(
    mock_client, mock_browser, session_settings, monkeypatch
):
    """Anthropic provider should fail fast when API key is missing."""
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = ""
    monkeypatch.setattr(cu_session_module, "_AsyncAnthropic", MagicMock())

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        debug_logger=None,
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
    """Anthropic rejects is_error tool_result blocks that include non-text content."""
    session_settings.cu_provider = "anthropic"
    session_settings.anthropic_api_key = "test-key"

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        anthropic_client=object(),
        debug_logger=None,
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
        model="claude-sonnet-4-6",
    )

    tool_result = payload["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("is_error") is True
    assert all(part.get("type") == "text" for part in tool_result["content"])
    assert payload["max_tokens"] == 16384


@pytest.mark.asyncio
async def test_openai_computer_use_calls_do_not_pass_request_timeout(
    mock_client, mock_browser, session_settings
):
    """OpenAI CU requests should rely on provider/client defaults for timeouts."""
    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="openai",
        debug_logger=None,
    )

    payload = {"model": "computer-use-preview", "input": "hello"}
    await session._create_response(payload)

    mock_client.responses.create.assert_awaited_once_with(**payload)


@pytest.mark.asyncio
async def test_anthropic_computer_use_calls_do_not_pass_request_timeout(
    mock_client, mock_browser, session_settings
):
    """Anthropic CU requests should rely on provider/client defaults for timeouts."""
    create = AsyncMock(return_value=DummyResponse({"id": "msg_1", "content": []}))
    anthropic_client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="anthropic",
        anthropic_client=anthropic_client,
        debug_logger=None,
    )

    payload = {"model": "claude-sonnet-4-6", "messages": []}
    await session._create_anthropic_response(payload)

    create.assert_awaited_once_with(**payload)


def test_computer_use_session_google_client_uses_vertex_project_location(
    mock_client, mock_browser, session_settings, monkeypatch
):
    """Google client should consume VERTEX_PROJECT and VERTEX_LOCATION when configured."""
    session_settings.cu_provider = "google"
    session_settings.vertex_project = "demo-project"
    session_settings.vertex_location = "us-east5"
    session_settings.vertex_api_key = ""
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="google",
        debug_logger=None,
    )
    session._ensure_google_client()

    client_factory.assert_called_once_with(
        vertexai=True,
        project="demo-project",
        location="us-east5",
    )


def test_computer_use_session_google_client_ignores_api_key_in_vertex_mode(
    mock_client, mock_browser, session_settings, monkeypatch
):
    """Google client should ignore API key when Vertex project/location mode is used."""
    session_settings.cu_provider = "google"
    session_settings.vertex_project = "demo-project"
    session_settings.vertex_location = "us-east5"
    session_settings.vertex_api_key = "vertex-key"
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="google",
        debug_logger=None,
    )
    session._ensure_google_client()

    client_factory.assert_called_once_with(
        vertexai=True,
        project="demo-project",
        location="us-east5",
    )


def test_computer_use_session_google_client_uses_api_key_mode(
    mock_client, mock_browser, session_settings, monkeypatch
):
    """Google client should use API-key mode when no project is provided."""
    session_settings.cu_provider = "google"
    session_settings.vertex_project = ""
    session_settings.vertex_api_key = "vertex-key"
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider="google",
        debug_logger=None,
    )
    session._ensure_google_client()

    client_factory.assert_called_once_with(api_key="vertex-key")
