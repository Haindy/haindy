"""
Tests for the Computer Use session orchestration.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.computer_use import ComputerUseSession
import src.agents.computer_use.session as cu_session_module
from src.agents.computer_use.session import ComputerUseSessionResult


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
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        desktop_coordinate_cache_path=tmp_path / "coords.json",
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
                        {"type": "output_text", "text": "Action completed successfully."}
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
    final_response = DummyResponse(
        {
            "id": "resp_obs_2",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Policy rejection acknowledged."}
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
    mock_browser.click.assert_not_called()


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


@pytest.mark.asyncio
async def test_computer_use_session_per_call_fallback_is_not_sticky(
    mock_client, mock_browser, session_settings
):
    """Google fallback to OpenAI should apply only to one run() call."""
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"

    session = ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        debug_logger=None,
        provider="google",
    )

    google_success = ComputerUseSessionResult(final_output="google-success")
    openai_fallback = ComputerUseSessionResult(final_output="openai-fallback")
    session._run_google = AsyncMock(  # type: ignore[assignment]
        side_effect=[RuntimeError("google failed"), google_success]
    )
    session._run_openai = AsyncMock(return_value=openai_fallback)  # type: ignore[assignment]

    first = await session.run(
        goal="Run once",
        initial_screenshot=b"initial",
        metadata={"step_number": 1},
    )
    second = await session.run(
        goal="Run twice",
        initial_screenshot=b"initial",
        metadata={"step_number": 2},
    )

    assert first.final_output == "openai-fallback"
    assert second.final_output == "google-success"
    assert session._provider == "google"
    assert session._run_google.await_count == 2  # type: ignore[attr-defined]
    assert session._run_openai.await_count == 1  # type: ignore[attr-defined]


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
