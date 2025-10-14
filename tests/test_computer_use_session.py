"""
Tests for the Computer Use session orchestration.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.computer_use import ComputerUseSession


class DummyResponse:
    """Simple stand-in for OpenAI Responses objects."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


@pytest.fixture
def session_settings():
    """Provide minimal settings required by the session."""
    return SimpleNamespace(
        openai_request_timeout_seconds=900,
        actions_computer_tool_action_timeout_ms=5000,
        actions_computer_tool_stabilization_wait_ms=0,
        actions_computer_tool_max_turns=5,
        actions_computer_tool_fail_fast_on_safety=True,
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
        browser=mock_browser,
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
    assert mock_client.responses.create.await_count == 2


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
        browser=mock_browser,
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
    assert mock_browser.click.await_count == 0
    mock_client.responses.create.assert_awaited_once()


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
        browser=mock_browser,
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
