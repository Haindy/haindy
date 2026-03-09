"""Google Computer Use helper and follow-up tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.agents.computer_use.session as cu_session_module
from src.agents.computer_use import ComputerUseSession
from src.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import make_session

pytest_plugins = ("tests.computer_use_session_support",)


def test_computer_use_session_google_client_uses_vertex_project_location(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    session_settings.vertex_project = "demo-project"
    session_settings.vertex_location = "us-east5"
    session_settings.vertex_api_key = ""
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
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
    session_settings.cu_provider = "google"
    session_settings.vertex_project = "demo-project"
    session_settings.vertex_location = "us-east5"
    session_settings.vertex_api_key = "vertex-key"
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
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
    session_settings.cu_provider = "google"
    session_settings.vertex_project = ""
    session_settings.vertex_api_key = "vertex-key"
    client_factory = MagicMock(return_value=object())
    monkeypatch.setattr(
        cu_session_module, "genai", SimpleNamespace(Client=client_factory)
    )

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
    )
    session._ensure_google_client()

    client_factory.assert_called_once_with(api_key="vertex-key")


def test_extract_google_pending_safety_checks() -> None:
    checks = ComputerUseSession._extract_google_pending_safety_checks(
        {
            "x": 500,
            "y": 538,
            "safety_decision": {
                "decision": "require_confirmation",
                "explanation": "Please confirm sign out.",
            },
        }
    )

    assert checks == [
        {
            "decision": "require_confirmation",
            "code": "require_confirmation",
            "message": "Please confirm sign out.",
        }
    ]


@pytest.mark.asyncio
async def test_google_follow_up_adds_safety_acknowledgement_and_call_id(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    turn = ComputerToolTurn(
        call_id="click_at",
        action_type="click_at",
        parameters={"x": 500, "y": 538},
        status="executed",
        pending_safety_checks=[
            {
                "decision": "require_confirmation",
                "code": "require_confirmation",
                "message": "Please confirm sign out.",
            }
        ],
        metadata={"google_function_call_id": "call_google_1"},
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Confirm sign out.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_result = payload["input"][0]
    assert payload["api_surface"] == "interactions"
    assert payload["previous_interaction_id"] == "int_prev"
    assert function_result["type"] == "function_result"
    assert function_result["call_id"] == "call_google_1"
    assert function_result["result"]["safety_acknowledgement"] is True
    assert function_result["result"]["current_url"] == "https://example.com"
    assert function_result["result"]["items"][0]["type"] == "text"
    assert (
        'google_function_call_id="call_google_1"'
        in function_result["result"]["items"][0]["text"]
    )
    assert function_result["result"]["items"][-1]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_google_follow_up_omits_function_response_id_without_google_call_id(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    turn = ComputerToolTurn(
        call_id="google_turn_1_call_2",
        action_type="click_at",
        parameters={"x": 500, "y": 538},
        status="executed",
        pending_safety_checks=[],
        metadata={
            "google_function_call_sequence": 2,
            "google_correlation_mode": "sequence_fallback",
            "google_function_call_fallback_id": "google_turn_1_call_2",
        },
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Tap settings.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_result = payload["input"][0]
    assert function_result["call_id"] == "google_turn_1_call_2"
    assert function_result["result"]["current_url"] == "https://example.com"
    call_text = function_result["result"]["items"][0]["text"]
    assert 'call_id="google_turn_1_call_2"' in call_text
    assert "google_function_call_sequence=2" in call_text
    assert 'google_correlation_mode="sequence_fallback"' in call_text
    assert 'google_function_call_fallback_id="google_turn_1_call_2"' in call_text


@pytest.mark.asyncio
async def test_google_follow_up_uses_original_google_function_call_name(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    turn = ComputerToolTurn(
        call_id="key_press",
        action_type="keypress",
        parameters={"key": "enter"},
        status="executed",
        pending_safety_checks=[],
        metadata={"google_function_call_name": "key_press"},
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Press enter.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_result = payload["input"][0]
    assert function_result["name"] == "key_press"


@pytest.mark.asyncio
async def test_google_follow_up_preserves_rich_grounding_fields(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    turn = ComputerToolTurn(
        call_id="call_google_rich",
        action_type="click_at",
        parameters={"x": 120, "y": 340},
        status="failed",
        pending_safety_checks=[],
        error_message="click failed",
        metadata={
            "x": 120,
            "y": 340,
            "clipboard_text": "copied value",
            "clipboard_truncated": False,
            "clipboard_error": "clipboard unavailable",
            "google_function_call_name": "click_at",
            "google_function_call_sequence": 3,
            "google_correlation_mode": "provider_id",
            "google_function_call_id": "google_call_3",
        },
    )

    payload, _, screenshot_bytes = await session._build_google_follow_up_request(
        goal="Tap the button.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_result = payload["input"][0]
    assert function_result["call_id"] == "google_call_3"
    assert function_result["name"] == "click_at"
    assert function_result["is_error"] is True
    assert function_result["result"]["status"] == "failed"
    assert function_result["result"]["current_url"] == "https://example.com"
    call_text = function_result["result"]["items"][0]["text"]
    assert 'call_id="call_google_rich"' in call_text
    assert "x=120" in call_text
    assert "y=340" in call_text
    assert 'clipboard_text="copied value"' in call_text
    assert "clipboard_truncated=false" in call_text
    assert 'clipboard_error="clipboard unavailable"' in call_text
    assert 'error="click failed"' in call_text
    assert "google_function_call_sequence=3" in call_text
    assert 'google_correlation_mode="provider_id"' in call_text
    assert 'google_function_call_id="google_call_3"' in call_text
    assert function_result["result"]["items"][-1]["type"] == "image"
    assert screenshot_bytes == b"fake_png_bytes"
