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
        history=[],
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_response = payload["contents"][0].parts[0].function_response
    assert function_response is not None
    assert function_response.id == "call_google_1"
    assert function_response.response["safety_acknowledgement"] == "true"


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
        history=[],
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_response = payload["contents"][0].parts[0].function_response
    assert function_response is not None
    assert function_response.id is None
    assert function_response.response["call_id"] == "google_turn_1_call_2"
    assert function_response.response["google_function_call_sequence"] == 2
    assert function_response.response["google_correlation_mode"] == "sequence_fallback"
    assert (
        function_response.response["google_function_call_fallback_id"]
        == "google_turn_1_call_2"
    )


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
        history=[],
        turns=[turn],
        metadata={},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_response = payload["contents"][0].parts[0].function_response
    assert function_response is not None
    assert function_response.name == "key_press"
