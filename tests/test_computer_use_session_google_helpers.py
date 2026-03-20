"""Google Computer Use helper and follow-up tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    assert function_result["result"]["url"] == "https://example.com"
    text_items = [item for item in payload["input"] if item["type"] == "text"]
    image_items = [item for item in payload["input"] if item["type"] == "image"]
    assert len(text_items) == 1
    assert 'google_function_call_id="call_google_1"' in text_items[0]["text"]
    assert image_items[-1]["mime_type"] == "image/png"


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
    assert function_result["result"]["url"] == "https://example.com"
    assert not any(item["type"] == "text" for item in payload["input"])


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
    assert function_result["result"]["url"] == "https://example.com"
    call_text = next(
        item["text"] for item in payload["input"] if item["type"] == "text"
    )
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
    assert payload["input"][-1]["type"] == "image"
    assert screenshot_bytes == b"fake_png_bytes"


@pytest.mark.asyncio
async def test_google_follow_up_execute_mode_uses_minimal_payload(
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
        call_id="call_google_execute",
        action_type="click_at",
        parameters={"x": 120, "y": 340},
        status="executed",
        pending_safety_checks=[],
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Tap the button.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={"interaction_mode": "execute"},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    texts = [item["text"] for item in payload["input"] if item["type"] == "text"]

    assert texts == []


@pytest.mark.asyncio
async def test_google_follow_up_preserves_observe_only_reminder_text(
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
        call_id="call_google_observe",
        action_type="scroll_document",
        parameters={"direction": "down"},
        status="executed",
        pending_safety_checks=[],
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Inspect the page.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={"interaction_mode": "observe_only"},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    texts = [item["text"] for item in payload["input"] if item["type"] == "text"]

    assert len(texts) == 1
    assert any("Observe-only mode is active" in text for text in texts)


@pytest.mark.asyncio
async def test_google_follow_up_requests_in_loop_localization_on_full_keyframe(
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
        call_id="call_google_localize",
        action_type="click_at",
        parameters={"x": 120, "y": 340},
        status="executed",
        pending_safety_checks=[],
    )

    payload, batch, _ = await session._build_google_follow_up_request(
        goal="Tap the button.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={"target": "Email"},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    assert batch.request_localization is True
    assert batch.localization_reason == "missing_session_cartography"
    texts = [item["text"] for item in payload["input"] if item["type"] == "text"]
    assert any("Refresh the session cartography now" in text for text in texts)


@pytest.mark.asyncio
async def test_google_follow_up_execute_mode_ignores_state_only_reporting_reminder(
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
        call_id="call_google_state_only",
        action_type="click_at",
        parameters={"x": 120, "y": 340},
        status="executed",
        pending_safety_checks=[],
    )

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Open Account Settings.",
        previous_interaction_id="int_prev",
        turns=[turn],
        metadata={
            "interaction_mode": "execute",
            "response_reporting_scope": "state_only",
        },
        environment="mobile_adb",
        model="gemini-3-flash-preview",
    )

    texts = [item["text"] for item in payload["input"] if item["type"] == "text"]

    assert texts == []


@pytest.mark.asyncio
async def test_google_follow_up_attaches_shared_context_only_to_first_result(
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
    turns = [
        ComputerToolTurn(
            call_id="call_google_a",
            action_type="click_at",
            parameters={"x": 120, "y": 340},
            status="executed",
            pending_safety_checks=[],
        ),
        ComputerToolTurn(
            call_id="call_google_b",
            action_type="type_text_at",
            parameters={"x": 120, "y": 340, "text": "done"},
            status="executed",
            pending_safety_checks=[],
        ),
    ]

    payload, _, _ = await session._build_google_follow_up_request(
        goal="Complete both actions.",
        previous_interaction_id="int_prev",
        turns=turns,
        metadata={"interaction_mode": "observe_only"},
        environment="desktop",
        model="gemini-3-flash-preview",
    )

    function_results = [
        item for item in payload["input"] if item["type"] == "function_result"
    ]
    text_items = [item["text"] for item in payload["input"] if item["type"] == "text"]

    assert len(function_results) == 2
    assert len(text_items) == 1
    assert "Observe-only mode is active" in text_items[0]
    assert "call_id=" not in text_items[0]


def test_wrap_goal_for_google_mobile_strips_preexisting_mobile_wrapper(
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

    goal = session._wrap_goal_for_mobile(
        "Select the Email option.",
        "mobile_adb",
        1080,
        2400,
    )
    wrapped = session._wrap_goal_for_google(goal, "mobile_adb")

    assert wrapped.count("MOBILE EXECUTION CONTEXT:") == 0
    assert wrapped.count("YOUR TASK:") == 1
    assert wrapped.endswith("Select the Email option.")


def test_apply_interaction_mode_guidance_limits_execute_mode_reporting(
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

    guided = session._apply_interaction_mode_guidance(
        "Open Account Settings.",
        {
            "interaction_mode": "execute",
            "response_reporting_scope": "state_only",
        },
    )

    assert "REPORTING RULES:" in guided
    assert "Do NOT quote or infer exact field values" in guided


def test_google_mobile_interaction_tools_match_documented_shape(
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

    tools = session._build_google_interaction_tools("mobile_adb")

    assert tools[0]["type"] == "computer_use"
    assert tools[0]["environment"] == "browser"
    assert set(tools[0]["excluded_predefined_functions"]) == {
        "open_web_browser",
        "search",
        "navigate",
        "hover_at",
        "go_forward",
        "key_combination",
        "scroll_document",
        "drag_and_drop",
    }
    function_names = [tool["name"] for tool in tools[1:]]
    assert function_names == [
        "open_app",
        "long_press_at",
        "go_home",
    ]


@pytest.mark.asyncio
async def test_google_mobile_custom_action_long_press_executes_hold(
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
        call_id="mobile_long_press",
        action_type="long_press_at",
        parameters={"x": 120, "y": 340, "duration_ms": 640},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={"step_number": 1},
        turn_index=1,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "executed"
    mock_browser.drag_mouse.assert_awaited_once_with(120, 340, 120, 340, steps=40)


@pytest.mark.asyncio
async def test_google_mobile_go_back_uses_android_back_key(
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
        call_id="mobile_back",
        action_type="go_back",
        parameters={},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={"step_number": 1},
        turn_index=1,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "executed"
    mock_browser.press_key.assert_awaited_once_with("back")


@pytest.mark.asyncio
async def test_google_mobile_custom_action_go_home(
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
        call_id="mobile_go_home",
        action_type="go_home",
        parameters={},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={"step_number": 1},
        turn_index=1,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "executed"
    mock_browser.press_key.assert_awaited_once_with("home")


@pytest.mark.asyncio
async def test_google_mobile_custom_action_open_app(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    mock_browser.launch_app = AsyncMock(return_value=None)
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )

    turn = ComputerToolTurn(
        call_id="mobile_open_app",
        action_type="open_app",
        parameters={"app_name": "PlayerUp"},
    )

    await session._execute_tool_action(
        turn=turn,
        metadata={
            "step_number": 1,
            "app_package": "co.playerup.app",
            "app_activity": ".MainActivity",
        },
        turn_index=1,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "executed"
    mock_browser.launch_app.assert_awaited_once_with(
        "co.playerup.app",
        ".MainActivity",
    )
