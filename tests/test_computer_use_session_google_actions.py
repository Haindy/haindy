"""Gemini 3.x Computer Use action normalization and execution tests."""

from __future__ import annotations

from unittest.mock import call

import pytest

from haindy.core.enhanced_types import ComputerToolTurn
from tests.computer_use_session_support import make_session

pytest_plugins = ("tests.computer_use_session_support",)


def _google_session(mock_client, mock_browser, session_settings):
    session_settings.cu_provider = "google"
    return make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )


async def _execute(
    session,
    turn: ComputerToolTurn,
    *,
    normalized_coords: bool = True,
    environment: str = "desktop",
) -> None:
    await session._execute_tool_action(
        turn=turn,
        metadata={"step_number": 1},
        turn_index=1,
        normalized_coords=normalized_coords,
        environment=environment,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "button", "click_count"),
    [
        ("triple_click", "left", 3),
        ("middle_click", "middle", 1),
    ],
)
async def test_google_modern_click_variants_execute_with_documented_semantics(
    mock_client,
    mock_browser,
    session_settings,
    action_type: str,
    button: str,
    click_count: int,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id=action_type,
        action_type=action_type,
        parameters={"type": action_type, "x": 500, "y": 250},
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    mock_browser.click.assert_awaited_once_with(
        513,
        192,
        button=button,
        click_count=click_count,
    )


@pytest.mark.asyncio
async def test_google_hotkey_executes_as_one_key_chord(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="hotkey",
        action_type="hotkey",
        parameters={"type": "hotkey", "keys": ["CTRL", "SHIFT", "P"]},
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    assert turn.action_type == "key_combination"
    mock_browser.press_key.assert_awaited_once_with("Control+Shift+P")


@pytest.mark.asyncio
async def test_google_take_screenshot_maps_to_screenshot_action(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="take_screenshot",
        action_type="take_screenshot",
        parameters={"type": "take_screenshot"},
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    assert turn.action_type == "screenshot"
    mock_browser.screenshot.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_google_direction_scroll_uses_magnitude_and_normalized_origin(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="scroll",
        action_type="scroll",
        parameters={
            "type": "scroll",
            "direction": "down",
            "magnitude_in_pixels": 320,
            "x": 500,
            "y": 250,
        },
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    mock_browser.scroll.assert_awaited_once_with(
        "down",
        320,
        origin=(513, 192),
    )
    assert turn.metadata["scroll_direction"] == "down"
    assert turn.metadata["scroll_magnitude"] == 320


@pytest.mark.asyncio
async def test_google_wait_converts_seconds_to_milliseconds(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="wait",
        action_type="wait",
        parameters={"type": "wait", "seconds": 1.25},
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    mock_browser.wait.assert_has_awaits([call(1250)])
    assert turn.metadata["duration_ms"] == 1250


@pytest.mark.asyncio
async def test_google_type_honors_press_enter(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="type",
        action_type="type",
        parameters={"type": "type", "text": "hello", "press_enter": True},
    )

    await _execute(session, turn)

    assert turn.status == "executed"
    mock_browser.type_text.assert_awaited_once_with("hello")
    mock_browser.press_key.assert_awaited_once_with("enter")
    assert turn.metadata["press_enter"] is True


@pytest.mark.asyncio
async def test_google_mobile_long_press_converts_seconds_to_hold_steps(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="long_press",
        action_type="long_press",
        parameters={"type": "long_press", "x": 120, "y": 340, "seconds": 2},
    )

    await _execute(
        session,
        turn,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "executed"
    assert turn.action_type == "long_press_at"
    mock_browser.drag_mouse.assert_awaited_once_with(
        120,
        340,
        120,
        340,
        steps=125,
    )
    assert turn.metadata["duration_ms"] == 2000


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["mouse_down", "mouse_up"])
async def test_google_mouse_hold_actions_fail_explicitly_without_driver_support(
    mock_client,
    mock_browser,
    session_settings,
    action_type: str,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id=action_type,
        action_type=action_type,
        parameters={"type": action_type, "x": 500, "y": 250},
    )

    await _execute(session, turn)

    assert turn.status == "failed"
    assert "independent mouse-button press and release" in (turn.error_message or "")
    mock_browser.click.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["key_down", "key_up"])
async def test_google_key_hold_actions_fail_explicitly_without_driver_support(
    mock_client,
    mock_browser,
    session_settings,
    action_type: str,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id=action_type,
        action_type=action_type,
        parameters={"type": action_type, "key": "Shift"},
    )

    await _execute(session, turn)

    assert turn.status == "failed"
    assert "independent key press and release" in (turn.error_message or "")
    mock_browser.press_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_google_list_apps_fails_explicitly_without_driver_inventory(
    mock_client,
    mock_browser,
    session_settings,
) -> None:
    session = _google_session(mock_client, mock_browser, session_settings)
    turn = ComputerToolTurn(
        call_id="list_apps",
        action_type="list_apps",
        parameters={"type": "list_apps"},
    )

    await _execute(
        session,
        turn,
        normalized_coords=False,
        environment="mobile_adb",
    )

    assert turn.status == "failed"
    assert "installed-application inventory" in (turn.error_message or "")
