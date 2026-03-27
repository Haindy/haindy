"""Google Computer Use step-scoped interaction flow tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.computer_use_session_support import make_google_client, make_session

pytest_plugins = ("tests.computer_use_session_support",)


def _google_function_call(
    call_id: str, name: str, arguments: dict[str, object]
) -> dict:
    return {
        "id": call_id,
        "status": "requires_action",
        "outputs": [
            {
                "type": "function_call",
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
        ],
    }


def _google_text_response(response_id: str, text: str) -> dict:
    return {
        "id": response_id,
        "status": "completed",
        "outputs": [{"type": "text", "text": text}],
    }


@pytest.mark.asyncio
async def test_google_step_actions_reuse_previous_interaction_chain(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"
    interactions_create = AsyncMock(
        side_effect=[
            _google_function_call("int_1", "click", {"x": 12, "y": 24}),
            _google_text_response("int_2", "Clicked."),
            _google_function_call("int_3", "type", {"text": "done"}),
            _google_text_response("int_4", "Typed."),
        ]
    )
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(interactions_create_mock=interactions_create),
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

    assert first.response_ids == ["int_1", "int_2"]
    assert second.response_ids == ["int_3", "int_4"]
    assert session.step_response_ids == ["int_1", "int_2", "int_3", "int_4"]

    first_request = interactions_create.await_args_list[0].kwargs
    follow_up_request = interactions_create.await_args_list[1].kwargs
    second_request = interactions_create.await_args_list[2].kwargs

    assert "previous_interaction_id" not in first_request
    assert first_request["input"][0]["type"] == "text"
    assert first_request["input"][1]["type"] == "image"
    assert (
        "IMPORTANT: You are controlling a single Firefox browser window"
        in (first_request["input"][0]["text"])
    )

    assert follow_up_request["previous_interaction_id"] == "int_1"
    assert follow_up_request["input"][0]["type"] == "function_result"

    assert second_request["previous_interaction_id"] == "int_2"
    assert len(second_request["input"]) == 1
    assert second_request["input"][0]["type"] == "text"
    assert second_request["input"][0]["text"] == "Type done into the field."


@pytest.mark.asyncio
async def test_google_step_reflection_uses_json_output_without_tools(
    mock_client, mock_browser, session_settings
) -> None:
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"
    interactions_create = AsyncMock(
        side_effect=[
            _google_function_call("int_1", "click", {"x": 5, "y": 6}),
            _google_text_response("int_2", "Clicked."),
            _google_text_response(
                "int_3",
                '{"verdict":"PASS","reasoning":"ok","actual_result":"done","confidence":0.9,"is_blocker":false,"blocker_reasoning":""}',
            ),
        ]
    )
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(interactions_create_mock=interactions_create),
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

    payload = interactions_create.await_args_list[2].kwargs
    assert payload["previous_interaction_id"] == "int_2"
    assert "tools" not in payload
    assert payload["response_mime_type"] == "application/json"
    assert payload["response_format"] == {"type": "object"}
    assert payload["input"][0]["text"].startswith("Respond with valid JSON")
    assert reflection["response_ids"] == ["int_3"]
    assert '"verdict":"PASS"' in reflection["raw_text"]
