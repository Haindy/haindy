"""Google Computer Use runtime and retry tests."""

from __future__ import annotations

import json
import logging
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import haindy.agents.computer_use.session as cu_session_module
from haindy.agents.computer_use.visual_state import encode_png
from tests.computer_use_session_support import make_google_client, make_session

pytest_plugins = ("tests.computer_use_session_support",)


async def _invoke_google_call_direct(call):
    return call()


@pytest.mark.asyncio
async def test_computer_use_session_google_failure_does_not_fallback_to_openai(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
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
async def test_google_prompt_block_marks_terminal_failure(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"
    session_settings.vertex_api_key = "dummy-key"

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
    )
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        return_value={
            "id": "resp_blocked",
            "candidates": None,
            "prompt_feedback": {
                "block_reason": "SAFETY",
                "block_reason_message": "Policy check blocked the prompt.",
            },
        }
    )

    result = await session.run(
        goal='Tap "Sign Up".',
        initial_screenshot=b"initial",
        metadata={"step_number": 2},
    )

    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "google_prompt_blocked"
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "system_notice"
    assert result.actions[0].parameters["block_reason"] == "SAFETY"
    assert "blocked before tool execution" in (result.terminal_failure_reason or "")
    mock_browser.click.assert_not_called()


@pytest.mark.asyncio
async def test_google_computer_use_retries_resource_exhausted_then_succeeds(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(
        side_effect=[
            RuntimeError(
                "429 RESOURCE_EXHAUSTED. {'error': {'status': 'RESOURCE_EXHAUSTED'}}"
            ),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            {"id": "ok"},
        ]
    )
    sleep_mock = AsyncMock()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    response = await session._create_google_response(
        {"model": "gemini-3-flash-preview", "contents": [], "config": {}}
    )

    assert response == {"id": "ok"}
    assert generate_content.call_count == 3
    assert sleep_mock.await_count == 2
    assert sleep_mock.await_args_list[0].args == (1.0,)
    assert sleep_mock.await_args_list[1].args == (5.0,)


@pytest.mark.asyncio
async def test_google_computer_use_retries_prompt_safety_block_then_succeeds(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(
        side_effect=[
            {
                "id": "blocked_1",
                "candidates": None,
                "prompt_feedback": {"block_reason": "SAFETY"},
            },
            {
                "id": "blocked_2",
                "candidates": None,
                "prompt_feedback": {"block_reason": "SAFETY"},
            },
            {
                "id": "ok",
                "candidates": [{"content": {"parts": []}}],
                "prompt_feedback": None,
            },
        ]
    )
    sleep_mock = AsyncMock()
    jitter_mock = MagicMock(side_effect=[0.05, 0.1])
    monkeypatch.setattr(cu_session_module.random, "uniform", jitter_mock)

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    response = await session._create_google_response(
        {"model": "gemini-3-flash-preview", "contents": [], "config": {}}
    )

    assert response["id"] == "ok"
    assert generate_content.call_count == 3
    assert sleep_mock.await_count == 2
    assert sleep_mock.await_args_list[0].args == (0.3,)
    assert sleep_mock.await_args_list[1].args == (0.85,)


@pytest.mark.asyncio
async def test_google_computer_use_returns_blocked_response_after_safety_retries(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(
        side_effect=[
            {
                "id": "blocked_1",
                "candidates": None,
                "prompt_feedback": {"block_reason": "SAFETY"},
            },
            {
                "id": "blocked_2",
                "candidates": None,
                "prompt_feedback": {"block_reason": "SAFETY"},
            },
            {
                "id": "blocked_3",
                "candidates": None,
                "prompt_feedback": {"block_reason": "SAFETY"},
            },
        ]
    )
    sleep_mock = AsyncMock()
    jitter_mock = MagicMock(side_effect=[0.01, 0.02])
    monkeypatch.setattr(cu_session_module.random, "uniform", jitter_mock)

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    response = await session._create_google_response(
        {"model": "gemini-3-flash-preview", "contents": [], "config": {}}
    )

    assert response["id"] == "blocked_3"
    assert response["prompt_feedback"]["block_reason"] == "SAFETY"
    assert generate_content.call_count == 3
    assert sleep_mock.await_count == 2
    assert sleep_mock.await_args_list[0].args == (0.26,)
    assert sleep_mock.await_args_list[1].args == (0.77,)


@pytest.mark.asyncio
async def test_google_computer_use_raises_after_retry_budget_exhausted(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(
        side_effect=[
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
        ]
    )
    sleep_mock = AsyncMock()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await session._create_google_response(
            {"model": "gemini-3-flash-preview", "contents": [], "config": {}}
        )

    assert generate_content.call_count == 4
    assert sleep_mock.await_count == 3
    assert sleep_mock.await_args_list[0].args == (1.0,)
    assert sleep_mock.await_args_list[1].args == (5.0,)
    assert sleep_mock.await_args_list[2].args == (10.0,)


@pytest.mark.asyncio
async def test_google_computer_use_non_retryable_error_does_not_retry(
    mock_client, mock_browser, session_settings, monkeypatch
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(side_effect=RuntimeError("google failed"))
    sleep_mock = AsyncMock()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="google failed"):
        await session._create_google_response(
            {"model": "gemini-3-flash-preview", "contents": [], "config": {}}
        )

    assert generate_content.call_count == 1
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_google_interactions_suppresses_known_package_warnings(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"

    class _Interactions:
        async def create(self, **kwargs):
            return {"id": "ok"}

    class _AioClient:
        @property
        def interactions(self):
            warnings.warn(
                "Interactions usage is experimental and may change in future versions.",
                UserWarning,
                stacklevel=1,
            )
            warnings.warn(
                "Async interactions client cannot use aiohttp, fallingback to httpx.",
                UserWarning,
                stacklevel=1,
            )
            return _Interactions()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=type("GoogleClient", (), {"aio": _AioClient()})(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = await session._create_google_response(
            {
                "api_surface": "interactions",
                "model": "gemini-3-flash-preview",
                "input": [],
                "tools": [],
            }
        )

    assert response == {"id": "ok"}
    assert caught == []


@pytest.mark.asyncio
async def test_google_interactions_preserves_unrelated_warnings(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"

    class _Interactions:
        async def create(self, **kwargs):
            warnings.warn("Unrelated interactions warning.", UserWarning, stacklevel=1)
            return {"id": "ok"}

    class _AioClient:
        @property
        def interactions(self):
            warnings.warn(
                "Interactions usage is experimental and may change in future versions.",
                UserWarning,
                stacklevel=1,
            )
            warnings.warn(
                "Async interactions client cannot use aiohttp, fallingback to httpx.",
                UserWarning,
                stacklevel=1,
            )
            return _Interactions()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=type("GoogleClient", (), {"aio": _AioClient()})(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = await session._create_google_response(
            {
                "api_surface": "interactions",
                "model": "gemini-3-flash-preview",
                "input": [],
                "tools": [],
            }
        )

    assert response == {"id": "ok"}
    assert [str(item.message) for item in caught] == ["Unrelated interactions warning."]


@pytest.mark.asyncio
async def test_google_interactions_failure_logs_payload_summary(
    mock_client, mock_browser, session_settings, caplog
):
    session_settings.cu_provider = "google"
    interactions_create = AsyncMock(
        side_effect=RuntimeError(
            "400 INVALID_ARGUMENT. {'error': {'message': 'Invalid input received.'}}"
        )
    )
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(interactions_create_mock=interactions_create),
    )

    with caplog.at_level(logging.ERROR, logger="haindy.agents.computer_use.session"):
        with pytest.raises(RuntimeError, match="Invalid input received"):
            await session._create_google_response(
                {
                    "api_surface": "interactions",
                    "model": "gemini-3-flash-preview",
                    "previous_interaction_id": "int_prev",
                    "input": [
                        {
                            "type": "function_result",
                            "name": "type_text_at",
                            "call_id": "call_1",
                            "result": {
                                "status": "executed",
                                "items": [
                                    {
                                        "type": "text",
                                        "text": 'current_url="android://screen"',
                                    },
                                    {
                                        "type": "image",
                                        "data": "ZmFrZQ==",
                                        "mime_type": "image/png",
                                    },
                                ],
                            },
                        }
                    ],
                    "tools": [{"type": "computer_use"}],
                }
            )

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Google Computer Use request failed"
    )
    assert record.payload_summary["previous_interaction_id"] == "int_prev"
    assert record.payload_summary["input_summary"][0]["type"] == "function_result"
    assert record.payload_summary["tool_types"] == ["computer_use"]


@pytest.mark.asyncio
async def test_google_computer_use_logs_non_retryable_failed_attempt(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(side_effect=RuntimeError("google failed"))

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="google failed"):
        await session._create_google_response(
            {"model": "gemini-3-flash-preview", "contents": [], "config": {}},
            agent="computer_use.google.initial",
            prompt="Tap Login",
            request_payload_for_log={"request": "sanitized"},
            metadata={"environment": "desktop"},
        )

    entry = json.loads(
        session_settings.model_log_path.read_text(encoding="utf-8").strip()
    )
    assert entry["outcome"] == "failure"
    assert entry["agent"] == "computer_use.google.initial"
    assert entry["metadata"]["provider"] == "google"
    assert entry["metadata"]["attempt_number"] == 1


@pytest.mark.asyncio
async def test_google_computer_use_suppresses_rate_limit_failure_entries(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"
    generate_content = MagicMock(
        side_effect=[
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            RuntimeError("429 RESOURCE_EXHAUSTED"),
        ]
    )
    sleep_mock = AsyncMock()

    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=make_google_client(generate_content),
    )
    session._invoke_google_request = _invoke_google_call_direct  # type: ignore[method-assign]
    session._sleep_google_retry = sleep_mock  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await session._create_google_response(
            {"model": "gemini-3-flash-preview", "contents": [], "config": {}},
            agent="computer_use.google.initial",
            prompt="Tap Login",
            request_payload_for_log={"request": "sanitized"},
            metadata={"environment": "desktop"},
        )

    assert (
        not session_settings.model_log_path.exists()
        or session_settings.model_log_path.read_text(encoding="utf-8") == ""
    )


@pytest.mark.asyncio
async def test_google_runtime_stores_in_session_localization_from_response(
    mock_client, mock_browser, session_settings
):
    session_settings.cu_provider = "google"
    session = make_session(
        mock_client=mock_client,
        mock_browser=mock_browser,
        session_settings=session_settings,
        provider="google",
        google_client=object(),
    )
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        return_value={
            "id": "resp_google_localize",
            "outputs": [
                {
                    "type": "text",
                    "text": (
                        "Located the target.\n"
                        "<haindy_cartography>"
                        + json.dumps(
                            {
                                "targets": [
                                    {
                                        "target_id": "target_1",
                                        "label": "Email",
                                        "bbox": {
                                            "x": 20,
                                            "y": 30,
                                            "width": 40,
                                            "height": 20,
                                        },
                                        "interaction_point": {"x": 35, "y": 40},
                                        "confidence": 0.9,
                                    }
                                ]
                            }
                        )
                        + "</haindy_cartography>"
                    ),
                }
            ],
        }
    )
    image = Image.new("RGB", (100, 100), color="black")
    screenshot = encode_png(image)

    session.begin_step_scope()

    result = await session.execute_step_action(
        goal="Open the Email field.",
        initial_screenshot=screenshot,
        metadata={"target": 'Email field labeled "Email"', "step_number": 3},
    )

    assert result.final_output == "Located the target."
    assert session._current_keyframe is not None
    assert session._current_keyframe.cartography is not None
    assert session._current_keyframe.cartography.targets[0].label == "Email"
