"""Shared fixtures and helpers for Computer Use session tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from haindy.agents.computer_use import ComputerUseSession


class DummyResponse:
    """Simple stand-in for model response objects."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, object]:
        return self._data


@pytest.fixture
def session_settings(tmp_path):
    """Provide minimal settings required by the session."""
    return SimpleNamespace(
        openai_request_timeout_seconds=900,
        actions_computer_tool_action_timeout_seconds=5.0,
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
        computer_use_model="gpt-5.4",
        google_cu_model="gemini-3-flash-preview",
        anthropic_api_key="",
        anthropic_cu_model="claude-sonnet-4-6",
        anthropic_cu_beta="computer-use-2025-11-24",
        anthropic_cu_max_tokens=16384,
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        openai_cu_transport="responses_http",
        cu_visual_mode="keyframe_patch",
        cu_cartography_model="",
        cu_keyframe_max_turns=3,
        cu_patch_max_area_ratio=0.35,
        cu_patch_margin_ratio=0.12,
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


def make_session(
    *,
    mock_client,
    mock_browser,
    session_settings,
    provider: str | None = None,
    model: str | None = None,
    google_client=None,
    anthropic_client=None,
    debug_logger=None,
) -> ComputerUseSession:
    """Build a session with standard test defaults."""
    return ComputerUseSession(
        client=mock_client,
        automation_driver=mock_browser,
        settings=session_settings,
        provider=provider,
        model=model,
        google_client=google_client,
        anthropic_client=anthropic_client,
        debug_logger=debug_logger,
    )


def openai_computer_call(
    call_id: str,
    action: dict[str, object] | list[dict[str, object]] | None = None,
    pending_safety_checks: list[dict[str, object]] | None = None,
    status: str = "completed",
) -> dict[str, object]:
    if isinstance(action, list):
        actions = action
    elif isinstance(action, dict):
        actions = [action]
    else:
        actions = []
    return {
        "type": "computer_call",
        "call_id": call_id,
        "actions": actions,
        "pending_safety_checks": pending_safety_checks or [],
        "status": status,
    }


def openai_message(text: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def openai_response(response_id: str, output: list[dict[str, object]]) -> DummyResponse:
    return DummyResponse({"id": response_id, "output": output})


def make_anthropic_client(create_mock: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    )


def make_google_client(
    generate_content_mock: MagicMock | None = None,
    interactions_create_mock: AsyncMock | None = None,
) -> SimpleNamespace:
    if generate_content_mock is None:
        generate_content_mock = MagicMock()
    if interactions_create_mock is None:
        interactions_create_mock = AsyncMock()
    return SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content_mock),
        aio=SimpleNamespace(
            interactions=SimpleNamespace(create=interactions_create_mock)
        ),
    )
