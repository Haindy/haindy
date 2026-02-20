from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.computer_use import ComputerUseSession


class DummyFunctionCall:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args


class DummyPart:
    def __init__(self, function_call: DummyFunctionCall) -> None:
        self.function_call = function_call


class DummyContent:
    def __init__(self, parts) -> None:
        self.parts = parts


class DummyCandidate:
    def __init__(self, content: DummyContent) -> None:
        self.content = content


class DummyGoogleResponse:
    def __init__(self, payload: dict, candidates=None) -> None:
        self._payload = payload
        self.candidates = candidates or []

    def model_dump(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_google_computer_use_provider_smoke(tmp_path: Path) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        google_cu_model="gemini-2.5-computer-use-preview-10-2025",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        desktop_coordinate_cache_path=tmp_path / "coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        browser=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )

    session._build_google_initial_request = (  # type: ignore[assignment]
        lambda *args, **kwargs: ([{"role": "user"}], "config")
    )
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        return_value=DummyGoogleResponse({"id": "resp_1", "candidates": []})
    )

    result = await session.run(
        goal="Observe the screen.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 1},
    )

    assert result.response_ids == ["resp_1"]
    assert result.actions == []


@pytest.mark.asyncio
async def test_google_computer_use_reports_max_turn_failure(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        openai_request_timeout_seconds=900,
        actions_computer_tool_action_timeout_ms=5000,
        actions_computer_tool_stabilization_wait_ms=0,
        actions_computer_tool_max_turns=1,
        actions_computer_tool_loop_detection_window=3,
        actions_computer_tool_fail_fast_on_safety=True,
        actions_computer_tool_allowed_domains=[],
        actions_computer_tool_blocked_domains=[],
        scroll_turn_multiplier=3.0,
        scroll_default_magnitude=450,
        scroll_max_magnitude=600,
        cu_provider="google",
        google_cu_model="gemini-2.5-computer-use-preview-10-2025",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        desktop_coordinate_cache_path=tmp_path / "coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.click.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    function_call = DummyFunctionCall("click", {"x": 100, "y": 100})
    response = DummyGoogleResponse(
        {"id": "resp_turn_1", "candidates": []},
        candidates=[DummyCandidate(DummyContent([DummyPart(function_call)]))],
    )

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        browser=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = (  # type: ignore[assignment]
        lambda *args, **kwargs: ([{"role": "user"}], "config")
    )
    session._create_google_response = AsyncMock(return_value=response)  # type: ignore[assignment]

    result = await session.run(
        goal="Click an element.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 1},
    )

    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "max_turns_exceeded"
    assert any(
        action.action_type == "system_notice" and action.status == "failed"
        for action in result.actions
    )
