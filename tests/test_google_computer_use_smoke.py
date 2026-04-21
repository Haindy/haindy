from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haindy.agents.computer_use import ComputerUseSession


class DummyFunctionCall:
    def __init__(self, name: str, args: dict, call_id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = call_id


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


def _stub_initial_interaction_request(*args, **kwargs):
    return (
        {
            "api_surface": "interactions",
            "model": "gemini-3-flash-preview",
            "input": [{"type": "text", "text": "stub"}],
            "tools": [{"type": "computer_use"}],
        },
        kwargs.get("screenshot_bytes"),
    )


@pytest.mark.asyncio
async def test_google_computer_use_provider_smoke(tmp_path: Path) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
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
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )

    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
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
async def test_google_observe_only_policy_violation_fails_immediately(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    function_call = DummyFunctionCall("click_at", {"x": 300, "y": 220})
    response = DummyGoogleResponse(
        {"id": "resp_obs_google_1", "candidates": []},
        candidates=[DummyCandidate(DummyContent([DummyPart(function_call)]))],
    )

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
    session._create_google_response = AsyncMock(return_value=response)  # type: ignore[assignment]

    result = await session.run(
        goal="Verify current state without interacting.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 2, "interaction_mode": "observe_only"},
        allowed_actions={"screenshot"},
    )

    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "observe_only_policy_violation"
    assert len(result.actions) == 1
    assert result.actions[0].status == "failed"
    assert result.actions[0].metadata.get("policy") == "observe_only"
    assert session._create_google_response.await_count == 1  # type: ignore[attr-defined]
    browser.click.assert_not_called()


@pytest.mark.asyncio
async def test_google_computer_use_reports_max_turn_failure(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        openai_request_timeout_seconds=900,
        actions_computer_tool_action_timeout_seconds=5.0,
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
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
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
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
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


@pytest.mark.asyncio
async def test_google_executes_duplicate_name_batch_with_ids(tmp_path: Path) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    first = DummyGoogleResponse(
        {"id": "resp_dup_1", "candidates": []},
        candidates=[
            DummyCandidate(
                DummyContent(
                    [
                        DummyPart(
                            DummyFunctionCall(
                                "key_combination",
                                {"keys": "ctrl+c"},
                                call_id="call_google_1",
                            )
                        ),
                        DummyPart(
                            DummyFunctionCall(
                                "key_combination",
                                {"keys": "ctrl+v"},
                                call_id="call_google_2",
                            )
                        ),
                    ]
                )
            )
        ],
    )
    second = DummyGoogleResponse({"id": "resp_dup_2", "candidates": []})

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        side_effect=[first, second]
    )

    result = await session.run(
        goal="Copy then paste.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 1},
    )

    assert len(result.actions) == 2
    assert result.actions[0].call_id == "call_google_1"
    assert result.actions[1].call_id == "call_google_2"
    assert result.actions[0].metadata["google_function_call_sequence"] == 1
    assert result.actions[1].metadata["google_function_call_sequence"] == 2
    assert result.actions[0].metadata["google_correlation_mode"] == "provider_id"
    assert result.actions[1].metadata["google_correlation_mode"] == "provider_id"
    assert result.terminal_status == "success"
    browser.press_key.assert_any_await("Control+c")
    browser.press_key.assert_any_await("Control+v")


@pytest.mark.asyncio
async def test_google_duplicate_name_batch_without_ids_reasks_then_executes_single_call(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    ambiguous = DummyGoogleResponse(
        {"id": "resp_ambiguous_1", "candidates": []},
        candidates=[
            DummyCandidate(
                DummyContent(
                    [
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+c"})
                        ),
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+v"})
                        ),
                    ]
                )
            )
        ],
    )
    single = DummyGoogleResponse(
        {"id": "resp_ambiguous_2", "candidates": []},
        candidates=[
            DummyCandidate(
                DummyContent(
                    [
                        DummyPart(
                            DummyFunctionCall(
                                "key_combination",
                                {"keys": "ctrl+c"},
                                call_id="call_after_reask",
                            )
                        )
                    ]
                )
            )
        ],
    )
    done = DummyGoogleResponse({"id": "resp_ambiguous_3", "candidates": []})

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        side_effect=[ambiguous, single, done]
    )

    result = await session.run(
        goal="Copy the selected value.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 1},
    )

    assert len(result.actions) == 1
    assert result.actions[0].call_id == "call_after_reask"
    assert result.terminal_status == "success"
    browser.press_key.assert_awaited_once_with("Control+c")
    assert session._create_google_response.await_count == 3  # type: ignore[attr-defined]
    reask_payload = session._create_google_response.await_args_list[1].args[0]  # type: ignore[attr-defined]
    reask_prompt = reask_payload["input"][0]["text"]
    assert "Return exactly one function call" in reask_prompt


@pytest.mark.asyncio
async def test_google_duplicate_name_batch_without_ids_fails_after_single_retry(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
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
        cu_provider="google",
        computer_use_model="computer-use-preview",
        google_cu_model="gemini-3-flash-preview",
        vertex_api_key="",
        vertex_project="",
        vertex_location="us-central1",
        cu_safety_policy="auto_approve",
        model_log_path=tmp_path / "model_logs" / "model_calls.jsonl",
        linux_coordinate_cache_path=tmp_path / "coords.json",
        mobile_coordinate_cache_path=tmp_path / "mobile_coords.json",
        max_screenshots=12,
    )
    browser = AsyncMock()
    browser.get_viewport_size.return_value = (800, 600)
    browser.screenshot.return_value = b"fake_png_bytes"
    browser.wait.return_value = None
    browser.start.return_value = None
    browser.get_page_url.return_value = "https://example.com"
    browser.get_page_title.return_value = "Example Page"

    ambiguous_first = DummyGoogleResponse(
        {"id": "resp_ambiguous_fail_1", "candidates": []},
        candidates=[
            DummyCandidate(
                DummyContent(
                    [
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+c"})
                        ),
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+v"})
                        ),
                    ]
                )
            )
        ],
    )
    ambiguous_second = DummyGoogleResponse(
        {"id": "resp_ambiguous_fail_2", "candidates": []},
        candidates=[
            DummyCandidate(
                DummyContent(
                    [
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+x"})
                        ),
                        DummyPart(
                            DummyFunctionCall("key_combination", {"keys": "ctrl+z"})
                        ),
                    ]
                )
            )
        ],
    )

    session = ComputerUseSession(
        client=None,  # type: ignore[arg-type]
        automation_driver=browser,
        settings=settings,
        google_client=object(),
        provider="google",
    )
    session._build_google_initial_request = _stub_initial_interaction_request  # type: ignore[assignment]
    session._create_google_response = AsyncMock(  # type: ignore[assignment]
        side_effect=[ambiguous_first, ambiguous_second]
    )

    result = await session.run(
        goal="Do keyboard steps.",
        initial_screenshot=b"initial_bytes",
        metadata={"step_number": 1},
    )

    assert result.terminal_status == "failed"
    assert result.terminal_failure_code == "google_ambiguous_function_call_batch"
    assert result.actions
    assert result.actions[-1].action_type == "system_notice"
    browser.press_key.assert_not_called()
