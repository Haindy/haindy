"""Unit tests for SituationalAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.situational_agent import SituationalAgent
from src.core.types import ActionType


def test_heuristic_web_context_is_sufficient() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment(
        "Target type: web\nOpen https://example.com and maximize window"
    )
    assert assessment.sufficient is True
    assert assessment.target_type == "web"
    assert assessment.setup.web_url == "https://example.com"
    assert assessment.setup.maximize is True


def test_heuristic_desktop_context_uses_visual_entry_action_without_os_identifiers() -> (
    None
):
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("Target type: desktop app for KeenBench")
    assert assessment.sufficient is True
    assert assessment.target_type == "desktop_app"
    assert not assessment.missing_items
    assert assessment.entry_actions
    assert assessment.entry_actions[0].action_type == ActionType.CLICK


def test_heuristic_desktop_context_with_app_name() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("app_name: Calculator")
    assert assessment.target_type == "desktop_app"
    assert assessment.sufficient is True
    assert assessment.setup.app_name == "Calculator"
    assert assessment.entry_actions
    assert "Calculator" in assessment.entry_actions[0].description


def test_parse_assessment_filters_deterministic_identifier_blockers() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "desktop_app",
        "sufficient": False,
        "missing_items": [
            "Exact Linux window/app name for KeenBench (window title/task switcher)",
            "WM_CLASS or process name",
        ],
        "setup": {"app_name": "KeenBench"},
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test KeenBench workflows",
        "Open KeenBench and verify workbench files",
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert assessment.entry_actions


def test_parse_assessment_treats_desktop_missing_items_as_non_blocking() -> None:
    agent = SituationalAgent()
    payload = {
        "target_type": "desktop_app",
        "sufficient": False,
        "missing_items": [
            (
                "Confirmation that KeenBench is on the Home screen with a visible "
                "New Workbench button"
            )
        ],
        "setup": {"app_name": "KeenBench"},
        "entry_actions": [],
        "notes": [],
    }

    assessment = agent._parse_assessment(
        payload,
        "Test KeenBench workflows",
        "KeenBench is already open on the desktop",
    )

    assert assessment.sufficient is True
    assert not assessment.missing_items
    assert assessment.entry_actions
    assert any("Non-blocking context gap" in note for note in assessment.notes)


@pytest.mark.asyncio
async def test_assess_context_persists_model_and_debug_logs(monkeypatch) -> None:
    model_logger = SimpleNamespace(log_call=AsyncMock())
    debug_logger = MagicMock()
    monkeypatch.setattr(
        "src.agents.situational_agent.get_debug_logger", lambda: debug_logger
    )

    agent = SituationalAgent(model_logger=model_logger)
    agent.call_openai = AsyncMock(
        return_value={
            "content": {
                "target_type": "desktop_app",
                "sufficient": True,
                "missing_items": [],
                "setup": {"app_name": "KeenBench"},
                "entry_actions": [],
                "notes": [],
            }
        }
    )

    assessment = await agent.assess_context(
        requirements="Validate KeenBench desktop workflow.",
        context_text="KeenBench is open in GNOME.",
    )

    assert assessment.sufficient is True
    model_logger.log_call.assert_awaited_once()
    logged_kwargs = model_logger.log_call.await_args.kwargs
    assert logged_kwargs["agent"] == "situational.assessment"
    assert "REQUIREMENTS:" in logged_kwargs["prompt"]
    assert "EXECUTION CONTEXT:" in logged_kwargs["prompt"]
    assert logged_kwargs["metadata"]["fallback_used"] is False

    debug_logger.log_ai_interaction.assert_called_once()
    debug_kwargs = debug_logger.log_ai_interaction.call_args.kwargs
    assert debug_kwargs["action_type"] == "situational_assessment"
    assert "REQUIREMENTS:" in debug_kwargs["prompt"]


@pytest.mark.asyncio
async def test_assess_context_logs_fallback_when_model_call_fails(monkeypatch) -> None:
    model_logger = SimpleNamespace(log_call=AsyncMock())
    debug_logger = MagicMock()
    monkeypatch.setattr(
        "src.agents.situational_agent.get_debug_logger", lambda: debug_logger
    )

    agent = SituationalAgent(model_logger=model_logger)
    agent.call_openai = AsyncMock(side_effect=RuntimeError("simulated API failure"))

    assessment = await agent.assess_context(
        requirements="Target desktop app flow.",
        context_text="app_name: Calculator",
    )

    assert assessment.target_type == "desktop_app"
    assert assessment.sufficient is True

    model_logger.log_call.assert_awaited_once()
    logged_kwargs = model_logger.log_call.await_args.kwargs
    assert logged_kwargs["metadata"]["fallback_used"] is True
    assert "simulated API failure" in (logged_kwargs["metadata"]["error"] or "")

    debug_logger.log_ai_interaction.assert_called_once()
    debug_response = debug_logger.log_ai_interaction.call_args.kwargs["response"]
    assert "Fallback note" in debug_response
