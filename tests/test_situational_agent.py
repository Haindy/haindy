"""Unit tests for SituationalAgent."""

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


def test_heuristic_desktop_context_uses_visual_entry_action_without_os_identifiers() -> None:
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
