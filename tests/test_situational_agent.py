"""Unit tests for SituationalAgent."""

from src.agents.situational_agent import SituationalAgent


def test_heuristic_web_context_is_sufficient() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment(
        "Target type: web\nOpen https://example.com and maximize window"
    )
    assert assessment.sufficient is True
    assert assessment.target_type == "web"
    assert assessment.setup.web_url == "https://example.com"
    assert assessment.setup.maximize is True


def test_heuristic_desktop_context_requires_app_details() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("Target type: desktop app")
    assert assessment.sufficient is False
    assert assessment.target_type == "desktop_app"
    assert assessment.missing_items


def test_heuristic_desktop_context_with_app_name() -> None:
    agent = SituationalAgent()
    assessment = agent._heuristic_assessment("app_name: Calculator")
    assert assessment.target_type == "desktop_app"
    assert assessment.sufficient is True
    assert assessment.setup.app_name == "Calculator"
