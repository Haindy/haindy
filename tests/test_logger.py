"""
Tests for HAINDY logging utilities.
"""

import logging

import pytest

from src.monitoring.logger import HumanReadableFormatter


@pytest.fixture()
def formatter() -> HumanReadableFormatter:
    """Provide a reusable formatter instance."""
    return HumanReadableFormatter()


def test_human_readable_formatter_orders_known_fields(formatter: HumanReadableFormatter) -> None:
    """Formatter should surface high-signal extras in the expected order."""
    record = logging.LogRecord(
        name="src.agents.test_runner",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg='Interpreting step with AI',
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.taskName = "Task-7"
    record.step_number = 2
    record.action = 'Type "artificial intelligence" in the search field.'
    record.expected_result = 'The text "artificial intelligence" appears in the search field.'

    output = formatter.format(record)

    assert output.startswith("[INFO] | 1970-01-01T00:00:00+00:00 | Test Runner | Interpreting step with AI")
    assert "Task:" not in output

    step_index = output.index("Step: 2")
    action_index = output.index('Action: Type "artificial intelligence" in the search field.')
    expected_index = output.index('Expected Result: The text "artificial intelligence" appears in the search field.')

    assert step_index < action_index < expected_index


def test_human_readable_formatter_handles_additional_fields(formatter: HumanReadableFormatter) -> None:
    """Formatter should humanize unknown extras and logger names."""
    record = logging.LogRecord(
        name="haindy.performance",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="Metric threshold exceeded",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.customField = "value"

    output = formatter.format(record)

    assert " | Performance | Metric threshold exceeded" in output
    assert "Custom Field: value" in output


def test_human_readable_formatter_prefers_agent_name(formatter: HumanReadableFormatter) -> None:
    """Formatter should use agent_name for the component label."""
    record = logging.LogRecord(
        name="src.monitoring.debug_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Prompt: Scroll through the article",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.agent_name = "ActionAgent"

    output = formatter.format(record)

    assert " | Action Agent | Prompt: Scroll through the article" in output
