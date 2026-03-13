"""
Tests for HAINDY logging utilities.
"""

import logging
from datetime import datetime, timezone

import pytest

from src.monitoring.logger import HumanReadableFormatter, get_logger, setup_logging


@pytest.fixture()
def formatter() -> HumanReadableFormatter:
    """Provide a reusable formatter instance."""
    return HumanReadableFormatter()


def test_human_readable_formatter_orders_known_fields(
    formatter: HumanReadableFormatter,
) -> None:
    """Formatter should surface high-signal extras in the expected order."""
    record = logging.LogRecord(
        name="src.agents.test_runner",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Interpreting step with AI",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.taskName = "Task-7"
    record.step_number = 2
    record.action = 'Type "artificial intelligence" in the search field.'
    record.expected_result = (
        'The text "artificial intelligence" appears in the search field.'
    )

    output = formatter.format(record)

    assert output.startswith(
        "[INFO] | 1970-01-01T00:00:00+00:00 | Test Runner | Interpreting step with AI"
    )
    assert "Task:" not in output

    step_index = output.index("Step: 2")
    action_index = output.index(
        'Action: Type "artificial intelligence" in the search field.'
    )
    expected_index = output.index(
        'Expected Result: The text "artificial intelligence" appears in the search field.'
    )

    assert step_index < action_index < expected_index


def test_human_readable_formatter_handles_additional_fields(
    formatter: HumanReadableFormatter,
) -> None:
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


def test_human_readable_formatter_prefers_agent_name(
    formatter: HumanReadableFormatter,
) -> None:
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


def test_human_readable_formatter_serializes_datetime_in_structured_extra(
    formatter: HumanReadableFormatter,
) -> None:
    """Formatter should not fail when structured extras contain datetimes."""
    record = logging.LogRecord(
        name="src.agents.computer_use.session",
        level=logging.ERROR,
        pathname=__file__,
        lineno=30,
        msg="Computer Use max turns reached (google)",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.last_response = {
        "id": "resp_123",
        "outputs": [
            {
                "timestamp": datetime(2026, 3, 9, 13, 0, 0, tzinfo=timezone.utc),
            }
        ],
    }

    output = formatter.format(record)

    assert "Computer Use max turns reached (google)" in output
    assert '"timestamp": "2026-03-09 13:00:00+00:00"' in output


def test_setup_logging_honors_debug_level_for_text_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text-mode console logging should emit DEBUG records when configured."""
    setup_logging(log_level="DEBUG", log_format="text", log_file=None)

    logger = get_logger("tests.logger")
    logger.debug("debug line should be visible")

    captured = capsys.readouterr()

    assert "debug line should be visible" in captured.out
    assert "[DEBUG]" in captured.out


def test_setup_logging_suppresses_noisy_third_party_debug_loggers() -> None:
    """Third-party wire/debug loggers should stay quiet in HAINDY debug mode."""
    setup_logging(log_level="DEBUG", log_format="text", log_file=None)

    assert logging.getLogger("google.genai").getEffectiveLevel() == logging.WARNING
    assert (
        logging.getLogger("PIL.PngImagePlugin").getEffectiveLevel() == logging.WARNING
    )
    assert logging.getLogger("tests.logger").getEffectiveLevel() == logging.DEBUG
