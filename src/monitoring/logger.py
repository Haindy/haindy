"""
Logging configuration and utilities for the HAINDY framework.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from src.config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "test_id"):
            log_data["test_id"] = record.test_id
        if hasattr(record, "step_id"):
            log_data["step_id"] = record.step_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class AgentLogAdapter(logging.LoggerAdapter):
    """Log adapter for agent-specific logging."""

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Add agent context to log records."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (defaults to settings)
        log_format: Log format 'json' or 'text' (defaults to settings)
        log_file: Optional log file path (defaults to settings)
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    level = log_level or settings.log_level
    format_type = log_format or settings.log_format
    file_path = log_file or settings.log_file

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper())

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure console handler
    if format_type == "json":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
    else:
        # Use Rich handler for pretty text output
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )

    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Configure file handler if specified
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)

        if format_type == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(numeric_level)

    # Configure specific loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log startup message
    logger = logging.getLogger("haindy")
    logger.info(
        f"HAINDY logging initialized",
        extra={
            "log_level": level,
            "log_format": format_type,
            "log_file": file_path,
        },
    )


def get_logger(name: str, **context: Any) -> logging.Logger:
    """
    Get a logger instance with optional context.

    Args:
        name: Logger name
        **context: Additional context to include in logs

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if context:
        return AgentLogAdapter(logger, context)

    return logger


def log_test_event(
    event_type: str,
    test_id: str,
    step_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a test execution event.

    Args:
        event_type: Type of event
        test_id: Test identifier
        step_id: Optional step identifier
        data: Additional event data
    """
    logger = logging.getLogger("haindy.test_events")
    
    extra = {
        "event_type": event_type,
        "test_id": test_id,
    }
    
    if step_id:
        extra["step_id"] = step_id
    
    if data:
        extra.update(data)
    
    logger.info(f"Test event: {event_type}", extra=extra)


def log_agent_communication(
    from_agent: str,
    to_agent: str,
    message_type: str,
    content: Dict[str, Any],
    correlation_id: Optional[str] = None,
) -> None:
    """
    Log inter-agent communication.

    Args:
        from_agent: Sending agent name
        to_agent: Receiving agent name
        message_type: Type of message
        content: Message content
        correlation_id: Optional correlation ID
    """
    logger = logging.getLogger("haindy.agent_communication")
    
    extra = {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "message_type": message_type,
        "content_summary": str(content)[:200],  # Truncate for logging
    }
    
    if correlation_id:
        extra["correlation_id"] = correlation_id
    
    logger.debug(
        f"Agent communication: {from_agent} -> {to_agent} ({message_type})",
        extra=extra,
    )


def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = "ms",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a performance metric.

    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        context: Additional context
    """
    logger = logging.getLogger("haindy.performance")
    
    extra = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
    }
    
    if context:
        extra.update(context)
    
    logger.info(f"Performance metric: {metric_name}={value}{unit}", extra=extra)