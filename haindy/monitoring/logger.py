"""
Logging configuration and utilities for the HAINDY framework.
"""

import json
import logging
import re
import sys
import uuid
from collections.abc import Iterable, MutableMapping
from datetime import datetime, timezone
from typing import Any

from haindy.config.settings import get_settings
from haindy.security.sanitizer import DataSanitizer

STANDARD_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "created",
    "msecs",
    "relativeCreated",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "lineno",
    "funcName",
    "exc_info",
    "exc_text",
    "stack_info",
    "thread",
    "threadName",
    "processName",
    "process",
    "getMessage",
    "message",
    "asctime",
}

_CURRENT_RUN_ID: str | None = None

HUMAN_LOG_EXTRA_FIELD_ORDER = [
    "taskName",
    "test_case",
    "step_number",
    "action",
    "description",
    "action_type",
    "expected_result",
    "verdict",
    "confidence",
    "is_blocker",
    "decomposed_actions",
]

# Fields written to JSONL but suppressed from STDOUT for readability.
HUMAN_LOG_SUPPRESSED_FIELDS = {
    "prompt_length",
    "screenshot",
    "screenshot_path",
    "screenshot_source",
    "intent",
}

HUMAN_LOG_FIELD_LABELS = {
    "taskName": "Task",
    "step_number": "Step",
    "test_case": "Test Case",
    "action": "Action",
    "expected_result": "Expected Result",
    "decomposed_actions": "Decomposed Actions",
    "is_blocker": "Blocker",
}


class JSONFormatter(logging.Formatter):
    """JSON log formatter with optional sanitization."""

    def __init__(self, *args: Any, sanitize: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sanitize = sanitize
        self.sanitizer = DataSanitizer() if sanitize else None

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Sanitize record if enabled
        if self.sanitize and self.sanitizer:
            record = self.sanitizer.sanitize_log_record(record)

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "run_id": get_run_id(),
        }

        # Add all extra fields dynamically
        # Get all attributes from the record
        # Add any extra fields that were passed
        for attr_name in dir(record):
            if (
                not attr_name.startswith("_")
                and attr_name not in STANDARD_LOG_RECORD_ATTRS
            ):
                attr_value = getattr(record, attr_name, None)
                if attr_value is not None and not callable(attr_value):
                    log_data[attr_name] = attr_value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Sanitize the entire log data if enabled
        if self.sanitize and self.sanitizer:
            log_data = self.sanitizer.sanitize_dict(log_data)

        return json.dumps(log_data)


class SanitizingHandler(logging.Handler):
    """Log handler that sanitizes messages before passing to wrapped handler."""

    def __init__(
        self, handler: logging.Handler, sanitizer: DataSanitizer | None = None
    ):
        super().__init__()
        self.handler = handler
        self.sanitizer = sanitizer or DataSanitizer()
        self.setLevel(handler.level)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit sanitized record to wrapped handler."""
        try:
            # Sanitize the record
            sanitized_record = self.sanitizer.sanitize_log_record(record)
            # Pass to wrapped handler
            self.handler.emit(sanitized_record)
        except Exception:
            self.handleError(record)


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN = "\033[32m"
_ANSI_CYAN = "\033[36m"
_ANSI_MAGENTA = "\033[35m"

_LEVEL_COLORS: dict[str, str] = {
    "ERROR": _ANSI_BOLD + _ANSI_RED,
    "CRITICAL": _ANSI_BOLD + _ANSI_RED,
    "WARNING": _ANSI_YELLOW,
}

# Message substrings that warrant special highlighting, checked in order.
_MESSAGE_HIGHLIGHTS: list[tuple[str, str]] = [
    ("Starting enhanced test plan execution", _ANSI_BOLD + _ANSI_MAGENTA),
    ("Starting test case execution", _ANSI_BOLD + _ANSI_CYAN),
    ("Executing test step", _ANSI_CYAN),
    ("Bug report created", _ANSI_BOLD + _ANSI_YELLOW),
    ("Blocker failure detected", _ANSI_BOLD + _ANSI_RED),
    ("Test case failure blocks further execution", _ANSI_BOLD + _ANSI_RED),
    ("Run trace written", _ANSI_GREEN),
]

_THIRD_PARTY_LOGGER_LEVELS: dict[str, int] = {
    # Network/client internals are too noisy at DEBUG and can dump large request
    # payload structures that do not help normal HAINDY debugging.
    "openai": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "asyncio": logging.WARNING,
    "google": logging.WARNING,
    "google.genai": logging.WARNING,
    # Pillow's PNG parser emits chunk-by-chunk debug spam when root logging is DEBUG.
    "PIL": logging.WARNING,
    "PIL.PngImagePlugin": logging.WARNING,
}


class HumanReadableFormatter(logging.Formatter):
    """Render log lines in a compact human-friendly layout."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        level = record.levelname
        logger_name, consumed_key = self._resolve_component(record)
        if not logger_name:
            logger_name = self._format_logger_name(record.name)
        message = record.getMessage()
        message_lines = message.splitlines() or [message]
        primary_message = message_lines[0] if message_lines else ""
        segments = [f"[{level}]", timestamp, logger_name, primary_message]

        consumed_fields = {consumed_key} if consumed_key else set()

        for label, value in self._iter_extra_fields(record, consumed_fields):
            segments.append(f"{label}: {value}")

        formatted = " | ".join(segments)

        if len(message_lines) > 1:
            continuation = "\n".join(f"    {line}" for line in message_lines[1:])
            formatted = f"{formatted}\n{continuation}"

        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"

        if record.stack_info:
            formatted = f"{formatted}\n{self.formatStack(record.stack_info)}"

        if sys.stdout.isatty():
            color = None
            msg = record.getMessage()
            for fragment, highlight in _MESSAGE_HIGHLIGHTS:
                if fragment in msg:
                    color = highlight
                    break
            if color is None:
                color = _LEVEL_COLORS.get(record.levelname)
            if color:
                formatted = f"{color}{formatted}{_ANSI_RESET}"

        return formatted

    def _iter_extra_fields(
        self,
        record: logging.LogRecord,
        consumed: set[str] | None = None,
    ) -> Iterable[tuple[str, str]]:
        seen = set(consumed or ())

        for field in HUMAN_LOG_EXTRA_FIELD_ORDER:
            if hasattr(record, field):
                if field == "taskName":
                    continue
                value = getattr(record, field)
                if value is not None:
                    seen.add(field)
                    yield self._label_for(field), self._stringify(value)

        for key, value in sorted(record.__dict__.items()):
            if key == "taskName":
                continue
            if key in seen or key in STANDARD_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            if key in HUMAN_LOG_SUPPRESSED_FIELDS:
                continue
            if value is None or callable(value):
                continue
            yield self._label_for(key), self._stringify(value)

    @staticmethod
    def _resolve_component(record: logging.LogRecord) -> tuple[str | None, str | None]:
        for candidate in ("component", "agent_name", "agent", "source"):
            value = getattr(record, candidate, None)
            if value:
                return HumanReadableFormatter._title_case(str(value)), candidate
        return None, None

    @staticmethod
    def _format_logger_name(logger_path: str) -> str:
        friendly = logger_path.split(".")[-1]
        friendly = friendly.replace("__", "_")
        if "_" in friendly:
            return friendly.replace("_", " ").title()
        return HumanReadableFormatter._title_case(friendly)

    @staticmethod
    def _label_for(field_name: str) -> str:
        if field_name in HUMAN_LOG_FIELD_LABELS:
            return HUMAN_LOG_FIELD_LABELS[field_name]
        return HumanReadableFormatter._title_case(field_name)

    @staticmethod
    def _title_case(raw: str) -> str:
        if "_" in raw:
            return raw.replace("_", " ").title()
        spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", raw).replace("-", " ")
        return spaced.strip().title()

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True, default=str)
        return str(value)


class AgentLogAdapter(logging.LoggerAdapter):
    """Log adapter for agent-specific logging."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Add agent context to log records."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def set_run_id(run_id: str) -> None:
    """Set the active run identifier for logging and tracing."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id


def get_run_id() -> str:
    """Return the active run identifier."""
    return _CURRENT_RUN_ID or "unknown"


def setup_logging(
    log_level: str | None = None,
    log_format: str | None = None,
    log_file: str | None = None,
    sanitize_logs: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (defaults to settings)
        log_format: Log format 'json' or 'text' (defaults to settings)
        log_file: Optional log file path (defaults to settings)
        sanitize_logs: Whether to sanitize sensitive data in logs

    Returns:
        Root logger instance
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
    console_handler: logging.Handler
    if format_type == "json":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter(sanitize=sanitize_logs))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(HumanReadableFormatter())

    # Honor the configured log level for both JSON and human-readable output.
    # The previous text-mode INFO clamp made `--debug` ineffective unless the
    # caller also switched to JSON logging.
    console_handler.setLevel(numeric_level)

    # Wrap with sanitizing handler if needed
    if sanitize_logs and format_type != "json":  # JSON formatter already sanitizes
        console_handler = SanitizingHandler(console_handler)

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

    # Keep noisy third-party libraries quiet even when HAINDY runs in debug mode.
    for logger_name, logger_level in _THIRD_PARTY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    # Log startup message
    logger = logging.getLogger("haindy")
    if get_run_id() == "unknown":
        set_run_id(_generate_run_id())
    logger.info(
        "HAINDY logging initialized",
        extra={
            "log_level": level,
            "log_format": format_type,
            "log_file": file_path,
            "sanitize_logs": sanitize_logs,
        },
    )

    return root_logger


def get_logger(name: str, **context: Any) -> logging.Logger | AgentLogAdapter:
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
    step_id: str | None = None,
    data: dict[str, Any] | None = None,
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
    content: dict[str, Any],
    correlation_id: str | None = None,
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
    context: dict[str, Any] | None = None,
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
