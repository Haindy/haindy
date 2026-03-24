"""Logging setup for tool-call CLI and daemon processes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from haindy.monitoring.logger import (
    _THIRD_PARTY_LOGGER_LEVELS,
    HumanReadableFormatter,
    JSONFormatter,
    SanitizingHandler,
    set_run_id,
)


def setup_tool_call_logging(
    *,
    log_path: Path,
    run_id: str,
    debug_to_stderr: bool = False,
) -> logging.Logger:
    """Configure root logging without ever writing logs to stdout."""

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    if debug_to_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.DEBUG)
        stderr_handler.setFormatter(HumanReadableFormatter())
        root_logger.addHandler(SanitizingHandler(stderr_handler))

    for logger_name, logger_level in _THIRD_PARTY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    set_run_id(run_id)
    return root_logger
