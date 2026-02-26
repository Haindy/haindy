"""
Monitoring module exports.
"""

from src.monitoring.analytics import (
    MetricsCollector,
    TestOutcome,
    end_test,
    get_analytics,
    record_api_call,
    record_automation_action,
    record_step,
    start_test,
)
from src.monitoring.logger import (
    JSONFormatter,
    SanitizingHandler,
    get_logger,
    log_agent_communication,
    log_performance_metric,
    log_test_event,
    setup_logging,
)
from src.monitoring.reporter import (
    ReportConfig,
    ReportGenerator,
    TestExecutionReport,
)

__all__ = [
    # Logger
    "setup_logging",
    "get_logger",
    "log_test_event",
    "log_agent_communication",
    "log_performance_metric",
    "JSONFormatter",
    "SanitizingHandler",
    # Analytics
    "MetricsCollector",
    "TestOutcome",
    "start_test",
    "end_test",
    "record_step",
    "record_api_call",
    "record_automation_action",
    "get_analytics",
    # Reporter
    "ReportGenerator",
    "TestExecutionReport",
    "ReportConfig",
]
