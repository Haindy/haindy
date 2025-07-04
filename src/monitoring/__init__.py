"""
Monitoring module exports.
"""

from src.monitoring.logger import (
    get_logger,
    log_agent_communication,
    log_performance_metric,
    log_test_event,
    setup_logging,
    JSONFormatter,
    SanitizingHandler,
)

from src.monitoring.analytics import (
    MetricsCollector,
    TestOutcome,
    start_test,
    end_test,
    record_step,
    record_api_call,
    record_browser_action,
    get_analytics,
)

from src.monitoring.reporter import (
    ReportGenerator,
    TestExecutionReport,
    ReportConfig,
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
    "record_browser_action",
    "get_analytics",
    
    # Reporter
    "ReportGenerator",
    "TestExecutionReport",
    "ReportConfig",
]