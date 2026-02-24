"""
Error handling and recovery mechanisms for HAINDY.

This module provides robust error handling, retry logic, and recovery strategies
for the multi-agent testing system.
"""

from .aggregator import ErrorAggregator, ErrorCategory, ErrorMetrics, ErrorReport
from .exceptions import (
    AgentError,
    AutomationError,
    CoordinationError,
    HAINDYError,
    HallucinationError,
    NonRetryableError,
    RecoveryError,
    RetryableError,
    ScopeTriageBlockedError,
    TimeoutError,
    ValidationError,
)
from .recovery import (
    ExponentialBackoffStrategy,
    LinearBackoffStrategy,
    RecoveryAction,
    RecoveryContext,
    RecoveryManager,
    RetryStrategy,
)
from .validation import (
    ActionValidator,
    ConfidenceScorer,
    HallucinationDetector,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
)

__all__ = [
    # Exceptions
    "HAINDYError",
    "AgentError",
    "AutomationError",
    "ValidationError",
    "RecoveryError",
    "HallucinationError",
    "RetryableError",
    "NonRetryableError",
    "TimeoutError",
    "CoordinationError",
    "ScopeTriageBlockedError",

    # Recovery
    "RetryStrategy",
    "ExponentialBackoffStrategy",
    "LinearBackoffStrategy",
    "RecoveryManager",
    "RecoveryContext",
    "RecoveryAction",

    # Validation
    "ActionValidator",
    "ConfidenceScorer",
    "HallucinationDetector",
    "ValidationResult",
    "ValidationRule",
    "ValidationSeverity",

    # Aggregation
    "ErrorAggregator",
    "ErrorCategory",
    "ErrorMetrics",
    "ErrorReport"
]
