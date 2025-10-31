"""
Error handling and recovery mechanisms for HAINDY.

This module provides robust error handling, retry logic, and recovery strategies
for the multi-agent testing system.
"""

from .exceptions import (
    HAINDYError,
    AgentError,
    BrowserError,
    ValidationError,
    RecoveryError,
    HallucinationError,
    RetryableError,
    NonRetryableError,
    TimeoutError,
    CoordinationError,
    ScopeTriageBlockedError,
)

from .recovery import (
    RetryStrategy,
    ExponentialBackoffStrategy,
    LinearBackoffStrategy,
    RecoveryManager,
    RecoveryContext,
    RecoveryAction
)

from .validation import (
    ActionValidator,
    ConfidenceScorer,
    HallucinationDetector,
    ValidationResult,
    ValidationRule,
    ValidationSeverity
)

from .aggregator import (
    ErrorAggregator,
    ErrorCategory,
    ErrorMetrics,
    ErrorReport
)

__all__ = [
    # Exceptions
    "HAINDYError",
    "AgentError",
    "BrowserError",
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
