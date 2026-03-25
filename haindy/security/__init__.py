"""
Security components for HAINDY.

This module provides rate limiting and data sanitization to ensure
safe and controlled test execution.
"""

from .rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
    SlidingWindowCounter,
    TokenBucket,
)
from .sanitizer import (
    DataSanitizer,
    RedactionMethod,
    SanitizationRule,
    SensitiveDataPattern,
    mask_sensitive_data,
    sanitize_dict,
    sanitize_string,
    set_literal_redactions,
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "RateLimitExceeded",
    "RateLimitConfig",
    "TokenBucket",
    "SlidingWindowCounter",
    # Data sanitization
    "DataSanitizer",
    "SensitiveDataPattern",
    "SanitizationRule",
    "RedactionMethod",
    "sanitize_dict",
    "sanitize_string",
    "set_literal_redactions",
    "mask_sensitive_data",
]
