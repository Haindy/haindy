"""
Evaluation utilities for test result analysis.

This module provides shared evaluation functionality used by agents
to analyze test outcomes, detect errors, and score confidence.
"""

from .error_detection import ErrorDetector, ErrorType, ErrorSeverity
from .confidence import ConfidenceScorer
from .validators import ValidationHelpers

__all__ = [
    "ErrorDetector",
    "ErrorType", 
    "ErrorSeverity",
    "ConfidenceScorer",
    "ValidationHelpers",
]