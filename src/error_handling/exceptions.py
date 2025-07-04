"""
Custom exception hierarchy for HAINDY error handling.

Provides a structured exception hierarchy that enables proper error categorization,
retry logic, and recovery strategies.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class HAINDYError(Exception):
    """Base exception for all HAINDY errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


class RetryableError(HAINDYError):
    """Base class for errors that can be retried."""
    
    def __init__(
        self,
        message: str,
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.retry_count = 0
    
    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.retry_count < self.max_retries


class NonRetryableError(HAINDYError):
    """Base class for errors that should not be retried."""
    pass


class AgentError(HAINDYError):
    """Error raised by AI agents during test execution."""
    
    def __init__(
        self,
        message: str,
        agent_name: str,
        agent_type: str,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.details.update({
            "agent_name": agent_name,
            "agent_type": agent_type
        })


class BrowserError(RetryableError):
    """Error related to browser automation."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.url = url
        self.selector = selector
        self.action = action
        self.details.update({
            "url": url,
            "selector": selector,
            "action": action
        })


class ValidationError(NonRetryableError):
    """Error raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_type: str,
        failed_rules: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.failed_rules = failed_rules or []
        self.details.update({
            "validation_type": validation_type,
            "failed_rules": failed_rules
        })


class RecoveryError(HAINDYError):
    """Error raised when recovery strategies fail."""
    
    def __init__(
        self,
        message: str,
        recovery_strategy: str,
        original_error: Optional[Exception] = None,
        **kwargs
    ):
        super().__init__(message, cause=original_error, **kwargs)
        self.recovery_strategy = recovery_strategy
        self.original_error = original_error
        self.details.update({
            "recovery_strategy": recovery_strategy,
            "original_error": str(original_error) if original_error else None
        })


class HallucinationError(NonRetryableError):
    """Error raised when AI agent hallucination is detected."""
    
    def __init__(
        self,
        message: str,
        agent_name: str,
        hallucination_type: str,
        confidence_score: float,
        evidence: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.hallucination_type = hallucination_type
        self.confidence_score = confidence_score
        self.evidence = evidence or []
        self.details.update({
            "agent_name": agent_name,
            "hallucination_type": hallucination_type,
            "confidence_score": confidence_score,
            "evidence": evidence
        })


class TimeoutError(RetryableError):
    """Error raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        timeout_ms: int,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.details.update({
            "operation": operation,
            "timeout_ms": timeout_ms
        })


class CoordinationError(HAINDYError):
    """Error raised during multi-agent coordination."""
    
    def __init__(
        self,
        message: str,
        agents_involved: List[str],
        coordination_phase: str,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.agents_involved = agents_involved
        self.coordination_phase = coordination_phase
        self.details.update({
            "agents_involved": agents_involved,
            "coordination_phase": coordination_phase
        })