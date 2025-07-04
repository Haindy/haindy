"""
Unit tests for error handling exceptions.
"""

import pytest
from datetime import datetime

from src.error_handling.exceptions import (
    HAINDYError, RetryableError, NonRetryableError, AgentError,
    BrowserError, ValidationError, RecoveryError, HallucinationError,
    TimeoutError, CoordinationError
)


class TestHAINDYError:
    """Test base exception class."""
    
    def test_basic_creation(self):
        """Test basic error creation."""
        error = HAINDYError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "HAINDYError"
        assert error.details == {}
        assert error.cause is None
        assert isinstance(error.timestamp, datetime)
    
    def test_with_details(self):
        """Test error with details."""
        details = {"key": "value", "count": 42}
        error = HAINDYError(
            "Test error",
            error_code="TEST001",
            details=details
        )
        assert error.error_code == "TEST001"
        assert error.details == details
    
    def test_with_cause(self):
        """Test error with cause."""
        cause = ValueError("Original error")
        error = HAINDYError("Wrapped error", cause=cause)
        assert error.cause == cause
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = HAINDYError(
            "Test error",
            error_code="TEST001",
            details={"key": "value"}
        )
        
        result = error.to_dict()
        assert result["error_type"] == "HAINDYError"
        assert result["error_code"] == "TEST001"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}
        assert "timestamp" in result
        assert result["cause"] is None


class TestRetryableError:
    """Test retryable error class."""
    
    def test_default_retry_settings(self):
        """Test default retry configuration."""
        error = RetryableError("Retry me")
        assert error.max_retries == 3
        assert error.retry_delay_ms == 1000
        assert error.retry_count == 0
        assert error.can_retry() is True
    
    def test_custom_retry_settings(self):
        """Test custom retry configuration."""
        error = RetryableError(
            "Retry me",
            max_retries=5,
            retry_delay_ms=2000
        )
        assert error.max_retries == 5
        assert error.retry_delay_ms == 2000
    
    def test_retry_counting(self):
        """Test retry counting logic."""
        error = RetryableError("Retry me", max_retries=2)
        
        assert error.can_retry() is True
        error.increment_retry()
        assert error.retry_count == 1
        assert error.can_retry() is True
        
        error.increment_retry()
        assert error.retry_count == 2
        assert error.can_retry() is False


class TestAgentError:
    """Test agent error class."""
    
    def test_agent_error_creation(self):
        """Test agent error with required fields."""
        error = AgentError(
            "Agent failed",
            agent_name="test_runner",
            agent_type="TestRunnerAgent"
        )
        
        assert error.message == "Agent failed"
        assert error.agent_name == "test_runner"
        assert error.agent_type == "TestRunnerAgent"
        assert error.details["agent_name"] == "test_runner"
        assert error.details["agent_type"] == "TestRunnerAgent"


class TestBrowserError:
    """Test browser error class."""
    
    def test_browser_error_creation(self):
        """Test browser error with optional fields."""
        error = BrowserError(
            "Click failed",
            url="https://example.com",
            selector="#submit-button",
            action="click"
        )
        
        assert error.message == "Click failed"
        assert error.url == "https://example.com"
        assert error.selector == "#submit-button"
        assert error.action == "click"
        assert isinstance(error, RetryableError)
    
    def test_browser_error_minimal(self):
        """Test browser error with minimal info."""
        error = BrowserError("Page not found")
        assert error.url is None
        assert error.selector is None
        assert error.action is None


class TestValidationError:
    """Test validation error class."""
    
    def test_validation_error(self):
        """Test validation error creation."""
        failed_rules = ["rule1", "rule2"]
        error = ValidationError(
            "Validation failed",
            validation_type="action_validation",
            failed_rules=failed_rules
        )
        
        assert error.validation_type == "action_validation"
        assert error.failed_rules == failed_rules
        assert isinstance(error, NonRetryableError)


class TestHallucinationError:
    """Test hallucination error class."""
    
    def test_hallucination_error(self):
        """Test hallucination error with evidence."""
        evidence = ["Element not in screenshot", "Impossible action"]
        error = HallucinationError(
            "Agent hallucinated",
            agent_name="action_agent",
            hallucination_type="PHANTOM_ELEMENT",
            confidence_score=0.85,
            evidence=evidence
        )
        
        assert error.agent_name == "action_agent"
        assert error.hallucination_type == "PHANTOM_ELEMENT"
        assert error.confidence_score == 0.85
        assert error.evidence == evidence
        assert isinstance(error, NonRetryableError)


class TestTimeoutError:
    """Test timeout error class."""
    
    def test_timeout_error(self):
        """Test timeout error creation."""
        error = TimeoutError(
            "Operation timed out",
            operation="page_load",
            timeout_ms=5000
        )
        
        assert error.operation == "page_load"
        assert error.timeout_ms == 5000
        assert isinstance(error, RetryableError)


class TestCoordinationError:
    """Test coordination error class."""
    
    def test_coordination_error(self):
        """Test coordination error with multiple agents."""
        agents = ["planner", "runner", "evaluator"]
        error = CoordinationError(
            "Agent communication failed",
            agents_involved=agents,
            coordination_phase="test_execution"
        )
        
        assert error.agents_involved == agents
        assert error.coordination_phase == "test_execution"