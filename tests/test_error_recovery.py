"""
Unit tests for error recovery and retry logic.
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.error_handling.recovery import (
    RecoveryAction, RecoveryContext, RetryStrategy,
    ExponentialBackoffStrategy, LinearBackoffStrategy,
    RecoveryManager
)
from src.error_handling.exceptions import (
    RetryableError, NonRetryableError, TimeoutError,
    RecoveryError
)


class TestRecoveryContext:
    """Test recovery context."""
    
    def test_context_creation(self):
        """Test context initialization."""
        error = ValueError("Test error")
        context = RecoveryContext(
            error=error,
            operation_name="test_op"
        )
        
        assert context.error == error
        assert context.operation_name == "test_op"
        assert context.attempt_number == 1
        assert context.previous_errors == []
        assert isinstance(context.start_time, datetime)
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        context = RecoveryContext(
            error=ValueError(),
            operation_name="test_op"
        )
        
        # Mock start time to be 1 second ago
        context.start_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        elapsed = context.elapsed_time
        
        assert elapsed.total_seconds() >= 1.0
        assert elapsed.total_seconds() < 2.0
    
    def test_is_retryable(self):
        """Test retryable error detection."""
        retryable = RetryableError("Can retry")
        non_retryable = NonRetryableError("Cannot retry")
        
        context1 = RecoveryContext(error=retryable, operation_name="op1")
        context2 = RecoveryContext(error=non_retryable, operation_name="op2")
        
        assert context1.is_retryable is True
        assert context2.is_retryable is False
    
    def test_add_attempt(self):
        """Test adding attempts."""
        error1 = ValueError("First error")
        error2 = ValueError("Second error")
        
        context = RecoveryContext(error=error1, operation_name="test")
        assert context.attempt_number == 1
        assert len(context.previous_errors) == 0
        
        context.add_attempt(error2)
        assert context.attempt_number == 2
        assert context.error == error2
        assert len(context.previous_errors) == 1
        assert str(context.previous_errors[0]) == "First error"


class TestExponentialBackoffStrategy:
    """Test exponential backoff strategy."""
    
    def test_default_settings(self):
        """Test default configuration."""
        strategy = ExponentialBackoffStrategy()
        assert strategy.base_delay_ms == 1000
        assert strategy.max_delay_ms == 60000
        assert strategy.max_attempts == 5
        assert strategy.multiplier == 2.0
        assert strategy.jitter is True
    
    def test_delay_calculation_no_jitter(self):
        """Test delay calculation without jitter."""
        strategy = ExponentialBackoffStrategy(
            base_delay_ms=100,
            multiplier=2.0,
            jitter=False
        )
        
        assert strategy.get_delay_ms(1) == 100  # 100 * 2^0
        assert strategy.get_delay_ms(2) == 200  # 100 * 2^1
        assert strategy.get_delay_ms(3) == 400  # 100 * 2^2
        assert strategy.get_delay_ms(4) == 800  # 100 * 2^3
    
    def test_max_delay_cap(self):
        """Test maximum delay capping."""
        strategy = ExponentialBackoffStrategy(
            base_delay_ms=1000,
            max_delay_ms=3000,
            jitter=False
        )
        
        assert strategy.get_delay_ms(1) == 1000
        assert strategy.get_delay_ms(2) == 2000
        assert strategy.get_delay_ms(3) == 3000  # Capped
        assert strategy.get_delay_ms(4) == 3000  # Still capped
    
    def test_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        strategy = ExponentialBackoffStrategy(
            base_delay_ms=1000,
            jitter=True
        )
        
        # Get multiple delays for same attempt
        delays = [strategy.get_delay_ms(2) for _ in range(10)]
        
        # Check all are within expected range (2000 ± 25%)
        for delay in delays:
            assert 1500 <= delay <= 2500
        
        # Check they're not all the same (jitter working)
        assert len(set(delays)) > 1
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        strategy = ExponentialBackoffStrategy(max_attempts=3)
        
        # Test with retryable error
        error = RetryableError("Test", max_retries=5)  # Higher than strategy max
        context = RecoveryContext(error=error, operation_name="test")
        
        # First attempt (initial)
        assert strategy.should_retry(context) is True
        
        # Second attempt 
        context.add_attempt(RetryableError("Test"))
        assert context.attempt_number == 2
        assert strategy.should_retry(context) is True
        
        # Third attempt - should still be allowed
        context.add_attempt(RetryableError("Test"))
        assert context.attempt_number == 3
        assert strategy.should_retry(context) is False  # Max attempts (3) reached
        
        # Test with non-retryable error
        non_retryable = NonRetryableError("Test")
        context2 = RecoveryContext(error=non_retryable, operation_name="test")
        assert strategy.should_retry(context2) is False


class TestLinearBackoffStrategy:
    """Test linear backoff strategy."""
    
    def test_delay_calculation(self):
        """Test linear delay calculation."""
        strategy = LinearBackoffStrategy(
            delay_increment_ms=500,
            max_delay_ms=2000
        )
        
        assert strategy.get_delay_ms(1) == 500
        assert strategy.get_delay_ms(2) == 1000
        assert strategy.get_delay_ms(3) == 1500
        assert strategy.get_delay_ms(4) == 2000  # Capped
        assert strategy.get_delay_ms(5) == 2000  # Still capped
    
    def test_should_retry(self):
        """Test retry decision."""
        strategy = LinearBackoffStrategy(max_attempts=2)
        
        error = RetryableError("Test")
        context = RecoveryContext(error=error, operation_name="test")
        
        assert strategy.should_retry(context) is True
        context.add_attempt(error)
        assert strategy.should_retry(context) is False  # Max attempts reached


class TestRecoveryManager:
    """Test recovery manager."""
    
    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test successful operation execution."""
        manager = RecoveryManager()
        
        async def successful_op(value):
            return value * 2
        
        result = await manager.execute_with_recovery(
            successful_op,
            "test_op",
            21
        )
        
        assert result == 42
        stats = manager.get_statistics()
        assert stats["test_op"]["successes"] == 1
        assert stats["test_op"]["failures"] == 0
        assert stats["test_op"]["retries"] == 0
    
    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test operation that fails then succeeds."""
        manager = RecoveryManager()
        
        call_count = 0
        
        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure", max_retries=5)
            return "success"
        
        # Use fast retry strategy for test with enough attempts
        strategy = LinearBackoffStrategy(
            delay_increment_ms=10,
            max_attempts=4  # Allow 4 attempts total (initial + 3 retries)
        )
        
        result = await manager.execute_with_recovery(
            flaky_op,
            "flaky_op",
            retry_strategy=strategy
        )
        
        assert result == "success"
        assert call_count == 3
        
        stats = manager.get_statistics()
        assert stats["flaky_op"]["successes"] == 1
        assert stats["flaky_op"]["retries"] == 2
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test non-retryable error handling."""
        manager = RecoveryManager()
        
        async def failing_op():
            raise NonRetryableError("Permanent failure")
        
        with pytest.raises(RecoveryError) as exc_info:
            await manager.execute_with_recovery(
                failing_op,
                "failing_op"
            )
        
        assert "All recovery attempts failed" in str(exc_info.value)
        assert exc_info.value.recovery_strategy == "ExponentialBackoffStrategy"
    
    @pytest.mark.asyncio
    async def test_fallback_function(self):
        """Test fallback function execution."""
        manager = RecoveryManager()
        
        async def failing_op():
            raise RetryableError("Always fails", max_retries=1)
        
        async def fallback_op():
            return "fallback_result"
        
        strategy = LinearBackoffStrategy(
            max_attempts=1,
            delay_increment_ms=10
        )
        
        result = await manager.execute_with_recovery(
            failing_op,
            "op_with_fallback",
            retry_strategy=strategy,
            fallback=fallback_op
        )
        
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_fallback_failure(self):
        """Test when fallback also fails."""
        manager = RecoveryManager()
        
        async def failing_op():
            raise RetryableError("Primary fails", max_retries=1)
        
        async def failing_fallback():
            raise ValueError("Fallback fails too")
        
        strategy = LinearBackoffStrategy(
            max_attempts=1,
            delay_increment_ms=10
        )
        
        with pytest.raises(RecoveryError) as exc_info:
            await manager.execute_with_recovery(
                failing_op,
                "double_failure",
                retry_strategy=strategy,
                fallback=failing_fallback
            )
        
        assert "Fallback failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout functionality."""
        manager = RecoveryManager()
        
        async def slow_op():
            await asyncio.sleep(0.2)
            return "done"
        
        # Should succeed with sufficient timeout
        result = await manager.execute_with_recovery(
            slow_op,
            "slow_op",
            timeout_ms=300
        )
        assert result == "done"
        
        # Should fail with insufficient timeout
        with pytest.raises(RecoveryError) as exc_info:
            await manager.execute_with_recovery(
                slow_op,
                "slow_op",
                timeout_ms=50
            )
        
        # Check for timeout error in chain
        assert isinstance(exc_info.value.cause, asyncio.TimeoutError)
    
    @pytest.mark.asyncio
    async def test_custom_error_handler(self):
        """Test custom error handler."""
        handled = False
        
        async def custom_handler(error, context):
            nonlocal handled
            handled = True
            return "handled"
        
        manager = RecoveryManager(
            error_handlers={ValueError: custom_handler}
        )
        
        async def op_with_value_error():
            raise ValueError("Test error")
        
        result = await manager.execute_with_recovery(
            op_with_value_error,
            "custom_handled"
        )
        
        assert result == "handled"
        assert handled is True
    
    def test_statistics_management(self):
        """Test statistics tracking."""
        manager = RecoveryManager()
        
        # Initially empty
        assert manager.get_statistics() == {}
        
        # Add some stats manually (normally done via execute_with_recovery)
        manager._record_success(
            RecoveryContext(error=None, operation_name="op1")
        )
        manager._record_failure(
            RecoveryContext(error=None, operation_name="op2")
        )
        
        stats = manager.get_statistics()
        assert stats["op1"]["successes"] == 1
        assert stats["op2"]["failures"] == 1
        
        # Reset statistics
        manager.reset_statistics()
        assert manager.get_statistics() == {}