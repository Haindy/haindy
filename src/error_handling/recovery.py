"""
Recovery strategies and retry logic for error handling.

Implements various retry strategies with backoff algorithms and
recovery mechanisms for different error scenarios.
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Type, TypeVar, Union,
    Awaitable, cast
)

from .exceptions import (
    HAINDYError, RecoveryError, RetryableError, NonRetryableError,
    TimeoutError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = auto()
    FALLBACK = auto()
    SKIP = auto()
    ABORT = auto()
    ESCALATE = auto()


@dataclass
class RecoveryContext:
    """Context information for recovery decisions."""
    error: Exception
    operation_name: str
    attempt_number: int = 1
    start_time: datetime = field(default_factory=datetime.utcnow)
    previous_errors: List[Exception] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since operation started."""
        return datetime.utcnow() - self.start_time
    
    @property
    def is_retryable(self) -> bool:
        """Check if the error is retryable."""
        return isinstance(self.error, RetryableError)
    
    def add_attempt(self, error: Exception) -> None:
        """Add a new attempt with its error."""
        self.attempt_number += 1
        self.previous_errors.append(error)
        self.error = error


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""
    
    @abstractmethod
    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay in milliseconds for the given attempt."""
        pass
    
    @abstractmethod
    def should_retry(self, context: RecoveryContext) -> bool:
        """Determine if operation should be retried."""
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff with jitter."""
    
    def __init__(
        self,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 60000,
        max_attempts: int = 5,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.max_attempts = max_attempts
        self.multiplier = multiplier
        self.jitter = jitter
    
    def get_delay_ms(self, attempt: int) -> int:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(
            self.base_delay_ms * (self.multiplier ** (attempt - 1)),
            self.max_delay_ms
        )
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return int(delay)
    
    def should_retry(self, context: RecoveryContext) -> bool:
        """Check if retry should be attempted."""
        if not context.is_retryable:
            return False
        
        if context.attempt_number >= self.max_attempts:
            logger.warning(
                f"Max attempts ({self.max_attempts}) reached for {context.operation_name}"
            )
            return False
        
        # Check if error implements retry logic
        if isinstance(context.error, RetryableError):
            return context.error.can_retry()
        
        return True


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff strategy."""
    
    def __init__(
        self,
        delay_increment_ms: int = 1000,
        max_delay_ms: int = 10000,
        max_attempts: int = 3
    ):
        self.delay_increment_ms = delay_increment_ms
        self.max_delay_ms = max_delay_ms
        self.max_attempts = max_attempts
    
    def get_delay_ms(self, attempt: int) -> int:
        """Calculate linear backoff delay."""
        delay = self.delay_increment_ms * attempt
        return min(delay, self.max_delay_ms)
    
    def should_retry(self, context: RecoveryContext) -> bool:
        """Check if retry should be attempted."""
        return (
            context.is_retryable and
            context.attempt_number < self.max_attempts
        )


class RecoveryManager:
    """Manages error recovery and retry logic."""
    
    def __init__(
        self,
        default_strategy: Optional[RetryStrategy] = None,
        error_handlers: Optional[Dict[Type[Exception], Callable]] = None
    ):
        self.default_strategy = default_strategy or ExponentialBackoffStrategy()
        self.error_handlers = error_handlers or {}
        self._recovery_stats: Dict[str, Dict[str, Any]] = {}
    
    async def execute_with_recovery(
        self,
        operation: Callable[..., Awaitable[T]],
        operation_name: str,
        *args,
        retry_strategy: Optional[RetryStrategy] = None,
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        timeout_ms: Optional[int] = None,
        **kwargs
    ) -> T:
        """
        Execute an operation with automatic retry and recovery.
        
        Args:
            operation: Async function to execute
            operation_name: Name for logging/tracking
            retry_strategy: Custom retry strategy (uses default if None)
            fallback: Fallback function if all retries fail
            timeout_ms: Overall timeout for all attempts
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Result from successful operation or fallback
            
        Raises:
            RecoveryError: If all recovery attempts fail
        """
        strategy = retry_strategy or self.default_strategy
        context = RecoveryContext(
            error=Exception("Not started"),
            operation_name=operation_name
        )
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                # Apply timeout if specified
                if timeout_ms:
                    remaining_ms = timeout_ms - int(
                        (asyncio.get_event_loop().time() - start_time) * 1000
                    )
                    if remaining_ms <= 0:
                        raise TimeoutError(
                            f"Operation {operation_name} timed out",
                            operation=operation_name,
                            timeout_ms=timeout_ms
                        )
                    
                    result = await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=remaining_ms / 1000
                    )
                else:
                    result = await operation(*args, **kwargs)
                
                # Success - record stats
                self._record_success(context)
                return result
                
            except Exception as e:
                context.add_attempt(e)
                
                # Check for custom error handler
                handler = self._get_error_handler(e)
                if handler:
                    try:
                        return await self._apply_handler(handler, e, context)
                    except Exception as handler_error:
                        logger.error(
                            f"Error handler failed for {operation_name}: {handler_error}"
                        )
                
                # Determine recovery action
                action = self._determine_recovery_action(e, context, strategy)
                
                if action == RecoveryAction.RETRY:
                    delay_ms = strategy.get_delay_ms(context.attempt_number)
                    logger.info(
                        f"Retrying {operation_name} after {delay_ms}ms "
                        f"(attempt {context.attempt_number + 1})"
                    )
                    await asyncio.sleep(delay_ms / 1000)
                    continue
                    
                elif action == RecoveryAction.FALLBACK and fallback:
                    logger.info(f"Using fallback for {operation_name}")
                    try:
                        return await fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        raise RecoveryError(
                            f"Fallback failed for {operation_name}",
                            recovery_strategy="fallback",
                            original_error=e
                        ) from fallback_error
                
                else:
                    # No recovery possible
                    self._record_failure(context)
                    raise RecoveryError(
                        f"All recovery attempts failed for {operation_name}",
                        recovery_strategy=strategy.__class__.__name__,
                        original_error=e
                    ) from e
    
    def _determine_recovery_action(
        self,
        error: Exception,
        context: RecoveryContext,
        strategy: RetryStrategy
    ) -> RecoveryAction:
        """Determine the appropriate recovery action."""
        # Non-retryable errors should not be retried
        if isinstance(error, NonRetryableError):
            return RecoveryAction.ABORT
        
        # Check strategy
        if strategy.should_retry(context):
            return RecoveryAction.RETRY
        
        # If we have a fallback available, try it
        return RecoveryAction.FALLBACK
    
    def _get_error_handler(
        self,
        error: Exception
    ) -> Optional[Callable]:
        """Get custom error handler for exception type."""
        for error_type, handler in self.error_handlers.items():
            if isinstance(error, error_type):
                return handler
        return None
    
    async def _apply_handler(
        self,
        handler: Callable,
        error: Exception,
        context: RecoveryContext
    ) -> Any:
        """Apply custom error handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(error, context)
        else:
            return handler(error, context)
    
    def _record_success(self, context: RecoveryContext) -> None:
        """Record successful operation statistics."""
        stats = self._recovery_stats.setdefault(
            context.operation_name,
            {"successes": 0, "failures": 0, "retries": 0}
        )
        stats["successes"] += 1
        if context.attempt_number > 1:
            stats["retries"] += context.attempt_number - 1
    
    def _record_failure(self, context: RecoveryContext) -> None:
        """Record failed operation statistics."""
        stats = self._recovery_stats.setdefault(
            context.operation_name,
            {"successes": 0, "failures": 0, "retries": 0}
        )
        stats["failures"] += 1
        stats["retries"] += context.attempt_number - 1
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get recovery statistics for all operations."""
        return self._recovery_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all recovery statistics."""
        self._recovery_stats.clear()