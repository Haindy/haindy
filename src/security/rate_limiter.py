"""
Rate limiting implementation for HAINDY.

Provides configurable rate limiting to prevent API abuse and ensure
controlled test execution.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Optional, Deque, Any
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Available rate limiting strategies."""
    TOKEN_BUCKET = auto()
    SLIDING_WINDOW = auto()
    FIXED_WINDOW = auto()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        limit: int,
        window_seconds: int,
        retry_after: Optional[float] = None
    ):
        super().__init__(message)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # API rate limits
    api_calls_per_minute: int = 60
    api_burst_size: int = 10
    
    # Browser action limits
    browser_actions_per_minute: int = 120
    browser_actions_burst: int = 20
    
    # Agent message limits
    agent_messages_per_minute: int = 300
    agent_messages_burst: int = 50
    
    # Screenshot limits
    screenshots_per_minute: int = 30
    screenshots_burst: int = 5
    
    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    
    # Global enable/disable
    enabled: bool = True


class TokenBucket:
    """Token bucket rate limiting algorithm."""
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens refilled per second
            initial_tokens: Starting tokens (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def acquire_or_wait(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.monotonic()
        
        while True:
            if await self.acquire(tokens):
                return True
            
            # Calculate wait time
            wait_time = self._calculate_wait_time(tokens)
            
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            await asyncio.sleep(wait_time)
    
    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
    
    def _calculate_wait_time(self, tokens: int) -> float:
        """Calculate time to wait for tokens."""
        if self.refill_rate <= 0:
            return float('inf')
        
        tokens_needed = tokens - self.tokens
        if tokens_needed <= 0:
            return 0
        
        return tokens_needed / self.refill_rate
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens (without refilling)."""
        return self.tokens


class SlidingWindowCounter:
    """Sliding window rate limiting algorithm."""
    
    def __init__(self, window_seconds: int, limit: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_seconds: Window size in seconds
            limit: Maximum requests in window
        """
        self.window_seconds = window_seconds
        self.limit = limit
        self.requests: Deque[float] = deque()
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)
            
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False
    
    def _cleanup_old_requests(self, now: float) -> None:
        """Remove requests outside the window."""
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        now = time.time()
        self._cleanup_old_requests(now)
        return len(self.requests)
    
    def time_until_next_allowed(self) -> float:
        """Calculate time until next request is allowed."""
        if len(self.requests) < self.limit:
            return 0
        
        if not self.requests:
            return 0
        
        oldest_request = self.requests[0]
        now = time.time()
        time_until_expire = (oldest_request + self.window_seconds) - now
        
        return max(0, time_until_expire)


class RateLimiter:
    """Main rate limiter coordinating different limits."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter with configuration."""
        self.config = config or RateLimitConfig()
        self._limiters: Dict[str, Any] = {}
        self._stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"allowed": 0, "rejected": 0})
        
        if self.config.enabled:
            self._setup_limiters()
    
    def _setup_limiters(self) -> None:
        """Set up individual limiters based on strategy."""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            # API limiter
            self._limiters["api"] = TokenBucket(
                capacity=self.config.api_burst_size,
                refill_rate=self.config.api_calls_per_minute / 60
            )
            
            # Browser actions limiter
            self._limiters["browser"] = TokenBucket(
                capacity=self.config.browser_actions_burst,
                refill_rate=self.config.browser_actions_per_minute / 60
            )
            
            # Agent messages limiter
            self._limiters["agent"] = TokenBucket(
                capacity=self.config.agent_messages_burst,
                refill_rate=self.config.agent_messages_per_minute / 60
            )
            
            # Screenshots limiter
            self._limiters["screenshot"] = TokenBucket(
                capacity=self.config.screenshots_burst,
                refill_rate=self.config.screenshots_per_minute / 60
            )
            
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            # API limiter
            self._limiters["api"] = SlidingWindowCounter(
                window_seconds=60,
                limit=self.config.api_calls_per_minute
            )
            
            # Browser actions limiter
            self._limiters["browser"] = SlidingWindowCounter(
                window_seconds=60,
                limit=self.config.browser_actions_per_minute
            )
            
            # Agent messages limiter
            self._limiters["agent"] = SlidingWindowCounter(
                window_seconds=60,
                limit=self.config.agent_messages_per_minute
            )
            
            # Screenshots limiter
            self._limiters["screenshot"] = SlidingWindowCounter(
                window_seconds=60,
                limit=self.config.screenshots_per_minute
            )
    
    async def check_api_call(self, wait: bool = False) -> bool:
        """
        Check if API call is allowed.
        
        Args:
            wait: Whether to wait if limit exceeded
            
        Returns:
            True if allowed, False otherwise
            
        Raises:
            RateLimitExceeded: If limit exceeded and wait=False
        """
        return await self._check_limit("api", wait)
    
    async def check_browser_action(self, wait: bool = False) -> bool:
        """Check if browser action is allowed."""
        return await self._check_limit("browser", wait)
    
    async def check_agent_message(self, wait: bool = False) -> bool:
        """Check if agent message is allowed."""
        return await self._check_limit("agent", wait)
    
    async def check_screenshot(self, wait: bool = False) -> bool:
        """Check if screenshot is allowed."""
        return await self._check_limit("screenshot", wait)
    
    async def _check_limit(self, limit_type: str, wait: bool) -> bool:
        """
        Check rate limit for a specific type.
        
        Args:
            limit_type: Type of limit to check
            wait: Whether to wait if limit exceeded
            
        Returns:
            True if allowed
            
        Raises:
            RateLimitExceeded: If limit exceeded and wait=False
        """
        if not self.config.enabled:
            return True
        
        limiter = self._limiters.get(limit_type)
        if not limiter:
            return True
        
        allowed = False
        
        if isinstance(limiter, TokenBucket):
            if wait:
                allowed = await limiter.acquire_or_wait(1, timeout=30)
            else:
                allowed = await limiter.acquire(1)
                
        elif isinstance(limiter, SlidingWindowCounter):
            allowed = await limiter.is_allowed()
            if not allowed and wait:
                wait_time = limiter.time_until_next_allowed()
                if wait_time <= 30:
                    await asyncio.sleep(wait_time)
                    allowed = await limiter.is_allowed()
        
        # Update statistics
        if allowed:
            self._stats[limit_type]["allowed"] += 1
        else:
            self._stats[limit_type]["rejected"] += 1
            
            if not wait:
                # Calculate retry after time
                retry_after = None
                if isinstance(limiter, TokenBucket):
                    retry_after = limiter._calculate_wait_time(1)
                elif isinstance(limiter, SlidingWindowCounter):
                    retry_after = limiter.time_until_next_allowed()
                
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {limit_type}",
                    limit=self._get_limit_for_type(limit_type),
                    window_seconds=60,
                    retry_after=retry_after
                )
        
        return allowed
    
    def _get_limit_for_type(self, limit_type: str) -> int:
        """Get configured limit for a type."""
        limits = {
            "api": self.config.api_calls_per_minute,
            "browser": self.config.browser_actions_per_minute,
            "agent": self.config.agent_messages_per_minute,
            "screenshot": self.config.screenshots_per_minute
        }
        return limits.get(limit_type, 0)
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limiting statistics."""
        stats = {}
        
        for limit_type, counts in self._stats.items():
            stats[limit_type] = {
                "allowed": counts["allowed"],
                "rejected": counts["rejected"],
                "total": counts["allowed"] + counts["rejected"],
                "rejection_rate": counts["rejected"] / max(1, counts["allowed"] + counts["rejected"])
            }
            
            # Add current state
            limiter = self._limiters.get(limit_type)
            if isinstance(limiter, TokenBucket):
                stats[limit_type]["available_tokens"] = limiter.available_tokens
                stats[limit_type]["capacity"] = limiter.capacity
            elif isinstance(limiter, SlidingWindowCounter):
                stats[limit_type]["current_count"] = limiter.current_count
                stats[limit_type]["limit"] = limiter.limit
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
    
    async def shutdown(self) -> None:
        """Clean shutdown of rate limiter."""
        # Log final statistics
        stats = self.get_statistics()
        for limit_type, data in stats.items():
            logger.info(
                f"Rate limiter {limit_type} - "
                f"Allowed: {data['allowed']}, "
                f"Rejected: {data['rejected']}, "
                f"Rejection rate: {data['rejection_rate']:.2%}"
            )