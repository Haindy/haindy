"""
Unit tests for rate limiting functionality.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from src.security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitConfig,
    RateLimitStrategy,
    TokenBucket,
    SlidingWindowCounter
)


class TestTokenBucket:
    """Test token bucket rate limiting."""
    
    @pytest.mark.asyncio
    async def test_token_bucket_creation(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10
    
    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test acquiring tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Acquire single token
        assert await bucket.acquire(1) is True
        assert bucket.tokens == 4
        
        # Acquire multiple tokens
        assert await bucket.acquire(3) is True
        assert 0.9 <= bucket.tokens <= 1.1  # Allow small drift
        
        # Try to acquire more than available
        assert await bucket.acquire(2) is False
        assert 0.9 <= bucket.tokens <= 1.1  # Allow small drift due to refill
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refilling over time."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)  # 10 tokens/second
        
        # Use all tokens
        await bucket.acquire(5)
        assert bucket.tokens == 0
        
        # Wait for refill
        await asyncio.sleep(0.2)  # Should refill ~2 tokens
        
        # Should be able to acquire 1-2 tokens
        assert await bucket.acquire(1) is True
    
    @pytest.mark.asyncio
    async def test_acquire_or_wait(self):
        """Test waiting for tokens."""
        bucket = TokenBucket(capacity=2, refill_rate=5.0)  # 5 tokens/second
        
        # Use all tokens
        await bucket.acquire(2)
        
        # Should wait and then acquire
        start = time.monotonic()
        assert await bucket.acquire_or_wait(1, timeout=1.0) is True
        elapsed = time.monotonic() - start
        
        # Should have waited approximately 0.2 seconds
        assert 0.1 < elapsed < 0.4
    
    @pytest.mark.asyncio
    async def test_acquire_or_wait_timeout(self):
        """Test timeout when waiting for tokens."""
        bucket = TokenBucket(capacity=1, refill_rate=0.1)  # Very slow refill
        
        # Use all tokens
        await bucket.acquire(1)
        
        # Should timeout
        assert await bucket.acquire_or_wait(1, timeout=0.1) is False


class TestSlidingWindowCounter:
    """Test sliding window rate limiting."""
    
    @pytest.mark.asyncio
    async def test_sliding_window_creation(self):
        """Test sliding window initialization."""
        window = SlidingWindowCounter(window_seconds=60, limit=10)
        assert window.window_seconds == 60
        assert window.limit == 10
        assert window.current_count == 0
    
    @pytest.mark.asyncio
    async def test_request_counting(self):
        """Test request counting within window."""
        window = SlidingWindowCounter(window_seconds=60, limit=3)
        
        # First requests should be allowed
        assert await window.is_allowed() is True
        assert window.current_count == 1
        
        assert await window.is_allowed() is True
        assert window.current_count == 2
        
        assert await window.is_allowed() is True
        assert window.current_count == 3
        
        # Next request should be denied
        assert await window.is_allowed() is False
        assert window.current_count == 3
    
    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Test request expiry after window."""
        # Use very short window for testing
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            
            window = SlidingWindowCounter(window_seconds=10, limit=2)
            
            # Add requests
            assert await window.is_allowed() is True
            assert await window.is_allowed() is True
            assert await window.is_allowed() is False  # Limit reached
            
            # Move time forward past window
            mock_time.return_value = 1011.0  # 11 seconds later
            
            # Old requests should be expired
            assert window.current_count == 0
            assert await window.is_allowed() is True


class TestRateLimiter:
    """Test main rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_creation(self):
        """Test rate limiter initialization."""
        config = RateLimitConfig(
            api_calls_per_minute=30,
            browser_actions_per_minute=60
        )
        limiter = RateLimiter(config)
        
        assert limiter.config.api_calls_per_minute == 30
        assert limiter.config.browser_actions_per_minute == 60
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting(self):
        """Test API call rate limiting."""
        config = RateLimitConfig(
            api_calls_per_minute=60,
            api_burst_size=2,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        limiter = RateLimiter(config)
        
        # Should allow burst
        assert await limiter.check_api_call() is True
        assert await limiter.check_api_call() is True
        
        # Should hit limit
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_api_call()
        
        assert exc_info.value.limit == 60
        assert exc_info.value.retry_after is not None
    
    @pytest.mark.asyncio
    async def test_browser_action_limiting(self):
        """Test browser action rate limiting."""
        config = RateLimitConfig(
            browser_actions_per_minute=120,
            browser_actions_burst=3
        )
        limiter = RateLimiter(config)
        
        # Should allow up to burst
        for _ in range(3):
            assert await limiter.check_browser_action() is True
        
        # Next should fail
        with pytest.raises(RateLimitExceeded):
            await limiter.check_browser_action()
    
    @pytest.mark.asyncio
    async def test_wait_for_limit(self):
        """Test waiting for rate limit."""
        config = RateLimitConfig(
            api_calls_per_minute=60,  # 1 per second
            api_burst_size=1
        )
        limiter = RateLimiter(config)
        
        # Use the burst
        assert await limiter.check_api_call() is True
        
        # Should wait and succeed
        start = time.monotonic()
        assert await limiter.check_api_call(wait=True) is True
        elapsed = time.monotonic() - start
        
        # Should have waited approximately 1 second
        assert 0.8 < elapsed < 1.5
    
    @pytest.mark.asyncio
    async def test_disabled_rate_limiting(self):
        """Test disabled rate limiting."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        # Should always allow when disabled
        for _ in range(100):
            assert await limiter.check_api_call() is True
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test rate limiter statistics."""
        config = RateLimitConfig(
            api_calls_per_minute=60,
            api_burst_size=2
        )
        limiter = RateLimiter(config)
        
        # Make some requests
        await limiter.check_api_call()
        await limiter.check_api_call()
        
        # This should be rejected
        try:
            await limiter.check_api_call()
        except RateLimitExceeded:
            pass
        
        stats = limiter.get_statistics()
        assert stats["api"]["allowed"] == 2
        assert stats["api"]["rejected"] == 1
        assert stats["api"]["rejection_rate"] == 1/3
    
    @pytest.mark.asyncio
    async def test_sliding_window_strategy(self):
        """Test sliding window strategy."""
        config = RateLimitConfig(
            api_calls_per_minute=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        limiter = RateLimiter(config)
        
        # Should track requests in window
        assert await limiter.check_api_call() is True
        
        stats = limiter.get_statistics()
        assert "api" in stats
        assert stats["api"]["allowed"] == 1