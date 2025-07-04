"""
Unit tests for analytics and metrics collection.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch

from src.monitoring.analytics import (
    MetricsCollector,
    TestMetrics,
    TestOutcome,
    MetricValue,
    start_test,
    end_test,
    record_step,
    get_analytics
)


class TestTestMetrics:
    """Test TestMetrics data class."""
    
    def test_metrics_creation(self):
        """Test metrics initialization."""
        test_id = uuid4()
        metrics = TestMetrics(
            test_id=test_id,
            test_name="test_example",
            start_time=datetime.utcnow()
        )
        
        assert metrics.test_id == test_id
        assert metrics.test_name == "test_example"
        assert metrics.outcome is None
        assert metrics.steps_total == 0
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.utcnow()
        end = start + timedelta(seconds=10)
        
        metrics = TestMetrics(
            test_id=uuid4(),
            test_name="test",
            start_time=start,
            end_time=end
        )
        
        assert metrics.duration_seconds == 10.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = TestMetrics(
            test_id=uuid4(),
            test_name="test",
            start_time=datetime.utcnow()
        )
        
        # No steps
        assert metrics.success_rate == 0.0
        
        # Some steps
        metrics.steps_total = 10
        metrics.steps_passed = 8
        metrics.steps_failed = 2
        
        assert metrics.success_rate == 0.8
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = TestMetrics(
            test_id=uuid4(),
            test_name="test",
            start_time=datetime.utcnow(),
            steps_total=5,
            steps_passed=4,
            steps_failed=1,
            api_calls=10,
            browser_actions=20
        )
        
        data = metrics.to_dict()
        assert data["test_name"] == "test"
        assert data["steps"]["total"] == 5
        assert data["steps"]["passed"] == 4
        assert data["steps"]["success_rate"] == 0.8
        assert data["resources"]["api_calls"] == 10


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """Test collector setup."""
        collector = MetricsCollector(window_size_minutes=30)
        assert collector.window_size == timedelta(minutes=30)
        assert len(collector.test_metrics) == 0
        assert len(collector.active_tests) == 0
    
    @pytest.mark.asyncio
    async def test_start_test(self):
        """Test starting test tracking."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        await collector.start_test(test_id, "test_example")
        
        assert test_id in collector.test_metrics
        assert test_id in collector.active_tests
        assert collector.test_metrics[test_id].test_name == "test_example"
        
        # Check counter was recorded
        assert len(collector.metrics["counter.tests.started"]) == 1
    
    @pytest.mark.asyncio
    async def test_end_test(self):
        """Test ending test tracking."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        # Start test
        await collector.start_test(test_id, "test_example")
        
        # End test
        await collector.end_test(test_id, TestOutcome.PASSED)
        
        assert test_id not in collector.active_tests
        assert collector.test_metrics[test_id].outcome == TestOutcome.PASSED
        assert collector.test_metrics[test_id].end_time is not None
        
        # Check counters
        assert len(collector.metrics["counter.tests.passed"]) == 1
    
    @pytest.mark.asyncio
    async def test_record_step_outcome(self):
        """Test recording step outcomes."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        await collector.start_test(test_id, "test_example")
        
        # Record successful step
        await collector.record_step_outcome(test_id, "step1", True, 100)
        
        metrics = collector.test_metrics[test_id]
        assert metrics.steps_total == 1
        assert metrics.steps_passed == 1
        assert metrics.steps_failed == 0
        
        # Record failed step
        await collector.record_step_outcome(test_id, "step2", False, 200)
        
        assert metrics.steps_total == 2
        assert metrics.steps_passed == 1
        assert metrics.steps_failed == 1
    
    @pytest.mark.asyncio
    async def test_record_api_call(self):
        """Test recording API calls."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        await collector.start_test(test_id, "test_example")
        await collector.record_api_call(test_id, "openai", 150, success=True)
        
        assert collector.test_metrics[test_id].api_calls == 1
        assert len(collector.metrics["counter.api.calls"]) == 1
        assert len(collector.metrics["timer.api.duration"]) == 1
    
    @pytest.mark.asyncio
    async def test_record_browser_action(self):
        """Test recording browser actions."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        await collector.start_test(test_id, "test_example")
        await collector.record_browser_action(test_id, "click", 50)
        
        assert collector.test_metrics[test_id].browser_actions == 1
        assert len(collector.metrics["counter.browser.actions"]) == 1
    
    @pytest.mark.asyncio
    async def test_metric_summary(self):
        """Test metric summary calculation."""
        collector = MetricsCollector()
        
        # Add some timer values
        for i in range(10):
            await collector.record_timer("test.duration", i + 1)
        
        summary = collector.get_metric_summary("timer.test.duration")
        
        assert summary["count"] == 10
        assert summary["mean"] == 5.5  # Average of 1-10
        assert summary["min"] == 1
        assert summary["max"] == 10
        assert summary["p50"] == 5.5
    
    @pytest.mark.asyncio
    async def test_rate_calculation(self):
        """Test rate calculation."""
        collector = MetricsCollector()
        
        # Record some counters
        for _ in range(60):
            await collector.record_counter("test.events")
        
        # Rate should be 60 per minute
        rate = collector.get_rate("test.events", window_minutes=1)
        assert 59 <= rate <= 61  # Allow small variance
    
    @pytest.mark.asyncio
    async def test_test_summary(self):
        """Test overall test summary."""
        collector = MetricsCollector()
        
        # Run some tests
        test_ids = []
        for i in range(5):
            test_id = uuid4()
            test_ids.append(test_id)
            await collector.start_test(test_id, f"test_{i}")
        
        # Complete some tests
        await collector.end_test(test_ids[0], TestOutcome.PASSED)
        await collector.end_test(test_ids[1], TestOutcome.PASSED)
        await collector.end_test(test_ids[2], TestOutcome.FAILED)
        
        summary = collector.get_test_summary()
        
        assert summary["total_tests"] == 3  # Only completed tests
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["active_tests"] == 2  # Still running
    
    @pytest.mark.asyncio
    async def test_performance_summary(self):
        """Test performance summary generation."""
        collector = MetricsCollector()
        test_id = uuid4()
        
        await collector.start_test(test_id, "test")
        
        # Record various metrics
        await collector.record_api_call(test_id, "api1", 100)
        await collector.record_browser_action(test_id, "click", 50)
        await collector.record_step_outcome(test_id, "step1", True, 200)
        
        perf_summary = collector.get_performance_summary()
        
        assert "api_calls" in perf_summary
        assert "browser_actions" in perf_summary
        assert "steps" in perf_summary
        assert perf_summary["steps"]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        
        # Add some data
        test_id = uuid4()
        await collector.start_test(test_id, "test")
        await collector.record_counter("test.counter")
        
        # Reset
        collector.reset()
        
        assert len(collector.metrics) == 0
        assert len(collector.test_metrics) == 0
        assert len(collector.active_tests) == 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @pytest.mark.asyncio
    async def test_global_functions(self):
        """Test global analytics functions."""
        # Get the global collector
        analytics = get_analytics()
        
        # Reset to ensure clean state
        analytics.reset()
        
        # Test convenience functions
        test_id = uuid4()
        
        await start_test(test_id, "global_test")
        assert test_id in analytics.test_metrics
        
        await record_step(test_id, "step1", True, 100)
        assert analytics.test_metrics[test_id].steps_passed == 1
        
        await end_test(test_id, TestOutcome.PASSED)
        assert analytics.test_metrics[test_id].outcome == TestOutcome.PASSED