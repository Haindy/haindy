"""
Analytics and metrics collection for HAINDY test execution.

Tracks success rates, performance metrics, and provides insights
into test execution patterns.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Deque, Set, Tuple
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    TIMER = auto()


class TestOutcome(Enum):
    """Possible test outcomes."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class TestMetrics:
    """Metrics for a single test execution."""
    test_id: UUID
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    outcome: Optional[TestOutcome] = None
    steps_total: int = 0
    steps_passed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    api_calls: int = 0
    browser_actions: int = 0
    screenshots_taken: int = 0
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get test duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if self.steps_total == 0:
            return 0.0
        return self.steps_passed / self.steps_total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": str(self.test_id),
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "outcome": self.outcome.value if self.outcome else None,
            "steps": {
                "total": self.steps_total,
                "passed": self.steps_passed,
                "failed": self.steps_failed,
                "skipped": self.steps_skipped,
                "success_rate": self.success_rate
            },
            "resources": {
                "api_calls": self.api_calls,
                "browser_actions": self.browser_actions,
                "screenshots": self.screenshots_taken
            },
            "errors": self.errors,
            "performance_metrics": self.performance_metrics
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, window_size_minutes: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            window_size_minutes: Size of sliding window for rate calculations
        """
        self.window_size = timedelta(minutes=window_size_minutes)
        self.metrics: Dict[str, Deque[MetricValue]] = defaultdict(lambda: deque(maxlen=10000))
        self.test_metrics: Dict[UUID, TestMetrics] = {}
        self.active_tests: Set[UUID] = set()
        self._lock = asyncio.Lock()
    
    async def start_test(
        self,
        test_id: UUID,
        test_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record test start."""
        async with self._lock:
            metrics = TestMetrics(
                test_id=test_id,
                test_name=test_name,
                start_time=datetime.utcnow()
            )
            self.test_metrics[test_id] = metrics
            self.active_tests.add(test_id)
            
            # Record as counter
            await self.record_counter("tests.started", tags={"test_name": test_name})
    
    async def end_test(
        self,
        test_id: UUID,
        outcome: TestOutcome,
        error_message: Optional[str] = None
    ) -> None:
        """Record test end."""
        async with self._lock:
            if test_id not in self.test_metrics:
                logger.warning(f"Ending unknown test: {test_id}")
                return
            
            metrics = self.test_metrics[test_id]
            metrics.end_time = datetime.utcnow()
            metrics.outcome = outcome
            
            if error_message:
                metrics.errors.append(error_message)
            
            self.active_tests.discard(test_id)
            
            # Record outcome
            await self.record_counter(
                f"tests.{outcome.value}",
                tags={"test_name": metrics.test_name}
            )
            
            # Record duration
            if metrics.duration_seconds:
                await self.record_timer(
                    "test.duration",
                    metrics.duration_seconds,
                    tags={"test_name": metrics.test_name, "outcome": outcome.value}
                )
    
    async def record_step_outcome(
        self,
        test_id: UUID,
        step_name: str,
        success: bool,
        duration_ms: Optional[int] = None
    ) -> None:
        """Record step execution outcome."""
        async with self._lock:
            if test_id not in self.test_metrics:
                return
            
            metrics = self.test_metrics[test_id]
            metrics.steps_total += 1
            
            if success:
                metrics.steps_passed += 1
            else:
                metrics.steps_failed += 1
            
            # Record step metrics
            outcome = "passed" if success else "failed"
            await self.record_counter(
                f"steps.{outcome}",
                tags={"test_id": str(test_id), "step_name": step_name}
            )
            
            if duration_ms:
                await self.record_timer(
                    "step.duration",
                    duration_ms / 1000,  # Convert to seconds
                    tags={"outcome": outcome}
                )
    
    async def record_api_call(
        self,
        test_id: UUID,
        api_name: str,
        duration_ms: int,
        success: bool = True
    ) -> None:
        """Record API call metrics."""
        async with self._lock:
            if test_id in self.test_metrics:
                self.test_metrics[test_id].api_calls += 1
            
            await self.record_counter(
                "api.calls",
                tags={"api": api_name, "success": str(success)}
            )
            
            await self.record_timer(
                "api.duration",
                duration_ms / 1000,
                tags={"api": api_name}
            )
    
    async def record_browser_action(
        self,
        test_id: UUID,
        action_type: str,
        duration_ms: int,
        success: bool = True
    ) -> None:
        """Record browser action metrics."""
        async with self._lock:
            if test_id in self.test_metrics:
                self.test_metrics[test_id].browser_actions += 1
            
            await self.record_counter(
                "browser.actions",
                tags={"action": action_type, "success": str(success)}
            )
            
            await self.record_timer(
                "browser.action_duration",
                duration_ms / 1000,
                tags={"action": action_type}
            )
    
    async def record_screenshot(self, test_id: UUID) -> None:
        """Record screenshot taken."""
        async with self._lock:
            if test_id in self.test_metrics:
                self.test_metrics[test_id].screenshots_taken += 1
            
            await self.record_counter("screenshots.taken")
    
    async def record_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a counter metric."""
        metric = MetricValue(value=value, tags=tags or {})
        self.metrics[f"counter.{name}"].append(metric)
    
    async def record_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a gauge metric."""
        metric = MetricValue(value=value, tags=tags or {})
        self.metrics[f"gauge.{name}"].append(metric)
    
    async def record_timer(
        self,
        name: str,
        duration_seconds: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timer metric."""
        metric = MetricValue(value=duration_seconds, tags=tags or {})
        self.metrics[f"timer.{name}"].append(metric)
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        values = [m.value for m in self.metrics.get(metric_name, [])]
        
        if not values:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": np.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def get_rate(self, metric_name: str, window_minutes: Optional[int] = None) -> float:
        """Calculate rate per minute for a counter."""
        window = timedelta(minutes=window_minutes) if window_minutes else self.window_size
        cutoff = datetime.utcnow() - window
        
        values = self.metrics.get(f"counter.{metric_name}", [])
        recent_values = [m for m in values if m.timestamp > cutoff]
        
        if not recent_values:
            return 0.0
        
        total = sum(m.value for m in recent_values)
        duration_minutes = window.total_seconds() / 60
        
        return total / duration_minutes
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test executions."""
        completed_tests = [
            m for m in self.test_metrics.values()
            if m.outcome is not None
        ]
        
        if not completed_tests:
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "active_tests": len(self.active_tests)
            }
        
        passed = sum(1 for t in completed_tests if t.outcome == TestOutcome.PASSED)
        failed = sum(1 for t in completed_tests if t.outcome == TestOutcome.FAILED)
        durations = [t.duration_seconds for t in completed_tests if t.duration_seconds]
        
        return {
            "total_tests": len(completed_tests),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(completed_tests) if completed_tests else 0.0,
            "avg_duration": np.mean(durations) if durations else 0.0,
            "active_tests": len(self.active_tests),
            "by_outcome": {
                outcome.value: sum(1 for t in completed_tests if t.outcome == outcome)
                for outcome in TestOutcome
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return {
            "api_calls": {
                "rate_per_minute": self.get_rate("api.calls"),
                "duration": self.get_metric_summary("timer.api.duration")
            },
            "browser_actions": {
                "rate_per_minute": self.get_rate("browser.actions"),
                "duration": self.get_metric_summary("timer.browser.action_duration")
            },
            "steps": {
                "success_rate": self._calculate_step_success_rate(),
                "duration": self.get_metric_summary("timer.step.duration")
            },
            "screenshots": {
                "rate_per_minute": self.get_rate("screenshots.taken")
            }
        }
    
    def _calculate_step_success_rate(self) -> float:
        """Calculate overall step success rate."""
        passed = sum(m.value for m in self.metrics.get("counter.steps.passed", []))
        failed = sum(m.value for m in self.metrics.get("counter.steps.failed", []))
        total = passed + failed
        
        return passed / total if total > 0 else 0.0
    
    def export_metrics(self, filepath: Path) -> None:
        """Export all metrics to JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_summary": self.get_test_summary(),
            "performance_summary": self.get_performance_summary(),
            "test_details": [
                metrics.to_dict() for metrics in self.test_metrics.values()
            ],
            "metric_summaries": {
                name: self.get_metric_summary(name)
                for name in self.metrics.keys()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.test_metrics.clear()
        self.active_tests.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


# Convenience functions
async def start_test(test_id: UUID, test_name: str, **metadata) -> None:
    """Start tracking a test."""
    await _metrics_collector.start_test(test_id, test_name, metadata)


async def end_test(test_id: UUID, outcome: TestOutcome, error: Optional[str] = None) -> None:
    """End test tracking."""
    await _metrics_collector.end_test(test_id, outcome, error)


async def record_step(test_id: UUID, step_name: str, success: bool, duration_ms: Optional[int] = None) -> None:
    """Record step outcome."""
    await _metrics_collector.record_step_outcome(test_id, step_name, success, duration_ms)


async def record_api_call(test_id: UUID, api_name: str, duration_ms: int, success: bool = True) -> None:
    """Record API call."""
    await _metrics_collector.record_api_call(test_id, api_name, duration_ms, success)


async def record_browser_action(test_id: UUID, action: str, duration_ms: int, success: bool = True) -> None:
    """Record browser action."""
    await _metrics_collector.record_browser_action(test_id, action, duration_ms, success)


def get_analytics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector