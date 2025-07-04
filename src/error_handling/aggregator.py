"""
Error aggregation and reporting for comprehensive error analysis.

Collects, aggregates, and reports errors across the test execution lifecycle.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .exceptions import HAINDYError

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories for error classification."""
    AGENT = auto()
    BROWSER = auto()
    VALIDATION = auto()
    COORDINATION = auto()
    TIMEOUT = auto()
    RECOVERY = auto()
    UNKNOWN = auto()


@dataclass
class ErrorMetrics:
    """Metrics for a specific error type."""
    count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    affected_agents: Set[str] = field(default_factory=set)
    affected_operations: Set[str] = field(default_factory=set)
    recovery_attempts: int = 0
    recovery_successes: int = 0
    
    def update(
        self,
        agent: Optional[str] = None,
        operation: Optional[str] = None,
        recovered: Optional[bool] = None
    ) -> None:
        """Update metrics with new error occurrence."""
        self.count += 1
        now = datetime.utcnow()
        
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now
        
        if agent:
            self.affected_agents.add(agent)
        if operation:
            self.affected_operations.add(operation)
        
        if recovered is not None:
            self.recovery_attempts += 1
            if recovered:
                self.recovery_successes += 1
    
    @property
    def recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.recovery_attempts == 0:
            return 0.0
        return self.recovery_successes / self.recovery_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "affected_agents": list(self.affected_agents),
            "affected_operations": list(self.affected_operations),
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "recovery_rate": self.recovery_rate
        }


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    test_id: str
    test_name: str
    start_time: datetime
    end_time: datetime
    total_errors: int
    errors_by_category: Dict[ErrorCategory, int]
    errors_by_type: Dict[str, ErrorMetrics]
    critical_errors: List[Dict[str, Any]]
    recovery_summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "duration": (self.end_time - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_errors": self.total_errors,
            "errors_by_category": {
                cat.name: count for cat, count in self.errors_by_category.items()
            },
            "errors_by_type": {
                error_type: metrics.to_dict()
                for error_type, metrics in self.errors_by_type.items()
            },
            "critical_errors": self.critical_errors,
            "recovery_summary": self.recovery_summary,
            "recommendations": self.recommendations
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ErrorAggregator:
    """Aggregates and analyzes errors during test execution."""
    
    def __init__(self, test_id: str, test_name: str):
        self.test_id = test_id
        self.test_name = test_name
        self.start_time = datetime.utcnow()
        self.errors: List[HAINDYError] = []
        self.error_metrics: Dict[str, ErrorMetrics] = defaultdict(ErrorMetrics)
        self.category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.critical_threshold = 5  # Errors become critical after this many occurrences
    
    def add_error(
        self,
        error: Exception,
        agent_name: Optional[str] = None,
        operation: Optional[str] = None,
        recovered: Optional[bool] = None
    ) -> None:
        """Add an error to the aggregator."""
        # Store error instance if it's a HAINDY error
        if isinstance(error, HAINDYError):
            self.errors.append(error)
        
        # Categorize error
        category = self._categorize_error(error)
        self.category_counts[category] += 1
        
        # Update metrics
        error_type = error.__class__.__name__
        metrics = self.error_metrics[error_type]
        metrics.update(agent=agent_name, operation=operation, recovered=recovered)
        
        # Log if becoming critical
        if metrics.count == self.critical_threshold:
            logger.warning(
                f"Error type {error_type} has reached critical threshold "
                f"({self.critical_threshold} occurrences)"
            )
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error."""
        error_type = error.__class__.__name__
        
        category_mapping = {
            "AgentError": ErrorCategory.AGENT,
            "BrowserError": ErrorCategory.BROWSER,
            "ValidationError": ErrorCategory.VALIDATION,
            "CoordinationError": ErrorCategory.COORDINATION,
            "TimeoutError": ErrorCategory.TIMEOUT,
            "RecoveryError": ErrorCategory.RECOVERY
        }
        
        return category_mapping.get(error_type, ErrorCategory.UNKNOWN)
    
    def get_critical_errors(self) -> List[Dict[str, Any]]:
        """Get list of critical errors."""
        critical_errors = []
        
        for error_type, metrics in self.error_metrics.items():
            if metrics.count >= self.critical_threshold:
                critical_errors.append({
                    "error_type": error_type,
                    "count": metrics.count,
                    "recovery_rate": metrics.recovery_rate,
                    "affected_agents": list(metrics.affected_agents),
                    "affected_operations": list(metrics.affected_operations)
                })
        
        return critical_errors
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of recovery attempts."""
        total_attempts = sum(m.recovery_attempts for m in self.error_metrics.values())
        total_successes = sum(m.recovery_successes for m in self.error_metrics.values())
        
        recoverable_errors = [
            (error_type, metrics)
            for error_type, metrics in self.error_metrics.items()
            if metrics.recovery_attempts > 0
        ]
        
        return {
            "total_recovery_attempts": total_attempts,
            "total_recovery_successes": total_successes,
            "overall_recovery_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
            "recoverable_error_types": len(recoverable_errors),
            "best_recovery": max(
                recoverable_errors,
                key=lambda x: x[1].recovery_rate,
                default=(None, None)
            )[0] if recoverable_errors else None,
            "worst_recovery": min(
                recoverable_errors,
                key=lambda x: x[1].recovery_rate,
                default=(None, None)
            )[0] if recoverable_errors else None
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        # Check for high error rates
        total_errors = sum(m.count for m in self.error_metrics.values())
        if total_errors > 20:
            recommendations.append(
                "High error rate detected. Consider reviewing test stability and environment setup."
            )
        
        # Check for specific error patterns
        for error_type, metrics in self.error_metrics.items():
            if metrics.count >= self.critical_threshold:
                if metrics.recovery_rate < 0.5:
                    recommendations.append(
                        f"Error '{error_type}' has low recovery rate ({metrics.recovery_rate:.1%}). "
                        f"Consider implementing better recovery strategies."
                    )
                
                if len(metrics.affected_agents) > 2:
                    recommendations.append(
                        f"Error '{error_type}' affects multiple agents. "
                        f"This may indicate a systemic issue."
                    )
        
        # Check category-specific issues
        if self.category_counts[ErrorCategory.BROWSER] > 10:
            recommendations.append(
                "High number of browser errors. Check browser stability and page load times."
            )
        
        if self.category_counts[ErrorCategory.VALIDATION] > 5:
            recommendations.append(
                "Multiple validation errors. Review test data and expected outcomes."
            )
        
        if self.category_counts[ErrorCategory.TIMEOUT] > 3:
            recommendations.append(
                "Timeout errors detected. Consider increasing timeout values or optimizing operations."
            )
        
        return recommendations
    
    def generate_report(self) -> ErrorReport:
        """Generate comprehensive error report."""
        end_time = datetime.utcnow()
        
        return ErrorReport(
            test_id=self.test_id,
            test_name=self.test_name,
            start_time=self.start_time,
            end_time=end_time,
            total_errors=len(self.errors),
            errors_by_category=dict(self.category_counts),
            errors_by_type=dict(self.error_metrics),
            critical_errors=self.get_critical_errors(),
            recovery_summary=self.get_recovery_summary(),
            recommendations=self.generate_recommendations()
        )
    
    def get_error_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of errors for visualization."""
        timeline = []
        
        for error in self.errors:
            if hasattr(error, 'timestamp'):
                timeline.append({
                    "timestamp": error.timestamp.isoformat(),
                    "error_type": error.__class__.__name__,
                    "message": str(error),
                    "category": self._categorize_error(error).name
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline
    
    def get_agent_error_summary(self) -> Dict[str, Dict[str, int]]:
        """Get error summary by agent."""
        agent_errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for error_type, metrics in self.error_metrics.items():
            for agent in metrics.affected_agents:
                agent_errors[agent][error_type] = metrics.count
        
        return dict(agent_errors)