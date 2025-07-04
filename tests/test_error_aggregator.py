"""
Unit tests for error aggregation and reporting.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.error_handling.aggregator import (
    ErrorCategory, ErrorMetrics, ErrorReport, ErrorAggregator
)
from src.error_handling.exceptions import (
    AgentError, BrowserError, ValidationError, TimeoutError,
    CoordinationError, RecoveryError
)


class TestErrorMetrics:
    """Test error metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics default values."""
        metrics = ErrorMetrics()
        assert metrics.count == 0
        assert metrics.first_seen is None
        assert metrics.last_seen is None
        assert metrics.affected_agents == set()
        assert metrics.affected_operations == set()
        assert metrics.recovery_attempts == 0
        assert metrics.recovery_successes == 0
    
    def test_update_metrics(self):
        """Test updating metrics."""
        metrics = ErrorMetrics()
        
        # First update
        metrics.update(agent="test_agent", operation="click", recovered=True)
        assert metrics.count == 1
        assert metrics.first_seen is not None
        assert metrics.last_seen is not None
        assert "test_agent" in metrics.affected_agents
        assert "click" in metrics.affected_operations
        assert metrics.recovery_attempts == 1
        assert metrics.recovery_successes == 1
        
        # Second update
        first_seen = metrics.first_seen
        metrics.update(agent="another_agent", operation="type", recovered=False)
        assert metrics.count == 2
        assert metrics.first_seen == first_seen  # Unchanged
        assert metrics.last_seen > first_seen
        assert len(metrics.affected_agents) == 2
        assert len(metrics.affected_operations) == 2
        assert metrics.recovery_attempts == 2
        assert metrics.recovery_successes == 1
    
    def test_recovery_rate(self):
        """Test recovery rate calculation."""
        metrics = ErrorMetrics()
        assert metrics.recovery_rate == 0.0  # No attempts
        
        metrics.update(recovered=True)
        assert metrics.recovery_rate == 1.0  # 1/1
        
        metrics.update(recovered=False)
        assert metrics.recovery_rate == 0.5  # 1/2
        
        metrics.update(recovered=True)
        assert metrics.recovery_rate == 2/3  # 2/3
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ErrorMetrics()
        metrics.update(agent="test", operation="op", recovered=True)
        
        data = metrics.to_dict()
        assert data["count"] == 1
        assert data["affected_agents"] == ["test"]
        assert data["affected_operations"] == ["op"]
        assert data["recovery_rate"] == 1.0
        assert "first_seen" in data
        assert "last_seen" in data


class TestErrorReport:
    """Test error report generation."""
    
    def test_report_creation(self):
        """Test creating error report."""
        metrics = {"TestError": ErrorMetrics()}
        metrics["TestError"].update()
        
        report = ErrorReport(
            test_id="test123",
            test_name="Test Run",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            total_errors=10,
            errors_by_category={ErrorCategory.AGENT: 5, ErrorCategory.BROWSER: 5},
            errors_by_type=metrics,
            critical_errors=[],
            recovery_summary={"total_recovery_attempts": 3},
            recommendations=["Fix agent errors"]
        )
        
        assert report.test_id == "test123"
        assert report.test_name == "Test Run"
        assert report.total_errors == 10
        assert len(report.errors_by_category) == 2
        assert len(report.recommendations) == 1
    
    def test_report_to_dict(self):
        """Test report serialization."""
        start = datetime.utcnow()
        end = start + timedelta(minutes=10)
        
        report = ErrorReport(
            test_id="test123",
            test_name="Test",
            start_time=start,
            end_time=end,
            total_errors=5,
            errors_by_category={ErrorCategory.AGENT: 5},
            errors_by_type={},
            critical_errors=[],
            recovery_summary={},
            recommendations=[]
        )
        
        data = report.to_dict()
        assert data["test_id"] == "test123"
        assert data["duration"] == 600  # 10 minutes in seconds
        assert data["total_errors"] == 5
        assert data["errors_by_category"]["AGENT"] == 5
    
    def test_save_to_file(self, tmp_path):
        """Test saving report to file."""
        report = ErrorReport(
            test_id="test123",
            test_name="Test",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            total_errors=0,
            errors_by_category={},
            errors_by_type={},
            critical_errors=[],
            recovery_summary={},
            recommendations=[]
        )
        
        filepath = tmp_path / "error_report.json"
        report.save_to_file(filepath)
        
        assert filepath.exists()
        
        # Verify content
        with open(filepath) as f:
            data = json.load(f)
            assert data["test_id"] == "test123"


class TestErrorAggregator:
    """Test error aggregator functionality."""
    
    def test_aggregator_initialization(self):
        """Test aggregator setup."""
        aggregator = ErrorAggregator("test123", "Test Run")
        assert aggregator.test_id == "test123"
        assert aggregator.test_name == "Test Run"
        assert aggregator.errors == []
        assert len(aggregator.error_metrics) == 0
        assert aggregator.critical_threshold == 5
    
    def test_add_error(self):
        """Test adding errors to aggregator."""
        aggregator = ErrorAggregator("test123", "Test")
        
        # Add agent error
        error1 = AgentError(
            "Agent failed",
            agent_name="test_agent",
            agent_type="TestAgent"
        )
        aggregator.add_error(error1, agent_name="test_agent", recovered=False)
        
        assert len(aggregator.errors) == 1
        assert "AgentError" in aggregator.error_metrics
        assert aggregator.category_counts[ErrorCategory.AGENT] == 1
        
        # Add browser error
        error2 = BrowserError("Click failed")
        aggregator.add_error(error2, operation="click", recovered=True)
        
        assert len(aggregator.errors) == 2
        assert "BrowserError" in aggregator.error_metrics
        assert aggregator.category_counts[ErrorCategory.BROWSER] == 1
    
    def test_error_categorization(self):
        """Test error category mapping."""
        aggregator = ErrorAggregator("test", "test")
        
        # Test known categories
        assert aggregator._categorize_error(AgentError("", "", "")) == ErrorCategory.AGENT
        assert aggregator._categorize_error(BrowserError("")) == ErrorCategory.BROWSER
        assert aggregator._categorize_error(ValidationError("", "")) == ErrorCategory.VALIDATION
        assert aggregator._categorize_error(TimeoutError("", "", 0)) == ErrorCategory.TIMEOUT
        assert aggregator._categorize_error(CoordinationError("", [], "")) == ErrorCategory.COORDINATION
        assert aggregator._categorize_error(RecoveryError("", "")) == ErrorCategory.RECOVERY
        
        # Test unknown category
        assert aggregator._categorize_error(ValueError("")) == ErrorCategory.UNKNOWN
    
    def test_critical_error_detection(self):
        """Test critical error threshold detection."""
        aggregator = ErrorAggregator("test", "test")
        aggregator.critical_threshold = 3
        
        # Add errors below threshold
        for i in range(2):
            aggregator.add_error(AgentError("Fail", "agent", "type"))
        
        critical = aggregator.get_critical_errors()
        assert len(critical) == 0
        
        # Add one more to reach threshold
        aggregator.add_error(AgentError("Fail", "agent", "type"))
        
        critical = aggregator.get_critical_errors()
        assert len(critical) == 1
        assert critical[0]["error_type"] == "AgentError"
        assert critical[0]["count"] == 3
    
    def test_recovery_summary(self):
        """Test recovery summary generation."""
        aggregator = ErrorAggregator("test", "test")
        
        # Add errors with recovery attempts
        for i in range(3):
            aggregator.add_error(
                BrowserError("Fail"),
                recovered=True
            )
        
        for i in range(2):
            aggregator.add_error(
                TimeoutError("Timeout", "op", 5000),
                recovered=False
            )
        
        summary = aggregator.get_recovery_summary()
        assert summary["total_recovery_attempts"] == 5
        assert summary["total_recovery_successes"] == 3
        assert summary["overall_recovery_rate"] == 0.6
        assert summary["recoverable_error_types"] == 2
        assert summary["best_recovery"] == "BrowserError"  # 100% success
        assert summary["worst_recovery"] == "TimeoutError"  # 0% success
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        aggregator = ErrorAggregator("test", "test")
        
        # No errors - no recommendations
        recommendations = aggregator.generate_recommendations()
        assert len(recommendations) == 0
        
        # Add many errors
        for i in range(25):
            aggregator.add_error(ValueError("Error"))
        
        recommendations = aggregator.generate_recommendations()
        assert any("High error rate" in r for r in recommendations)
        
        # Add browser errors
        for i in range(12):
            aggregator.add_error(BrowserError("Browser fail"))
        
        recommendations = aggregator.generate_recommendations()
        assert any("browser errors" in r for r in recommendations)
        
        # Add timeout errors
        for i in range(4):
            aggregator.add_error(TimeoutError("Timeout", "op", 1000))
        
        recommendations = aggregator.generate_recommendations()
        assert any("Timeout errors" in r for r in recommendations)
    
    def test_error_timeline(self):
        """Test error timeline generation."""
        aggregator = ErrorAggregator("test", "test")
        
        # Add errors with timestamps
        error1 = AgentError("First", "agent1", "type")
        error2 = BrowserError("Second")
        
        aggregator.add_error(error1)
        aggregator.add_error(error2)
        
        timeline = aggregator.get_error_timeline()
        assert len(timeline) == 2
        assert timeline[0]["error_type"] == "AgentError"
        assert timeline[1]["error_type"] == "BrowserError"
        assert all("timestamp" in event for event in timeline)
        assert all("category" in event for event in timeline)
    
    def test_agent_error_summary(self):
        """Test agent-specific error summary."""
        aggregator = ErrorAggregator("test", "test")
        
        # Add errors for different agents
        aggregator.add_error(
            AgentError("Fail1", "agent1", "type"),
            agent_name="agent1"
        )
        aggregator.add_error(
            AgentError("Fail2", "agent1", "type"),
            agent_name="agent1"
        )
        aggregator.add_error(
            BrowserError("Browser fail"),
            agent_name="agent2"
        )
        
        summary = aggregator.get_agent_error_summary()
        assert "agent1" in summary
        assert "agent2" in summary
        assert summary["agent1"]["AgentError"] == 2
        assert summary["agent2"]["BrowserError"] == 1
    
    def test_generate_full_report(self):
        """Test full report generation."""
        aggregator = ErrorAggregator("test123", "Full Test")
        
        # Add various errors
        for i in range(3):
            aggregator.add_error(
                AgentError(f"Agent fail {i}", "agent", "type"),
                recovered=i < 2
            )
        
        for i in range(2):
            aggregator.add_error(
                BrowserError(f"Browser fail {i}"),
                recovered=True
            )
        
        report = aggregator.generate_report()
        
        assert report.test_id == "test123"
        assert report.test_name == "Full Test"
        assert report.total_errors == 5
        assert len(report.errors_by_category) > 0
        assert len(report.errors_by_type) == 2
        assert "total_recovery_attempts" in report.recovery_summary
        assert isinstance(report.recommendations, list)