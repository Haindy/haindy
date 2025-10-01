"""
Test execution reporting for HAINDY.

Generates comprehensive reports in various formats (HTML, JSON, Markdown)
with execution details, metrics, and insights.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID

from jinja2 import Environment, Template

from src.core.types import TestState, TestStep, ActionResult
from src.error_handling.aggregator import ErrorReport
from src.journal.models import ExecutionJournal
from .analytics import MetricsCollector, TestMetrics, TestOutcome

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    include_screenshots: bool = True
    include_detailed_steps: bool = True
    include_error_details: bool = True
    include_performance_metrics: bool = True
    include_journal_entries: bool = False
    sanitize_sensitive_data: bool = True
    max_screenshot_size_kb: int = 500


class TestExecutionReport:
    """Comprehensive test execution report."""
    
    def __init__(
        self,
        test_metrics: TestMetrics,
        error_report: Optional[ErrorReport] = None,
        journal: Optional[ExecutionJournal] = None,
        config: Optional[ReportConfig] = None
    ):
        self.test_metrics = test_metrics
        self.error_report = error_report
        self.journal = journal
        self.config = config or ReportConfig()
        self.generated_at = datetime.now(timezone.utc)
        self.bug_reports: List[Dict[str, Any]] = []  # Will be populated after initialization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        report = {
            "metadata": {
                "report_version": "1.0",
                "generated_at": self.generated_at.isoformat(),
                "test_id": str(self.test_metrics.test_id),
                "test_name": self.test_metrics.test_name
            },
            "summary": {
                "outcome": self.test_metrics.outcome.value if self.test_metrics.outcome else "unknown",
                "duration_seconds": self.test_metrics.duration_seconds,
                "start_time": self.test_metrics.start_time.isoformat(),
                "end_time": self.test_metrics.end_time.isoformat() if self.test_metrics.end_time else None,
                "success_rate": self.test_metrics.success_rate
            },
            "steps": {
                "total": self.test_metrics.steps_total,
                "passed": self.test_metrics.steps_passed,
                "failed": self.test_metrics.steps_failed,
                "skipped": self.test_metrics.steps_skipped
            },
            "resources": {
                "api_calls": self.test_metrics.api_calls,
                "browser_actions": self.test_metrics.browser_actions,
                "screenshots": self.test_metrics.screenshots_taken
            }
        }
        
        # Add performance metrics
        if self.config.include_performance_metrics:
            report["performance"] = self.test_metrics.performance_metrics
        
        # Add error details
        if self.config.include_error_details and self.error_report:
            report["errors"] = {
                "total": self.error_report.total_errors,
                "by_category": {
                    cat.name: count 
                    for cat, count in self.error_report.errors_by_category.items()
                },
                "critical": self.error_report.critical_errors,
                "recovery_summary": self.error_report.recovery_summary,
                "recommendations": self.error_report.recommendations
            }
        
        # Add journal entries
        if self.config.include_journal_entries and self.journal:
            report["execution_journal"] = {
                "entries": len(self.journal.entries),
                "summary": self.journal.get_summary() if hasattr(self.journal, 'get_summary') else {}
            }
        
        # Add bug reports
        if self.bug_reports:
            report["bug_reports"] = self.bug_reports
        
        return report
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_html(self) -> str:
        """Generate HTML report."""
        from src.monitoring.debug_logger import get_debug_logger
        
        template = Template(HTML_REPORT_TEMPLATE)
        template_data = self.to_dict()
        
        # Add AI conversations if available
        debug_logger = get_debug_logger()
        if debug_logger:
            template_data['ai_conversations'] = debug_logger.get_ai_conversations_html()
        
        return template.render(report=template_data, ai_conversations=template_data.get('ai_conversations'))
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        data = self.to_dict()
        
        md = f"# Test Execution Report: {data['metadata']['test_name']}\n\n"
        md += f"**Generated:** {data['metadata']['generated_at']}\n"
        md += f"**Test ID:** {data['metadata']['test_id']}\n\n"
        
        # Summary
        md += "## Summary\n\n"
        summary = data['summary']
        md += f"- **Outcome:** {summary['outcome'].upper()}\n"
        md += f"- **Duration:** {summary['duration_seconds']:.2f}s\n"
        md += f"- **Success Rate:** {summary['success_rate']*100:.1f}%\n\n"
        
        # Steps
        md += "## Test Steps\n\n"
        steps = data['steps']
        md += f"- Total: {steps['total']}\n"
        md += f"- Passed: {steps['passed']} ✓\n"
        md += f"- Failed: {steps['failed']} ✗\n"
        md += f"- Skipped: {steps['skipped']} ⚠\n\n"
        
        # Resources
        md += "## Resource Usage\n\n"
        resources = data['resources']
        md += f"- API Calls: {resources['api_calls']}\n"
        md += f"- Browser Actions: {resources['browser_actions']}\n"
        md += f"- Screenshots: {resources['screenshots']}\n\n"
        
        # Errors
        if 'errors' in data:
            md += "## Errors and Recovery\n\n"
            errors = data['errors']
            md += f"Total errors: {errors['total']}\n\n"
            
            if errors['critical']:
                md += "### Critical Errors\n\n"
                for error in errors['critical']:
                    md += f"- **{error['error_type']}**: {error['count']} occurrences\n"
                md += "\n"
            
            if errors['recommendations']:
                md += "### Recommendations\n\n"
                for rec in errors['recommendations']:
                    md += f"- {rec}\n"
                md += "\n"
        
        # Bug Reports
        if 'bug_reports' in data and data['bug_reports']:
            md += "## Bug Reports\n\n"
            md += f"Found {len(data['bug_reports'])} bug(s) during test execution:\n\n"
            
            for bug in data['bug_reports']:
                md += f"### {bug['description']}\n\n"
                md += f"- **Severity:** {bug['severity'].upper()}\n"
                md += f"- **Step:** {bug['step_number']}\n"
                md += f"- **Type:** {bug['error_type']}\n"
                md += f"- **Expected:** {bug['expected_result']}\n"
                md += f"- **Actual:** {bug['actual_result']}\n"
                
                if bug.get('error_details'):
                    md += f"- **Error Details:** {bug['error_details']}\n"
                
                if bug.get('reproduction_steps'):
                    md += "\n**Steps to Reproduce:**\n"
                    for i, step in enumerate(bug['reproduction_steps'], 1):
                        md += f"{i}. {step}\n"
                
                md += "\n"
        
        return md
    
    def save(self, output_dir: Path, formats: List[str] = ["json", "html", "markdown"]) -> Dict[str, Path]:
        """
        Save report in multiple formats.
        
        Args:
            output_dir: Directory to save reports
            formats: List of formats to generate
            
        Returns:
            Dict mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.generated_at.strftime("%Y%m%d_%H%M%S")
        base_name = f"test_report_{self.test_metrics.test_name}_{timestamp}"
        
        saved_files = {}
        
        if "json" in formats:
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                f.write(self.to_json())
            saved_files["json"] = json_path
        
        if "html" in formats:
            html_path = output_dir / f"{base_name}.html"
            with open(html_path, 'w') as f:
                f.write(self.to_html())
            saved_files["html"] = html_path
        
        if "markdown" in formats:
            md_path = output_dir / f"{base_name}.md"
            with open(md_path, 'w') as f:
                f.write(self.to_markdown())
            saved_files["markdown"] = md_path
        
        logger.info(f"Saved test report to {output_dir} in formats: {list(saved_files.keys())}")
        
        return saved_files


class ReportGenerator:
    """Generates various types of reports."""
    
    def __init__(
        self,
        analytics: MetricsCollector,
        output_dir: Path = Path("reports"),
        config: Optional[ReportConfig] = None
    ):
        self.analytics = analytics
        self.output_dir = Path(output_dir)
        self.config = config or ReportConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_test_report(
        self,
        test_id: UUID,
        error_report: Optional[ErrorReport] = None,
        journal: Optional[ExecutionJournal] = None
    ) -> Optional[TestExecutionReport]:
        """Generate report for a specific test."""
        if test_id not in self.analytics.test_metrics:
            logger.warning(f"No metrics found for test {test_id}")
            return None
        
        test_metrics = self.analytics.test_metrics[test_id]
        report = TestExecutionReport(
            test_metrics=test_metrics,
            error_report=error_report,
            journal=journal,
            config=self.config
        )
        
        return report
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report for all tests."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_summary": self.analytics.get_test_summary(),
            "performance_summary": self.analytics.get_performance_summary(),
            "tests": [
                {
                    "test_id": str(metrics.test_id),
                    "test_name": metrics.test_name,
                    "outcome": metrics.outcome.value if metrics.outcome else "active",
                    "duration": metrics.duration_seconds,
                    "success_rate": metrics.success_rate,
                    "errors": len(metrics.errors)
                }
                for metrics in self.analytics.test_metrics.values()
            ]
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        perf_data = self.analytics.get_performance_summary()
        
        # Add trends
        perf_data["trends"] = {
            "api_call_rate_trend": self._calculate_trend("api.calls"),
            "browser_action_trend": self._calculate_trend("browser.actions"),
            "error_rate_trend": self._calculate_trend("tests.failed")
        }
        
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "performance_metrics": perf_data,
            "recommendations": self._generate_performance_recommendations(perf_data)
        }
    
    def _calculate_trend(self, metric_name: str, windows: List[int] = [5, 15, 60]) -> Dict[str, float]:
        """Calculate metric trends over different time windows."""
        trends = {}
        for window in windows:
            rate = self.analytics.get_rate(metric_name, window)
            trends[f"{window}min"] = rate
        return trends
    
    def _generate_performance_recommendations(self, perf_data: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        # Check API call rate
        api_rate = perf_data["api_calls"]["rate_per_minute"]
        if api_rate > 100:
            recommendations.append(
                f"High API call rate ({api_rate:.1f}/min). Consider implementing caching or batching."
            )
        
        # Check step duration
        step_duration = perf_data["steps"]["duration"]
        if step_duration.get("p95", 0) > 5.0:
            recommendations.append(
                f"Slow step execution (p95: {step_duration['p95']:.1f}s). Review slow steps for optimization."
            )
        
        # Check success rate
        success_rate = perf_data["steps"]["success_rate"]
        if success_rate < 0.9:
            recommendations.append(
                f"Low step success rate ({success_rate*100:.1f}%). Review failing steps and improve reliability."
            )
        
        return recommendations
    
    def save_all_reports(self) -> Dict[str, Path]:
        """Generate and save all report types."""
        saved_files = {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Summary report
        summary = self.generate_summary_report()
        summary_path = self.output_dir / f"summary_report_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files["summary"] = summary_path
        
        # Performance report
        perf = self.generate_performance_report()
        perf_path = self.output_dir / f"performance_report_{timestamp}.json"
        with open(perf_path, 'w') as f:
            json.dump(perf, f, indent=2)
        saved_files["performance"] = perf_path
        
        # Individual test reports
        test_dir = self.output_dir / f"tests_{timestamp}"
        test_dir.mkdir(exist_ok=True)
        
        for test_id, metrics in self.analytics.test_metrics.items():
            if metrics.outcome:  # Only completed tests
                report = self.generate_test_report(test_id)
                if report:
                    test_files = report.save(test_dir)
                    saved_files[f"test_{test_id}"] = test_files
        
        logger.info(f"Generated {len(saved_files)} reports in {self.output_dir}")
        
        return saved_files


# HTML template for reports
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report: {{ report.metadata.test_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .passed { color: #4caf50; }
        .failed { color: #f44336; }
        .skipped { color: #ff9800; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .recommendation {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 4px solid #2196f3;
        }
        .bug-report {
            background: #ffebee;
            border: 1px solid #ffcdd2;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .bug-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .severity-critical { color: #d32f2f; font-weight: bold; }
        .severity-high { color: #f44336; font-weight: bold; }
        .severity-medium { color: #ff9800; }
        .severity-low { color: #2196f3; }
        .debug-info {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }
        .screenshots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .screenshot-container {
            text-align: center;
        }
        .screenshot-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .confidence-score {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .confidence-high { background: #c8e6c9; color: #1b5e20; }
        .confidence-medium { background: #fff3cd; color: #856404; }
        .confidence-low { background: #ffcdd2; color: #c62828; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Execution Report</h1>
        <h2>{{ report.metadata.test_name }}</h2>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value {% if report.summary.outcome == 'passed' %}passed{% else %}failed{% endif %}">
                    {{ report.summary.outcome|upper }}
                </div>
                <div class="metric-label">Test Outcome</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f"|format(report.summary.duration_seconds) }}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(report.summary.success_rate * 100) }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <h2>Test Steps</h2>
        <table>
            <tr>
                <th>Status</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            <tr>
                <td class="passed">Passed</td>
                <td>{{ report.steps.passed }}</td>
                <td>{{ "%.1f"|format((report.steps.passed / report.steps.total * 100) if report.steps.total else 0) }}%</td>
            </tr>
            <tr>
                <td class="failed">Failed</td>
                <td>{{ report.steps.failed }}</td>
                <td>{{ "%.1f"|format((report.steps.failed / report.steps.total * 100) if report.steps.total else 0) }}%</td>
            </tr>
            <tr>
                <td class="skipped">Skipped</td>
                <td>{{ report.steps.skipped }}</td>
                <td>{{ "%.1f"|format((report.steps.skipped / report.steps.total * 100) if report.steps.total else 0) }}%</td>
            </tr>
        </table>
        
        {% if report.errors %}
        <h2>Errors and Recommendations</h2>
        <p>Total errors encountered: {{ report.errors.total }}</p>
        
        {% if report.errors.recommendations %}
        <h3>Recommendations</h3>
        {% for rec in report.errors.recommendations %}
        <div class="recommendation">{{ rec }}</div>
        {% endfor %}
        {% endif %}
        {% endif %}
        
        {% if report.bug_reports %}
        <h2>Bug Reports</h2>
        <p>Found {{ report.bug_reports|length }} bug(s) during test execution:</p>
        
        {% for bug in report.bug_reports %}
        <div class="bug-report">
            <div class="bug-header">
                <h3>{{ bug.description }}</h3>
                <span class="severity-{{ bug.severity }}">{{ bug.severity|upper }}</span>
            </div>
            <div class="bug-details">
                <p><strong>Step:</strong> Step {{ bug.step_number }}</p>
                <p><strong>Error Type:</strong> {{ bug.error_type }}</p>
                <p><strong>Expected:</strong> {{ bug.expected_result }}</p>
                <p><strong>Actual:</strong> {{ bug.actual_result }}</p>
                {% if bug.error_details %}
                <p><strong>Error Details:</strong> {{ bug.error_details }}</p>
                {% endif %}
                
                {% if bug.reproduction_steps %}
                <h4>Steps to Reproduce:</h4>
                <ol>
                {% for step in bug.reproduction_steps %}
                    <li>{{ step }}</li>
                {% endfor %}
                </ol>
                {% endif %}
            </div>
        </div>
        {% endfor %}
        {% endif %}
        
        <!-- AI Conversations Section -->
        {% if ai_conversations %}
        <h2>🤖 AI Conversations</h2>
        <div style="margin: 20px 0;">
            {{ ai_conversations|safe }}
        </div>
        {% endif %}
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #666;">
            Generated at {{ report.metadata.generated_at }} | HAINDY Test Report v{{ report.metadata.report_version }}
        </footer>
    </div>
</body>
</html>
"""


class TestReporter:
    """
    High-level test reporter that integrates with the workflow coordinator.
    
    Converts TestState objects into comprehensive reports.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the test reporter."""
        self.config = config or ReportConfig()
        self.metrics_collector = MetricsCollector()
        self.report_generator = ReportGenerator(
            analytics=self.metrics_collector,
            config=self.config
        )
    
    async def generate_report(
        self,
        test_state: TestState,
        output_dir: Path,
        format: str = "html",
        action_storage: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Optional[Path]]:
        """
        Generate a test report from TestState.
        
        Args:
            test_state: The test state containing execution results
            output_dir: Directory to save the report
            format: Report format (html, json, markdown)
            action_storage: Optional action storage data to save alongside report
            
        Returns:
            Tuple of (report_path, actions_path) where actions_path may be None
        """
        # Use enhanced reporter for HTML format
        if format == "html":
            from src.monitoring.enhanced_reporter import EnhancedReporter
            enhanced_reporter = EnhancedReporter()
            return enhanced_reporter.generate_report(test_state, output_dir, action_storage)
        
        # For other formats, use the original implementation
        # Convert TestState to TestMetrics
        test_metrics = self._convert_to_metrics(test_state)
        
        # Create test execution report
        test_report = TestExecutionReport(
            test_metrics=test_metrics,
            error_report=self._extract_error_report(test_state),
            journal=None,  # TODO: Extract journal if available
            config=self.config
        )
        
        # Add bug reports to the report data
        test_report.bug_reports = self._extract_bug_reports(test_state)
        
        # Generate the report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{test_state.test_report.test_plan_id}_{timestamp}.{format}"
        output_path = output_dir / filename
        
        if format == "json":
            report_data = test_report.to_json()
        elif format == "markdown":
            report_data = test_report.to_markdown()
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        # Save the report
        output_path.write_text(report_data)
        logger.info(f"Generated {format} report: {output_path}")
        
        # Save actions file if provided
        actions_path = None
        if action_storage and action_storage.get("test_cases"):
            # Generate actions filename based on report filename
            test_plan_id = str(test_state.test_report.test_plan_id)
            actions_filename = f"{test_plan_id}_{timestamp}-actions.json"
            actions_path = output_dir / actions_filename
            
            # Write actions data with pretty formatting
            with open(actions_path, 'w') as f:
                json.dump(action_storage, f, indent=2, default=str)
            
            logger.info(f"Generated actions file: {actions_path}")
            
            # Log both paths
            logger.info(f"Test Report: {output_path}")
            logger.info(f"Actions Log: {actions_path}")
        else:
            # Log only report path if no actions
            logger.info(f"Test Report: {output_path}")
        
        return output_path, actions_path
    
    def _convert_to_metrics(self, test_state: TestState) -> TestMetrics:
        """Convert TestState to TestMetrics."""
        # Calculate metrics from test report
        test_report = test_state.test_report
        total_steps = sum(tc.steps_total for tc in test_report.test_cases)
        passed_steps = sum(tc.steps_completed for tc in test_report.test_cases)
        failed_steps = sum(tc.steps_failed for tc in test_report.test_cases)
        
        # Determine overall outcome
        from src.core.types import TestStatus
        if test_state.status == TestStatus.PASSED and failed_steps == 0:
            outcome = TestOutcome.PASSED
        elif test_state.status == TestStatus.FAILED or failed_steps > 0:
            outcome = TestOutcome.FAILED
        elif test_state.status == TestStatus.SKIPPED:
            outcome = TestOutcome.ERROR
        else:
            outcome = TestOutcome.PASSED  # Default
        
        # Calculate duration
        now = datetime.now(timezone.utc)
        start_time = now
        end_time = now
        
        # Calculate duration from start/end times
        if test_state.start_time and test_state.end_time:
            duration = (test_state.end_time - test_state.start_time).total_seconds()
        elif test_state.start_time:
            duration = (now - test_state.start_time).total_seconds()
        else:
            duration = 0.0
        
        # Get test info from test report
        test_id = test_state.test_report.test_plan_id
        test_name = test_state.test_report.test_plan_name
        start_time = test_state.test_report.started_at
        end_time = test_state.test_report.completed_at or now
        skipped_steps = sum(tc.steps_total - tc.steps_completed - tc.steps_failed for tc in test_state.test_report.test_cases)
        
        return TestMetrics(
            test_id=test_id,
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            outcome=outcome,
            steps_total=total_steps,
            steps_passed=passed_steps,
            steps_failed=failed_steps,
            steps_skipped=skipped_steps,
            api_calls=0,  # TODO: Track API calls
            browser_actions=0,  # TODO: Track browser actions
            screenshots_taken=0,  # TODO: Track screenshots
            errors=[],  # TODO: Track errors
            performance_metrics={}
        )
    
    def _extract_error_report(self, test_state: TestState) -> Optional[ErrorReport]:
        """Extract error report from test state."""
        # TODO: Implement proper error tracking in TestState
        # For now, return a basic error report based on error_count
        if test_state.error_count == 0:
            return None
        
        from src.error_handling.aggregator import ErrorCategory
        return ErrorReport(
            test_id=str(test_state.test_plan.plan_id),
            test_name=test_state.test_plan.name,
            start_time=test_state.start_time or datetime.now(timezone.utc),
            end_time=test_state.end_time or datetime.now(timezone.utc),
            total_errors=test_state.error_count,
            errors_by_category={},
            errors_by_type={},
            critical_errors=[],
            recovery_summary={"total_attempts": 0, "successful_recoveries": 0},
            recommendations=[]
        )
    
    def _extract_bug_reports(self, test_state: TestState) -> List[Dict[str, Any]]:
        """Extract bug reports from test state."""
        if not test_state.test_report or not test_state.test_report.bugs:
            return []
        
        bug_reports = []
        for bug in test_state.test_report.bugs:
            bug_reports.append({
                "bug_id": str(bug.bug_id),
                "step_id": str(bug.step_id),
                "test_case_id": str(bug.test_case_id),
                "step_number": bug.step_number,
                "description": bug.description,
                "severity": bug.severity.value,
                "error_type": bug.error_type,
                "expected_result": bug.expected_result,
                "actual_result": bug.actual_result,
                "screenshot_path": bug.screenshot_path,
                "error_details": bug.error_details,
                "reproduction_steps": bug.reproduction_steps
            })
        
        return bug_reports