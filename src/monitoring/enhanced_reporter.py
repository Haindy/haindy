"""
Enhanced test reporting with detailed error information and bug reports.

This module provides comprehensive reporting capabilities for the refactored
architecture, including detailed error screenshots and AI reasoning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Template

from src.core.enhanced_types import BugReport, EnhancedTestState, EnhancedActionResult
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


ENHANCED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report: {{ test_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        h3 { color: #888; }
        
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
        .warning { color: #ff9800; }
        
        .bug-report {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            background: #fff9f9;
        }
        .bug-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .bug-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #d32f2f;
        }
        .severity {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .severity-critical { background: #ffcdd2; color: #b71c1c; }
        .severity-high { background: #ffecb3; color: #e65100; }
        .severity-medium { background: #fff9c4; color: #f57f17; }
        .severity-low { background: #f1f8e9; color: #33691e; }
        
        .screenshots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .screenshot-container {
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        .screenshot-label {
            background: #f5f5f5;
            padding: 8px;
            font-weight: 500;
            text-align: center;
        }
        .screenshot-img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .reasoning-section {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .reasoning-title {
            font-weight: 600;
            margin-bottom: 8px;
        }
        .reasoning-content {
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
        }
        
        .recommendations {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 4px solid #2196f3;
        }
        .recommendation-item {
            margin: 5px 0;
        }
        
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
        
        .step-details {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .confidence-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .confidence-high { background: #c8e6c9; color: #1b5e20; }
        .confidence-medium { background: #fff9c4; color: #f57f17; }
        .confidence-low { background: #ffcdd2; color: #b71c1c; }
        
        .execution-timeline {
            margin: 20px 0;
        }
        .timeline-item {
            display: flex;
            align-items: start;
            margin: 10px 0;
        }
        .timeline-time {
            min-width: 100px;
            color: #666;
            font-size: 0.9em;
        }
        .timeline-content {
            flex: 1;
            margin-left: 20px;
        }
        
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enhanced Test Execution Report</h1>
        <h2>{{ test_name }}</h2>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value {% if status == 'completed' %}passed{% else %}failed{% endif %}">
                    {{ status|upper }}
                </div>
                <div class="metric-label">Test Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ duration }}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ success_rate }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">{{ bug_count }}</div>
                <div class="metric-label">Bug Reports</div>
            </div>
        </div>
        
        <h2>Test Steps Summary</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Description</th>
                <th>Action</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Time (ms)</th>
            </tr>
            {% for step in steps %}
            <tr>
                <td>{{ step.number }}</td>
                <td>{{ step.description }}</td>
                <td>{{ step.action_type }}</td>
                <td class="{% if step.success %}passed{% else %}failed{% endif %}">
                    {{ "PASSED" if step.success else "FAILED" }}
                </td>
                <td>
                    <span class="confidence-badge confidence-{{ step.confidence_level }}">
                        {{ "%.0f"|format(step.confidence * 100) }}%
                    </span>
                </td>
                <td>{{ "%.0f"|format(step.execution_time) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        {% if bug_reports %}
        <h2>Bug Reports</h2>
        {% for bug in bug_reports %}
        <div class="bug-report">
            <div class="bug-header">
                <div class="bug-title">Step {{ bug.step_number }}: {{ bug.step_description }}</div>
                <span class="severity severity-{{ bug.severity }}">{{ bug.severity|upper }}</span>
            </div>
            
            <div class="step-details">
                <p><strong>Action Attempted:</strong> {{ bug.action_attempted }}</p>
                <p><strong>Expected:</strong> {{ bug.expected_outcome }}</p>
                <p><strong>Actual:</strong> {{ bug.actual_outcome }}</p>
                <p><strong>Error:</strong> {{ bug.error_message }}</p>
            </div>
            
            {% if bug.screenshots %}
            <div class="screenshots">
                {% if bug.screenshots.grid_highlighted %}
                <div class="screenshot-container">
                    <div class="screenshot-label">Grid Selection (Highlighted)</div>
                    <img src="{{ bug.screenshots.grid_highlighted }}" alt="Grid selection" class="screenshot-img">
                </div>
                {% endif %}
                {% if bug.screenshots.before %}
                <div class="screenshot-container">
                    <div class="screenshot-label">Before Action</div>
                    <img src="{{ bug.screenshots.before }}" alt="Before" class="screenshot-img">
                </div>
                {% endif %}
                {% if bug.screenshots.after %}
                <div class="screenshot-container">
                    <div class="screenshot-label">After Action</div>
                    <img src="{{ bug.screenshots.after }}" alt="After" class="screenshot-img">
                </div>
                {% endif %}
            </div>
            {% endif %}
            
            <div class="reasoning-section">
                <div class="reasoning-title">AI Reasoning</div>
                <div class="reasoning-content">Validation: {{ bug.ai_reasoning.validation }}

Coordinate Selection: {{ bug.ai_reasoning.coordinates }}

Analysis: {{ bug.ai_reasoning.analysis }}</div>
            </div>
            
            {% if bug.recommended_fixes %}
            <div class="recommendations">
                <strong>Recommendations:</strong>
                {% for fix in bug.recommended_fixes %}
                <div class="recommendation-item">• {{ fix }}</div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% endfor %}
        {% endif %}
        
        <h2>Execution Timeline</h2>
        <div class="execution-timeline">
            {% for event in timeline %}
            <div class="timeline-item">
                <div class="timeline-time">{{ event.time }}</div>
                <div class="timeline-content">
                    <strong>{{ event.action }}</strong>
                    {% if event.details %}
                    <div style="color: #666; font-size: 0.9em;">{{ event.details }}</div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <footer>
            Generated at {{ generated_at }} | HAINDY Enhanced Test Report v2.0
        </footer>
    </div>
</body>
</html>
"""


class EnhancedReporter:
    """Generate enhanced reports from refactored test execution."""
    
    def __init__(self):
        """Initialize the enhanced reporter."""
        self.logger = logger
    
    def generate_html_report(
        self,
        test_state: EnhancedTestState,
        output_path: Path
    ) -> Path:
        """
        Generate comprehensive HTML report with bug details.
        
        Args:
            test_state: Enhanced test state with execution details
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Calculate metrics from test report
        test_report = test_state.test_report
        total_steps = sum(tc.steps_total for tc in test_report.test_cases)
        completed_steps = sum(tc.steps_completed for tc in test_report.test_cases)
        failed_steps = sum(tc.steps_failed for tc in test_report.test_cases)
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Calculate duration
        if test_state.start_time and test_state.end_time:
            duration = (test_state.end_time - test_state.start_time).total_seconds()
        else:
            duration = 0
        
        # Build step details
        steps = []
        for i, result in enumerate(test_state.execution_history):
            confidence_level = "high" if result.coordinate_confidence >= 0.8 else \
                             "medium" if result.coordinate_confidence >= 0.6 else "low"
            
            steps.append({
                "number": i + 1,
                "description": test_state.test_plan.steps[i].description if i < len(test_state.test_plan.steps) else "Unknown",
                "action_type": result.action_type,
                "success": result.execution_success and result.validation_passed,
                "confidence": result.coordinate_confidence,
                "confidence_level": confidence_level,
                "execution_time": result.execution_time_ms
            })
        
        # Build timeline
        timeline = []
        for result in test_state.execution_history:
            timeline.append({
                "time": result.timestamp.strftime("%H:%M:%S"),
                "action": f"{result.action_type} on {result.grid_cell}",
                "details": f"Validation: {result.validation_status.value}, "
                          f"Execution: {'Success' if result.execution_success else 'Failed'}"
            })
        
        # Prepare bug reports for template
        bug_reports_data = []
        for bug in test_state.bug_reports:
            bug_reports_data.append({
                "step_number": bug.step_number,
                "step_description": bug.step_description,
                "severity": bug.severity,
                "action_attempted": bug.action_attempted,
                "expected_outcome": bug.expected_outcome,
                "actual_outcome": bug.actual_outcome,
                "error_message": bug.error_message,
                "screenshots": bug.screenshots,
                "ai_reasoning": bug.ai_reasoning,
                "recommended_fixes": bug.recommended_fixes
            })
        
        # Render template
        template = Template(ENHANCED_HTML_TEMPLATE)
        html_content = template.render(
            test_name=test_state.test_plan.name,
            status=test_state.status.value,
            duration=f"{duration:.1f}",
            success_rate=f"{success_rate:.0f}",
            bug_count=len(test_state.bug_reports),
            steps=steps,
            bug_reports=bug_reports_data,
            timeline=timeline,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)
        
        logger.info(f"Generated enhanced HTML report: {output_path}")
        
        return output_path
    
    def print_terminal_summary(self, test_state: EnhancedTestState) -> None:
        """Print a detailed summary to terminal."""
        print("\n" + "="*80)
        print(f"TEST EXECUTION SUMMARY: {test_state.test_plan.name}")
        print("="*80)
        
        # Basic metrics
        total_steps = len(test_state.test_plan.steps)
        print(f"\nStatus: {test_state.status.value.upper()}")
        print(f"Steps: {len(test_state.completed_steps)}/{total_steps} completed")
        print(f"Failures: {len(test_state.failed_steps)}")
        print(f"Bug Reports: {len(test_state.bug_reports)}")
        
        # Duration
        if test_state.start_time and test_state.end_time:
            duration = (test_state.end_time - test_state.start_time).total_seconds()
            print(f"Duration: {duration:.1f}s")
        
        # Bug report summary
        if test_state.bug_reports:
            print("\n" + "-"*80)
            print("BUG REPORTS:")
            print("-"*80)
            
            for bug in test_state.bug_reports:
                print(f"\n[{bug.severity.upper()}] Step {bug.step_number}: {bug.step_description}")
                print(f"  Error: {bug.error_message}")
                print(f"  Category: {bug.category}")
                
                if bug.screenshots:
                    print("  Screenshots saved:")
                    for name, path in bug.screenshots.items():
                        print(f"    - {name}: {path}")
                
                if bug.recommended_fixes:
                    print("  Recommendations:")
                    for fix in bug.recommended_fixes:
                        print(f"    - {fix}")
        
        # AI failure analysis
        if test_state.metadata.get("failure_analysis"):
            print("\n" + "-"*80)
            print("AI FAILURE ANALYSIS:")
            print("-"*80)
            print(test_state.metadata["failure_analysis"])
        
        print("\n" + "="*80 + "\n")