"""
Simple HTML reporter for test execution with enhanced error reporting.

This reporter works directly with TestState and TestStepResult to generate
comprehensive HTML reports with bug details.
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from jinja2 import Template

from src.core.types import StepResult as TestStepResult
from src.core.types import TestState
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


HTML_TEMPLATE = """
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
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #333; }
        h1 { margin-bottom: 10px; }
        .test-info { color: #666; margin-bottom: 20px; }
        
        /* Summary Section */
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Status Colors */
        .passed { color: #4caf50; }
        .failed { color: #f44336; }
        .skipped { color: #ff9800; }
        .in-progress { color: #2196f3; }
        
        /* Steps Table */
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
            position: sticky;
            top: 0;
        }
        tr:hover {
            background: #f9f9f9;
        }
        
        /* Bug Reports */
        .bug-reports {
            margin-top: 40px;
        }
        .bug-report {
            background: #fff;
            border: 2px solid #ffcdd2;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            position: relative;
        }
        .bug-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .bug-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #d32f2f;
        }
        .bug-metadata {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        /* Severity Badges */
        .severity-badge {
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85em;
            font-weight: 500;
            text-transform: uppercase;
        }
        .severity-critical {
            background: #d32f2f;
            color: white;
        }
        .severity-high {
            background: #f44336;
            color: white;
        }
        .severity-medium {
            background: #ff9800;
            color: white;
        }
        .severity-low {
            background: #2196f3;
            color: white;
        }
        
        /* Confidence Scores */
        .confidence-scores {
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .confidence-item {
            background: #f5f5f5;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .confidence-value {
            font-weight: 600;
            margin-left: 5px;
        }
        .confidence-high { color: #4caf50; }
        .confidence-medium { color: #ff9800; }
        .confidence-low { color: #f44336; }
        
        /* Bug Details */
        .bug-section {
            margin: 20px 0;
        }
        .bug-section h4 {
            margin: 10px 0;
            color: #666;
            font-size: 1.1em;
        }
        .error-message {
            background: #ffebee;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #f44336;
            font-family: monospace;
            margin: 10px 0;
        }
        .debug-info {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            overflow-x: auto;
            margin: 10px 0;
        }
        
        /* Screenshots */
        .screenshots {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .screenshot-container {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .screenshot-label {
            font-weight: 600;
            margin-bottom: 10px;
            color: #666;
        }
        .screenshot-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .screenshot-container img:hover {
            transform: scale(1.02);
        }
        
        /* AI Analysis */
        .ai-analysis {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #4caf50;
            margin: 10px 0;
        }
        .anomalies {
            background: #fff3cd;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
        .recommendations {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #2196f3;
            margin: 10px 0;
        }
        
        /* Grid Info */
        .grid-info {
            display: inline-block;
            background: #e8eaf6;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            margin: 0 5px;
        }
        
        /* Lists */
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        li {
            margin: 5px 0;
        }
        
        /* Footer */
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Execution Report</h1>
        <div class="test-info">
            <strong>{{ test_name }}</strong> | 
            Generated: {{ generated_at }} | 
            Duration: {{ duration }}s
        </div>
        
        <!-- Summary Section -->
        <div class="summary">
            <div class="metric">
                <div class="metric-value {{ status_class }}">
                    {{ status|upper }}
                </div>
                <div class="metric-label">Test Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ total_steps }}</div>
                <div class="metric-label">Total Steps</div>
            </div>
            <div class="metric">
                <div class="metric-value passed">{{ passed_steps }}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">{{ failed_steps }}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ success_rate }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <!-- Test Steps -->
        <h2>Test Steps</h2>
        <table>
            <thead>
                <tr>
                    <th width="60">Step #</th>
                    <th>Description</th>
                    <th>Action</th>
                    <th width="100">Status</th>
                    <th>Result</th>
                    <th width="100">Mode</th>
                </tr>
            </thead>
            <tbody>
                {% for step in steps %}
                <tr>
                    <td>{{ step.number }}</td>
                    <td>{{ step.description }}</td>
                    <td>{{ step.action }}</td>
                    <td><span class="{{ step.status_class }}">{{ step.status|upper }}</span></td>
                    <td>{{ step.result }}</td>
                    <td>{{ step.mode }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <!-- Bug Reports -->
        {% if bug_reports %}
        <div class="bug-reports">
            <h2>Bug Reports</h2>
            {% for bug in bug_reports %}
            <div class="bug-report">
                <div class="bug-header">
                    <div>
                        <div class="bug-title">Step {{ bug.step_number }}: {{ bug.error_message }}</div>
                        <div style="color: #666; margin-top: 5px;">{{ bug.attempted_action }}</div>
                    </div>
                    <div class="bug-metadata">
                        <span class="severity-badge severity-{{ bug.severity }}">{{ bug.severity }}</span>
                        <span>{{ bug.failure_type }} failure</span>
                    </div>
                </div>
                
                <!-- Confidence Scores -->
                {% if bug.confidence_scores %}
                <div class="confidence-scores">
                    {% for phase, score in bug.confidence_scores.items() %}
                    <div class="confidence-item">
                        {{ phase|title }}:
                        <span class="confidence-value {% if score >= 0.8 %}confidence-high{% elif score >= 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                            {{ "%.0f"|format(score * 100) }}%
                        </span>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <!-- Expected vs Actual -->
                <div class="bug-section">
                    <h4>Expected vs Actual</h4>
                    <div class="debug-info">
                        <strong>Expected:</strong> {{ bug.expected_outcome }}<br>
                        <strong>Actual:</strong> {{ bug.actual_outcome }}
                    </div>
                </div>
                
                <!-- Error Details -->
                {% if bug.detailed_error %}
                <div class="bug-section">
                    <h4>Error Details</h4>
                    <div class="error-message">{{ bug.detailed_error }}</div>
                </div>
                {% endif %}
                
                <!-- Grid Information -->
                {% if bug.grid_cell_targeted %}
                <div class="bug-section">
                    <h4>Grid Interaction</h4>
                    <div>
                        Targeted cell: <span class="grid-info">{{ bug.grid_cell_targeted }}</span>
                        {% if bug.coordinates_used %}
                        at offset ({{ "%.2f"|format(bug.coordinates_used.offset_x) }}, {{ "%.2f"|format(bug.coordinates_used.offset_y) }})
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <!-- Screenshots -->
                {% if bug.screenshots %}
                <div class="bug-section">
                    <h4>Screenshots</h4>
                    <div class="screenshots">
                        {% for screenshot in bug.screenshots %}
                        <div class="screenshot-container">
                            <div class="screenshot-label">{{ screenshot.label }}</div>
                            <img src="data:image/png;base64,{{ screenshot.data }}" 
                                 alt="{{ screenshot.label }}"
                                 onclick="window.open(this.src, '_blank')">
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <!-- UI Anomalies -->
                {% if bug.ui_anomalies %}
                <div class="bug-section">
                    <h4>Detected Anomalies</h4>
                    <div class="anomalies">
                        <ul>
                            {% for anomaly in bug.ui_anomalies %}
                            <li>{{ anomaly }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                {% endif %}
                
                <!-- AI Recommendations -->
                {% if bug.suggested_fixes %}
                <div class="bug-section">
                    <h4>Recommended Fixes</h4>
                    <div class="recommendations">
                        <ul>
                            {% for fix in bug.suggested_fixes %}
                            <li>{{ fix }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                {% endif %}
                
                <!-- Browser State -->
                <div class="bug-section">
                    <h4>Browser State</h4>
                    <div class="debug-info">
                        <strong>URL Before:</strong> {{ bug.url_before or "N/A" }}<br>
                        <strong>URL After:</strong> {{ bug.url_after or "N/A" }}<br>
                        <strong>Page Title Before:</strong> {{ bug.page_title_before or "N/A" }}<br>
                        <strong>Page Title After:</strong> {{ bug.page_title_after or "N/A" }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="footer">
            Generated by HAINDY Test Automation System
        </div>
    </div>
</body>
</html>
"""


class SimpleHTMLReporter:
    """Generate HTML reports from test execution results."""
    
    def __init__(self):
        """Initialize the reporter."""
        self.logger = logger
    
    def generate_report(
        self,
        test_state: TestState,
        execution_history: List[TestStepResult],
        output_path: Path
    ) -> Path:
        """
        Generate HTML report from test state and execution history.
        
        Args:
            test_state: The test state with plan information
            execution_history: List of test step results
            output_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        # Prepare template data
        template_data = self._prepare_template_data(test_state, execution_history)
        
        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)
        
        self.logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    def _prepare_template_data(
        self,
        test_state: TestState,
        execution_history: List[TestStepResult]
    ) -> dict:
        """Prepare data for template rendering."""

        def _encode_image(image_bytes: Optional[bytes]) -> Optional[str]:
            if not image_bytes:
                return None
            return base64.b64encode(image_bytes).decode()

        def _build_bug_entry(bug: Optional[Any]) -> Optional[dict]:
            if bug is None:
                return None

            bug_data = {
                "step_number": getattr(bug, "step_number", None),
                "error_message": getattr(bug, "error_message", "Unknown error"),
                "attempted_action": getattr(bug, "attempted_action", "N/A"),
                "expected_outcome": getattr(bug, "expected_outcome", "N/A"),
                "actual_outcome": getattr(bug, "actual_outcome", "N/A"),
                "severity": getattr(bug, "severity", "medium"),
                "failure_type": getattr(bug, "failure_type", "unknown"),
                "detailed_error": getattr(bug, "detailed_error", None),
                "confidence_scores": getattr(bug, "confidence_scores", {}) or {},
                "grid_cell_targeted": getattr(bug, "grid_cell_targeted", None),
                "coordinates_used": getattr(bug, "coordinates_used", None),
                "ui_anomalies": getattr(bug, "ui_anomalies", []) or [],
                "suggested_fixes": getattr(bug, "suggested_fixes", []) or [],
                "url_before": getattr(bug, "url_before", None),
                "url_after": getattr(bug, "url_after", None),
                "page_title_before": getattr(bug, "page_title_before", None),
                "page_title_after": getattr(bug, "page_title_after", None),
                "screenshots": [],
            }

            screenshots = [
                ("Grid Screenshot (Highlighted)", getattr(bug, "grid_screenshot", None)),
                ("Before Action", getattr(bug, "screenshot_before", None)),
                ("After Action", getattr(bug, "screenshot_after", None)),
            ]
            for label, image in screenshots:
                encoded = _encode_image(image)
                if encoded:
                    bug_data["screenshots"].append({"label": label, "data": encoded})

            return bug_data

        status_class_map = {
            "pending": "in-progress",
            "in_progress": "in-progress",
            "passed": "passed",
            "completed": "passed",
            "failed": "failed",
            "skipped": "skipped",
            "blocked": "skipped",
        }

        test_report = test_state.test_report
        duration = 0
        if test_state.start_time and test_state.end_time:
            duration = round((test_state.end_time - test_state.start_time).total_seconds(), 2)

        if test_report:
            total_steps = sum(tc.steps_total for tc in test_report.test_cases)
            passed_steps = sum(tc.steps_completed for tc in test_report.test_cases)
            failed_steps = sum(tc.steps_failed for tc in test_report.test_cases)
            success_rate = int((passed_steps / total_steps * 100) if total_steps > 0 else 0)

            steps: List[dict] = []
            for test_case in test_report.test_cases:
                for step_result in test_case.step_results:
                    status_value = getattr(step_result, "status", "in_progress")
                    if hasattr(status_value, "value"):
                        status_value = status_value.value
                    if status_value in {"passed", "success", "completed"}:
                        status = "passed"
                    elif status_value in {"failed", "error", "blocked"}:
                        status = "failed"
                    else:
                        status = "in-progress"

                    steps.append(
                        {
                            "number": getattr(step_result, "step_number", len(steps) + 1),
                            "description": getattr(
                                step_result, "step_description", getattr(test_case, "name", "Step")
                            ),
                            "action": getattr(step_result, "action_taken", "N/A"),
                            "status": status,
                            "status_class": status,
                            "result": getattr(step_result, "actual_result", "N/A") or "N/A",
                            "mode": getattr(step_result, "execution_mode", "enhanced"),
                        }
                    )

            bug_reports = []
            for bug in test_report.bugs:
                bug_entry = _build_bug_entry(bug)
                if bug_entry:
                    bug_reports.append(bug_entry)

            test_name = test_report.test_plan_name

        else:
            total_steps = len(getattr(test_state.test_plan, "steps", []))
            passed_steps = len(getattr(test_state, "completed_steps", []))
            failed_steps = len(getattr(test_state, "failed_steps", []))
            success_rate = int((passed_steps / total_steps * 100) if total_steps > 0 else 0)

            steps: List[dict] = []
            for index, step_result in enumerate(execution_history, start=1):
                step = getattr(step_result, "step", None)
                step_number = getattr(step, "step_number", index)
                description = getattr(step, "description", f"Step {step_number}")
                action_taken = getattr(step_result, "action_taken", None)
                action = action_taken or getattr(step, "action", "N/A")
                success = getattr(step_result, "success", False)
                status = "passed" if success else "failed"

                steps.append(
                    {
                        "number": step_number,
                        "description": description,
                        "action": action,
                        "status": status,
                        "status_class": status,
                        "result": getattr(step_result, "actual_result", "N/A") or "N/A",
                        "mode": getattr(step_result, "execution_mode", "unknown"),
                    }
                )

            if not steps and getattr(test_state.test_plan, "steps", []):
                for index, step in enumerate(test_state.test_plan.steps, start=1):
                    steps.append(
                        {
                            "number": getattr(step, "step_number", index),
                            "description": getattr(step, "description", f"Step {index}"),
                            "action": getattr(step, "action", "N/A"),
                            "status": "skipped",
                            "status_class": "skipped",
                            "result": "Not executed",
                            "mode": "unknown",
                        }
                    )

            bug_reports = []
            for entry in execution_history:
                bug = None
                creator = getattr(entry, "create_bug_report", None)
                if callable(creator):
                    try:
                        bug = creator(test_state.test_plan.name)
                    except TypeError:
                        bug = creator()
                elif hasattr(entry, "bug_report"):
                    bug = getattr(entry, "bug_report")

                bug_entry = _build_bug_entry(bug)
                if bug_entry:
                    bug_reports.append(bug_entry)

            test_name = getattr(test_state.test_plan, "name", "Test Plan")

        return {
            "test_name": test_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration,
            "status": test_state.status.value,
            "status_class": status_class_map.get(test_state.status.value, "in-progress"),
            "total_steps": total_steps,
            "passed_steps": passed_steps,
            "failed_steps": failed_steps,
            "success_rate": success_rate,
            "steps": steps,
            "bug_reports": bug_reports
        }
