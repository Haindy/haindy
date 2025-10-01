"""
Enhanced HTML test report generator with hierarchical structure.

This module creates comprehensive test reports organized by the test plan
hierarchy with expandable/collapsible sections and rich details.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from jinja2 import Environment, Template

from src.core.types import TestState, TestStatus, BugSeverity

logger = logging.getLogger(__name__)


ENHANCED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report: {{ test_plan_name }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        
        /* Container styling */
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Summary box */
        .summary-box {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
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
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Status colors */
        .status-passed { color: #4caf50; }
        .status-failed { color: #f44336; }
        .status-skipped { color: #ff9800; }
        .status-in-progress { color: #2196f3; }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 16px;
            overflow: hidden;
        }
        
        .card-header {
            padding: 16px 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #e0e0e0;
            transition: background-color 0.2s;
        }
        
        .card-header:hover {
            background-color: #f5f5f5;
        }
        
        .card-header.status-passed {
            border-left: 4px solid #4caf50;
        }
        
        .card-header.status-failed {
            border-left: 4px solid #f44336;
        }
        
        .card-header.status-skipped {
            border-left: 4px solid #ff9800;
        }
        
        .card-title {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
        }
        
        .card-title-text {
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .card-metadata {
            display: flex;
            gap: 20px;
            align-items: center;
            color: #666;
            font-size: 0.9em;
        }
        
        .expand-icon {
            transition: transform 0.2s;
            font-size: 1.2em;
        }
        
        .expand-icon.expanded {
            transform: rotate(90deg);
        }
        
        .card-body {
            padding: 20px;
            display: none;
        }
        
        .card-body.expanded {
            display: block;
        }
        
        /* Icons */
        .icon {
            font-size: 1.2em;
            margin-right: 5px;
        }
        
        .icon-passed { color: #4caf50; }
        .icon-failed { color: #f44336; }
        .icon-skipped { color: #ff9800; }
        
        /* Nested cards */
        .nested-container {
            margin-left: 20px;
        }
        
        /* Details sections */
        .detail-section {
            margin: 16px 0;
            padding: 16px;
            background: #f9f9f9;
            border-radius: 6px;
        }
        
        .detail-section h4 {
            margin: 0 0 12px 0;
            color: #555;
            font-size: 1em;
        }
        
        /* Screenshots */
        .screenshot-container {
            display: flex;
            gap: 16px;
            margin: 16px 0;
            flex-wrap: wrap;
        }
        
        .screenshot-item {
            text-align: center;
        }
        
        .screenshot-item img {
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .screenshot-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 4px;
        }
        
        /* AI Conversation */
        .ai-conversation {
            background: #f5f5f5;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0;
        }
        
        .ai-message {
            margin: 12px 0;
            padding: 12px;
            border-radius: 6px;
        }
        
        .ai-message.user {
            background: #e3f2fd;
            margin-right: 20%;
        }
        
        .ai-message.assistant {
            background: #f5f5f5;
            margin-left: 20%;
            border: 1px solid #e0e0e0;
        }
        
        .ai-message-role {
            font-weight: 600;
            color: #666;
            font-size: 0.85em;
            margin-bottom: 4px;
        }
        
        /* Bug report */
        .bug-report {
            background: #ffebee;
            border: 1px solid #ffcdd2;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
        }
        
        .bug-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .bug-title {
            font-weight: 600;
            color: #c62828;
        }
        
        .severity-badge {
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85em;
            font-weight: 600;
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
        
        /* Action details */
        .action-list {
            margin: 16px 0;
        }
        
        .action-item {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin: 8px 0;
            padding: 16px;
        }
        
        .action-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .action-type {
            font-weight: 600;
            color: #1976d2;
        }
        
        .action-details {
            font-size: 0.9em;
            color: #666;
            margin: 8px 0;
        }
        
        /* Browser calls */
        .browser-calls {
            background: #f5f5f5;
            border-radius: 4px;
            padding: 12px;
            margin: 8px 0;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85em;
            overflow-x: auto;
        }
        
        /* Test plan metadata */
        .metadata-section {
            background: #f9f9f9;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0;
        }
        
        .metadata-item {
            display: flex;
            margin: 8px 0;
        }
        
        .metadata-label {
            font-weight: 600;
            color: #666;
            min-width: 150px;
        }
        
        .metadata-value {
            color: #333;
        }
        
        /* Utility classes */
        .text-muted {
            color: #666;
        }
        
        .mb-0 { margin-bottom: 0; }
        .mt-16 { margin-top: 16px; }
        .mb-16 { margin-bottom: 16px; }
        
        /* Code blocks */
        .code-block {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }
        
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        th {
            background: #f5f5f5;
            font-weight: 600;
            color: #666;
        }
        
        /* Footer */
        .footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Summary Box -->
        <div class="summary-box">
            <h1>Test Execution Report</h1>
            <div class="summary-grid">
                <div class="metric">
                    <div class="metric-value status-{{ overall_status|lower }}">
                        {{ overall_status|upper }}
                    </div>
                    <div class="metric-label">Test Outcome</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ duration }}s</div>
                    <div class="metric-label">Duration</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ success_rate }}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>
        </div>
        
        <!-- Test Plan Container -->
        <div class="card">
            <div class="card-header" onclick="toggleSection('test-plan')">
                <div class="card-title">
                    <span class="expand-icon expanded" id="test-plan-icon">▶</span>
                    <h2 class="card-title-text mb-0">{{ test_plan_name }}</h2>
                </div>
            </div>
            
            <div class="card-body expanded" id="test-plan-body">
                <!-- Test Plan Metadata -->
                <div class="metadata-section">
                    <h3>Test Plan Details</h3>
                    <div class="metadata-item">
                        <span class="metadata-label">Plan ID:</span>
                        <span class="metadata-value">{{ test_plan_id }}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Description:</span>
                        <span class="metadata-value">{{ test_plan_description }}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Created:</span>
                        <span class="metadata-value">{{ test_plan_created }}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Started:</span>
                        <span class="metadata-value">{{ test_started }}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Completed:</span>
                        <span class="metadata-value">{{ test_completed }}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Environment:</span>
                        <span class="metadata-value"><pre>{{ environment }}</pre></span>
                    </div>
                </div>
                
                <!-- Test Cases -->
                <h3>Test Cases</h3>
                <div class="nested-container">
                    {% for test_case in test_cases %}
                    <div class="card">
                        <div class="card-header status-{{ test_case.status|lower }}" onclick="toggleSection('case-{{ loop.index }}')">
                            <div class="card-title">
                                <span class="expand-icon {% if test_case.status == 'failed' %}expanded{% endif %}" id="case-{{ loop.index }}-icon">▶</span>
                                <span class="icon icon-{{ test_case.status|lower }}">
                                    {% if test_case.status == 'passed' %}✓
                                    {% elif test_case.status == 'failed' %}✗
                                    {% elif test_case.status == 'skipped' %}⊖
                                    {% else %}○
                                    {% endif %}
                                </span>
                                <span class="card-title-text">{{ test_case.name }}</span>
                            </div>
                            <div class="card-metadata">
                                <span>{{ test_case.steps_completed }}/{{ test_case.steps_total }} steps</span>
                                <span>{{ test_case.duration }}s</span>
                            </div>
                        </div>
                        
                        <div class="card-body {% if test_case.status == 'failed' %}expanded{% endif %}" id="case-{{ loop.index }}-body">
                            <!-- Test Case Details -->
                            <div class="metadata-section">
                                <div class="metadata-item">
                                    <span class="metadata-label">Priority:</span>
                                    <span class="metadata-value">{{ test_case.priority|upper }}</span>
                                </div>
                                {% if test_case.prerequisites %}
                                <div class="metadata-item">
                                    <span class="metadata-label">Prerequisites:</span>
                                    <span class="metadata-value">{{ test_case.prerequisites|join(', ') }}</span>
                                </div>
                                {% endif %}
                                {% if test_case.error_message %}
                                <div class="metadata-item">
                                    <span class="metadata-label">Error:</span>
                                    <span class="metadata-value" style="color: #f44336;">{{ test_case.error_message }}</span>
                                </div>
                                {% endif %}
                            </div>
                            
                            <!-- Test Steps -->
                            <h4>Test Steps</h4>
                            <div class="nested-container">
                                {% for step in test_case.steps %}
                                <div class="card">
                                    <div class="card-header status-{{ step.status|lower }}" onclick="toggleSection('step-{{ loop.index0 }}-{{ loop.index }}')">
                                        <div class="card-title">
                                            <span class="expand-icon {% if step.status == 'failed' %}expanded{% endif %}" id="step-{{ loop.index0 }}-{{ loop.index }}-icon">▶</span>
                                            <span class="icon icon-{{ step.status|lower }}">
                                                {% if step.status == 'passed' %}✓
                                                {% elif step.status == 'failed' %}✗
                                                {% elif step.status == 'skipped' %}⊖
                                                {% else %}○
                                                {% endif %}
                                            </span>
                                            <span class="card-title-text">Step {{ step.number }}: {{ step.action }}</span>
                                        </div>
                                        <div class="card-metadata">
                                            <span>{{ step.duration }}s</span>
                                            {% if step.confidence %}
                                            <span>Confidence: {{ (step.confidence * 100)|round(1) }}%</span>
                                            {% endif %}
                                        </div>
                                    </div>
                                    
                                    <div class="card-body {% if step.status == 'failed' %}expanded{% endif %}" id="step-{{ loop.index0 }}-{{ loop.index }}-body">
                                        <!-- Step Details -->
                                        <div class="detail-section">
                                            <h4>Step Details</h4>
                                            <div class="metadata-item">
                                                <span class="metadata-label">Expected Result:</span>
                                                <span class="metadata-value">{{ step.expected_result }}</span>
                                            </div>
                                            <div class="metadata-item">
                                                <span class="metadata-label">Actual Result:</span>
                                                <span class="metadata-value">{{ step.actual_result }}</span>
                                            </div>
                                            {% if step.error_message %}
                                            <div class="metadata-item">
                                                <span class="metadata-label">Error:</span>
                                                <span class="metadata-value" style="color: #f44336;">{{ step.error_message }}</span>
                                            </div>
                                            {% endif %}
                                        </div>
                                        
                                        <!-- Bug Report (if failed) -->
                                        {% if step.bug_report %}
                                        <div class="bug-report">
                                            <div class="bug-header">
                                                <h4 class="bug-title">{{ step.bug_report.description }}</h4>
                                                <span class="severity-badge severity-{{ step.bug_report.severity }}">
                                                    {{ step.bug_report.severity|upper }}
                                                </span>
                                            </div>
                                            <div class="bug-details">
                                                <p><strong>Error Type:</strong> {{ step.bug_report.error_type }}</p>
                                                <p><strong>Expected:</strong> {{ step.bug_report.expected_result }}</p>
                                                <p><strong>Actual:</strong> {{ step.bug_report.actual_result }}</p>
                                                {% if step.bug_report.error_details %}
                                                <p><strong>Details:</strong> {{ step.bug_report.error_details }}</p>
                                                {% endif %}
                                                
                                                {% if step.bug_report.reproduction_steps %}
                                                <h5>Steps to Reproduce:</h5>
                                                <ol>
                                                    {% for repro_step in step.bug_report.reproduction_steps %}
                                                    <li>{{ repro_step }}</li>
                                                    {% endfor %}
                                                </ol>
                                                {% endif %}
                                            </div>
                                        </div>
                                        {% endif %}
                                        
                                        <!-- Test Runner AI Conversation -->
                                        {% if step.test_runner_conversation %}
                                        <div class="detail-section">
                                            <h4>Test Runner AI Analysis</h4>
                                            <div class="ai-conversation">
                                                <div class="ai-message user">
                                                    <div class="ai-message-role">Test Runner Query</div>
                                                    <div class="code-block">{{ step.test_runner_conversation.prompt }}</div>
                                                </div>
                                                <div class="ai-message assistant">
                                                    <div class="ai-message-role">AI Response</div>
                                                    <div class="code-block">{{ step.test_runner_conversation.response|tojson }}</div>
                                                </div>
                                            </div>
                                        </div>
                                        {% endif %}
                                        
                                        <!-- Step Screenshots -->
                                        {% if step.screenshots %}
                                        <div class="detail-section">
                                            <h4>Step Screenshots</h4>
                                            <div class="screenshot-container">
                                                {% if step.screenshots.before %}
                                                <div class="screenshot-item">
                                                    <a href="file://{{ step.screenshots.before }}" target="_blank">
                                                        <img src="file://{{ step.screenshots.before }}" alt="Before">
                                                    </a>
                                                    <div class="screenshot-label">Before</div>
                                                </div>
                                                {% endif %}
                                                {% if step.screenshots.after %}
                                                <div class="screenshot-item">
                                                    <a href="file://{{ step.screenshots.after }}" target="_blank">
                                                        <img src="file://{{ step.screenshots.after }}" alt="After">
                                                    </a>
                                                    <div class="screenshot-label">After</div>
                                                </div>
                                                {% endif %}
                                            </div>
                                        </div>
                                        {% endif %}
                                        
                                        <!-- Actions -->
                                        {% if step.actions %}
                                        <div class="detail-section">
                                            <h4>Actions Performed</h4>
                                            <div class="action-list">
                                                {% for action in step.actions %}
                                                <div class="action-item">
                                                    <div class="action-header">
                                                        <span class="action-type">{{ action.action_type|upper }}</span>
                                                        <span class="text-muted">{{ action.duration }}ms</span>
                                                    </div>
                                                    <div class="action-details">
                                                        <p><strong>Target:</strong> {{ action.target }}</p>
                                                        {% if action.value %}
                                                        <p><strong>Value:</strong> {{ action.value }}</p>
                                                        {% endif %}
                                                        <p><strong>Description:</strong> {{ action.description }}</p>
                                                        {% if action.result %}
                                                        <p><strong>Result:</strong> 
                                                            <span class="{% if action.result.success %}status-completed{% else %}status-failed{% endif %}">
                                                                {% if action.result.success %}Success{% else %}Failed{% endif %}
                                                            </span>
                                                            {% if action.result.error %} - {{ action.result.error }}{% endif %}
                                                        </p>
                                                        {% endif %}
                                                    </div>
                                                    
                                                    <!-- Action AI Conversation -->
                                                    {% if action.ai_conversation %}
                                                    <div class="mt-16">
                                                        <strong>Action Agent AI Analysis:</strong>
                                                        <div class="ai-conversation">
                                                            {% for message in action.ai_conversation.messages %}
                                                            <div class="ai-message {{ message.role }}">
                                                                <div class="ai-message-role">{{ message.role|title }}</div>
                                                                <div class="code-block">
                                                                    {% if message.content is string %}
                                                                        {{ message.content }}
                                                                    {% else %}
                                                                        {% for item in message.content %}
                                                                            {% if item.type == 'text' %}
                                                                                {{ item.text }}
                                                                            {% endif %}
                                                                        {% endfor %}
                                                                    {% endif %}
                                                                </div>
                                                            </div>
                                                            {% endfor %}
                                                        </div>
                                                    </div>
                                                    {% endif %}
                                                    
                                                    <!-- Browser Calls -->
                                                    {% if action.browser_calls %}
                                                    <div class="mt-16">
                                                        <strong>Browser Calls:</strong>
                                                        <div class="browser-calls">
                                                            {% for call in action.browser_calls %}
                                                            {{ call.timestamp }}: {{ call.method }}({{ call.args|join(', ') }}){% if call.result %} → {{ call.result }}{% endif %}
                                                            {% endfor %}
                                                        </div>
                                                    </div>
                                                    {% endif %}
                                                    
                                                    <!-- Action Screenshots -->
                                                    {% if action.screenshots %}
                                                    <div class="mt-16">
                                                        <strong>Action Screenshots:</strong>
                                                        <div class="screenshot-container">
                                                            {% if action.screenshots.before %}
                                                            <div class="screenshot-item">
                                                                <a href="file://{{ action.screenshots.before }}" target="_blank">
                                                                    <img src="file://{{ action.screenshots.before }}" alt="Before Action">
                                                                </a>
                                                                <div class="screenshot-label">Before Action</div>
                                                            </div>
                                                            {% endif %}
                                                            {% if action.screenshots.after %}
                                                            <div class="screenshot-item">
                                                                <a href="file://{{ action.screenshots.after }}" target="_blank">
                                                                    <img src="file://{{ action.screenshots.after }}" alt="After Action">
                                                                </a>
                                                                <div class="screenshot-label">After Action</div>
                                                            </div>
                                                            {% endif %}
                                                            {% if action.screenshots.grid_overlay %}
                                                            <div class="screenshot-item">
                                                                <a href="file://{{ action.screenshots.grid_overlay }}" target="_blank">
                                                                    <img src="file://{{ action.screenshots.grid_overlay }}" alt="Grid Overlay">
                                                                </a>
                                                                <div class="screenshot-label">Grid Overlay</div>
                                                            </div>
                                                            {% endif %}
                                                        </div>
                                                    </div>
                                                    {% endif %}
                                                </div>
                                                {% endfor %}
                                            </div>
                                        </div>
                                        {% endif %}
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            Generated at {{ generated_at }} | HAINDY Test Report v2.0
        </div>
    </div>
    
    <script>
        function toggleSection(id) {
            const body = document.getElementById(id + '-body');
            const icon = document.getElementById(id + '-icon');
            
            if (body.classList.contains('expanded')) {
                body.classList.remove('expanded');
                icon.classList.remove('expanded');
            } else {
                body.classList.add('expanded');
                icon.classList.add('expanded');
            }
        }
    </script>
</body>
</html>
"""


class EnhancedReporter:
    """Enhanced HTML report generator with hierarchical structure."""
    
    def __init__(self):
        """Initialize the enhanced reporter."""
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self,
        test_state: TestState,
        output_dir: Path,
        action_storage: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Optional[Path]]:
        """
        Generate an enhanced HTML report.
        
        Args:
            test_state: The test state containing execution results
            output_dir: Directory to save the report
            action_storage: Optional action storage data
            
        Returns:
            Tuple of (report_path, actions_path)
        """
        # Extract data for template
        template_data = self._extract_template_data(test_state, action_storage)
        
        # Generate HTML
        template = Template(ENHANCED_HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Save report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{test_state.test_report.test_plan_id}_{timestamp}.html"
        output_path = output_dir / filename
        output_path.write_text(html_content)
        
        self.logger.info(f"Generated enhanced HTML report: {output_path}")
        
        # Save actions file if provided
        actions_path = None
        if action_storage and action_storage.get("test_cases"):
            actions_filename = f"{test_state.test_report.test_plan_id}_{timestamp}-actions.json"
            actions_path = output_dir / actions_filename
            
            with open(actions_path, 'w') as f:
                json.dump(action_storage, f, indent=2, default=str)
            
            self.logger.info(f"Generated actions file: {actions_path}")
        
        return output_path, actions_path
    
    def _extract_template_data(
        self,
        test_state: TestState,
        action_storage: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract data from test state for template rendering."""
        test_report = test_state.test_report
        
        # Calculate overall metrics
        total_steps = sum(tc.steps_total for tc in test_report.test_cases)
        completed_steps = sum(tc.steps_completed for tc in test_report.test_cases)
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Calculate duration
        if test_report.completed_at and test_report.started_at:
            duration = (test_report.completed_at - test_report.started_at).total_seconds()
        else:
            duration = 0
        
        # Build test cases data
        test_cases_data = []
        for tc_idx, tc in enumerate(test_report.test_cases):
            # Match test case with action storage
            tc_actions = None
            if action_storage:
                for stored_tc in action_storage.get("test_cases", []):
                    if stored_tc.get("test_case_id") == str(tc.case_id):
                        tc_actions = stored_tc
                        break
            
            # Calculate test case duration
            tc_duration = 0
            if tc.completed_at and tc.started_at:
                tc_duration = (tc.completed_at - tc.started_at).total_seconds()
            
            # Build steps data
            steps_data = []
            for step in tc.step_results:
                # Find matching bug report
                bug_report = None
                for bug in test_report.bugs:
                    if bug.step_id == step.step_id:
                        bug_report = bug
                        break
                
                # Find matching actions from storage
                step_actions = None
                test_runner_conversation = None
                
                if tc_actions:
                    for stored_step in tc_actions.get("steps", []):
                        if stored_step.get("step_id") == str(step.step_id):
                            step_actions = stored_step.get("actions", [])
                            test_runner_conversation = stored_step.get("test_runner_interpretation")
                            break
                
                # Process actions
                actions_data = []
                if step_actions:
                    for action in step_actions:
                        # Calculate action duration
                        action_duration = 0
                        if action.get("timestamp_start") and action.get("timestamp_end"):
                            start = datetime.fromisoformat(action["timestamp_start"])
                            end = datetime.fromisoformat(action["timestamp_end"])
                            action_duration = int((end - start).total_seconds() * 1000)
                        
                        # Clean AI conversation to remove image data
                        ai_conversation = action.get("ai_conversation", {}).get("action_agent_execution")
                        if ai_conversation:
                            ai_conversation = self._clean_ai_conversation(ai_conversation)
                        
                        actions_data.append({
                            "action_type": action.get("action_type", "unknown"),
                            "target": action.get("target", ""),
                            "value": action.get("value"),
                            "description": action.get("description", ""),
                            "duration": action_duration,
                            "result": action.get("result"),
                            "ai_conversation": ai_conversation,
                            "browser_calls": action.get("browser_calls", []),
                            "screenshots": action.get("screenshots", {})
                        })
                
                # Calculate step duration
                step_duration = (step.completed_at - step.started_at).total_seconds()
                
                # Handle screenshots with absolute paths
                screenshots = None
                if step.screenshot_before or step.screenshot_after:
                    screenshots = {}
                    if step.screenshot_before:
                        screenshots["before"] = Path(step.screenshot_before).absolute()
                    if step.screenshot_after:
                        screenshots["after"] = Path(step.screenshot_after).absolute()
                
                steps_data.append({
                    "id": str(step.step_id),
                    "number": step.step_number,
                    "status": step.status.value,
                    "action": step.action,
                    "expected_result": step.expected_result,
                    "actual_result": step.actual_result,
                    "error_message": step.error_message,
                    "confidence": step.confidence,
                    "duration": round(step_duration, 2),
                    "screenshots": screenshots,
                    "bug_report": {
                        "description": bug_report.description,
                        "severity": bug_report.severity.value,
                        "error_type": bug_report.error_type,
                        "expected_result": bug_report.expected_result,
                        "actual_result": bug_report.actual_result,
                        "error_details": bug_report.error_details,
                        "reproduction_steps": bug_report.reproduction_steps
                    } if bug_report else None,
                    "test_runner_conversation": test_runner_conversation,
                    "actions": actions_data
                })
            
            # Get test case from test plan for metadata
            test_case_meta = None
            for plan_tc in test_state.test_plan.test_cases:
                if plan_tc.case_id == tc.case_id:
                    test_case_meta = plan_tc
                    break
            
            test_cases_data.append({
                "id": str(tc.case_id),
                "name": tc.name,
                "status": tc.status.value,
                "priority": test_case_meta.priority.value if test_case_meta else "medium",
                "prerequisites": test_case_meta.prerequisites if test_case_meta else [],
                "steps_total": tc.steps_total,
                "steps_completed": tc.steps_completed,
                "steps_failed": tc.steps_failed,
                "error_message": tc.error_message,
                "duration": round(tc_duration, 2),
                "steps": steps_data
            })
        
        # Build template data
        return {
            "test_plan_id": str(test_report.test_plan_id),
            "test_plan_name": test_report.test_plan_name,
            "test_plan_description": test_state.test_plan.description if test_state.test_plan else "",
            "test_plan_created": test_state.test_plan.created_at.isoformat() if test_state.test_plan and hasattr(test_state.test_plan, 'created_at') else "N/A",
            "test_started": test_report.started_at.isoformat(),
            "test_completed": test_report.completed_at.isoformat() if test_report.completed_at else "In Progress",
            "environment": json.dumps(test_report.environment, indent=2) if test_report.environment else "{}",
            "overall_status": test_report.status.value,
            "duration": round(duration, 2),
            "success_rate": round(success_rate, 1),
            "test_cases": test_cases_data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _clean_ai_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Clean AI conversation by removing base64 image data."""
        if not conversation or "messages" not in conversation:
            return conversation
        
        cleaned_conversation = {"messages": []}
        
        for message in conversation.get("messages", []):
            cleaned_message = {
                "role": message.get("role", ""),
                "content": message.get("content", "")
            }
            
            # If content is a list (multimodal), process each item
            if isinstance(cleaned_message["content"], list):
                cleaned_content = []
                for item in cleaned_message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            cleaned_content.append(item)
                        elif item.get("type") == "image_url":
                            # Replace image data with a placeholder
                            cleaned_content.append({
                                "type": "text",
                                "text": "[IMAGE: Screenshot provided to AI]"
                            })
                    else:
                        cleaned_content.append(item)
                cleaned_message["content"] = cleaned_content
            
            cleaned_conversation["messages"].append(cleaned_message)
        
        return cleaned_conversation