"""
Unit tests for Test Plan Formatters.
"""

import json
from datetime import datetime, timezone

import pytest

from src.agents.formatters import TestPlanFormatter
from src.core.types import TestCase, TestCasePriority, TestPlan, TestStep


class TestTestPlanFormatter:
    """Tests for TestPlanFormatter class."""
    
    @pytest.fixture
    def sample_test_plan(self):
        """Create a sample test plan for testing."""
        return TestPlan(
            name="Sample Test Plan",
            description="A sample test plan for testing formatters",
            requirements_source="User Story #123",
            test_cases=[
                TestCase(
                    test_id="TC001",
                    name="Happy Path Test",
                    description="Test the happy path scenario",
                    priority=TestCasePriority.CRITICAL,
                    prerequisites=["System is running", "User account exists"],
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Navigate to home page",
                            expected_result="Home page is displayed",
                            description="Go to the home page"
                        ),
                        TestStep(
                            step_number=2,
                            action="Click on login button",
                            expected_result="Login form is displayed",
                            description="Open login form",
                            dependencies=[1]
                        )
                    ],
                    postconditions=["User is logged in"],
                    tags=["smoke", "critical"]
                ),
                TestCase(
                    test_id="TC002",
                    name="Error Handling Test",
                    description="Test error scenarios",
                    priority=TestCasePriority.HIGH,
                    prerequisites=["System is running"],
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Enter invalid data",
                            expected_result="Error message is displayed",
                            description="Test validation"
                        )
                    ],
                    postconditions=["Error is handled gracefully"],
                    tags=["negative", "validation"]
                )
            ],
            tags=["functional", "ui"],
            estimated_duration_seconds=300
        )
    
    def test_to_json_pretty(self, sample_test_plan):
        """Test converting test plan to pretty JSON."""
        json_output = TestPlanFormatter.to_json(sample_test_plan, pretty=True)
        
        # Parse JSON to verify it's valid
        parsed = json.loads(json_output)
        
        assert parsed["name"] == "Sample Test Plan"
        assert parsed["description"] == "A sample test plan for testing formatters"
        assert parsed["requirements_source"] == "User Story #123"
        assert len(parsed["test_cases"]) == 2
        assert parsed["tags"] == ["functional", "ui"]
        assert parsed["estimated_duration_seconds"] == 300
        
        # Check first test case
        tc1 = parsed["test_cases"][0]
        assert tc1["test_id"] == "TC001"
        assert tc1["name"] == "Happy Path Test"
        assert tc1["priority"] == "critical"
        assert len(tc1["steps"]) == 2
        
        # Check formatting (pretty print should have newlines and indentation)
        assert "\n" in json_output
        assert "    " in json_output
    
    def test_to_json_compact(self, sample_test_plan):
        """Test converting test plan to compact JSON."""
        json_output = TestPlanFormatter.to_json(sample_test_plan, pretty=False)
        
        # Parse JSON to verify it's valid
        parsed = json.loads(json_output)
        
        assert parsed["name"] == "Sample Test Plan"
        # Compact JSON should not have unnecessary whitespace
        assert "\n" not in json_output
        # JSON will have spaces after colons and commas, but should be minimal
        # With the hierarchical structure, we have more fields so more spaces
        assert json_output.count(" ") < 200  # Should have minimal spaces for structure
    
    def test_to_markdown(self, sample_test_plan):
        """Test converting test plan to Markdown."""
        markdown_output = TestPlanFormatter.to_markdown(sample_test_plan)
        
        # Check header
        assert "# Test Plan: Sample Test Plan" in markdown_output
        
        # Check metadata
        assert "**Description**: A sample test plan for testing formatters" in markdown_output
        assert "**Requirements Source**: User Story #123" in markdown_output
        assert "**Tags**: functional, ui" in markdown_output
        assert "**Estimated Duration**: 5 minutes" in markdown_output
        
        # Check summary
        assert "## Summary" in markdown_output
        assert "**Total Test Cases**: 2" in markdown_output
        assert "**Total Test Steps**: 3" in markdown_output
        assert "Critical: 1" in markdown_output
        assert "High: 1" in markdown_output
        
        # Check test cases
        assert "### TC001: Happy Path Test" in markdown_output
        assert "**Priority**: Critical" in markdown_output
        assert "#### Prerequisites" in markdown_output
        assert "- System is running" in markdown_output
        assert "- User account exists" in markdown_output
        
        # Check steps
        assert "#### Test Steps" in markdown_output
        assert "1. **Navigate to home page**" in markdown_output
        assert "_Expected Result_: Home page is displayed" in markdown_output
        assert "2. **Click on login button**" in markdown_output
        assert "_Depends on_: Step(s) 1" in markdown_output
        
        # Check postconditions
        assert "#### Postconditions" in markdown_output
        assert "- User is logged in" in markdown_output
        
        # Check second test case
        assert "### TC002: Error Handling Test" in markdown_output
        assert "**Priority**: High" in markdown_output
    
    def test_from_json(self):
        """Test parsing test plan from JSON."""
        json_str = """{
            "plan_id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "Test Plan",
            "description": "Description",
            "requirements_source": "Requirements",
            "test_cases": [
                {
                    "case_id": "123e4567-e89b-12d3-a456-426614174001",
                    "test_id": "TC001",
                    "name": "Test Case",
                    "description": "Test description",
                    "priority": "high",
                    "prerequisites": [],
                    "steps": [
                        {
                            "step_id": "123e4567-e89b-12d3-a456-426614174002",
                            "step_number": 1,
                            "action": "Do something",
                            "expected_result": "Something happens",
                            "description": "Step description",
                            "dependencies": [],
                            "optional": false,
                            "max_retries": 3
                        }
                    ],
                    "postconditions": [],
                    "tags": []
                }
            ],
            "created_at": "2025-01-10T10:00:00Z",
            "created_by": "HAINDY Test Planner",
            "tags": [],
            "estimated_duration_seconds": null
        }"""
        
        test_plan = TestPlanFormatter.from_json(json_str)
        
        assert test_plan.name == "Test Plan"
        assert test_plan.description == "Description"
        assert test_plan.requirements_source == "Requirements"
        assert len(test_plan.test_cases) == 1
        assert test_plan.test_cases[0].test_id == "TC001"
    
    def test_to_dict(self, sample_test_plan):
        """Test converting test plan to dictionary."""
        plan_dict = TestPlanFormatter.to_dict(sample_test_plan)
        
        assert plan_dict["name"] == "Sample Test Plan"
        assert plan_dict["description"] == "A sample test plan for testing formatters"
        assert plan_dict["requirements_source"] == "User Story #123"
        assert len(plan_dict["test_cases"]) == 2
        
        # Check test case structure
        tc1 = plan_dict["test_cases"][0]
        assert tc1["id"] == "TC001"
        assert tc1["name"] == "Happy Path Test"
        assert tc1["priority"] == "critical"
        assert len(tc1["steps"]) == 2
        
        # Check step structure
        step1 = tc1["steps"][0]
        assert step1["step_number"] == 1
        assert step1["action"] == "Navigate to home page"
        assert step1["expected_result"] == "Home page is displayed"
        assert step1["dependencies"] == []
    
    def test_markdown_no_optional_fields(self):
        """Test Markdown generation with minimal test plan."""
        minimal_plan = TestPlan(
            name="Minimal Plan",
            description="A minimal test plan",
            requirements_source="Basic requirements",
            test_cases=[
                TestCase(
                    test_id="TC001",
                    name="Basic Test",
                    description="A basic test",
                    priority=TestCasePriority.MEDIUM,
                    prerequisites=[],
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Do something",
                            expected_result="It works",
                            description="Basic step"
                        )
                    ],
                    postconditions=[],
                    tags=[]
                )
            ],
            tags=[]
        )
        
        markdown_output = TestPlanFormatter.to_markdown(minimal_plan)
        
        # Should still have basic structure
        assert "# Test Plan: Minimal Plan" in markdown_output
        assert "### TC001: Basic Test" in markdown_output
        
        # Should not have optional sections
        assert "**Tags**:" not in markdown_output
        assert "#### Prerequisites" not in markdown_output
        assert "#### Postconditions" not in markdown_output
        assert "_Optional_:" not in markdown_output