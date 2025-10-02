"""
Formatters for test plans - JSON and Markdown output generation.
"""

import json
from datetime import datetime
from typing import Any, Dict

from src.core.types import TestCase, TestPlan, TestStep


class TestPlanFormatter:
    """Handles formatting of test plans into various output formats."""
    
    @staticmethod
    def to_json(test_plan: TestPlan, pretty: bool = True) -> str:
        """
        Convert a TestPlan to JSON string.
        
        Args:
            test_plan: The test plan to convert
            pretty: Whether to pretty-print the JSON
            
        Returns:
            JSON string representation of the test plan
        """
        # Convert to dict and handle UUID/datetime serialization
        plan_dict = test_plan.model_dump()
        
        # Convert UUIDs and datetimes to strings
        plan_dict["plan_id"] = str(plan_dict["plan_id"])
        plan_dict["created_at"] = plan_dict["created_at"].isoformat()
        
        # Handle test cases
        for case in plan_dict.get("test_cases", []):
            case["case_id"] = str(case["case_id"])
            # Handle steps within each case
            for step in case.get("steps", []):
                step["step_id"] = str(step["step_id"])
                # Remove deprecated fields
                if "action_instruction" in step:
                    del step["action_instruction"]
        
        # Remove deprecated fields
        if "steps" in plan_dict:
            del plan_dict["steps"]
        
        if pretty:
            return json.dumps(plan_dict, indent=2, ensure_ascii=False)
        else:
            return json.dumps(plan_dict, ensure_ascii=False)
    
    @staticmethod
    def to_markdown(test_plan: TestPlan) -> str:
        """
        Convert a TestPlan to Markdown format.
        
        Args:
            test_plan: The test plan to convert
            
        Returns:
            Markdown string representation of the test plan
        """
        lines = []
        
        # Header
        lines.append(f"# Test Plan: {test_plan.name}")
        lines.append("")
        
        # Metadata
        lines.append(f"**Description**: {test_plan.description}")
        lines.append(f"**Requirements Source**: {test_plan.requirements_source}")
        if test_plan.tags:
            lines.append(f"**Tags**: {', '.join(test_plan.tags)}")
        if test_plan.estimated_duration_seconds:
            duration_min = test_plan.estimated_duration_seconds // 60
            lines.append(f"**Estimated Duration**: {duration_min} minutes")
        lines.append("")
        
        # Summary
        total_steps = sum(len(tc.steps) for tc in test_plan.test_cases)
        lines.append(f"## Summary")
        lines.append(f"- **Total Test Cases**: {len(test_plan.test_cases)}")
        lines.append(f"- **Total Test Steps**: {total_steps}")
        
        # Priority breakdown
        priority_counts = {}
        for tc in test_plan.test_cases:
            priority_counts[tc.priority.value] = priority_counts.get(tc.priority.value, 0) + 1
        
        if priority_counts:
            lines.append(f"- **Priority Distribution**:")
            for priority in ["critical", "high", "medium", "low"]:
                if priority in priority_counts:
                    lines.append(f"  - {priority.capitalize()}: {priority_counts[priority]}")
        lines.append("")
        
        # Test Cases
        lines.append("## Test Cases")
        lines.append("")
        
        for tc in test_plan.test_cases:
            # Test case header
            lines.append(f"### {tc.test_id}: {tc.name}")
            lines.append("")
            lines.append(f"**Priority**: {tc.priority.value.capitalize()}")
            lines.append(f"**Description**: {tc.description}")
            
            # Prerequisites
            if tc.prerequisites:
                lines.append("")
                lines.append("#### Prerequisites")
                for prereq in tc.prerequisites:
                    lines.append(f"- {prereq}")
            
            # Test steps
            lines.append("")
            lines.append("#### Test Steps")
            
            for step in tc.steps:
                lines.append(f"{step.step_number}. **{step.action}**")
                lines.append(f"   - _Expected Result_: {step.expected_result}")
                
                if step.dependencies:
                    deps_str = ", ".join(str(d) for d in step.dependencies)
                    lines.append(f"   - _Depends on_: Step(s) {deps_str}")
                
                if step.optional:
                    lines.append(f"   - _Optional_: Yes")
            
            # Postconditions
            if tc.postconditions:
                lines.append("")
                lines.append("#### Postconditions")
                for postcond in tc.postconditions:
                    lines.append(f"- {postcond}")
            
            # Tags
            if tc.tags:
                lines.append("")
                lines.append(f"**Tags**: {', '.join(tc.tags)}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Remove trailing separator
        if lines and lines[-1] == "---":
            lines.pop()
            if lines and lines[-1] == "":
                lines.pop()
        
        return "\n".join(lines)
    
    @staticmethod
    def from_json(json_str: str) -> TestPlan:
        """
        Parse a TestPlan from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            TestPlan object
        """
        data = json.loads(json_str)
        
        # Convert string UUIDs back to UUID objects would be handled by Pydantic
        # Convert ISO datetime strings would be handled by Pydantic
        
        return TestPlan(**data)
    
    @staticmethod
    def to_dict(test_plan: TestPlan) -> Dict[str, Any]:
        """
        Convert TestPlan to a clean dictionary format for AI consumption.
        
        Args:
            test_plan: The test plan to convert
            
        Returns:
            Dictionary representation suitable for AI agents
        """
        return {
            "name": test_plan.name,
            "description": test_plan.description,
            "requirements_source": test_plan.requirements_source,
            "created_at": test_plan.created_at.isoformat(),
            "test_cases": [
                {
                    "id": tc.test_id,
                    "name": tc.name,
                    "description": tc.description,
                    "priority": tc.priority.value,
                    "prerequisites": tc.prerequisites,
                    "steps": [
                        {
                            "step_number": step.step_number,
                            "action": step.action,
                            "expected_result": step.expected_result,
                            "dependencies": step.dependencies,
                            "optional": step.optional
                        }
                        for step in tc.steps
                    ],
                    "postconditions": tc.postconditions,
                    "tags": tc.tags
                }
                for tc in test_plan.test_cases
            ],
            "tags": test_plan.tags,
            "estimated_duration_seconds": test_plan.estimated_duration_seconds
        }
