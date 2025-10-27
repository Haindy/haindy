"""Test Planner Agent implementation.

Analyzes requirements/PRDs and creates structured test plans.
"""

import inspect
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from src.agents.base_agent import BaseAgent
from src.agents.formatters import TestPlanFormatter
from src.config.agent_prompts import TEST_PLANNER_SYSTEM_PROMPT
from src.core.types import ActionInstruction, ActionType, TestCase, TestCasePriority, TestPlan, TestStep, StepIntent
from src.models.openai_client import ResponseStreamObserver
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class TestPlannerAgent(BaseAgent):
    """
    AI agent that analyzes requirements and creates structured test plans.
    
    This agent takes high-level requirements or PRDs and transforms them
    into detailed, executable test plans with clear steps and expected outcomes.
    """
    
    def __init__(self, name: str = "TestPlanner", **kwargs):
        """Initialize the Test Planner Agent."""
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_PLANNER_SYSTEM_PROMPT
    
    async def _get_completion(
        self,
        messages: List[Dict[str, str]],
        stream_observer: Optional[ResponseStreamObserver] = None,
        **kwargs,
    ) -> Dict:
        """Get completion from OpenAI."""
        response = await self.call_openai(
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            response_format=kwargs.get("response_format"),
            stream=True,
            stream_observer=stream_observer,
        )

        if stream_observer is not None:
            usage = response.get("usage")
            if isinstance(usage, dict):
                normalized_usage = {
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                }
                maybe = stream_observer.on_usage_total(normalized_usage)
                if inspect.isawaitable(maybe):
                    await maybe

        return response

    async def create_test_plan(
        self,
        requirements: str,
        context: Optional[Dict] = None,
        curated_scope: Optional[str] = None,
        ambiguous_points: Optional[List[str]] = None,
        stream_observer: Optional[ResponseStreamObserver] = None,
    ) -> TestPlan:
        """
        Create a structured test plan from requirements.

        Args:
            requirements: High-level requirements, user story, or PRD
            context: Optional context about the application or domain
            stream_observer: Optional observer for streaming updates
            
        Returns:
            TestPlan: Structured test plan with steps and expected outcomes
        """
        logger.info("Creating test plan from requirements", extra={
            "requirements_length": len(requirements),
            "has_context": context is not None
        })
        
        # Build the user message
        user_message = self._build_requirements_message(
            requirements=requirements,
            context=context,
            curated_scope=curated_scope,
            ambiguous_points=ambiguous_points,
        )
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Get test plan from AI
        response = await self._get_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent planning
            stream_observer=stream_observer,
        )

        # Parse and validate the response
        test_plan = self._parse_test_plan_response(response)

        # Save test plan permanently
        self._save_test_plan(test_plan)

        # Calculate total steps
        total_steps = sum(len(tc.steps) for tc in test_plan.test_cases)

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logger.info("Test plan created successfully", extra={
            "plan_name": test_plan.name,
            "num_test_cases": len(test_plan.test_cases),
            "total_steps": total_steps,
            "has_tags": bool(test_plan.tags),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })

        return test_plan
    
    def _build_requirements_message(
        self,
        requirements: str,
        context: Optional[Dict] = None,
        curated_scope: Optional[str] = None,
        ambiguous_points: Optional[List[str]] = None,
    ) -> str:
        """Build the user message with requirements, curated scope, and context."""
        message_parts: List[str] = []

        if curated_scope:
            message_parts.append(
                "Use the curated testing scope below. Do NOT introduce functionality beyond it."
            )
            message_parts.append("")
            message_parts.append(curated_scope)

            normalized_ambiguities = [
                item.strip() for item in (ambiguous_points or []) if item.strip()
            ]
            if normalized_ambiguities:
                message_parts.append("")
                message_parts.append(
                    "Ambiguous points identified by scope triage "
                    "(surface them in plan notes; do not resolve automatically):"
                )
                for point in normalized_ambiguities:
                    message_parts.append(f"- {point}")

            message_parts.append("")
            message_parts.append(
                "Full requirements package is included below for reference only."
            )
            message_parts.append(
                "Never infer additional features that are not part of the curated scope."
            )
        else:
            message_parts.append(
                "Please create a test plan for the following requirements:"
            )

        message_parts.append("")
        message_parts.append(requirements)

        if context:
            message_parts.append("")
            message_parts.append("Additional Context:")
            for key, value in context.items():
                message_parts.append(f"- {key}: {value}")

        message_parts.append(
            """\n\nProvide the test plan in the following JSON format:
{
    "name": "Test plan name",
    "description": "Overall description of what is being tested",
    "requirements_source": "Source of requirements (e.g., 'PRD v1.2', 'User Story #123', URL)",
    "test_cases": [
        {
            "test_id": "TC001",
            "name": "Test case name (e.g., 'Happy Path Login')",
            "description": "Detailed description of this specific test scenario",
            "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
            "steps": [
                {
                    "step_number": 1,
                    "action": "Clear action description (e.g., 'Navigate to login page')",
                    "expected_result": "What should happen after this action",
                    "intent": "validation",
                    "dependencies": [],
                    "optional": false
                }
            ],
            "postconditions": ["Expected state after test completion"],
            "tags": []
        }
    ],
    "tags": [],
    "estimated_duration_seconds": 300
}

IMPORTANT: 
- Create multiple test cases to cover different scenarios
- Each test case should test ONE specific flow or scenario
- Ignore non-functional or infrastructure rules that are not user-facing functionality
- Include both positive (happy path) and realistic negative (user-observable) test cases only
- Assign test case IDs sequentially without gaps (TC001, TC002, ...)
- Tag each step with an 'intent': use 'setup' for preparatory actions with minimal verification, 'validation' for standard checks, and 'group_assert' when bundling multiple related assertions into one step
- Do NOT assign or output priority fields
- Respond with well-formed json that matches the structure above."""
        )

        return "\n".join(message_parts)
    
    def _parse_test_plan_response(self, response: Dict) -> TestPlan:
        """Parse the AI response into a TestPlan object."""
        try:
            # Extract the content - it's already a dict when using JSON response format
            content = response.get("content", {})
            import json
            
            # If content is a string, parse it. If it's already a dict, use it directly
            if isinstance(content, str):
                plan_data = json.loads(content)
            else:
                plan_data = content
            
            # Parse test cases
            test_cases = []
            for case_data in plan_data.get("test_cases", []):
                # Parse priority
                priority_map = {
                    "critical": TestCasePriority.CRITICAL,
                    "high": TestCasePriority.HIGH,
                    "medium": TestCasePriority.MEDIUM,
                    "low": TestCasePriority.LOW
                }
                priority = priority_map.get(
                    case_data.get("priority", "medium").lower(),
                    TestCasePriority.MEDIUM
                )
                
                # Parse steps for this test case
                steps = []
                for step_data in case_data.get("steps", []):
                    intent_value = step_data.get("intent")
                    try:
                        intent = StepIntent(intent_value) if intent_value else StepIntent.VALIDATION
                    except ValueError:
                        intent = StepIntent.VALIDATION
                    
                    step = TestStep(
                        step_number=step_data["step_number"],
                        description=step_data.get("description", step_data.get("action", "")),
                        action=step_data.get("action", ""),
                        expected_result=step_data.get("expected_result", ""),
                        dependencies=step_data.get("dependencies", []),
                        optional=step_data.get("optional", False),
                        intent=intent,
                        max_retries=step_data.get("max_retries", 3)
                    )
                    steps.append(step)
                
                # Create test case
                test_case = TestCase(
                    test_id=case_data.get("test_id", f"TC{len(test_cases)+1:03d}"),
                    name=case_data.get("name", "Unnamed Test Case"),
                    description=case_data.get("description", ""),
                    priority=priority,
                    prerequisites=case_data.get("prerequisites", []),
                    steps=steps,
                    postconditions=case_data.get("postconditions", []),
                    tags=case_data.get("tags", [])
                )
                test_cases.append(test_case)
            
            # Create test plan
            test_plan = TestPlan(
                name=plan_data.get("name", "Unnamed Test Plan"),
                description=plan_data.get("description", ""),
                requirements_source=plan_data.get("requirements_source", "Unknown"),
                test_cases=test_cases,
                tags=plan_data.get("tags", []),
                estimated_duration_seconds=plan_data.get("estimated_duration_seconds")
            )
            
            return test_plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse test plan response", extra={
                "error": str(e),
                "response": response
            })
            raise ValueError(f"Failed to parse test plan response: {e}")
    
    def _save_test_plan(self, test_plan: TestPlan) -> None:
        """Save test plan to permanent storage."""
        # Create base directory for all generated test plans
        base_dir = Path("generated_test_plans")
        base_dir.mkdir(exist_ok=True)
        
        # Create directory for this specific test plan
        plan_dir = base_dir / str(test_plan.plan_id)
        plan_dir.mkdir(exist_ok=True)
        
        # Use formatter to generate both formats
        formatter = TestPlanFormatter()
        
        # Save as JSON
        json_path = plan_dir / f"test_plan_{test_plan.plan_id}.json"
        with open(json_path, "w") as f:
            f.write(formatter.to_json(test_plan))
        
        # Save as Markdown
        md_path = plan_dir / f"test_plan_{test_plan.plan_id}.md"
        with open(md_path, "w") as f:
            f.write(formatter.to_markdown(test_plan))
        
        logger.info("Test plan saved permanently", extra={
            "plan_id": str(test_plan.plan_id),
            "plan_dir": str(plan_dir),
            "json_path": str(json_path),
            "md_path": str(md_path)
        })
    
    async def refine_test_plan(self, test_plan: TestPlan, feedback: str) -> TestPlan:
        """
        Refine an existing test plan based on feedback.
        
        Args:
            test_plan: Existing test plan to refine
            feedback: Feedback or additional requirements
            
        Returns:
            TestPlan: Refined test plan
        """
        logger.info("Refining test plan based on feedback", extra={
            "plan_name": test_plan.name,
            "feedback_length": len(feedback)
        })
        
        # Build refinement message
        current_plan = self._serialize_test_plan(test_plan)
        user_message = f"""Current test plan:
{current_plan}

Feedback/Additional Requirements:
{feedback}

Please provide an updated test plan that addresses the feedback while maintaining the same JSON format."""
        
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Get refined plan from AI
        response = await self._get_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse the refined plan
        refined_plan = self._parse_test_plan_response(response)
        
        # Calculate total steps for new structure
        total_steps = sum(len(tc.steps) for tc in refined_plan.test_cases)
        
        logger.info("Test plan refined successfully", extra={
            "plan_name": refined_plan.name,
            "num_test_cases": len(refined_plan.test_cases),
            "total_steps": total_steps
        })
        
        return refined_plan
    
    def _serialize_test_plan(self, test_plan: TestPlan) -> str:
        """Serialize a test plan to a readable string format."""
        formatter = TestPlanFormatter()
        return formatter.to_markdown(test_plan)
    
    async def extract_test_scenarios(self, requirements: str) -> List[Dict[str, str]]:
        """
        Extract multiple test scenarios from complex requirements.
        
        Args:
            requirements: Complex requirements potentially containing multiple scenarios
            
        Returns:
            List of test scenarios with names and descriptions
        """
        logger.info("Extracting test scenarios from requirements")
        
        prompt = """Given the following requirements, identify and list all distinct test scenarios that should be tested separately.

For each scenario, provide:
1. A clear, concise name
2. A brief description
3. Priority (high/medium/low)
4. Type (functional/edge_case/error_handling)

Output as JSON object with a "scenarios" array."""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": requirements}
        ]
        
        response = await self._get_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        try:
            import json
            content = response.get("content", "{}")
            scenarios_data = json.loads(content)
            scenarios = scenarios_data.get("scenarios", [])
            
            logger.info(f"Extracted {len(scenarios)} test scenarios")
            return scenarios
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to extract test scenarios", extra={"error": str(e)})
            return []
