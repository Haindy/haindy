"""Test Planner Agent implementation.

Analyzes requirements/PRDs and creates structured test plans.
"""

import inspect
import re
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.formatters import TestPlanFormatter
from src.config.agent_prompts import TEST_PLANNER_SYSTEM_PROMPT
from src.core.types import (
    StepIntent,
    TestCase,
    TestCasePriority,
    TestPlan,
    TestStep,
)
from src.models.openai_client import ResponseStreamObserver
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class TestPlannerAgent(BaseAgent):
    """
    AI agent that analyzes requirements and creates structured test plans.

    This agent takes high-level requirements or PRDs and transforms them
    into detailed, executable test plans with clear steps and expected outcomes.
    """

    __test__ = False

    def __init__(self, name: str = "TestPlanner", **kwargs: Any) -> None:
        """Initialize the Test Planner Agent."""
        super().__init__(name=name, **kwargs)
        self.system_prompt = TEST_PLANNER_SYSTEM_PROMPT

    async def _get_completion(
        self,
        messages: list[dict[str, str]],
        stream_observer: ResponseStreamObserver | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        context: dict[str, Any] | None = None,
        curated_scope: str | None = None,
        ambiguous_points: list[str] | None = None,
        stream_observer: ResponseStreamObserver | None = None,
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
        logger.info(
            "Creating test plan from requirements",
            extra={
                "requirements_length": len(requirements),
                "has_context": context is not None,
            },
        )

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
            {"role": "user", "content": user_message},
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

        logger.info(
            "Test plan created successfully",
            extra={
                "plan_name": test_plan.name,
                "num_test_cases": len(test_plan.test_cases),
                "total_steps": total_steps,
                "has_tags": bool(test_plan.tags),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        return test_plan

    def _build_requirements_message(
        self,
        requirements: str,
        context: dict[str, Any] | None = None,
        curated_scope: str | None = None,
        ambiguous_points: list[str] | None = None,
    ) -> str:
        """Build the user message with requirements, curated scope, and context."""
        message_parts: list[str] = []

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
            "setup_steps": [
                {
                    "step_number": 1,
                    "action": "Action to reach the starting state for this test case (e.g., 'Sign in with existing user credentials')",
                    "expected_result": "What the screen should look like after this setup action",
                    "intent": "setup",
                    "dependencies": [],
                    "optional": false
                }
            ],
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
            "cleanup_steps": [
                {
                    "step_number": 1,
                    "action": "Action to clean up after the test case (e.g., 'Log out of the account', 'Clear app data')",
                    "expected_result": "What the screen should look like after cleanup",
                    "intent": "setup",
                    "dependencies": [],
                    "optional": true
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
- setup_steps: actions executed BEFORE the main steps to reach the test case's starting state. Do NOT duplicate actions that are part of the main steps. Choose one of three patterns based on the test case preconditions and the state left by the previous test case:
  1. EMPTY (continuation): the test case starts exactly where the previous one ended. No setup needed.
  2. SOFT RESET (navigation/sign-out): the test case needs a different screen or sign-in state, but session data can remain. Emit one or more steps to navigate or sign out.
  3. HARD RESET (clean app state): the test case requires a completely clean app state — no active session, no cached data (e.g., preconditions say "app freshly launched", "user not signed in", or "new unregistered user"). Describe this as a step with action "Reset the app to a clean state with no active user session" and expected result "App launches to the initial signed-out screen". Do not write technical commands; the executor will handle the mechanism.
- setup_steps atomicity: each setup_step must be as atomic as a regular step — one action, one expected result. Never collapse a multi-action flow (e.g., "register a new user, pick a role, verify email, reach screen X") into a single setup step. When a test case's preconditions require completing a flow that was already defined in an earlier test case, copy each action from that flow individually as separate setup_steps in the same order. The executor performs each setup_step as a single, unambiguous interaction.
- cleanup_steps: actions executed AFTER the main steps to restore state for the next test case. Use for things like logging out, clearing app data/cache, or navigating back to a known screen. Leave empty if no cleanup is needed. Cleanup steps are always optional and non-blocking -- a failed cleanup never fails the test case.
- Respond with well-formed json that matches the structure above."""
        )

        return "\n".join(message_parts)

    def _parse_test_plan_response(self, response: dict[str, Any]) -> TestPlan:
        """Parse the AI response into a TestPlan object."""
        try:
            # Extract the content - it's already a dict when using JSON response format
            content = response.get("content", {})
            import json

            # If content is a string, parse it. If it's already a dict, use it directly
            if isinstance(content, str):
                plan_data: dict[str, Any] = json.loads(content)
            elif isinstance(content, dict):
                plan_data = content
            else:
                raise ValueError("Test plan response content must be a JSON object")

            # Parse test cases
            test_cases: list[TestCase] = []
            for case_data in plan_data.get("test_cases", []):
                # Parse priority
                priority_map = {
                    "critical": TestCasePriority.CRITICAL,
                    "high": TestCasePriority.HIGH,
                    "medium": TestCasePriority.MEDIUM,
                    "low": TestCasePriority.LOW,
                }
                priority = priority_map.get(
                    str(case_data.get("priority", "medium")).lower(),
                    TestCasePriority.MEDIUM,
                )

                # Parse steps for this test case
                steps: list[TestStep] = []
                for default_step_number, step_data in enumerate(
                    case_data.get("steps", []),
                    start=1,
                ):
                    intent_value = step_data.get("intent")
                    try:
                        intent = (
                            StepIntent(intent_value)
                            if intent_value
                            else StepIntent.VALIDATION
                        )
                    except ValueError:
                        intent = StepIntent.VALIDATION

                    step_number = self._parse_step_number(
                        step_data.get("step_number"),
                        default=default_step_number,
                    )
                    dependencies = self._parse_step_dependencies(
                        step_data.get("dependencies")
                    )
                    # Prevent impossible self-dependencies from malformed model output.
                    dependencies = [
                        dep_step_number
                        for dep_step_number in dependencies
                        if dep_step_number != step_number
                    ]

                    step = TestStep(
                        step_number=step_number,
                        description=step_data.get(
                            "description", step_data.get("action", "")
                        ),
                        action=step_data.get("action", ""),
                        expected_result=step_data.get("expected_result", ""),
                        dependencies=dependencies,
                        optional=step_data.get("optional", False),
                        intent=intent,
                        max_retries=step_data.get("max_retries", 3),
                    )
                    steps.append(step)

                # Parse setup_steps for this test case
                setup_steps: list[TestStep] = []
                for default_setup_num, setup_data in enumerate(
                    case_data.get("setup_steps", []),
                    start=1,
                ):
                    setup_intent_value = setup_data.get("intent")
                    try:
                        setup_intent = (
                            StepIntent(setup_intent_value)
                            if setup_intent_value
                            else StepIntent.SETUP
                        )
                    except ValueError:
                        setup_intent = StepIntent.SETUP

                    setup_step_number = self._parse_step_number(
                        setup_data.get("step_number"),
                        default=default_setup_num,
                    )

                    setup_step = TestStep(
                        step_number=setup_step_number,
                        description=setup_data.get(
                            "description", setup_data.get("action", "")
                        ),
                        action=setup_data.get("action", ""),
                        expected_result=setup_data.get("expected_result", ""),
                        dependencies=self._parse_step_dependencies(
                            setup_data.get("dependencies")
                        ),
                        optional=setup_data.get("optional", False),
                        intent=setup_intent,
                        max_retries=setup_data.get("max_retries", 3),
                    )
                    setup_steps.append(setup_step)

                # Parse cleanup_steps for this test case
                cleanup_steps: list[TestStep] = []
                for default_cleanup_num, cleanup_data in enumerate(
                    case_data.get("cleanup_steps", []),
                    start=1,
                ):
                    cleanup_intent_value = cleanup_data.get("intent")
                    try:
                        cleanup_intent = (
                            StepIntent(cleanup_intent_value)
                            if cleanup_intent_value
                            else StepIntent.SETUP
                        )
                    except ValueError:
                        cleanup_intent = StepIntent.SETUP

                    cleanup_step_number = self._parse_step_number(
                        cleanup_data.get("step_number"),
                        default=default_cleanup_num,
                    )

                    cleanup_step = TestStep(
                        step_number=cleanup_step_number,
                        description=cleanup_data.get(
                            "description", cleanup_data.get("action", "")
                        ),
                        action=cleanup_data.get("action", ""),
                        expected_result=cleanup_data.get("expected_result", ""),
                        dependencies=self._parse_step_dependencies(
                            cleanup_data.get("dependencies")
                        ),
                        optional=cleanup_data.get("optional", True),
                        intent=cleanup_intent,
                        max_retries=cleanup_data.get("max_retries", 3),
                    )
                    cleanup_steps.append(cleanup_step)

                # Create test case
                test_case = TestCase(
                    test_id=case_data.get("test_id", f"TC{len(test_cases) + 1:03d}"),
                    name=case_data.get("name", "Unnamed Test Case"),
                    description=case_data.get("description", ""),
                    priority=priority,
                    prerequisites=case_data.get("prerequisites", []),
                    setup_steps=setup_steps,
                    steps=steps,
                    cleanup_steps=cleanup_steps,
                    postconditions=case_data.get("postconditions", []),
                    tags=case_data.get("tags", []),
                )
                test_cases.append(test_case)

            # Create test plan
            test_plan = TestPlan(
                name=plan_data.get("name", "Unnamed Test Plan"),
                description=plan_data.get("description", ""),
                requirements_source=plan_data.get("requirements_source", "Unknown"),
                test_cases=test_cases,
                tags=plan_data.get("tags", []),
                estimated_duration_seconds=plan_data.get("estimated_duration_seconds"),
            )

            return test_plan

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                "Failed to parse test plan response",
                extra={"error": str(e), "response": response},
            )
            raise ValueError(f"Failed to parse test plan response: {e}") from e

    @staticmethod
    def _parse_step_number(value: object, default: int) -> int:
        """Normalize LLM-provided step number values to a positive integer."""
        parsed_numbers = TestPlannerAgent._extract_step_numbers(value)
        return parsed_numbers[0] if parsed_numbers else default

    @staticmethod
    def _parse_step_dependencies(value: object) -> list[int]:
        """Normalize LLM-provided dependency values to ordered, unique step numbers."""
        if value is None:
            return []
        return TestPlannerAgent._extract_step_numbers(value)

    @staticmethod
    def _extract_step_numbers(value: object) -> list[int]:
        """Extract positive step numbers from loose values like 'Step 1' or [1, 'step 2']."""
        parsed_numbers: list[int] = []
        raw_values = value if isinstance(value, (list, tuple, set)) else [value]

        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                continue

            if isinstance(raw_value, int):
                if raw_value > 0:
                    parsed_numbers.append(raw_value)
                continue

            if isinstance(raw_value, float):
                if raw_value.is_integer() and raw_value > 0:
                    parsed_numbers.append(int(raw_value))
                continue

            if isinstance(raw_value, str):
                parsed_numbers.extend(
                    int(match) for match in re.findall(r"\d+", raw_value)
                )

        unique_numbers: list[int] = []
        seen_numbers: set[int] = set()
        for number in parsed_numbers:
            if number <= 0 or number in seen_numbers:
                continue
            seen_numbers.add(number)
            unique_numbers.append(number)

        return unique_numbers

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

        logger.info(
            "Test plan saved permanently",
            extra={
                "plan_id": str(test_plan.plan_id),
                "plan_dir": str(plan_dir),
                "json_path": str(json_path),
                "md_path": str(md_path),
            },
        )

    def persist_test_plan(self, test_plan: TestPlan) -> None:
        """Persist a test plan artifact to disk."""
        self._save_test_plan(test_plan)

    async def refine_test_plan(self, test_plan: TestPlan, feedback: str) -> TestPlan:
        """
        Refine an existing test plan based on feedback.

        Args:
            test_plan: Existing test plan to refine
            feedback: Feedback or additional requirements

        Returns:
            TestPlan: Refined test plan
        """
        logger.info(
            "Refining test plan based on feedback",
            extra={"plan_name": test_plan.name, "feedback_length": len(feedback)},
        )

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
            {"role": "user", "content": user_message},
        ]

        # Get refined plan from AI
        response = await self._get_completion(
            messages=messages, response_format={"type": "json_object"}, temperature=0.3
        )

        # Parse the refined plan
        refined_plan = self._parse_test_plan_response(response)

        # Calculate total steps for new structure
        total_steps = sum(len(tc.steps) for tc in refined_plan.test_cases)

        logger.info(
            "Test plan refined successfully",
            extra={
                "plan_name": refined_plan.name,
                "num_test_cases": len(refined_plan.test_cases),
                "total_steps": total_steps,
            },
        )

        return refined_plan

    def _serialize_test_plan(self, test_plan: TestPlan) -> str:
        """Serialize a test plan to a readable string format."""
        formatter = TestPlanFormatter()
        return formatter.to_markdown(test_plan)

    async def extract_test_scenarios(self, requirements: str) -> list[dict[str, str]]:
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
            {"role": "user", "content": requirements},
        ]

        response = await self._get_completion(
            messages=messages, response_format={"type": "json_object"}, temperature=0.3
        )

        try:
            import json

            content = response.get("content", "{}")
            scenarios_data = json.loads(content)
            scenarios_raw = scenarios_data.get("scenarios", [])
            scenarios: list[dict[str, str]] = []
            if isinstance(scenarios_raw, list):
                for item in scenarios_raw:
                    if not isinstance(item, dict):
                        continue
                    scenarios.append(
                        {
                            "name": str(item.get("name", "")),
                            "description": str(item.get("description", "")),
                            "priority": str(item.get("priority", "")),
                            "type": str(item.get("type", "")),
                        }
                    )

            logger.info(f"Extracted {len(scenarios)} test scenarios")
            return scenarios

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to extract test scenarios", extra={"error": str(e)})
            return []
