"""
Test Planner Agent implementation.

Analyzes requirements/PRDs and creates structured test plans.
"""

from typing import Dict, List, Optional
from uuid import UUID

from src.agents.base_agent import BaseAgent
from src.config.agent_prompts import TEST_PLANNER_SYSTEM_PROMPT
from src.core.types import ActionInstruction, ActionType, TestPlan, TestStep
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
    
    async def _get_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """Get completion from OpenAI."""
        response = await self.call_openai(
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            response_format=kwargs.get("response_format"),
        )
        
        return response
    
    async def create_test_plan(self, requirements: str, context: Optional[Dict] = None) -> TestPlan:
        """
        Create a structured test plan from requirements.
        
        Args:
            requirements: High-level requirements, user story, or PRD
            context: Optional context about the application or domain
            
        Returns:
            TestPlan: Structured test plan with steps and expected outcomes
        """
        logger.info("Creating test plan from requirements", extra={
            "requirements_length": len(requirements),
            "has_context": context is not None
        })
        
        # Build the user message
        user_message = self._build_requirements_message(requirements, context)
        
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Get test plan from AI
        response = await self._get_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent planning
        )
        
        # Parse and validate the response
        test_plan = self._parse_test_plan_response(response)
        
        # Debug: Save test plan
        import json
        with open("debug_test_plan.json", "w") as f:
            plan_dict = test_plan.model_dump()
            # Convert UUIDs to strings
            plan_dict["plan_id"] = str(plan_dict["plan_id"])
            plan_dict["created_at"] = plan_dict["created_at"].isoformat()
            for step in plan_dict["steps"]:
                step["step_id"] = str(step["step_id"])
                step["dependencies"] = [str(d) for d in step["dependencies"]]
            json.dump(plan_dict, f, indent=2)
        
        logger.info("Test plan created successfully", extra={
            "plan_name": test_plan.name,
            "num_steps": len(test_plan.steps),
            "has_tags": bool(test_plan.tags)
        })
        
        return test_plan
    
    def _build_requirements_message(self, requirements: str, context: Optional[Dict] = None) -> str:
        """Build the user message with requirements and context."""
        message = f"Please create a test plan for the following requirements:\n\n{requirements}"
        
        if context:
            message += "\n\nAdditional Context:"
            for key, value in context.items():
                message += f"\n- {key}: {value}"
        
        message += """\n\nProvide the test plan in the following JSON format:
{
    "name": "Test plan name",
    "description": "Brief description of what is being tested",
    "requirements": "The original requirements being tested",
    "steps": [
        {
            "step_number": 1,
            "description": "Step description",
            "action": "click/type/navigate/wait/verify",
            "target": "Description of what to interact with",
            "value": "Value to type (if action is 'type')",
            "expected_result": "What should happen",
            "dependencies": [],
            "optional": false
        }
    ],
    "tags": ["functional", "regression", "smoke"],
    "estimated_duration_seconds": 300
}"""
        
        return message
    
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
            
            # Create test steps
            steps = []
            step_dependencies = {}  # Track dependencies by step number
            
            for step_data in plan_data.get("steps", []):
                # Map action string to ActionType
                action_type_map = {
                    "click": ActionType.CLICK,
                    "type": ActionType.TYPE,
                    "navigate": ActionType.NAVIGATE,
                    "wait": ActionType.WAIT,
                    "verify": ActionType.ASSERT,
                    "assert": ActionType.ASSERT
                }
                action_type = action_type_map.get(
                    step_data.get("action", "verify").lower(),
                    ActionType.ASSERT
                )
                
                # Create action instruction with required fields
                action_instruction = ActionInstruction(
                    action_type=action_type,
                    description=step_data.get("description", ""),
                    target=step_data.get("target", ""),
                    value=step_data.get("value"),
                    expected_outcome=step_data.get("expected_result", "")
                )
                
                # Create test step
                step = TestStep(
                    step_number=step_data["step_number"],
                    description=step_data["description"],
                    action_instruction=action_instruction,
                    optional=step_data.get("optional", False),
                    max_retries=step_data.get("max_retries", 3)
                )
                steps.append(step)
                
                # Store dependencies for later resolution
                step_dependencies[step.step_number] = step_data.get("dependencies", [])
            
            # Resolve step dependencies (convert step numbers to UUIDs)
            step_by_number = {step.step_number: step for step in steps}
            for step in steps:
                dep_numbers = step_dependencies.get(step.step_number, [])
                for dep_num in dep_numbers:
                    if dep_num in step_by_number:
                        step.dependencies.append(step_by_number[dep_num].step_id)
            
            # Create test plan
            # Handle requirements as either string or list
            requirements = plan_data.get("requirements", "")
            if isinstance(requirements, list):
                requirements = "\n".join(requirements)
            
            test_plan = TestPlan(
                name=plan_data["name"],
                description=plan_data["description"],
                requirements=requirements,
                steps=steps,
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
        
        logger.info("Test plan refined successfully", extra={
            "plan_name": refined_plan.name,
            "num_steps": len(refined_plan.steps)
        })
        
        return refined_plan
    
    def _serialize_test_plan(self, test_plan: TestPlan) -> str:
        """Serialize a test plan to a readable string format."""
        lines = [
            f"Name: {test_plan.name}",
            f"Description: {test_plan.description}",
            f"Requirements: {test_plan.requirements[:100]}..." if len(test_plan.requirements) > 100 else f"Requirements: {test_plan.requirements}",
            f"Tags: {', '.join(test_plan.tags) if test_plan.tags else 'None'}",
            "\nSteps:"
        ]
        
        for step in test_plan.steps:
            lines.append(f"  {step.step_number}. {step.description}")
            lines.append(f"     Action: {step.action_instruction.action_type.value} - {step.action_instruction.target}")
            if step.action_instruction.value:
                lines.append(f"     Value: {step.action_instruction.value}")
            lines.append(f"     Expected: {step.action_instruction.expected_outcome}")
            if step.dependencies:
                lines.append(f"     Dependencies: {len(step.dependencies)} steps")
            if step.optional:
                lines.append(f"     Optional: Yes")
        
        if test_plan.estimated_duration_seconds:
            lines.append(f"\nEstimated Duration: {test_plan.estimated_duration_seconds} seconds")
        
        return "\n".join(lines)
    
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