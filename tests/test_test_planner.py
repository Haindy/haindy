"""
Unit tests for Test Planner Agent.
"""

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.agents.test_planner import TestPlannerAgent
from src.core.types import ActionInstruction, ActionType, TestPlan, TestStep


class TestTestPlannerAgent:
    """Tests for TestPlannerAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a test planner agent instance."""
        agent = TestPlannerAgent()
        # Mock the OpenAI client
        agent._client = MagicMock()
        return agent

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI response for test plan generation."""
        return {
            "content": json.dumps({
                "name": "User Login Flow Test",
                "description": "Test user login functionality",
                "requirements": "Test the login functionality",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Navigate to login page",
                        "action": "navigate",
                        "target": "Login page URL",
                        "expected_result": "Login page is displayed",
                        "dependencies": [],
                        "optional": False
                    },
                    {
                        "step_number": 2,
                        "description": "Enter username",
                        "action": "type",
                        "target": "Username field",
                        "value": "testuser",
                        "expected_result": "Username is entered in the field",
                        "dependencies": [1],
                        "optional": False
                    },
                    {
                        "step_number": 3,
                        "description": "Enter password",
                        "action": "type",
                        "target": "Password field",
                        "value": "password123",
                        "expected_result": "Password is entered in the field",
                        "dependencies": [1],
                        "optional": False
                    },
                    {
                        "step_number": 4,
                        "description": "Click login button",
                        "action": "click",
                        "target": "Login button",
                        "expected_result": "User is logged in and redirected",
                        "dependencies": [2, 3],
                        "optional": False
                    }
                ],
                "tags": ["functional", "login"],
                "estimated_duration_seconds": 60
            })
        }

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "TestPlanner"
        assert agent.system_prompt is not None
        assert "Test Planning Specialist" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_create_test_plan_basic(self, agent, mock_openai_response):
        """Test creating a basic test plan."""
        # Mock the OpenAI call
        agent.call_openai = AsyncMock(return_value=mock_openai_response)
        
        requirements = "Test the login functionality"
        test_plan = await agent.create_test_plan(requirements)

        assert isinstance(test_plan, TestPlan)
        assert test_plan.name == "User Login Flow Test"
        assert test_plan.description == "Test user login functionality"
        assert test_plan.requirements == "Test the login functionality"
        assert len(test_plan.steps) == 4
        assert len(test_plan.tags) == 2

    @pytest.mark.asyncio
    async def test_create_test_plan_with_context(self, agent, mock_openai_response):
        """Test creating a test plan with additional context."""
        agent.call_openai = AsyncMock(return_value=mock_openai_response)
        
        requirements = "Test the login functionality"
        context = {
            "application": "E-commerce platform",
            "user_type": "Customer"
        }
        test_plan = await agent.create_test_plan(requirements, context)

        assert isinstance(test_plan, TestPlan)
        assert test_plan.name == "User Login Flow Test"

    @pytest.mark.asyncio
    async def test_test_plan_steps_structure(self, agent, mock_openai_response):
        """Test that test plan steps have correct structure."""
        agent.call_openai = AsyncMock(return_value=mock_openai_response)
        
        requirements = "Test the login functionality"
        test_plan = await agent.create_test_plan(requirements)

        # Check first step
        first_step = test_plan.steps[0]
        assert isinstance(first_step, TestStep)
        assert first_step.step_number == 1
        assert first_step.description == "Navigate to login page"
        assert first_step.action_instruction.action_type == ActionType.NAVIGATE
        assert first_step.action_instruction.target == "Login page URL"
        assert len(first_step.dependencies) == 0
        assert first_step.optional is False

        # Check step with dependencies
        last_step = test_plan.steps[3]
        assert last_step.step_number == 4
        assert len(last_step.dependencies) == 2  # Should have 2 dependencies

    def test_parse_test_plan_response_error(self, agent):
        """Test error handling when parsing invalid response."""
        invalid_response = {"content": "not valid json"}
        
        with pytest.raises(ValueError, match="Failed to parse test plan response"):
            agent._parse_test_plan_response(invalid_response)

    @pytest.mark.asyncio
    async def test_refine_test_plan(self, agent):
        """Test refining an existing test plan."""
        # Create initial test plan
        action_instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click login button",
            target="Login button",
            expected_outcome="User logged in"
        )
        
        initial_plan = TestPlan(
            name="Basic Login Test",
            description="Simple login test",
            requirements="Test login",
            steps=[
                TestStep(
                    step_number=1,
                    description="Login step",
                    action_instruction=action_instruction,
                    optional=False
                )
            ],
            tags=["login"]
        )

        # Mock refined response
        refined_response = json.dumps({
            "name": "Enhanced Login Test",
            "description": "Comprehensive login test with error cases",
            "requirements": "Test login with error handling",
            "steps": [
                {
                    "step_number": 1,
                    "description": "Navigate to login",
                    "action": "navigate",
                    "target": "Login page",
                    "expected_result": "Login page shown",
                    "dependencies": [],
                    "optional": False
                },
                {
                    "step_number": 2,
                    "description": "Test invalid login",
                    "action": "click",
                    "target": "Login button",
                    "expected_result": "Error message shown",
                    "dependencies": [1],
                    "optional": False
                }
            ],
            "tags": ["login", "error-handling"],
            "estimated_duration_seconds": 120
        })

        agent.call_openai = AsyncMock(return_value={"content": refined_response})
        
        feedback = "Add error case testing"
        refined_plan = await agent.refine_test_plan(initial_plan, feedback)

        assert refined_plan.name == "Enhanced Login Test"
        assert len(refined_plan.steps) == 2
        assert len(refined_plan.tags) == 2

    def test_serialize_test_plan(self, agent):
        """Test serializing a test plan to string."""
        action_instruction = ActionInstruction(
            action_type=ActionType.CLICK,
            description="Click submit button",
            target="Submit button",
            value=None,
            expected_outcome="Form submitted"
        )
        
        test_plan = TestPlan(
            name="Test Plan",
            description="A test plan",
            requirements="Test requirements",
            steps=[
                TestStep(
                    step_number=1,
                    description="Click submit",
                    action_instruction=action_instruction,
                    optional=False
                )
            ],
            tags=["functional", "ui"]
        )

        serialized = agent._serialize_test_plan(test_plan)
        
        assert "Name: Test Plan" in serialized
        assert "Description: A test plan" in serialized
        assert "Requirements: Test requirements" in serialized
        assert "Tags: functional, ui" in serialized
        assert "1. Click submit" in serialized
        assert "Action: click - Submit button" in serialized
        assert "Expected: Form submitted" in serialized

    @pytest.mark.asyncio
    async def test_extract_test_scenarios(self, agent):
        """Test extracting multiple test scenarios from requirements."""
        scenarios_response = json.dumps({
            "scenarios": [
                {
                    "name": "Happy Path Login",
                    "description": "Test successful login",
                    "priority": "high",
                    "type": "functional"
                },
                {
                    "name": "Invalid Credentials",
                    "description": "Test login with wrong password",
                    "priority": "high",
                    "type": "error_handling"
                },
                {
                    "name": "Account Lockout",
                    "description": "Test account lockout after failed attempts",
                    "priority": "medium",
                    "type": "edge_case"
                }
            ]
        })

        agent.call_openai = AsyncMock(return_value={"content": scenarios_response})
        
        requirements = "Test complete login functionality including error cases"
        scenarios = await agent.extract_test_scenarios(requirements)

        assert len(scenarios) == 3
        assert scenarios[0]["name"] == "Happy Path Login"
        assert scenarios[1]["type"] == "error_handling"
        assert scenarios[2]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_extract_test_scenarios_error(self, agent):
        """Test error handling when extracting scenarios fails."""
        agent.call_openai = AsyncMock(return_value={"content": "invalid json"})
        
        requirements = "Test requirements"
        scenarios = await agent.extract_test_scenarios(requirements)
        
        assert scenarios == []  # Should return empty list on error

    def test_build_requirements_message(self, agent):
        """Test building requirements message."""
        requirements = "Test login"
        message = agent._build_requirements_message(requirements)
        
        assert "Please create a test plan" in message
        assert requirements in message
        assert "JSON format" in message

    def test_build_requirements_message_with_context(self, agent):
        """Test building requirements message with context."""
        requirements = "Test login"
        context = {"app": "E-commerce", "version": "2.0"}
        message = agent._build_requirements_message(requirements, context)
        
        assert requirements in message
        assert "Additional Context:" in message
        assert "app: E-commerce" in message
        assert "version: 2.0" in message

    @pytest.mark.asyncio
    async def test_openai_call_parameters(self, agent):
        """Test that OpenAI is called with correct parameters."""
        mock_response = json.dumps({
            "name": "Test",
            "description": "Test",
            "requirements": "Test",
            "steps": [],
            "tags": []
        })
        
        agent.call_openai = AsyncMock(return_value={"content": mock_response})
        
        await agent.create_test_plan("Test requirements")
        
        # Verify call parameters
        agent.call_openai.assert_called_once()
        call_args = agent.call_openai.call_args
        assert "response_format" in call_args.kwargs
        assert call_args.kwargs["response_format"] == {"type": "json_object"}
        assert call_args.kwargs["temperature"] == 0.3