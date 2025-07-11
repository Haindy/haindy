"""
Unit tests for Test Planner Agent.
"""

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.agents.test_planner import TestPlannerAgent
from src.agents.formatters import TestPlanFormatter
from src.core.types import ActionInstruction, ActionType, TestCase, TestCasePriority, TestPlan, TestStep


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
                "name": "User Login Flow Test Plan",
                "description": "Comprehensive test plan for user login functionality",
                "requirements_source": "Test the login functionality",
                "test_cases": [
                    {
                        "test_id": "TC001",
                        "name": "Happy Path Login",
                        "description": "Test successful login with valid credentials",
                        "priority": "critical",
                        "prerequisites": ["Valid test user account exists", "Application is accessible"],
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "Navigate to login page",
                                "expected_result": "Login page is displayed with username and password fields",
                                "dependencies": [],
                                "optional": False
                            },
                            {
                                "step_number": 2,
                                "action": "Enter username 'testuser' in the username field",
                                "expected_result": "Username is entered in the field",
                                "dependencies": [],
                                "optional": False
                            },
                            {
                                "step_number": 3,
                                "action": "Enter password 'password123' in the password field",
                                "expected_result": "Password is entered in the field (masked)",
                                "dependencies": [],
                                "optional": False
                            },
                            {
                                "step_number": 4,
                                "action": "Click the login button",
                                "expected_result": "User is logged in and redirected to dashboard",
                                "dependencies": [],
                                "optional": False
                            }
                        ],
                        "postconditions": ["User is logged in", "User session is active"],
                        "tags": ["smoke", "critical-path"]
                    },
                    {
                        "test_id": "TC002",
                        "name": "Invalid Credentials Login",
                        "description": "Test login with incorrect password",
                        "priority": "high",
                        "prerequisites": ["Application is accessible"],
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "Navigate to login page",
                                "expected_result": "Login page is displayed",
                                "dependencies": [],
                                "optional": False
                            },
                            {
                                "step_number": 2,
                                "action": "Enter username 'testuser' and wrong password",
                                "expected_result": "Credentials are entered",
                                "dependencies": [],
                                "optional": False
                            },
                            {
                                "step_number": 3,
                                "action": "Click login button",
                                "expected_result": "Error message 'Invalid credentials' is displayed",
                                "dependencies": [],
                                "optional": False
                            }
                        ],
                        "postconditions": ["User remains on login page", "Error message is visible"],
                        "tags": ["negative-test", "error-handling"]
                    }
                ],
                "tags": ["functional", "login", "authentication"],
                "estimated_duration_seconds": 120
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
        assert test_plan.name == "User Login Flow Test Plan"
        assert test_plan.description == "Comprehensive test plan for user login functionality"
        assert test_plan.requirements_source == "Test the login functionality"
        assert len(test_plan.test_cases) == 2
        assert len(test_plan.tags) == 3
        
        # Check first test case
        tc1 = test_plan.test_cases[0]
        assert tc1.test_id == "TC001"
        assert tc1.name == "Happy Path Login"
        assert tc1.priority == TestCasePriority.CRITICAL
        assert len(tc1.steps) == 4
        assert len(tc1.prerequisites) == 2
        assert len(tc1.postconditions) == 2

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
        assert test_plan.name == "User Login Flow Test Plan"
        assert len(test_plan.test_cases) == 2

    @pytest.mark.asyncio
    async def test_test_plan_steps_structure(self, agent, mock_openai_response):
        """Test that test plan steps have correct structure."""
        agent.call_openai = AsyncMock(return_value=mock_openai_response)
        
        requirements = "Test the login functionality"
        test_plan = await agent.create_test_plan(requirements)

        # Check first test case steps
        tc1 = test_plan.test_cases[0]
        first_step = tc1.steps[0]
        assert isinstance(first_step, TestStep)
        assert first_step.step_number == 1
        assert first_step.action == "Navigate to login page"
        assert first_step.expected_result == "Login page is displayed with username and password fields"
        assert len(first_step.dependencies) == 0
        assert first_step.optional is False

        # Check other steps in first test case
        assert tc1.steps[1].action == "Enter username 'testuser' in the username field"
        assert tc1.steps[2].action == "Enter password 'password123' in the password field"
        assert tc1.steps[3].action == "Click the login button"
        
        # Check second test case
        tc2 = test_plan.test_cases[1]
        assert tc2.test_id == "TC002"
        assert len(tc2.steps) == 3
        assert tc2.priority == TestCasePriority.HIGH

    def test_parse_test_plan_response_error(self, agent):
        """Test error handling when parsing invalid response."""
        invalid_response = {"content": "not valid json"}
        
        with pytest.raises(ValueError, match="Failed to parse test plan response"):
            agent._parse_test_plan_response(invalid_response)

    @pytest.mark.asyncio
    async def test_refine_test_plan(self, agent):
        """Test refining an existing test plan."""
        # Create initial test plan
        initial_plan = TestPlan(
            name="Basic Login Test",
            description="Simple login test",
            requirements_source="Test login",
            test_cases=[
                TestCase(
                    test_id="TC001",
                    name="Basic Login",
                    description="Simple login test",
                    priority=TestCasePriority.HIGH,
                    prerequisites=[],
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Login to application",
                            expected_result="User is logged in",
                            description="Basic login step"
                        )
                    ],
                    postconditions=[],
                    tags=["login"]
                )
            ],
            tags=["login"]
        )

        # Mock refined response
        refined_response = json.dumps({
            "name": "Enhanced Login Test Plan",
            "description": "Comprehensive login test with error cases",
            "requirements_source": "Test login with error handling",
            "test_cases": [
                {
                    "test_id": "TC001",
                    "name": "Happy Path Login",
                    "description": "Test successful login",
                    "priority": "critical",
                    "prerequisites": ["Valid user exists"],
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "Navigate to login page",
                            "expected_result": "Login page shown",
                            "dependencies": [],
                            "optional": False
                        },
                        {
                            "step_number": 2,
                            "action": "Enter valid credentials and login",
                            "expected_result": "User logged in successfully",
                            "dependencies": [],
                            "optional": False
                        }
                    ],
                    "postconditions": ["User is logged in"],
                    "tags": ["smoke"]
                },
                {
                    "test_id": "TC002",
                    "name": "Invalid Login",
                    "description": "Test login with invalid credentials",
                    "priority": "high",
                    "prerequisites": [],
                    "steps": [
                        {
                            "step_number": 1,
                            "action": "Navigate to login page",
                            "expected_result": "Login page shown",
                            "dependencies": [],
                            "optional": False
                        },
                        {
                            "step_number": 2,
                            "action": "Enter invalid credentials",
                            "expected_result": "Error message shown",
                            "dependencies": [],
                            "optional": False
                        }
                    ],
                    "postconditions": ["User remains on login page"],
                    "tags": ["negative"]
                }
            ],
            "tags": ["login", "error-handling", "authentication"],
            "estimated_duration_seconds": 120
        })

        agent.call_openai = AsyncMock(return_value={"content": refined_response})
        
        feedback = "Add error case testing"
        refined_plan = await agent.refine_test_plan(initial_plan, feedback)

        assert refined_plan.name == "Enhanced Login Test Plan"
        assert len(refined_plan.test_cases) == 2
        assert len(refined_plan.tags) == 3

    def test_serialize_test_plan(self, agent):
        """Test serializing a test plan to string."""
        test_plan = TestPlan(
            name="Test Plan",
            description="A test plan",
            requirements_source="Test requirements document",
            test_cases=[
                TestCase(
                    test_id="TC001",
                    name="Submit Form Test",
                    description="Test form submission",
                    priority=TestCasePriority.HIGH,
                    prerequisites=["Form is loaded"],
                    steps=[
                        TestStep(
                            step_number=1,
                            action="Click submit button",
                            expected_result="Form is submitted successfully",
                            description="Submit the form"
                        )
                    ],
                    postconditions=["Form data is saved"],
                    tags=["form", "submission"]
                )
            ],
            tags=["functional", "ui"]
        )

        serialized = agent._serialize_test_plan(test_plan)
        
        # The serialized output should be markdown
        assert "# Test Plan: Test Plan" in serialized
        assert "**Description**: A test plan" in serialized
        assert "**Requirements Source**: Test requirements document" in serialized
        assert "**Tags**: functional, ui" in serialized
        assert "### TC001: Submit Form Test" in serialized
        assert "**Priority**: High" in serialized
        assert "1. **Click submit button**" in serialized
        assert "_Expected Result_: Form is submitted successfully" in serialized

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
            "requirements_source": "Test",
            "test_cases": [],
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