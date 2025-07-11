"""
Tests for the Enhanced Test Runner Agent.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.agents.test_runner import TestRunner
from src.core.types import (
    ActionInstruction,
    ActionType,
    BugReport,
    BugSeverity,
    GridAction,
    GridCoordinate,
    StepResult,
    TestCase,
    TestCasePriority,
    TestCaseResult,
    TestPlan,
    TestReport,
    TestState,
    TestStatus,
    TestStep,
    TestSummary,
)


@pytest.fixture
def mock_browser_driver():
    """Mock browser driver."""
    driver = AsyncMock()
    driver.navigate = AsyncMock()
    driver.screenshot = AsyncMock(return_value=b"mock_screenshot")
    driver.click = AsyncMock()
    driver.type = AsyncMock()
    driver.type_text = AsyncMock()
    driver.wait = AsyncMock()
    driver.get_viewport_size = AsyncMock(return_value=(1920, 1080))
    return driver


@pytest.fixture
def mock_action_agent():
    """Mock action agent."""
    agent = AsyncMock()
    
    # Mock the enhanced action result structure
    from src.core.enhanced_types import (
        AIAnalysis,
        ValidationResult,
        CoordinateResult,
        ExecutionResult,
        EnhancedActionResult
    )
    
    # Track execution order
    execution_count = [0]
    
    async def mock_execute_action(test_step, test_context, screenshot=None):
        """Mock execute_action that returns appropriate results based on action type."""
        # Use execution order to determine the correct outcome
        # This ensures we return the right outcome regardless of how the test runner
        # modifies the test step during execution
        outcomes_by_order = [
            "Login page is displayed",  # Step 1 - Navigate
            "Username is entered",  # Step 2 - Type username
            "Password is entered",  # Step 3 - Type password
            "User is logged in and redirected to dashboard",  # Step 4 - Click login
            "User menu dropdown is displayed",  # Step 5 - Click profile (logout case)
            "User is logged out and redirected to home page"  # Step 6 - Click logout
        ]
        
        idx = execution_count[0]
        execution_count[0] += 1
        
        if idx < len(outcomes_by_order):
            outcome = outcomes_by_order[idx]
        else:
            outcome = "Action completed successfully"
        
        return EnhancedActionResult(
            test_step_id=test_step.step_id,
            test_step=test_step,
            test_context=test_context,
            validation=ValidationResult(
                valid=True,
                reasoning="Action is valid and appropriate",
                confidence=0.95,
                concerns=[],
                suggestions=[]
            ),
            coordinates=CoordinateResult(
                grid_cell="M23",
                grid_coordinates=(960, 540),
                offset_x=0.5,
                offset_y=0.5,
                confidence=0.95,
                reasoning="Element found successfully",
                refined=False,
                refinement_details=None
            ),
            execution=ExecutionResult(
                success=True,
                execution_time_ms=523.4,
                error_message=None,
                error_traceback=None,
                browser_logs=[],
                network_activity=[]
            ),
            ai_analysis=AIAnalysis(
                success=True,
                confidence=0.95,
                actual_outcome=outcome,
                matches_expected=True,
                ui_changes=["Page updated as expected"],
                recommendations=[],
                anomalies=[]
            ),
            overall_success=True
        )
    
    agent.execute_action = AsyncMock(side_effect=mock_execute_action)
    return agent


@pytest.fixture
def sample_hierarchical_test_plan():
    """Create a sample hierarchical test plan with test cases."""
    plan_id = uuid4()
    
    # Create test steps for login test case
    login_steps = [
        TestStep(
            step_number=1,
            description="Navigate to login page",
            action="Navigate to https://example.com/login",
            expected_result="Login page is displayed",
            action_instruction=ActionInstruction(
                action_type=ActionType.NAVIGATE,
                description="Navigate to the login page",
                target="https://example.com/login",
                expected_outcome="Login page displayed"
            )
        ),
        TestStep(
            step_number=2,
            description="Enter username",
            action="Type 'testuser' in the username field",
            expected_result="Username is entered",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Enter username",
                target="username field",
                value="testuser",
                expected_outcome="Username entered"
            )
        ),
        TestStep(
            step_number=3,
            description="Enter password",
            action="Type password in the password field",
            expected_result="Password is entered",
            action_instruction=ActionInstruction(
                action_type=ActionType.TYPE,
                description="Enter password",
                target="password field",
                value="testpass",
                expected_outcome="Password entered"
            )
        ),
        TestStep(
            step_number=4,
            description="Submit login",
            action="Click the login button",
            expected_result="User is logged in and redirected to dashboard",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click login button",
                target="login button",
                expected_outcome="Login successful"
            )
        )
    ]
    
    # Create test steps for logout test case
    logout_steps = [
        TestStep(
            step_number=1,
            description="Open user menu",
            action="Click on the user profile icon",
            expected_result="User menu dropdown is displayed",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click user profile icon",
                target="user profile icon",
                expected_outcome="User menu opened"
            )
        ),
        TestStep(
            step_number=2,
            description="Click logout",
            action="Click the logout option",
            expected_result="User is logged out and redirected to home page",
            action_instruction=ActionInstruction(
                action_type=ActionType.CLICK,
                description="Click logout option",
                target="logout button",
                expected_outcome="User logged out"
            )
        )
    ]
    
    # Create test cases
    login_case = TestCase(
        test_id="TC001",
        name="User Login Test",
        description="Test user can login with valid credentials",
        priority=TestCasePriority.HIGH,
        prerequisites=["Application is accessible", "Test user account exists"],
        steps=login_steps,
        postconditions=["User is logged in", "Dashboard is displayed"],
        tags=["authentication", "login"]
    )
    
    logout_case = TestCase(
        test_id="TC002",
        name="User Logout Test",
        description="Test user can logout successfully",
        priority=TestCasePriority.MEDIUM,
        prerequisites=["User is logged in"],
        steps=logout_steps,
        postconditions=["User is logged out", "Home page is displayed"],
        tags=["authentication", "logout"]
    )
    
    # Create test plan
    test_plan = TestPlan(
        plan_id=plan_id,
        name="Authentication Flow Test",
        description="Test user authentication including login and logout",
        requirements_source="PRD v1.2 - Authentication Requirements",
        test_cases=[login_case, logout_case],
        tags=["authentication", "smoke-test"]
    )
    
    return test_plan


@pytest.fixture
def test_runner(mock_browser_driver, mock_action_agent, tmp_path):
    """Create a TestRunner instance for testing."""
    runner = TestRunner(
        browser_driver=mock_browser_driver,
        action_agent=mock_action_agent,
        report_dir=str(tmp_path / "test_reports")
    )
    runner._client = AsyncMock()
    return runner


class TestEnhancedTestRunner:
    """Test cases for TestRunner."""
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_success(
        self, test_runner, sample_hierarchical_test_plan, tmp_path
    ):
        """Test successful test plan execution with all test cases passing."""
        # Mock AI responses
        test_runner.call_openai = AsyncMock()
        
        # Create a list of responses in the order they will be called
        # The pattern is: interpret step 1, verify step 1, interpret step 2, verify step 2, etc.
        responses = []
        
        # Login Test Case - Step 1
        responses.append({"content": json.dumps({"actions": [{"type": "navigate", "target": "https://example.com/login", "critical": True, "description": "Navigate to login page"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "Login page is displayed", "confidence": 0.95, "reason": ""})})
        
        # Login Test Case - Step 2
        responses.append({"content": json.dumps({"actions": [{"type": "type", "target": "username field", "value": "testuser", "critical": True, "description": "Type username in the field"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "Username is entered", "confidence": 0.95, "reason": ""})})
        
        # Login Test Case - Step 3
        responses.append({"content": json.dumps({"actions": [{"type": "type", "target": "password field", "value": "testpass", "critical": True, "description": "Type password in the field"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "Password is entered", "confidence": 0.95, "reason": ""})})
        
        # Login Test Case - Step 4
        responses.append({"content": json.dumps({"actions": [{"type": "click", "target": "login button", "critical": True, "description": "Click the login button"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "User is logged in and redirected to dashboard", "confidence": 0.95, "reason": ""})})
        
        # Logout Test Case - Step 1
        responses.append({"content": json.dumps({"actions": [{"type": "click", "target": "user profile icon", "critical": True, "description": "Click on the user profile icon"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "User menu dropdown is displayed", "confidence": 0.95, "reason": ""})})
        
        # Logout Test Case - Step 2
        responses.append({"content": json.dumps({"actions": [{"type": "click", "target": "logout button", "critical": True, "description": "Click the logout option"}]})})
        responses.append({"content": json.dumps({"success": True, "actual_outcome": "User is logged out and redirected to home page", "confidence": 0.95, "reason": ""})})
        
        # Create a call counter
        call_count = [0]
        
        def mock_call_openai(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            # Debug what call is being made
            messages = kwargs.get('messages', [])
            if messages:
                content = messages[0].get('content', '')[:100]  # First 100 chars
                print(f"DEBUG: AI call {idx}: {content}...")
            
            if idx < len(responses):
                response = responses[idx]
                print(f"DEBUG: Returning response {idx}: {response['content'][:100]}...")
                return response
            print(f"WARNING: Ran out of mock responses at index {idx}")
            return {"content": json.dumps({"success": False, "error": "No more responses"})}
        
        test_runner.call_openai = AsyncMock(side_effect=mock_call_openai)
        
        # Execute test plan
        report = await test_runner.execute_test_plan(
            sample_hierarchical_test_plan,
            initial_url="https://example.com"
        )
        
        # Verify report structure
        assert isinstance(report, TestReport)
        assert report.test_plan_id == sample_hierarchical_test_plan.plan_id
        assert report.test_plan_name == "Authentication Flow Test"
        assert report.status == TestStatus.COMPLETED
        assert len(report.test_cases) == 2
        
        # Verify test case results
        login_result = report.test_cases[0]
        assert login_result.test_id == "TC001"
        assert login_result.status == TestStatus.COMPLETED
        assert login_result.steps_completed == 4
        assert login_result.steps_failed == 0
        
        logout_result = report.test_cases[1]
        assert logout_result.test_id == "TC002"
        assert logout_result.status == TestStatus.COMPLETED
        assert logout_result.steps_completed == 2
        assert logout_result.steps_failed == 0
        
        # Verify summary
        assert report.summary is not None
        assert report.summary.total_test_cases == 2
        assert report.summary.completed_test_cases == 2
        assert report.summary.failed_test_cases == 0
        assert report.summary.success_rate == 1.0
        
        # Verify no bugs reported
        assert len(report.bugs) == 0
        
        # Verify report files were saved
        report_dir = Path(tmp_path / "test_reports")
        json_files = list(report_dir.glob("*.json"))
        md_files = list(report_dir.glob("*.md"))
        assert len(json_files) > 0
        assert len(md_files) > 0
    
    @pytest.mark.asyncio
    async def test_execute_test_plan_with_failure(
        self, test_runner, sample_hierarchical_test_plan, mock_action_agent
    ):
        """Test test plan execution with a critical failure."""
        # Make the login button click fail
        from src.core.enhanced_types import (
            AIAnalysis,
            ValidationResult,
            CoordinateResult,
            ExecutionResult,
            EnhancedActionResult
        )
        
        async def mock_execute_with_failure(test_step, test_context, screenshot=None):
            # Fail on step 4 of login test case (click login button)
            if "login button" in test_step.action:
                return EnhancedActionResult(
                    test_step_id=test_step.step_id,
                    test_step=test_step,
                    test_context=test_context,
                    validation=ValidationResult(
                        valid=True,
                        reasoning="Action is valid",
                        confidence=0.95,
                        concerns=[],
                        suggestions=[]
                    ),
                    coordinates=CoordinateResult(
                        grid_cell="M23",
                        grid_coordinates=(960, 540),
                        offset_x=0.5,
                        offset_y=0.5,
                        confidence=0.95,
                        reasoning="Element found",
                        refined=False,
                        refinement_details=None
                    ),
                    execution=ExecutionResult(
                        success=False,
                        execution_time_ms=1000.0,
                        error_message="Element not found: login button",
                        error_traceback=None,
                        browser_logs=[],
                        network_activity=[]
                    ),
                    ai_analysis=AIAnalysis(
                        success=False,
                        confidence=0.9,
                        actual_outcome="Login button not found on page",
                        matches_expected=False,
                        ui_changes=[],
                        recommendations=["Check if page loaded correctly"],
                        anomalies=["Login button missing"]
                    ),
                    overall_success=False,
                    failure_phase="execution"
                )
            # All other actions succeed
            return EnhancedActionResult(
                test_step_id=test_step.step_id,
                test_step=test_step,
                test_context=test_context,
                validation=ValidationResult(
                    valid=True,
                    reasoning="Valid",
                    confidence=0.95,
                    concerns=[],
                    suggestions=[]
                ),
                coordinates=CoordinateResult(
                    grid_cell="M23",
                    grid_coordinates=(960, 540),
                    offset_x=0.5,
                    offset_y=0.5,
                    confidence=0.95,
                    reasoning="Element found",
                    refined=False,
                    refinement_details=None
                ),
                execution=ExecutionResult(
                    success=True,
                    execution_time_ms=500.0,
                    error_message=None,
                    error_traceback=None,
                    browser_logs=[],
                    network_activity=[]
                ),
                ai_analysis=AIAnalysis(
                    success=True,
                    confidence=0.9,
                    actual_outcome="Success",
                    matches_expected=True,
                    ui_changes=[],
                    recommendations=[],
                    anomalies=[]
                ),
                overall_success=True
            )
        
        mock_action_agent.execute_action = AsyncMock(side_effect=mock_execute_with_failure)
        
        # Mock AI responses - carefully ordered
        ai_responses = [
            # Step 1 - Navigate (interpret & verify)
            {"content": json.dumps({"actions": [{"type": "navigate", "target": "https://example.com/login", "critical": True}]})},
            {"content": json.dumps({"success": True, "actual_outcome": "Page loaded", "confidence": 0.95})},
            # Step 2 - Type username (interpret & verify)
            {"content": json.dumps({"actions": [{"type": "type", "target": "username field", "value": "testuser", "critical": True}]})},
            {"content": json.dumps({"success": True, "actual_outcome": "Username entered", "confidence": 0.95})},
            # Step 3 - Type password (interpret & verify)
            {"content": json.dumps({"actions": [{"type": "type", "target": "password field", "value": "testpass", "critical": True}]})},
            {"content": json.dumps({"success": True, "actual_outcome": "Password entered", "confidence": 0.95})},
            # Step 4 - Click login (interpret only - action will fail)
            {"content": json.dumps({"actions": [{"type": "click", "target": "login button", "critical": True}]})},
            # Bug classification for failed step 4
            {"content": json.dumps({"error_type": "element_not_found", "severity": "high", "reasoning": "Login button missing"})},
            # Blocker analysis for failed step 4
            {"content": json.dumps({"is_blocker": True, "reasoning": "Cannot proceed without login"})},
            # Cascade failure decision after test case 1 fails
            {"content": json.dumps({"continue": False, "reasoning": "Login failure blocks all subsequent tests"})},
        ]
        
        # Add extra responses for any additional calls
        # The test seems to be making more calls than expected
        for _ in range(10):
            ai_responses.append({"content": json.dumps({"error": "Unexpected call"})})
        
        test_runner.call_openai = AsyncMock(side_effect=ai_responses)
        
        # Execute test plan
        report = await test_runner.execute_test_plan(sample_hierarchical_test_plan)
        
        # Debug output
        print(f"\nDEBUG: Report status: {report.status}")
        print(f"DEBUG: Test cases: {len(report.test_cases)}")
        for i, tc in enumerate(report.test_cases):
            print(f"DEBUG: Test case {i}: {tc.name} - {tc.status} (steps: {tc.steps_completed}/{tc.steps_total}, failed: {tc.steps_failed})")
        
        # Verify failure handling
        assert report.status == TestStatus.FAILED
        assert len(report.test_cases) == 2
        
        # First test case should fail
        login_result = report.test_cases[0]
        assert login_result.status == TestStatus.FAILED
        assert login_result.steps_completed == 3  # First 3 steps succeeded
        assert login_result.steps_failed == 1  # Login button click failed
        
        # Second test case should be blocked (prerequisites not met)
        logout_result = report.test_cases[1]
        assert logout_result.status == TestStatus.BLOCKED
        
        # Verify bug was reported
        assert len(report.bugs) == 1
        bug = report.bugs[0]
        assert bug.severity == BugSeverity.HIGH
        assert bug.error_type == "element_not_found"
        assert "login button" in bug.description.lower()
        assert bug.step_number == 4
        
        # Verify summary reflects failures
        assert report.summary.failed_test_cases == 1
        assert report.summary.success_rate < 1.0
    
    @pytest.mark.asyncio
    async def test_intelligent_step_interpretation(self, test_runner):
        """Test that steps are intelligently decomposed into multiple actions."""
        # Create a complex step
        complex_step = TestStep(
            step_number=1,
            description="Add product to cart",
            action="Search for 'laptop', select first result, and add to cart",
            expected_result="Product is added to cart and cart count shows 1"
        )
        
        test_case = TestCase(
            test_id="TC001",
            name="Shopping Test",
            description="Test shopping flow",
            priority=TestCasePriority.HIGH,
            steps=[complex_step]
        )
        
        # Mock AI to decompose into multiple actions
        test_runner.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "actions": [
                    {
                        "type": "type",
                        "target": "search box",
                        "value": "laptop",
                        "description": "Type search query",
                        "critical": True
                    },
                    {
                        "type": "click",
                        "target": "search button",
                        "description": "Submit search",
                        "critical": True
                    },
                    {
                        "type": "wait",
                        "target": "search results",
                        "description": "Wait for results to load",
                        "critical": False
                    },
                    {
                        "type": "click",
                        "target": "first product",
                        "description": "Click first search result",
                        "critical": True
                    },
                    {
                        "type": "scroll",
                        "target": "add to cart button",
                        "description": "Scroll to add to cart button",
                        "critical": False
                    },
                    {
                        "type": "click",
                        "target": "add to cart button",
                        "description": "Add product to cart",
                        "critical": True
                    },
                    {
                        "type": "assert",
                        "target": "cart count",
                        "value": "1",
                        "description": "Verify cart count",
                        "critical": True
                    }
                ]
            })
        })
        
        # Execute step interpretation
        actions = await test_runner._interpret_step(
            complex_step, test_case, TestCaseResult(
                case_id=test_case.case_id,
                test_id=test_case.test_id,
                name=test_case.name,
                status=TestStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
                steps_total=1,
                steps_completed=0,
                steps_failed=0
            )
        )
        
        # Verify decomposition
        assert len(actions) == 7
        assert actions[0]["type"] == "type"
        assert actions[0]["value"] == "laptop"
        assert actions[6]["type"] == "assert"
        assert actions[6]["value"] == "1"
        
        # Verify critical vs non-critical actions
        assert actions[2]["critical"] is False  # Wait is non-critical
        assert actions[4]["critical"] is False  # Scroll is non-critical
        assert actions[5]["critical"] is True   # Add to cart is critical
    
    @pytest.mark.asyncio
    async def test_bug_report_generation(self, test_runner):
        """Test detailed bug report generation for failures."""
        # Create a failed step result
        step = TestStep(
            step_number=2,
            description="Submit form",
            action="Click submit button",
            expected_result="Form is submitted successfully"
        )
        
        test_case = TestCase(
            case_id=uuid4(),
            test_id="TC001",
            name="Form Submission Test",
            description="Test form submission",
            priority=TestCasePriority.HIGH,
            steps=[step]
        )
        
        step_result = StepResult(
            step_id=step.step_id,
            step_number=2,
            status=TestStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            action="Click submit button",
            expected_result="Form is submitted successfully",
            actual_result="Submit button not found",
            error_message="Element not found: submit button",
            screenshot_after="/screenshots/error.png"
        )
        
        # Set up test runner context
        test_runner._current_test_plan = TestPlan(
            plan_id=uuid4(),
            name="Test Plan",
            description="Test",
            requirements_source="PRD",
            test_cases=[test_case]
        )
        
        # Create a test case result for the function
        case_result = TestCaseResult(
            case_id=test_case.case_id,
            test_id=test_case.test_id,
            name=test_case.name,
            status=TestStatus.IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            steps_total=1,
            steps_completed=0,
            steps_failed=1,
            step_results=[step_result]  # Include the failed step
        )
        
        # Generate bug report
        bug_report = await test_runner._create_bug_report(
            step_result, step, test_case, case_result
        )
        
        # Verify bug report
        assert bug_report is not None
        assert bug_report.severity == BugSeverity.HIGH
        assert bug_report.error_type == "element_not_found"
        assert bug_report.step_number == 2
        assert "submit button" in bug_report.description
        assert bug_report.expected_result == "Form is submitted successfully"
        assert bug_report.actual_result == "Submit button not found"
        assert bug_report.screenshot_path == "/screenshots/error.png"
        assert len(bug_report.reproduction_steps) > 0
    
    @pytest.mark.asyncio
    async def test_living_document_updates(self, test_runner, tmp_path):
        """Test that reports are saved periodically as living documents."""
        # Create a simple test plan
        test_case = TestCase(
            test_id="TC001",
            name="Test Case",
            description="Test",
            priority=TestCasePriority.MEDIUM,
            steps=[
                TestStep(
                    step_number=1,
                    description="Step 1",
                    action="Do something",
                    expected_result="Success"
                )
            ]
        )
        
        test_plan = TestPlan(
            name="Test Plan",
            description="Test",
            requirements_source="PRD",
            test_cases=[test_case]
        )
        
        # Mock minimal AI responses
        test_runner.call_openai = AsyncMock()
        test_runner.call_openai.return_value = {
            "content": json.dumps({
                "actions": [{"type": "click", "target": "button", "critical": True}]
            })
        }
        
        # Start execution
        test_runner._test_report = TestReport(
            test_plan_id=test_plan.plan_id,
            test_plan_name=test_plan.name,
            started_at=datetime.now(timezone.utc),
            status=TestStatus.IN_PROGRESS
        )
        
        # Save report multiple times (simulating periodic saves)
        await test_runner._save_report()
        
        # Verify files exist
        report_dir = Path(test_runner.report_dir)
        json_files = list(report_dir.glob("*.json"))
        md_files = list(report_dir.glob("*.md"))
        
        assert len(json_files) == 1
        assert len(md_files) == 1
        
        # Read and verify JSON content
        with open(json_files[0], "r") as f:
            saved_report = json.load(f)
        
        assert saved_report["test_plan_name"] == "Test Plan"
        assert saved_report["status"] == "in_progress"
        
        # Update report and save again
        test_runner._test_report.status = TestStatus.COMPLETED
        await test_runner._save_report()
        
        # Verify updated content
        with open(json_files[0], "r") as f:
            updated_report = json.load(f)
        
        assert updated_report["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_failure_classification(self, test_runner):
        """Test that failures are correctly classified as blockers or non-blockers."""
        # Test navigation error (should be blocker)
        nav_result = StepResult(
            step_id=uuid4(),
            step_number=1,
            status=TestStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            action="Navigate to page",
            expected_result="Page loads",
            actual_result="Navigation failed",
            error_message="Navigation timeout"
        )
        
        is_blocker = await test_runner._is_blocker_failure(nav_result)
        assert is_blocker is True
        
        # Test element not found (should check with AI)
        test_runner.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "is_blocker": True,
                "reasoning": "Cannot proceed without this element"
            })
        })
        
        element_result = StepResult(
            step_id=uuid4(),
            step_number=2,
            status=TestStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            action="Click button",
            expected_result="Button clicked",
            actual_result="Button not found",
            error_message="Element not found"
        )
        
        is_blocker = await test_runner._is_blocker_failure(element_result)
        assert is_blocker is True
        test_runner.call_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summary_calculation(self, test_runner):
        """Test summary statistics calculation."""
        # Create test report with mixed results
        test_runner._test_report = TestReport(
            test_plan_id=uuid4(),
            test_plan_name="Test Plan",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            status=TestStatus.COMPLETED,
            test_cases=[
                TestCaseResult(
                    case_id=uuid4(),
                    test_id="TC001",
                    name="Test 1",
                    status=TestStatus.COMPLETED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    steps_total=3,
                    steps_completed=3,
                    steps_failed=0
                ),
                TestCaseResult(
                    case_id=uuid4(),
                    test_id="TC002",
                    name="Test 2",
                    status=TestStatus.FAILED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    steps_total=2,
                    steps_completed=1,
                    steps_failed=1
                )
            ],
            bugs=[
                BugReport(
                    step_id=uuid4(),
                    test_case_id=uuid4(),
                    test_plan_id=uuid4(),
                    step_number=1,
                    description="Bug 1",
                    severity=BugSeverity.CRITICAL,
                    error_type="error",
                    expected_result="Success",
                    actual_result="Failure"
                ),
                BugReport(
                    step_id=uuid4(),
                    test_case_id=uuid4(),
                    test_plan_id=uuid4(),
                    step_number=2,
                    description="Bug 2",
                    severity=BugSeverity.MEDIUM,
                    error_type="error",
                    expected_result="Success",
                    actual_result="Failure"
                )
            ]
        )
        
        # Calculate summary
        summary = test_runner._calculate_summary()
        
        # Verify calculations
        assert summary.total_test_cases == 2
        assert summary.completed_test_cases == 1
        assert summary.failed_test_cases == 1
        assert summary.total_steps == 5
        assert summary.completed_steps == 4
        assert summary.failed_steps == 1
        assert summary.success_rate == 0.8  # 4/5 steps
        assert summary.critical_bugs == 1
        assert summary.medium_bugs == 1
        assert summary.high_bugs == 0
        assert summary.low_bugs == 0
    
    @pytest.mark.asyncio  
    async def test_cascade_failure_handling(self, test_runner, sample_hierarchical_test_plan):
        """Test that cascade failure handling works correctly."""
        # Store original method
        original_execute = test_runner._execute_test_case
        
        # Create results
        failed_result = TestCaseResult(
            case_id=sample_hierarchical_test_plan.test_cases[0].case_id,
            test_id="TC001",
            name="User Login Test",
            status=TestStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            steps_total=4,
            steps_completed=3,
            steps_failed=1,
            error_message="Login button not found"
        )
        
        # Mock to return our results and add them to the report
        async def mock_execute_test_case(test_case):
            if test_case.test_id == "TC001":
                # Add to report like the real method does
                test_runner._test_report.test_cases.append(failed_result)
                return failed_result
            else:
                # This shouldn't be called if cascade works
                raise AssertionError("Second test case should not be executed")
        
        test_runner._execute_test_case = mock_execute_test_case
        
        # Mock the cascade decision - should NOT continue
        test_runner.call_openai = AsyncMock(return_value={
            "content": json.dumps({
                "continue": False,
                "reasoning": "Login failure prevents logout test"
            })
        })
        
        # Execute test plan
        report = await test_runner.execute_test_plan(sample_hierarchical_test_plan)
        
        # Debug output
        print(f"\nDEBUG: Report has {len(report.test_cases)} test cases")
        for i, tc in enumerate(report.test_cases):
            print(f"  {i}: {tc.test_id} - {tc.name} - {tc.status}")
        
        # Verify cascade handling
        assert len(report.test_cases) == 2
        assert report.status == TestStatus.FAILED
        
        # First test case should be failed
        assert report.test_cases[0].status == TestStatus.FAILED
        assert report.test_cases[0].test_id == "TC001"
        
        # Second test case should be blocked
        assert report.test_cases[1].status == TestStatus.BLOCKED
        assert report.test_cases[1].test_id == "TC002"
        assert "Blocked due to failure" in report.test_cases[1].error_message