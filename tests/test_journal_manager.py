"""
Tests for the journal manager.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.types import (
    ActionInstruction,
    ActionType,
    TestPlan,
    TestStep
)
from src.journal import (
    ExecutionJournal,
    ExecutionMode,
    JournalActionResult,
    JournalManager
)


@pytest.fixture
def temp_journal_dir(tmp_path):
    """Create a temporary journal directory."""
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    return journal_dir


@pytest.fixture
def journal_manager(temp_journal_dir):
    """Create a JournalManager instance for testing."""
    return JournalManager(journal_dir=temp_journal_dir)


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    return TestPlan(
        plan_id=uuid4(),
        name="Test Login Flow",
        description="Test user login",
        requirements="User should be able to login",
        steps=[
            TestStep(
                step_id=uuid4(),
                step_number=1,
                description="Navigate to login",
                action_instruction=ActionInstruction(
                    action_type=ActionType.NAVIGATE,
                    description="Go to login page",
                    expected_outcome="Login page displayed"
                )
            ),
            TestStep(
                step_id=uuid4(),
                step_number=2,
                description="Click login button",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK,
                    description="Click the login button",
                    expected_outcome="Login form shown"
                )
            )
        ]
    )


@pytest.fixture
def sample_action_result():
    """Create a sample action result."""
    return JournalActionResult(
        success=True,
        action=ActionType.CLICK,
        confidence=0.95,
        coordinates=(500, 300),
        grid_coordinates={
            "initial_selection": "M23",
            "initial_confidence": 0.7,
            "refined_coordinates": "M23+offset(0.7,0.4)",
            "final_confidence": 0.95
        },
        playwright_command="await page.click('#login-btn')",
        selectors={"primary": "#login-btn", "fallback": "button[type=submit]"},
        actual_outcome="Login form displayed"
    )


class TestJournalManager:
    """Test cases for JournalManager."""
    
    @pytest.mark.asyncio
    async def test_create_journal(self, journal_manager, sample_test_plan):
        """Test creating a new journal."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        assert isinstance(journal, ExecutionJournal)
        assert journal.test_plan_id == sample_test_plan.plan_id
        assert journal.test_name == sample_test_plan.name
        assert journal.journal_id in journal_manager._active_journals
    
    @pytest.mark.asyncio
    async def test_record_action_visual(self, journal_manager, sample_test_plan, sample_action_result):
        """Test recording a visual action."""
        # Create journal
        journal = await journal_manager.create_journal(sample_test_plan)
        
        # Record action
        entry = await journal_manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Login Test",
            step=sample_test_plan.steps[1],  # Click step
            action_result=sample_action_result,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=1500
        )
        
        assert entry.test_scenario == "Login Test"
        assert entry.action_taken.startswith("Click the login button")
        assert entry.grid_coordinates is not None
        assert entry.execution_mode == ExecutionMode.VISUAL
        assert entry.success is True
        assert entry.agent_confidence == 0.95
        
        # Check journal was updated
        assert len(journal.entries) == 1
        assert journal.total_steps == 1
        assert journal.successful_steps == 1
    
    @pytest.mark.asyncio
    async def test_record_action_scripted(self, journal_manager, sample_test_plan, sample_action_result):
        """Test recording a scripted action."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        entry = await journal_manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Login Test",
            step=sample_test_plan.steps[1],
            action_result=sample_action_result,
            execution_mode=ExecutionMode.SCRIPTED,
            execution_time_ms=250,
            screenshot_before="before.png",
            screenshot_after="after.png"
        )
        
        assert entry.execution_mode == ExecutionMode.SCRIPTED
        assert entry.scripted_command == "await page.click('#login-btn')"
        assert entry.selectors is not None
        assert entry.screenshot_before == "before.png"
        assert entry.screenshot_after == "after.png"
        assert entry.execution_time_ms == 250
    
    @pytest.mark.asyncio
    async def test_pattern_creation_high_confidence(self, journal_manager, sample_test_plan, sample_action_result):
        """Test pattern creation for high confidence actions."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        # Record high confidence action
        await journal_manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Test",
            step=sample_test_plan.steps[1],
            action_result=sample_action_result,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=1000
        )
        
        # Check pattern was created
        assert len(journal_manager._pattern_library) == 1
        pattern = list(journal_manager._pattern_library.values())[0]
        assert pattern.pattern_type.value == "click"
        assert pattern.playwright_command == sample_action_result.playwright_command
        assert pattern.success_count == 1
    
    @pytest.mark.asyncio
    async def test_pattern_not_created_low_confidence(self, journal_manager, sample_test_plan):
        """Test pattern not created for low confidence actions."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        # Low confidence result
        low_conf_result = JournalActionResult(
            success=True,
            action=ActionType.CLICK,
            confidence=0.6,  # Below threshold
            playwright_command="await page.click('#btn')"
        )
        
        await journal_manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Test",
            step=sample_test_plan.steps[1],
            action_result=low_conf_result,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=1000
        )
        
        # No pattern should be created
        assert len(journal_manager._pattern_library) == 0
    
    @pytest.mark.asyncio
    async def test_find_matching_pattern(self, journal_manager, sample_test_plan, sample_action_result):
        """Test finding matching patterns."""
        # Create and record pattern
        journal = await journal_manager.create_journal(sample_test_plan)
        await journal_manager.record_action(
            journal_id=journal.journal_id,
            test_scenario="Test",
            step=sample_test_plan.steps[1],
            action_result=sample_action_result,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=1000
        )
        
        # Find matching pattern
        context = {
            "url": "https://example.com/login",
            "element_text": "Login",
            "element_type": "button"
        }
        
        match = await journal_manager.find_matching_pattern(
            sample_test_plan.steps[1],
            context
        )
        
        assert match is not None
        assert match.pattern_type.value == "click"
    
    @pytest.mark.asyncio
    async def test_finalize_journal(self, journal_manager, sample_test_plan, temp_journal_dir):
        """Test finalizing a journal."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        # Add some entries
        for i in range(3):
            result = JournalActionResult(
                success=True,
                action=ActionType.CLICK,
                confidence=0.9,
                actual_outcome="Success"
            )
            await journal_manager.record_action(
                journal_id=journal.journal_id,
                test_scenario="Test",
                step=sample_test_plan.steps[1],
                action_result=result,
                execution_mode=ExecutionMode.VISUAL,
                execution_time_ms=100
            )
        
        # Finalize
        finalized = await journal_manager.finalize_journal(journal.journal_id)
        
        assert finalized.end_time is not None
        assert journal.journal_id not in journal_manager._active_journals
        
        # Check journal was saved
        journal_file = temp_journal_dir / f"{journal.journal_id}.json"
        assert journal_file.exists()
    
    @pytest.mark.asyncio
    async def test_get_journal(self, journal_manager, sample_test_plan):
        """Test getting a journal."""
        # Active journal
        journal = await journal_manager.create_journal(sample_test_plan)
        retrieved = await journal_manager.get_journal(journal.journal_id)
        assert retrieved == journal
        
        # Finalize and get from disk
        await journal_manager.finalize_journal(journal.journal_id)
        retrieved = await journal_manager.get_journal(journal.journal_id)
        assert retrieved is not None
        assert retrieved.journal_id == journal.journal_id
    
    @pytest.mark.asyncio
    async def test_pattern_library_stats(self, journal_manager, sample_test_plan):
        """Test getting pattern library statistics."""
        journal = await journal_manager.create_journal(sample_test_plan)
        
        # Add various patterns
        for i in range(5):
            result = JournalActionResult(
                success=True,
                action=ActionType.CLICK if i < 3 else ActionType.TYPE,
                confidence=0.9,
                playwright_command=f"await page.click('#btn{i}')",
                actual_outcome="Success"
            )
            
            step = TestStep(
                step_id=uuid4(),
                step_number=i+1,
                description=f"Step {i+1}",
                action_instruction=ActionInstruction(
                    action_type=ActionType.CLICK if i < 3 else ActionType.TYPE,
                    description=f"Action {i+1}",
                    expected_outcome="Success"
                )
            )
            
            await journal_manager.record_action(
                journal_id=journal.journal_id,
                test_scenario="Test",
                step=step,
                action_result=result,
                execution_mode=ExecutionMode.VISUAL,
                execution_time_ms=100
            )
        
        stats = await journal_manager.get_pattern_library_stats()
        
        assert stats["total_patterns"] == 5
        assert stats["patterns_by_type"]["click"] == 3
        assert stats["patterns_by_type"]["type"] == 2
        assert len(stats["most_successful"]) <= 5
        assert len(stats["most_used"]) <= 5
    
    @pytest.mark.asyncio
    async def test_shutdown(self, journal_manager, sample_test_plan):
        """Test journal manager shutdown."""
        # Create active journals
        journal1 = await journal_manager.create_journal(sample_test_plan)
        journal2 = await journal_manager.create_journal(sample_test_plan)
        
        # Add patterns
        result = JournalActionResult(
            success=True,
            action=ActionType.CLICK,
            confidence=0.9,
            playwright_command="await page.click('#btn')"
        )
        await journal_manager.record_action(
            journal_id=journal1.journal_id,
            test_scenario="Test",
            step=sample_test_plan.steps[1],
            action_result=result,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=100
        )
        
        # Shutdown
        await journal_manager.shutdown()
        
        # All journals should be finalized
        assert len(journal_manager._active_journals) == 0
        
        # Pattern library should be saved
        pattern_file = journal_manager.journal_dir / "pattern_library.json"
        assert pattern_file.exists()
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_journal_id(self, journal_manager, sample_test_plan, sample_action_result):
        """Test error handling for invalid journal ID."""
        with pytest.raises(ValueError, match="No active journal found"):
            await journal_manager.record_action(
                journal_id=uuid4(),  # Non-existent ID
                test_scenario="Test",
                step=sample_test_plan.steps[0],
                action_result=sample_action_result,
                execution_mode=ExecutionMode.VISUAL,
                execution_time_ms=100
            )