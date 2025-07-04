"""
Tests for journal data models.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.journal.models import (
    ActionRecord,
    ExecutionJournal,
    ExecutionMode,
    JournalEntry,
    PatternMatch,
    PatternType,
    ScriptedCommand
)


class TestJournalEntry:
    """Test cases for JournalEntry model."""
    
    def test_create_journal_entry(self):
        """Test creating a journal entry."""
        entry = JournalEntry(
            test_scenario="Login Test",
            step_reference="Step 1: Navigate to login",
            action_taken="Navigated to login page",
            expected_result="Login page displayed",
            actual_result="Login page loaded successfully",
            success=True,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=1500,
            agent_confidence=0.95
        )
        
        assert entry.test_scenario == "Login Test"
        assert entry.success is True
        assert entry.execution_mode == ExecutionMode.VISUAL
        assert entry.agent_confidence == 0.95
        assert isinstance(entry.entry_id, type(uuid4()))
        assert isinstance(entry.timestamp, datetime)
    
    def test_journal_entry_with_grid_coordinates(self):
        """Test journal entry with grid coordinates."""
        grid_coords = {
            "initial_selection": "M23",
            "initial_confidence": 0.7,
            "refined_coordinates": "M23+offset(0.7,0.4)",
            "final_confidence": 0.95
        }
        
        entry = JournalEntry(
            test_scenario="Test",
            step_reference="Step 1",
            action_taken="Clicked button",
            grid_coordinates=grid_coords,
            expected_result="Success",
            actual_result="Success",
            execution_mode=ExecutionMode.VISUAL
        )
        
        assert entry.grid_coordinates == grid_coords
        assert entry.scripted_command is None
    
    def test_journal_entry_with_scripted_command(self):
        """Test journal entry with scripted command."""
        entry = JournalEntry(
            test_scenario="Test",
            step_reference="Step 1",
            action_taken="Clicked button",
            scripted_command="await page.click('#submit-btn')",
            selectors={"primary": "#submit-btn", "fallback": "button[type=submit]"},
            expected_result="Success",
            actual_result="Success",
            execution_mode=ExecutionMode.SCRIPTED
        )
        
        assert entry.scripted_command == "await page.click('#submit-btn')"
        assert entry.selectors["primary"] == "#submit-btn"
        assert entry.grid_coordinates is None


class TestActionRecord:
    """Test cases for ActionRecord model."""
    
    def test_create_action_record(self):
        """Test creating an action record."""
        record = ActionRecord(
            pattern_type=PatternType.CLICK,
            visual_signature={"action_type": "click", "description": "Click login"},
            playwright_command="await page.click('#login-btn')",
            selectors={"primary": "#login-btn"},
            success_count=5,
            failure_count=1,
            avg_execution_time_ms=250.5
        )
        
        assert record.pattern_type == PatternType.CLICK
        assert record.success_count == 5
        assert record.failure_count == 1
        assert record.avg_execution_time_ms == 250.5
        assert isinstance(record.record_id, type(uuid4()))
    
    def test_action_record_with_fallbacks(self):
        """Test action record with fallback commands."""
        record = ActionRecord(
            pattern_type=PatternType.TYPE,
            visual_signature={},
            playwright_command="await page.fill('#username', 'test')",
            selectors={"primary": "#username"},
            fallback_commands=[
                "await page.fill('input[name=username]', 'test')",
                "await page.fill('input[type=text]:first', 'test')"
            ]
        )
        
        assert len(record.fallback_commands) == 2
        assert record.fallback_commands[0] == "await page.fill('input[name=username]', 'test')"


class TestPatternMatch:
    """Test cases for PatternMatch model."""
    
    def test_create_pattern_match(self):
        """Test creating a pattern match."""
        pattern_id = uuid4()
        match = PatternMatch(
            pattern_id=pattern_id,
            confidence=0.85,
            match_type="similar",
            adjustments={"selector": "modified"}
        )
        
        assert match.pattern_id == pattern_id
        assert match.confidence == 0.85
        assert match.match_type == "similar"
        assert match.adjustments["selector"] == "modified"
    
    def test_pattern_match_confidence_validation(self):
        """Test pattern match confidence validation."""
        # Valid confidence
        match = PatternMatch(
            pattern_id=uuid4(),
            confidence=0.5,
            match_type="partial"
        )
        assert match.confidence == 0.5
        
        # Invalid confidence should raise
        with pytest.raises(ValueError):
            PatternMatch(
                pattern_id=uuid4(),
                confidence=1.5,  # > 1.0
                match_type="exact"
            )


class TestScriptedCommand:
    """Test cases for ScriptedCommand model."""
    
    def test_create_scripted_command(self):
        """Test creating a scripted command."""
        cmd = ScriptedCommand(
            command_type="click",
            command="await page.click('#submit')",
            selectors=["#submit", "button[type=submit]", "button:has-text('Submit')"],
            parameters={"timeout": 30000},
            retry_count=3
        )
        
        assert cmd.command_type == "click"
        assert cmd.command == "await page.click('#submit')"
        assert len(cmd.selectors) == 3
        assert cmd.timeout_ms == 30000
        assert cmd.retry_count == 3
        assert cmd.allow_visual_fallback is True
    
    def test_scripted_command_fallback_settings(self):
        """Test scripted command fallback settings."""
        cmd = ScriptedCommand(
            command_type="type",
            command="await page.fill('#input', 'text')",
            fallback_threshold=0.9,
            allow_visual_fallback=False
        )
        
        assert cmd.fallback_threshold == 0.9
        assert cmd.allow_visual_fallback is False


class TestExecutionJournal:
    """Test cases for ExecutionJournal model."""
    
    def test_create_execution_journal(self):
        """Test creating an execution journal."""
        test_plan_id = uuid4()
        journal = ExecutionJournal(
            test_plan_id=test_plan_id,
            test_name="Login Test"
        )
        
        assert journal.test_plan_id == test_plan_id
        assert journal.test_name == "Login Test"
        assert len(journal.entries) == 0
        assert journal.total_steps == 0
        assert isinstance(journal.journal_id, type(uuid4()))
        assert isinstance(journal.start_time, datetime)
    
    def test_add_entry_to_journal(self):
        """Test adding entries to journal."""
        journal = ExecutionJournal(
            test_plan_id=uuid4(),
            test_name="Test"
        )
        
        # Add successful entry
        entry1 = JournalEntry(
            test_scenario="Test",
            step_reference="Step 1",
            action_taken="Action 1",
            expected_result="Success",
            actual_result="Success",
            success=True,
            execution_mode=ExecutionMode.SCRIPTED,
            execution_time_ms=100
        )
        journal.add_entry(entry1)
        
        assert journal.total_steps == 1
        assert journal.successful_steps == 1
        assert journal.failed_steps == 0
        assert journal.scripted_interactions == 1
        assert journal.visual_interactions == 0
        assert journal.avg_step_time_ms == 100
        
        # Add failed visual entry
        entry2 = JournalEntry(
            test_scenario="Test",
            step_reference="Step 2",
            action_taken="Action 2",
            expected_result="Success",
            actual_result="Failed",
            success=False,
            execution_mode=ExecutionMode.VISUAL,
            execution_time_ms=300
        )
        journal.add_entry(entry2)
        
        assert journal.total_steps == 2
        assert journal.successful_steps == 1
        assert journal.failed_steps == 1
        assert journal.scripted_interactions == 1
        assert journal.visual_interactions == 1
        assert journal.avg_step_time_ms == 200  # (100 + 300) / 2
    
    def test_finalize_journal(self):
        """Test finalizing a journal."""
        journal = ExecutionJournal(
            test_plan_id=uuid4(),
            test_name="Test"
        )
        
        assert journal.end_time is None
        
        journal.finalize()
        
        assert journal.end_time is not None
        assert journal.end_time >= journal.start_time
    
    def test_journal_summary(self):
        """Test getting journal summary."""
        journal = ExecutionJournal(
            test_plan_id=uuid4(),
            test_name="Test Suite"
        )
        
        # Add some entries
        for i in range(5):
            journal.add_entry(JournalEntry(
                test_scenario="Test",
                step_reference=f"Step {i+1}",
                action_taken=f"Action {i+1}",
                expected_result="Success",
                actual_result="Success",
                success=i != 2,  # Make step 3 fail
                execution_mode=ExecutionMode.SCRIPTED if i < 3 else ExecutionMode.VISUAL,
                execution_time_ms=100 * (i + 1)
            ))
        
        # Add discovered patterns
        journal.discovered_patterns = [
            ActionRecord(pattern_type=PatternType.CLICK, visual_signature={}, playwright_command="cmd1"),
            ActionRecord(pattern_type=PatternType.TYPE, visual_signature={}, playwright_command="cmd2")
        ]
        journal.reused_patterns = [uuid4(), uuid4(), uuid4()]
        
        journal.finalize()
        
        summary = journal.get_summary()
        
        assert summary["test_name"] == "Test Suite"
        assert summary["total_steps"] == 5
        assert summary["success_rate"] == 0.8  # 4/5
        assert summary["execution_modes"]["scripted"] == 3
        assert summary["execution_modes"]["visual"] == 2
        assert summary["patterns"]["discovered"] == 2
        assert summary["patterns"]["reused"] == 3
        assert summary["performance"]["avg_step_time_ms"] == 300  # (100+200+300+400+500)/5
        assert summary["performance"]["total_time_seconds"] is not None
    
    def test_journal_with_patterns(self):
        """Test journal with pattern tracking."""
        journal = ExecutionJournal(
            test_plan_id=uuid4(),
            test_name="Pattern Test"
        )
        
        # Add discovered pattern
        pattern = ActionRecord(
            pattern_type=PatternType.CLICK,
            visual_signature={"action": "click"},
            playwright_command="await page.click('#btn')"
        )
        journal.discovered_patterns.append(pattern)
        
        # Track pattern reuse
        journal.reused_patterns.append(pattern.record_id)
        journal.reused_patterns.append(pattern.record_id)
        
        assert len(journal.discovered_patterns) == 1
        assert len(journal.reused_patterns) == 2
        assert journal.discovered_patterns[0].pattern_type == PatternType.CLICK