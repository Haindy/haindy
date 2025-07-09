"""
Journal manager for recording and managing test execution journals.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from src.core.types import TestPlan, TestStep
from src.core.enhanced_types import EnhancedActionResult
from src.journal.models import (
    ActionRecord,
    ExecutionJournal,
    ExecutionMode,
    JournalActionResult,
    JournalEntry,
    PatternType
)
from src.journal.pattern_matcher import PatternMatcher
from src.journal.adapters import enhanced_to_journal_action_result, extract_journal_context
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class JournalManager:
    """
    Manages execution journals for test runs.
    
    Records all test actions, manages pattern library,
    and provides journal persistence.
    """
    
    def __init__(self, journal_dir: Optional[Path] = None):
        """
        Initialize the journal manager.
        
        Args:
            journal_dir: Directory to store journal files
        """
        self.journal_dir = journal_dir or Path("data/journals")
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # Active journals
        self._active_journals: Dict[UUID, ExecutionJournal] = {}
        
        # Pattern library
        self._pattern_library: Dict[UUID, ActionRecord] = {}
        self._pattern_matcher = PatternMatcher()
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Load existing patterns
        self._load_pattern_library()
        
        logger.info(f"Journal manager initialized with directory: {self.journal_dir}")
    
    async def create_journal(self, test_plan: TestPlan) -> ExecutionJournal:
        """
        Create a new execution journal for a test plan.
        
        Args:
            test_plan: The test plan to journal
            
        Returns:
            New execution journal
        """
        async with self._lock:
            journal = ExecutionJournal(
                test_plan_id=test_plan.plan_id,
                test_name=test_plan.name
            )
            
            self._active_journals[journal.journal_id] = journal
            logger.info(f"Created journal for test: {test_plan.name}")
            
            return journal
    
    async def record_action(
        self,
        journal_id: UUID,
        test_scenario: str,
        step: TestStep,
        action_result: Union[JournalActionResult, EnhancedActionResult],
        execution_mode: ExecutionMode,
        execution_time_ms: Optional[int] = None,
        screenshot_before: Optional[str] = None,
        screenshot_after: Optional[str] = None
    ) -> JournalEntry:
        """
        Record an action in the journal.
        
        Args:
            journal_id: ID of the journal
            test_scenario: Current test scenario name
            step: The test step executed
            action_result: Result of the action (JournalActionResult or EnhancedActionResult)
            execution_mode: How the action was executed
            execution_time_ms: Execution time in milliseconds (extracted from EnhancedActionResult if not provided)
            screenshot_before: Path to before screenshot
            screenshot_after: Path to after screenshot
            
        Returns:
            The created journal entry
        """
        async with self._lock:
            journal = self._active_journals.get(journal_id)
            if not journal:
                raise ValueError(f"No active journal found: {journal_id}")
            
            # Convert EnhancedActionResult to JournalActionResult if needed
            if isinstance(action_result, EnhancedActionResult):
                # Extract execution time from enhanced result if not provided
                if execution_time_ms is None and action_result.execution:
                    execution_time_ms = int(action_result.execution.execution_time_ms)
                
                # Convert to journal format
                journal_result = enhanced_to_journal_action_result(action_result)
                
                # Extract additional context
                journal_context = extract_journal_context(action_result)
                
                logger.debug("Converted EnhancedActionResult to JournalActionResult", extra={
                    "enhanced_success": action_result.overall_success,
                    "journal_success": journal_result.success,
                    "execution_time_ms": execution_time_ms
                })
            else:
                # Already in journal format
                journal_result = action_result
                journal_context = {}
            
            # Fallback execution time if still not available
            if execution_time_ms is None:
                execution_time_ms = 0
            
            # Create journal entry
            entry = JournalEntry(
                test_scenario=test_scenario,
                step_reference=f"Step {step.step_number}: {step.description}",
                action_taken=self._describe_action(step, journal_result),
                grid_coordinates=journal_result.grid_coordinates if execution_mode == ExecutionMode.VISUAL else None,
                scripted_command=journal_result.playwright_command,
                selectors=journal_result.selectors,
                execution_mode=execution_mode,
                execution_time_ms=execution_time_ms,
                agent_confidence=journal_result.confidence,
                expected_result=step.action_instruction.expected_outcome,
                actual_result=journal_result.actual_outcome or "Unknown",
                success=journal_result.success,
                error_message=journal_result.error_message,
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after
            )
            
            # Add to journal
            journal.add_entry(entry)
            
            # Record pattern if successful and confidence is high
            if journal_result.success and journal_result.confidence > 0.85:
                pattern = await self._create_action_pattern(step, journal_result, entry)
                if pattern:
                    self._pattern_library[pattern.record_id] = pattern
                    journal.discovered_patterns.append(pattern)
            
            logger.debug(f"Recorded action: {entry.action_taken}")
            return entry
    
    async def find_matching_pattern(
        self,
        step: TestStep,
        context: Dict[str, Any]
    ) -> Optional[ActionRecord]:
        """
        Find a matching pattern for a test step.
        
        Args:
            step: The test step to match
            context: Current execution context
            
        Returns:
            Matching action record if found
        """
        # Extract pattern features from step
        pattern_type = self._get_pattern_type(step)
        features = {
            "action_type": step.action_instruction.action_type,
            "description": step.action_instruction.description,
            "element_text": context.get("element_text"),
            "url_pattern": context.get("url"),
            "element_type": context.get("element_type")
        }
        
        # Find best match
        best_match = None
        best_confidence = 0.0
        
        for pattern in self._pattern_library.values():
            if pattern.pattern_type != pattern_type:
                continue
            
            match = self._pattern_matcher.match_pattern(pattern, features)
            if match and match.confidence > best_confidence:
                best_match = pattern
                best_confidence = match.confidence
        
        if best_match and best_confidence > 0.7:
            logger.info(f"Found matching pattern with confidence: {best_confidence}")
            return best_match
        
        return None
    
    async def finalize_journal(self, journal_id: UUID) -> ExecutionJournal:
        """
        Finalize and save a journal.
        
        Args:
            journal_id: ID of the journal to finalize
            
        Returns:
            The finalized journal
        """
        async with self._lock:
            journal = self._active_journals.get(journal_id)
            if not journal:
                raise ValueError(f"No active journal found: {journal_id}")
            
            # Finalize journal
            journal.finalize()
            
            # Save to disk
            await self._save_journal(journal)
            
            # Save updated pattern library
            await self._save_pattern_library()
            
            # Remove from active journals
            del self._active_journals[journal_id]
            
            logger.info(f"Finalized journal: {journal.test_name}")
            return journal
    
    async def get_journal(self, journal_id: UUID) -> Optional[ExecutionJournal]:
        """Get a journal by ID."""
        # Check active journals first
        if journal_id in self._active_journals:
            return self._active_journals[journal_id]
        
        # Try to load from disk
        journal_file = self.journal_dir / f"{journal_id}.json"
        if journal_file.exists():
            return await self._load_journal(journal_file)
        
        return None
    
    async def get_pattern_library_stats(self) -> Dict[str, Any]:
        """Get statistics about the pattern library."""
        stats = {
            "total_patterns": len(self._pattern_library),
            "patterns_by_type": {},
            "most_successful": [],
            "most_used": []
        }
        
        # Count by type
        for pattern in self._pattern_library.values():
            pattern_type = pattern.pattern_type
            if pattern_type not in stats["patterns_by_type"]:
                stats["patterns_by_type"][pattern_type] = 0
            stats["patterns_by_type"][pattern_type] += 1
        
        # Find most successful (highest success rate)
        patterns_with_usage = [
            p for p in self._pattern_library.values()
            if p.success_count + p.failure_count > 0
        ]
        
        patterns_with_usage.sort(
            key=lambda p: p.success_count / (p.success_count + p.failure_count),
            reverse=True
        )
        stats["most_successful"] = [
            {
                "pattern_id": str(p.record_id),
                "type": p.pattern_type,
                "success_rate": p.success_count / (p.success_count + p.failure_count),
                "total_uses": p.success_count + p.failure_count
            }
            for p in patterns_with_usage[:5]
        ]
        
        # Find most used
        patterns_with_usage.sort(
            key=lambda p: p.success_count + p.failure_count,
            reverse=True
        )
        stats["most_used"] = [
            {
                "pattern_id": str(p.record_id),
                "type": p.pattern_type,
                "total_uses": p.success_count + p.failure_count,
                "success_rate": p.success_count / (p.success_count + p.failure_count)
            }
            for p in patterns_with_usage[:5]
        ]
        
        return stats
    
    def _describe_action(self, step: TestStep, result: JournalActionResult) -> str:
        """Create a human-readable description of the action taken."""
        action_type = step.action_instruction.action_type
        description = step.action_instruction.description
        
        if result.grid_coordinates:
            coords = result.grid_coordinates
            if coords.get("refined_coordinates"):
                return f"{description} at {coords['refined_coordinates']} (refined from {coords['initial_selection']})"
            else:
                return f"{description} at {coords['initial_selection']}"
        else:
            return description
    
    def _get_pattern_type(self, step: TestStep) -> PatternType:
        """Determine pattern type from test step."""
        action_type = step.action_instruction.action_type.lower()
        
        mapping = {
            "click": PatternType.CLICK,
            "type": PatternType.TYPE,
            "navigate": PatternType.NAVIGATE,
            "scroll": PatternType.SCROLL,
            "wait": PatternType.WAIT,
            "screenshot": PatternType.SCREENSHOT
        }
        
        return mapping.get(action_type, PatternType.CLICK)
    
    async def _create_action_pattern(
        self,
        step: TestStep,
        result: JournalActionResult,
        entry: JournalEntry
    ) -> Optional[ActionRecord]:
        """Create an action pattern from successful execution."""
        if not result.playwright_command:
            return None
        
        pattern = ActionRecord(
            pattern_type=self._get_pattern_type(step),
            visual_signature={
                "action_type": step.action_instruction.action_type,
                "description": step.action_instruction.description,
                "grid_coordinates": result.grid_coordinates
            },
            playwright_command=result.playwright_command,
            selectors=result.selectors or {},
            element_text=result.element_text,
            success_count=1,
            avg_execution_time_ms=float(entry.execution_time_ms),
            last_used=datetime.now(timezone.utc)
        )
        
        logger.debug(f"Created action pattern: {pattern.pattern_type}")
        return pattern
    
    async def _save_journal(self, journal: ExecutionJournal) -> None:
        """Save journal to disk."""
        journal_file = self.journal_dir / f"{journal.journal_id}.json"
        
        try:
            with open(journal_file, "w") as f:
                json.dump(journal.model_dump(), f, indent=2, default=str)
            logger.debug(f"Saved journal to: {journal_file}")
        except Exception as e:
            logger.error(f"Failed to save journal: {e}")
    
    async def _load_journal(self, journal_file: Path) -> Optional[ExecutionJournal]:
        """Load journal from disk."""
        try:
            with open(journal_file, "r") as f:
                data = json.load(f)
            return ExecutionJournal(**data)
        except Exception as e:
            logger.error(f"Failed to load journal: {e}")
            return None
    
    def _load_pattern_library(self) -> None:
        """Load pattern library from disk."""
        pattern_file = self.journal_dir / "pattern_library.json"
        
        if pattern_file.exists():
            try:
                with open(pattern_file, "r") as f:
                    data = json.load(f)
                
                for pattern_data in data.get("patterns", []):
                    pattern = ActionRecord(**pattern_data)
                    self._pattern_library[pattern.record_id] = pattern
                
                logger.info(f"Loaded {len(self._pattern_library)} patterns")
            except Exception as e:
                logger.error(f"Failed to load pattern library: {e}")
    
    async def _save_pattern_library(self) -> None:
        """Save pattern library to disk."""
        pattern_file = self.journal_dir / "pattern_library.json"
        
        try:
            data = {
                "patterns": [
                    pattern.model_dump()
                    for pattern in self._pattern_library.values()
                ],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(pattern_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug("Saved pattern library")
        except Exception as e:
            logger.error(f"Failed to save pattern library: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the journal manager."""
        # Finalize any active journals
        journal_ids = list(self._active_journals.keys())
        for journal_id in journal_ids:
            try:
                await self.finalize_journal(journal_id)
            except Exception as e:
                logger.error(f"Failed to finalize journal {journal_id}: {e}")
        
        # Save pattern library
        await self._save_pattern_library()
        
        logger.info("Journal manager shutdown complete")