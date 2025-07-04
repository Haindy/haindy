"""
Data models for execution journaling and scripted automation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# Import ActionType from core
from src.core.types import ActionType


class ExecutionMode(str, Enum):
    """Mode of test execution."""
    VISUAL = "visual"      # AI-driven visual interaction
    SCRIPTED = "scripted"  # Direct WebDriver commands
    HYBRID = "hybrid"      # Mix of both modes


class PatternType(str, Enum):
    """Types of action patterns."""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


class JournalActionResult(BaseModel):
    """Action result for journaling system."""
    
    success: bool
    action: Optional[ActionType] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    coordinates: Optional[Tuple[int, int]] = None
    grid_coordinates: Optional[Dict[str, Any]] = None
    playwright_command: Optional[str] = None
    selectors: Optional[Dict[str, str]] = None
    input_text: Optional[str] = None
    element_text: Optional[str] = None
    actual_outcome: Optional[str] = None
    error_message: Optional[str] = None


class JournalEntry(BaseModel):
    """A single test execution journal entry."""
    
    entry_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    test_scenario: str
    step_reference: str
    action_taken: str
    
    # Grid coordinates for visual mode
    grid_coordinates: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Grid coordinates used for visual interaction"
    )
    
    # Scripted command for replay mode
    scripted_command: Optional[str] = Field(
        default=None,
        description="Playwright command for direct execution"
    )
    
    # Selectors discovered during execution
    selectors: Optional[Dict[str, str]] = Field(
        default=None,
        description="CSS/XPath selectors found for the element"
    )
    
    # Execution details
    execution_mode: ExecutionMode = ExecutionMode.VISUAL
    execution_time_ms: int = 0
    agent_confidence: float = 0.0
    
    # Results
    expected_result: str
    actual_result: str
    success: bool = True
    error_message: Optional[str] = None
    
    # Evidence
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    
    # Pattern matching data
    pattern_id: Optional[UUID] = Field(
        default=None,
        description="ID of matched pattern if reused"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ActionRecord(BaseModel):
    """Recorded action for pattern matching and reuse."""
    
    record_id: UUID = Field(default_factory=uuid4)
    pattern_type: PatternType
    
    # Visual pattern data
    visual_signature: Dict[str, Any] = Field(
        default_factory=dict,
        description="Visual characteristics for pattern matching"
    )
    
    # Scripted replay data
    playwright_command: str
    selectors: Dict[str, str] = Field(default_factory=dict)
    fallback_commands: List[str] = Field(
        default_factory=list,
        description="Alternative commands if primary fails"
    )
    
    # Context and metadata
    url_pattern: Optional[str] = None
    element_type: Optional[str] = None
    element_text: Optional[str] = None
    
    # Performance data
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time_ms: float = 0.0
    last_used: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class PatternMatch(BaseModel):
    """Result of pattern matching attempt."""
    
    pattern_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    match_type: str  # exact, similar, partial
    adjustments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Required adjustments to apply pattern"
    )


class ScriptedCommand(BaseModel):
    """A scripted command ready for execution."""
    
    command_type: str  # click, type, navigate, etc.
    command: str       # The actual Playwright command
    selectors: List[str] = Field(
        default_factory=list,
        description="Ordered list of selectors to try"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30000
    retry_count: int = 3
    
    # Fallback to visual mode criteria
    fallback_threshold: float = Field(
        default=0.8,
        description="Confidence threshold to trigger visual fallback"
    )
    allow_visual_fallback: bool = True


class ExecutionJournal(BaseModel):
    """Complete execution journal for a test run."""
    
    journal_id: UUID = Field(default_factory=uuid4)
    test_plan_id: UUID
    test_name: str
    
    # Execution timeline
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Entries
    entries: List[JournalEntry] = Field(default_factory=list)
    
    # Patterns discovered
    discovered_patterns: List[ActionRecord] = Field(default_factory=list)
    reused_patterns: List[UUID] = Field(default_factory=list)
    
    # Statistics
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    visual_interactions: int = 0
    scripted_interactions: int = 0
    
    # Performance metrics
    total_execution_time_ms: int = 0
    avg_step_time_ms: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def add_entry(self, entry: JournalEntry) -> None:
        """Add an entry to the journal."""
        self.entries.append(entry)
        self.total_steps += 1
        
        if entry.success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
        
        if entry.execution_mode == ExecutionMode.VISUAL:
            self.visual_interactions += 1
        elif entry.execution_mode == ExecutionMode.SCRIPTED:
            self.scripted_interactions += 1
        
        self.total_execution_time_ms += entry.execution_time_ms
        if self.total_steps > 0:
            self.avg_step_time_ms = self.total_execution_time_ms / self.total_steps
    
    def finalize(self) -> None:
        """Finalize the journal at test completion."""
        self.end_time = datetime.utcnow()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the journal."""
        duration = None
        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "test_name": self.test_name,
            "total_steps": self.total_steps,
            "success_rate": self.successful_steps / self.total_steps if self.total_steps > 0 else 0,
            "execution_modes": {
                "visual": self.visual_interactions,
                "scripted": self.scripted_interactions
            },
            "patterns": {
                "discovered": len(self.discovered_patterns),
                "reused": len(self.reused_patterns)
            },
            "performance": {
                "total_time_seconds": duration,
                "avg_step_time_ms": self.avg_step_time_ms
            }
        }