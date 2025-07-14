# Phase 17: Usability and Persistence Improvements

## Overview
This phase focuses on making HAINDY actually usable by implementing proper document persistence, structured test artifacts, and reliable agent behaviors.

## Goals
1. **Test Plan Persistence**: Transform test plans from ephemeral objects to persisted documents
2. **Action Decomposition Storage**: Save the Test Runner's action decompositions for debugging and replay
3. **Structured Reporting**: Create hierarchical reports that mirror the test plan structure
4. **Agent Reliability**: Fine-tune system prompts for consistent, reliable behavior

## Success Criteria
- [ ] Test plans are saved as permanent files (not debug files)
- [ ] Action decompositions are stored and linked to test plans
- [ ] Reports show clear test case → step hierarchy with pass/fail at each level
- [ ] All three agents produce reliable, consistent outputs
- [ ] End-to-end test execution works smoothly without manual intervention

## Implementation Tasks

### 1. Test Plan Document Management
- Create a `TestPlanManager` class to handle test plan persistence
- Save test plans in a structured directory (e.g., `test_plans/YYYY-MM-DD/`)
- Use test plan ID consistently across all artifacts
- Implement test plan versioning for modifications

### 2. Action Decomposition Storage
- Capture Test Runner's step interpretations
- Save decomposed actions with the test plan ID
- Create linkage between original steps and decomposed actions
- Enable action replay from saved decompositions

### 3. Hierarchical Report Structure
- Redesign report to mirror test plan structure exactly
- Show test cases as top-level items
- Display steps under each test case with clear pass/fail indicators
- Keep AI conversations and debug info below the main status view
- Implement both HTML and Markdown report formats with proper hierarchy

### 4. System Prompt Refinement

#### Test Planner Agent
- Ensure consistent test case/step structure
- Improve handling of various input formats
- Add validation for generated test plans
- Remove any tendencies to generate technical selectors

#### Test Runner Agent
- Fix click-before-type issue for input fields
- Improve step interpretation accuracy
- Better handling of expected outcomes per action
- Consistent error categorization

#### Action Agent
- Improve visual element identification
- Better handling of input field interactions
- More accurate grid coordinate targeting
- Clearer validation messages

### 5. File Structure
```
haindy/
├── test_plans/
│   └── 2025-01-14/
│       ├── test_plan_<uuid>.json
│       ├── test_plan_<uuid>_actions.json
│       └── test_plan_<uuid>.md
├── test_reports/
│   └── 2025-01-14/
│       ├── report_<test_plan_uuid>_<timestamp>.html
│       └── report_<test_plan_uuid>_<timestamp>.json
└── test_artifacts/
    └── <test_plan_uuid>/
        ├── screenshots/
        ├── ai_conversations.jsonl
        └── execution_log.jsonl
```

## Technical Design

### Test Plan Persistence
```python
class TestPlanManager:
    """Manages test plan storage and retrieval."""
    
    def save_test_plan(self, test_plan: TestPlan) -> Path:
        """Save test plan to structured directory."""
        # Create date-based directory
        # Save as JSON with proper formatting
        # Generate human-readable markdown version
        # Return path to saved plan
    
    def load_test_plan(self, plan_id: UUID) -> TestPlan:
        """Load test plan by ID."""
        # Search for plan across date directories
        # Load and validate JSON
        # Return TestPlan object
    
    def save_action_decomposition(self, plan_id: UUID, actions: Dict) -> Path:
        """Save Test Runner's decomposed actions."""
        # Link to test plan via ID
        # Store with timestamps and context
        # Enable future replay capability
```

### Report Structure Enhancement
```python
class EnhancedReportGenerator:
    """Generate structured test reports."""
    
    def generate_report(self, test_state: TestState) -> None:
        """Generate hierarchical test report."""
        # Top level: Test Plan summary
        # Second level: Test Cases with status
        # Third level: Steps with pass/fail
        # Fourth level: Action details (collapsed by default)
        # Bottom: AI conversations and debug info
```

## Risks and Mitigations
1. **Risk**: Breaking changes to existing functionality
   - **Mitigation**: Implement changes incrementally with backward compatibility

2. **Risk**: Performance impact from file I/O
   - **Mitigation**: Implement async file operations and caching

3. **Risk**: Storage growth from artifacts
   - **Mitigation**: Implement retention policies and compression

## Timeline
- **Estimated Duration**: 3-4 days
- **Day 1**: Test plan persistence and management
- **Day 2**: Action decomposition storage and linking
- **Day 3**: Report structure enhancement
- **Day 4**: System prompt refinement and testing

## Definition of Done
- [ ] Test plans are automatically saved to structured directories
- [ ] Action decompositions are captured and linked to test plans
- [ ] Reports show clear hierarchical structure with appropriate detail levels
- [ ] All agents behave reliably without manual intervention
- [ ] Documentation updated with new workflows
- [ ] Integration tests pass with new persistence layer