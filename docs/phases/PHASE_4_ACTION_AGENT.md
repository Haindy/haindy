# Phase 4 - Action Agent

## Tasks
Screenshot analysis, adaptive grid refinement, precision coordinate determination.

## ETA
2-3 days

## Status
Completed

## Overview

The Action Agent is responsible for converting visual screenshots and text instructions into precise grid-based coordinates for browser interaction. It implements the adaptive grid refinement system that enables reliable cross-application compatibility without DOM dependencies.

## Agent Definition

From the multi-agent architecture (Section 4.1):

```python
class ActionAgent(BaseAgent):
    """Screenshot + instruction → Adaptive grid coordinates with refinement"""
    def determine_action(self, screenshot: bytes, instruction: str) -> GridAction: pass
    def refine_coordinates(self, cropped_region: bytes, initial_coords: GridCoords) -> RefinedGridAction: pass
```

## Role and Responsibilities

### Primary Functions
- **Visual Analysis**: Analyzes screenshots to identify UI elements
- **Coordinate Mapping**: Converts visual locations to grid coordinates
- **Adaptive Refinement**: Implements precision targeting through grid refinement
- **Confidence Scoring**: Provides confidence levels for action accuracy

### Agent Specialization
From Section 4.3, the Action Agent specializes as:
- Adaptive grid specialist
- Visual refinement expert
- Confidence-based precision targeting

## Adaptive Grid System

### Grid Configuration (Section 5.2)
- **Base Grid**: 60×60 overlay (optimized through testing)
- **Grid Cells**: Numbered/lettered for reference (A1-Z60, AA1-AZ60, etc.)
- **Visual Overlay**: Semi-transparent grid lines for debugging

### DOM-Free Approach (Section 5.1)
The Action Agent explicitly avoids DOM-based interaction methods:
- No XPath, CSS selectors, or element IDs
- Pure visual-based interaction
- Enables no-code accessibility
- Platform-agnostic approach

## Adaptive Refinement Strategy

### Refinement Workflow (Section 5.3)

```
Step 1: Initial Analysis
- AI receives full screenshot with 60×60 grid overlay
- Action Agent analyzes: "Click the 'Add to Cart' button"
- Initial selection: Grid cell M23 (confidence: 70%)

Step 2: Refinement Trigger
- Confidence below threshold (80%) triggers refinement
- System crops 3×3 region: L22, L23, L24, M22, M23, M24, N22, N23, N24
- Cropped region analyzed at higher resolution

Step 3: Precision Targeting
- AI re-analyzes cropped region: "Button clearly visible in center-right of M23"
- Refined coordinates: M23 + offset (0.7, 0.4) relative to cell
- New confidence: 95%

Step 4: Action Execution
- Click executed at refined coordinates
- Action logged with both initial and refined coordinate details
```

### Coordinate System (Section 5.4)
- **Absolute Mapping**: Grid cells mapped to pixel coordinates
- **Fractional Coordinates**: Support for sub-cell precision (e.g., A1.5.7)
- **Adaptive Scaling**: Based on viewport dimensions

## Workflow Position

In the multi-agent execution flow (Section 4.2):

```
TestRunnerAgent: "Step 1: Navigate to product page"
    ↓
ActionAgent: Analyzes screenshot → "Click grid cell B7"
    ↓
BrowserDriver: Executes action
```

## Implementation Details

### Visual Feedback
- **Development Mode**: Grid overlay visible with labels
- **Production Mode**: Hidden overlay, transparent calculation
- **Debugging**: Screenshots include grid references

### Confidence Threshold System (Section 8.2)

| Confidence Level | Action Taken | Escalation |
|------------------|--------------|------------|
| 95-100% | Execute immediately | None |
| 80-94% | Execute with monitoring | Log for review |
| 60-79% | Trigger adaptive refinement | Retry with refinement |
| 40-59% | Request human guidance | Pause for intervention |
| 0-39% | Fail gracefully | Escalate to human |

## Execution Journaling

### Logging Format (Section 6.1)
```json
{
  "grid_coordinates": {
    "initial_selection": "M23",
    "initial_confidence": 0.70,
    "refinement_applied": true,
    "refined_coordinates": "M23+offset(0.7,0.4)",
    "final_confidence": 0.95
  }
}
```

## Error Handling and Validation

### Hallucination Mitigation (Section 8.3)
- Action decisions validated by Evaluator Agent
- Multi-factor confidence calculation
- Visual clarity assessment
- Context matching
- Historical success tracking

### Validation Process
From Section 4.4:
- Agent validates its own outputs
- Cross-agent validation with Evaluator
- Maximum 3 retry attempts with coordination
- Alternative action paths on failure

## Technical Implementation

### File Structure
Located in: `src/agents/action_agent.py`
- Inherits from `BaseAgent` class
- Implements `determine_action` and `refine_coordinates` methods
- Uses grid utilities from `src/grid/`

### Grid System Components
- `src/grid/overlay.py`: Grid overlay utilities
- `src/grid/coordinator.py`: Coordinate mapping and refinement
- `src/grid/refinement.py`: Zoom-in and sub-grid analysis

## Integration with Architecture Refactor

Following the architecture refactor, the Action Agent now owns the execution lifecycle:
- Enhanced debugging capabilities
- Improved error handling
- Better coordination with browser driver

## Best Practices

1. **Confidence-Based Decisions**: Always check confidence before execution
2. **Refinement When Needed**: Use adaptive refinement for precision
3. **Clear Logging**: Document both initial and refined coordinates
4. **Visual Validation**: Ensure screenshots are clear before analysis
5. **Fallback Strategies**: Have alternative approaches for low confidence

## Success Metrics

- Achieves >80% accuracy in coordinate determination
- Successfully triggers refinement when needed
- Maintains high confidence scores (>85% average)
- Works across 5+ different web applications
- Enables reliable cross-platform interaction