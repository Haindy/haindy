# Autonomous AI Testing Agent — High-Level Implementation Plan

## 1  MVP Goals

| # | Goal | Success criterion |
|---|------|-------------------|
| 1 | Accept high-level requirements (PRD, test scenarios) and autonomously execute testing workflows. | Agent successfully completes 3/5 different test scenarios without human intervention. |
| 2 | Multi-agent coordination: Test planning → execution → evaluation → iteration. | Each agent fulfills its role with >80% accuracy in its domain. |
| 3 | Grid-based browser interaction for reliable cross-application compatibility. | Successfully navigates and interacts with 5+ different web applications. |
| 4 | Comprehensive test execution reporting with evidence collection. | Generates detailed reports with screenshots, steps, and outcomes for each test run. |

---

## 2  Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Orchestrator | **Python 3.10+** | Best ecosystem for AI & agent coordination. |
| Browser driver | **Playwright-Python (Chromium)** | WebSocket/CDP, native screenshots & absolute mouse clicks. |
| AI Agents | **4x OpenAI GPT-4o mini instances** | Multi-agent system: Planner, Runner, Action, Evaluator agents. |
| Agent coordination | **Custom multi-agent framework** | Manages agent communication, state, and workflow orchestration. |
| Grid system | **Adaptive grid overlay + coordinate mapping** | DOM-free visual interaction with adaptive refinement (initially 60×60, refined as needed). |
| Test scenarios | **Natural language requirements** | PRDs, user stories, test case descriptions as input. |
| Packaging | `setuptools` + `pyproject.toml` | Simple `pip` distribution. |
| Logs / metrics | JSONL + PNG + [`rich`](https://github.com/Textualize/rich) console | Human-readable and machine-parsable. |

---

## 3  Suggested Folder Layout

```
autonomous-ai-testing-agent/
│
├─ src/
│   ├─ agents/             # AI agent implementations
│   │   ├─ __init__.py
│   │   ├─ base_agent.py   # Abstract base class for all agents
│   │   ├─ test_planner.py # Requirements → Test plan agent
│   │   ├─ test_runner.py  # Test orchestration & coordination agent
│   │   ├─ action_agent.py # Screenshot + instruction → grid coordinates
│   │   └─ evaluator.py    # Screenshot + expectation → success/failure
│   ├─ orchestration/      # Multi-agent coordination
│   │   ├─ __init__.py
│   │   ├─ coordinator.py  # Agent workflow management
│   │   ├─ communication.py # Inter-agent message passing
│   │   └─ state_manager.py # Test execution state tracking
│   ├─ browser/            # Browser automation layer
│   │   ├─ __init__.py
│   │   ├─ driver.py       # Playwright wrapper (click, scroll, screenshot)
│   │   └─ controller.py   # High-level browser operations
│   ├─ grid/
│   │   ├─ __init__.py
│   │   ├─ overlay.py      # Adaptive grid overlay utilities with refinement
│   │   ├─ coordinator.py  # Grid coordinate mapping and adaptive refinement
│   │   ├─ refinement.py   # Zoom-in and sub-grid analysis logic
│   │   └─ utils.py
│   ├─ models/
│   │   ├─ __init__.py
│   │   ├─ openai_client.py # OpenAI API wrapper
│   │   └─ prompts.py      # System prompts for each agent type
│   ├─ core/               # abstraction layers & interfaces
│   │   ├─ __init__.py
│   │   ├─ interfaces.py   # Agent, BrowserDriver, TestExecutor abstracts
│   │   ├─ middleware.py   # logging, monitoring, error handling middleware
│   │   └─ types.py        # Data models (TestPlan, TestStep, ActionResult)
│   ├─ error_handling/
│   │   ├─ __init__.py
│   │   ├─ recovery.py     # retry logic, hallucination detection
│   │   └─ validation.py   # confidence scoring, action validation
│   ├─ security/
│   │   ├─ __init__.py
│   │   ├─ rate_limiter.py # DoS prevention
│   │   └─ sanitizer.py    # sensitive data protection
│   ├─ monitoring/
│   │   ├─ __init__.py
│   │   ├─ logger.py       # structured logging (stdout + file)
│   │   ├─ analytics.py    # success/failure tracking
│   │   └─ reporter.py     # test execution reports generation
│   └─ config/
│       ├─ __init__.py
│       ├─ settings.py     # environment-specific configs
│       ├─ agent_prompts.py # System prompts configuration
│       └─ validation.py   # config validation
│
├─ tests/                  # unit tests & integration tests
├─ test_scenarios/         # example test scenarios & PRDs
├─ data/                   # captured screenshots & logs
├─ reports/                # test execution reports & analytics
├─ docs/                   # this file and extra docs
└─ pyproject.toml
```

---

## 4  Multi-Agent Architecture

### **4.1 AI Agent System**
```python
# Four specialized AI agents working in coordination
class TestPlannerAgent(BaseAgent):
    """Analyzes requirements/PRDs → Creates structured test plans"""
    def create_test_plan(self, requirements: str) -> TestPlan: pass

class TestRunnerAgent(BaseAgent):
    """Orchestrates test execution → Decides next steps"""  
    def get_next_action(self, test_plan: TestPlan, current_state: TestState) -> ActionInstruction: pass
    def evaluate_step_result(self, step_result: StepResult) -> TestState: pass

class ActionAgent(BaseAgent):
    """Screenshot + instruction → Adaptive grid coordinates with refinement"""
    def determine_action(self, screenshot: bytes, instruction: str) -> GridAction: pass
    def refine_coordinates(self, cropped_region: bytes, initial_coords: GridCoords) -> RefinedGridAction: pass

class EvaluatorAgent(BaseAgent):
    """Screenshot + expectation → Success/failure assessment"""
    def evaluate_result(self, screenshot: bytes, expected_outcome: str) -> EvaluationResult: pass
```

### **4.2 Agent Coordination Workflow**
```python
# Multi-agent execution flow
Human Input: "Test checkout flow for Product X"
    ↓
TestPlannerAgent: Creates structured test plan (8 steps)
    ↓
TestRunnerAgent: "Step 1: Navigate to product page"
    ↓
ActionAgent: Analyzes screenshot → "Click grid cell B7"
    ↓
BrowserDriver: Executes action, waits, captures screenshot  
    ↓
EvaluatorAgent: "Success - product page loaded correctly"
    ↓
TestRunnerAgent: Processes result → "Step 2: Add to cart"
    ↓
[Loop continues until test completion]
```

### **4.3 System Prompts & Agent Specialization**
- **Test Planner**: Understands business requirements, creates comprehensive test plans
- **Test Runner**: Maintains test context, coordinates execution, handles branching logic, manages scripted automation fallbacks
- **Action Agent**: Adaptive grid specialist, visual refinement expert, confidence-based precision targeting
- **Evaluator**: Result validation, UI state assessment, error detection, confidence scoring

### **4.4 Error Handling & Recovery**
- **Agent-level error handling**: Each agent validates its own outputs
- **Cross-agent validation**: Results validated by subsequent agents
- **Retry logic**: Max 3 attempts per action with agent coordination
- **Fallback strategies**: Alternative action paths when primary approaches fail

### **4.5 State Management**
- **Test execution state**: Current step, completed steps, remaining steps
- **Browser state tracking**: Page loaded, navigation state, UI changes
- **Agent communication state**: Message passing, result sharing
- **Test plan state**: Dynamic plan updates based on discovered UI changes

### **4.6 Security & Safety**
- **Rate limiting**: Coordinated API throttling across all agents
- **Sensitive data protection**: PII detection before screenshot sharing
- **Sandboxed execution**: Browser isolation, controlled navigation scope

### **4.7 Observability & Debugging**
- **Multi-agent logging**: Track communication between agents
- **Test execution reports**: Complete workflow documentation
- **Agent performance metrics**: Individual agent success rates
- **Visual evidence**: Screenshot captures at each decision point
- **Execution timeline**: Full audit trail with timestamps and decisions

---

## 5  Grid-Based Interaction and Adaptive Refinement

### **5.1 DOM-Free Interaction Approach**

**Core Philosophy**: The system explicitly avoids DOM-based interaction methods (XPath, CSS selectors, element IDs) at all stages of development, including the MVP and future roadmap.

**Rationale**:
- **Brittleness Elimination**: DOM-based selectors introduce maintenance overhead and break easily with UI changes
- **No-Code Accessibility**: Enables users without technical expertise to use the system effectively
- **Platform Extensibility**: Visual-only approach enables future expansion to mobile apps, desktop applications, and platforms without reliable DOM access
- **Reliability**: Visual interaction patterns are more stable across different frameworks and implementations

### **5.2 Adaptive Grid System**

**Initial Grid Configuration**:
- Base grid: 60×60 overlay (tentative, subject to optimization through testing)
- Grid cells numbered/lettered for easy reference (A1-Z60, AA1-AZ60, etc.)
- Visual overlay with semi-transparent grid lines for debugging/development

**Adaptive Refinement Strategy**:
When initial grid selection yields uncertain or borderline results:

1. **Automatic Zoom-In**: Agent crops the selected cell and its 8 neighboring cells
2. **Sub-Grid Analysis**: AI re-analyzes the cropped 3×3 region with higher granularity
3. **Precision Targeting**: Agent pinpoints exact coordinates or corners within the refined area
4. **Confidence Validation**: Higher precision improves confidence scores significantly beyond initial grid limitations

### **5.3 Example Adaptive Refinement Workflow**

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

### **5.4 Grid System Implementation Details**

**Coordinate Mapping**:
- Grid cells mapped to absolute pixel coordinates
- Support for fractional coordinates within cells (e.g., A1.5.7 = 50% right, 70% down in cell A1)
- Adaptive scaling based on viewport dimensions

**Visual Feedback**:
- Development mode: Grid overlay visible with cell labels
- Production mode: Grid overlay hidden, coordinates calculated transparently
- Screenshot capture includes grid reference for debugging

---

## 6  Detailed Execution Journaling

### **6.1 Structured Natural Language Logging**

Each test execution step must generate comprehensive, human-readable logs with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "test_scenario": "E-commerce Checkout Flow",
  "step_reference": "Step 3: Add product to cart",
  "action_taken": "Clicked 'Add to Cart' button using adaptive grid refinement",
  "grid_coordinates": {
    "initial_selection": "M23",
    "initial_confidence": 0.70,
    "refinement_applied": true,
    "refined_coordinates": "M23+offset(0.7,0.4)",
    "final_confidence": 0.95
  },
  "expected_result": "Product added to cart, cart counter increments",
  "actual_result": "Cart counter changed from 0 to 1, success notification appeared",
  "agent_confidence": 0.95,
  "screenshot_before": "screenshots/step_3_before.png",
  "screenshot_after": "screenshots/step_3_after.png",
  "execution_time_ms": 1247,
  "success": true
}
```

### **6.2 Caching and Reuse Strategy**

**Action Pattern Recognition**:
- Successful actions stored as reusable patterns
- Pattern matching for similar UI elements across different test runs
- Adaptive pattern library builds over time

**Execution Journal Benefits**:
- Human-readable test execution history
- Debugging aid for failed test scenarios
- Pattern recognition for UI consistency
- Performance optimization through action caching

---

## 7  Just-In-Time Scripted Automation

### **7.1 Dual-Mode Execution Strategy**

**Primary Mode: Scripted Replay**
- Each successful AI action recorded as explicit WebDriver (Playwright) API calls
- Subsequent test executions attempt direct WebDriver command replay first
- Significantly faster execution for stable UI elements

**Fallback Mode: Visual AI Interaction**
- When scripted commands fail (UI changes, element not found, etc.)
- System seamlessly reverts to visual-grid AI interaction
- Updated action recorded for future use

### **7.2 Action Recording Format**

```python
# Example recorded action
{
  "action_type": "click",
  "playwright_command": "page.click('button[data-testid=\"add-to-cart\"]')",
  "visual_backup": {
    "grid_coordinates": "M23+offset(0.7,0.4)",
    "screenshot_reference": "add_to_cart_button_reference.png",
    "confidence_threshold": 0.85
  },
  "success_criteria": "Cart counter increments",
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### **7.3 Hybrid Execution Flow**

```
1. Load Test Scenario
2. Check for existing scripted actions
3. Attempt scripted replay (WebDriver commands)
4. IF scripted action fails:
   a. Capture current screenshot
   b. Invoke visual AI interaction
   c. Record new action for future use
5. Continue with next step
```

---

## 8  Hierarchical Agent Validation Strategy

### **8.1 Layered Validation Architecture**

**Bottom-Up Validation**:
- Action-level agents return results with confidence scores
- Each agent validates its own output before passing upstream
- Confidence thresholds trigger retry or escalation

**Top-Down Confirmation**:
- Higher-level agents (Test Runner, Planner) verify lower-level outputs
- Contextual consistency checks across agent decisions
- Cross-agent validation prevents cascading errors

### **8.2 Confidence Threshold System**

| Confidence Level | Action Taken | Escalation |
|------------------|--------------|------------|
| 95-100% | Execute immediately | None |
| 80-94% | Execute with monitoring | Log for review |
| 60-79% | Trigger adaptive refinement | Retry with refinement |
| 40-59% | Request human guidance | Pause for intervention |
| 0-39% | Fail gracefully | Escalate to human |

### **8.3 Hallucination Mitigation**

**Cross-Agent Verification**:
- Action Agent decisions validated by Evaluator Agent
- Test Runner Agent maintains execution context consistency
- Planner Agent validates step sequence logical flow

**Confidence Scoring**:
- Multi-factor confidence calculation (visual clarity, context match, historical success)
- Confidence degradation triggers additional validation layers
- Human-in-the-loop triggers for persistent low confidence

**Validation Checkpoints**:
- Pre-action validation: "Is this action appropriate for current context?"
- Post-action validation: "Did the action achieve expected outcome?"
- Sequence validation: "Does current state align with test plan progression?"

---

## 9  Development Phases

| Phase | Tasks | ETA |
|-------|-------|-----|
| **0 — Prep** | Repo scaffold, `pre-commit`, multi-agent project structure. | 1 day |
| **1 — Core foundation** | Base agent class, OpenAI client, data models (TestPlan, TestStep, etc.). | 2 days |
| **2 — Browser & Grid system** | Playwright wrapper, adaptive grid overlay, coordinate mapping with refinement. | 2-3 days |
| **3 — Test Planner Agent** | Requirements analysis, test plan generation, system prompts. | 2-3 days |
| **4 — Action Agent** | Screenshot analysis, adaptive grid refinement, precision coordinate determination. | 2-3 days |
| **5 — Evaluator Agent** | Result assessment, success/failure detection, UI state validation. | 2 days |
| **6 — Test Runner Agent** | Test orchestration, step coordination, execution flow management. | 3 days |
| **7 — Agent Coordination** | Inter-agent communication, state management, workflow orchestration. | 2-3 days |
| **8 — Execution Journaling & Scripted Automation** | Detailed logging, action recording, just-in-time script generation. | 2-3 days |
| **9 — Error Handling & Recovery** | Agent-level error handling, retry logic, fallback strategies. | 2 days |
| **10 — Security & Monitoring** | Rate limiting, logging, analytics, execution reporting. | 2 days |
| **11 — End-to-end Integration** | Complete multi-agent workflow testing and debugging. | 3 days |
| **12 — Test Scenarios** | Create 5 comprehensive test scenarios, measure success rates. | 2-3 days |
| **13 — Packaging & docs** | CLI interface, documentation, release v0.1.0. | 1-2 days |

_Total: roughly **28–36 working days**._

---

## 10  Design Principles

1. **Agent specialization** – Each AI agent has a focused role and expertise domain.  
2. **Coordinated intelligence** – Agents collaborate but maintain independent decision-making.  
3. **Fail-fast with recovery** – Agent-level validation with cross-agent verification and rollback.  
4. **Observable by default** – Every agent communication and decision must be traceable.  
5. **Human-interpretable** – Test plans, actions, and results must be understandable to humans.
6. **Modular & extensible** – Easy to add new agent types or replace existing ones.
7. **Production-ready** – Built for reliability, scale, and real-world testing scenarios.

---

## 11  Post-MVP Roadmap

| Milestone | Description |
|-----------|-------------|
| **v0.2** | Multi-tab support, complex form handling, file upload capabilities. |
| **v0.3** | Specialized agents: API testing agent, mobile testing agent, accessibility testing agent. |
| **v0.4** | Learning & adaptation: Agent fine-tuning based on execution history and failure analysis. |
| **v0.5** | REST API for remote test execution, CI/CD integration, collaborative test planning. |
| **v0.6** | Mobile and desktop application testing, visual regression testing, cross-platform grid adaptation. |

---

## 12  Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Browser update breaks CDP API. | Pin Playwright ≥ 1.45; daily CI smoke tests. |
| API rate limits or downtime. | Implement retry logic, exponential backoff, coordinate rate limiting across agents. |
| Agent coordination failures. | Circuit breaker pattern, fallback to single-agent mode, timeout handling. |
| Infinite loops in test execution. | Global `step_limit`, watchdog timers, agent-level loop detection. |
| Agent hallucination cascade. | Cross-agent validation, confidence thresholds, human-in-loop triggers. |
| Miss-clicks in dense UIs. | Dynamic grid scaling, element detection fallbacks, agent coordination for complex interactions. |

---

## 13  Immediate Next Steps

1. Add this `PLAN.md` to `docs/`.  
2. Create `conda`/`venv` with Python 3.10.  
3. `pip install playwright openai && playwright install chromium`.  
4. Build minimal browser wrapper; verify grid overlay and coordinate mapping.  
5. Implement first Test Planner Agent with basic requirements → test plan functionality.
6. Create initial system prompts for each agent type.
7. Test single-agent workflows before building inter-agent coordination. 