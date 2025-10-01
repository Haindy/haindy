# Autonomous AI Testing Agent — High-Level Implementation Plan

## 1  MVP Goals

| # | Goal | Success criterion | Status |
|---|------|-------------------|---------|
| 1 | Accept high-level requirements (PRD, test scenarios) and autonomously execute testing workflows. | Agent successfully completes 3/5 different test scenarios without human intervention. | 🔄 In Progress |
| 2 | Multi-agent coordination: Test planning → execution → evaluation → iteration. | Each agent fulfills its role with >80% accuracy in its domain. | ✅ Achieved |
| 3 | Grid-based browser interaction for reliable cross-application compatibility. | Successfully navigates and interacts with 5+ different web applications. | ✅ Achieved |
| 4 | Comprehensive test execution reporting with evidence collection. | Generates detailed reports with screenshots, steps, and outcomes for each test run. | ✅ Achieved |
| 5 | Debug visibility and AI interaction logging. | Complete visibility into AI prompts, responses, and decision-making process. | ✅ Achieved |

---

## 2  Tech Stack

| Layer | Technology | Why | Status |
|-------|------------|-----|--------|
| Orchestrator | **Python 3.10+** | Best ecosystem for AI & agent coordination. | ✅ |
| Browser driver | **Playwright-Python (Chromium)** | WebSocket/CDP, native screenshots & absolute mouse clicks. | ✅ |
| AI Agents | **GPT-5 family** | All agents run on GPT-5 with role-specific reasoning levels and modalities. | ✅ Modern model mix |
| Agent coordination | **Custom multi-agent framework** | Manages agent communication, state, and workflow orchestration. | ✅ |
| Grid system | **Adaptive grid overlay + coordinate mapping** | DOM-free visual interaction with adaptive refinement (60×60 grid). | ✅ |
| Test scenarios | **Natural language requirements** | PRDs, user stories, test case descriptions as input. | ✅ |
| Debug logging | **JSONL + HTML reports** | AI conversations, screenshots, comprehensive debugging. | ✅ |
| Packaging | `setuptools` + `pyproject.toml` | Simple `pip` distribution. | ✅ |

---

### Model Configuration (Oct 2025)

| Agent | Model | Reasoning Level | Temperature |
|-------|-------|-----------------|-------------|
| Test Planner | `gpt-5` | High | 0.35 |
| Test Runner | `gpt-5` | Medium | 0.55 |
| Action Agent | `gpt-5` | Low (vision-enabled, escalates as needed) | 0.25 |

Per-agent overrides live in `.env` (`HAINDY_TEST_*` variables) and fall back to
global `OPENAI_MODEL`/`OPENAI_TEMPERATURE` for compatibility.

## 3  Folder Structure

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
# Three specialized AI agents working in coordination
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

# Note: Evaluation is now handled within Action Agent
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
TestRunnerAgent: Processes result → "Step 2: Add to cart"
    ↓
[Loop continues until test completion]
```

### **4.3 System Prompts & Agent Specialization**
- **Test Planner**: Understands business requirements, creates comprehensive test plans
- **Test Runner**: Maintains test context, coordinates execution, handles branching logic, manages scripted automation fallbacks
- **Action Agent**: Adaptive grid specialist, visual refinement expert, confidence-based precision targeting
- **Action Agent**: Also handles result validation, UI state assessment, error detection

---

## 5  Key System Features

### **5.1 DOM-Free Visual Interaction**
- Explicitly avoids DOM-based interaction methods (XPath, CSS selectors)
- Enables no-code accessibility and platform extensibility
- Visual-only approach for reliability across different frameworks

### **5.2 Adaptive Grid System**
- Base 60×60 grid overlay with automatic refinement
- Confidence-based precision targeting
- Support for fractional coordinates within cells

### **5.3 Comprehensive Observability**
- Structured execution journaling with screenshots
- AI decision logging and debugging
- HTML reports with embedded evidence

### **5.4 Dual-Mode Execution**
- Primary: Scripted replay for stable UI elements
- Fallback: Visual AI interaction when scripts fail
- Automatic action recording for future optimization

---

## 6  Design Principles

1. **Agent specialization** – Each AI agent has a focused role and expertise domain.  
2. **Coordinated intelligence** – Agents collaborate but maintain independent decision-making.  
3. **Fail-fast with recovery** – Agent-level validation with cross-agent verification and rollback.  
4. **Observable by default** – Every agent communication and decision must be traceable.  
5. **Human-interpretable** – Test plans, actions, and results must be understandable to humans.
6. **Modular & extensible** – Easy to add new agent types or replace existing ones.
7. **Production-ready** – Built for reliability, scale, and real-world testing scenarios.

---

## 7  Development Phases

### ✅ Completed Phases (20/23)

For detailed information about completed phases, see the individual phase documents:

| Phase | Description | Documentation |
|-------|-------------|---------------|
| **Phase 0** | Repository Scaffold | [PHASE_0_PREP.md](./phases/PHASE_0_PREP.md) |
| **Phase 1** | Core Foundation | [PHASE_1_CORE_FOUNDATION.md](./phases/PHASE_1_CORE_FOUNDATION.md) |
| **Phase 2** | Browser & Grid System | [PHASE_2_BROWSER_GRID.md](./phases/PHASE_2_BROWSER_GRID.md) |
| **Phase 3** | Test Planner Agent | [PHASE_3_TEST_PLANNER.md](./phases/PHASE_3_TEST_PLANNER.md) |
| **Phase 4** | Action Agent | [PHASE_4_ACTION_AGENT.md](./phases/PHASE_4_ACTION_AGENT.md) |
| **Phase 5** | Evaluator Agent | [PHASE_5_EVALUATOR.md](./phases/PHASE_5_EVALUATOR.md) |
| **Phase 6** | Test Runner Agent | [PHASE_6_TEST_RUNNER.md](./phases/PHASE_6_TEST_RUNNER.md) |
| **Phase 7** | Agent Coordination | [PHASE_7_AGENT_COORDINATION.md](./phases/PHASE_7_AGENT_COORDINATION.md) |
| **Phase 8** | Execution Journaling | [PHASE_8_EXECUTION_JOURNALING.md](./phases/PHASE_8_EXECUTION_JOURNALING.md) |
| **Phase 9** | Error Handling | [PHASE_9_ERROR_HANDLING_RECOVERY.md](./phases/PHASE_9_ERROR_HANDLING_RECOVERY.md) |
| **Phase 10** | Security & Monitoring | [PHASE_10_SECURITY_MONITORING.md](./phases/PHASE_10_SECURITY_MONITORING.md) |
| **Phase 10b** | Test Suite Cleanup | [PHASE_10B_TEST_SUITE_CLEANUP.md](./phases/PHASE_10B_TEST_SUITE_CLEANUP.md) |
| **Phase 11** | End-to-end Integration | [PHASE_11_END_TO_END_INTEGRATION.md](./phases/PHASE_11_END_TO_END_INTEGRATION.md) |
| **Phase 11b** | CLI Improvements | [PHASE_11B_CLI_IMPROVEMENTS.md](./phases/PHASE_11B_CLI_IMPROVEMENTS.md) |
| **Phase 13** | Conversation-Based AI Interactions | [PHASE_13_CONVERSATION_BASED_AI_INTERACTIONS.md](./phases/PHASE_13_CONVERSATION_BASED_AI_INTERACTIONS.md) |
| **Phase 14** | Test Planner Agent Refinement | [PHASE_14_TEST_PLANNER_REFINEMENT.md](./phases/PHASE_14_TEST_PLANNER_REFINEMENT.md) |
| **Phase 15** | Test Runner Agent Enhancement | [PHASE_15_TEST_RUNNER_ENHANCEMENT.md](./phases/PHASE_15_TEST_RUNNER_ENHANCEMENT.md) |
| **Phase 16** | Evaluator Agent Removal | [PHASE_16_EVALUATOR_REASSESSMENT.md](./phases/PHASE_16_EVALUATOR_REASSESSMENT.md) |

Additional completed work:
- **Architecture Refactor**: Action Agent owns execution lifecycle
- **Debug Enhancement**: AI interaction logging & screenshot management

### 🔄 In Progress

| Phase | Status | Target | Documentation |
|-------|--------|--------|---------------|
| **Phase 17** | Usability & Persistence | 3-4 days | [PHASE_17_USABILITY_AND_PERSISTENCE.md](./phases/PHASE_17_USABILITY_AND_PERSISTENCE.md) |

### 📅 Upcoming Phases

| Phase | Status | ETA | Documentation |
|-------|--------|-----|---------------|
| **Phase 12** | Test Scenarios | 1/5 scenarios working | [PHASE_12_TEST_SCENARIOS.md](./phases/PHASE_12_TEST_SCENARIOS.md) |
| **Phase 18** | Additional Action Types | TBD | [PHASE_18_ADDITIONAL_ACTION_TYPES.md](./phases/PHASE_18_ADDITIONAL_ACTION_TYPES.md) |
| **Phase 19** | Packaging & Documentation | 1-2 days | [PHASE_19_PACKAGING_DOCUMENTATION.md](./phases/PHASE_19_PACKAGING_DOCUMENTATION.md) |

---

## 8  Post-MVP Roadmap

| Milestone | Description |
|-----------|-------------|
| **v0.2** | Multi-tab support, complex form handling, file upload capabilities. |
| **v0.3** | Specialized agents: API testing agent, mobile testing agent, accessibility testing agent. |
| **v0.4** | Learning & adaptation: Agent fine-tuning based on execution history and failure analysis. |
| **v0.5** | REST API for remote test execution, CI/CD integration, collaborative test planning. |
| **v0.6** | Mobile and desktop application testing, visual regression testing, cross-platform grid adaptation. |

---

## 9  Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Browser update breaks CDP API. | Pin Playwright ≥ 1.45; daily CI smoke tests. |
| API rate limits or downtime. | Implement retry logic, exponential backoff, coordinate rate limiting across agents. |
| Agent coordination failures. | Circuit breaker pattern, fallback to single-agent mode, timeout handling. |
| Infinite loops in test execution. | Global `step_limit`, watchdog timers, agent-level loop detection. |
| Agent hallucination cascade. | Cross-agent validation, confidence thresholds, human-in-loop triggers. |
| Miss-clicks in dense UIs. | Dynamic grid scaling, element detection fallbacks, agent coordination for complex interactions. |

---

## 10  Immediate Next Steps

1. ~~Complete Phase 16 Evaluator Agent Reassessment~~ ✅ DONE
2. ~~Adopt GPT-5 / GPT-4.1 model configuration for all agents~~ ✅ DONE
3. Complete Phase 17 Usability & Persistence Improvements
4. Complete Phase 12 test scenarios (4 remaining)
5. Resume Phase 18 Additional Action Types
6. Complete Phase 19 Packaging & Documentation
7. Package and release v0.1.0

---

## 11  Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 81% | 80%+ ✅ |
| Code Quality | Zero warnings | Maintained ✅ |
| Test Success Rate | 20% (1/5 scenarios) | 60%+ (3/5 scenarios) |
| AI Vision Accuracy | Stabilized with GPT-4.1-mini-vision | High |
