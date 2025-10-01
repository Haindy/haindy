# HAINDY - Autonomous AI Testing Agent

## Important Rules

1. **No emojis**: Do not use emojis in comments, PR descriptions, commit messages, or any other content.
2. **No backward compatibility**: This is a new tool and not in prod. There is no reason to require backwards compatibility or to mark things as deprecated. Just replace old code with new code.
3. **No timeouts on test runs**: NEVER use timeouts when running tests or any long-running operations. This is not a time-sensitive operation and tests are designed to run for extended periods of time.

## Project Overview

HAINDY is an autonomous AI testing agent that uses a multi-agent architecture to accept high-level requirements and autonomously execute testing workflows on web applications. The system employs four specialized AI agents coordinating together to plan, execute, and evaluate tests.

## Implementation Plan

**Important**: The comprehensive implementation plan with phase tracking and technical details is at `docs/HAINDY_PLAN.md`. This includes:
- MVP goals and success criteria
- Detailed tech stack decisions
- Multi-agent architecture design
- Implementation phases with progress tracking
- Post-MVP roadmap

## Development Runbook

**Important**: Before running tests or demos, refer to the comprehensive runbook at `docs/RUNBOOK.md` for detailed instructions on:
- Environment setup
- Running tests for specific phases
- Executing demo scripts
- Code quality checks
- Troubleshooting common issues

## Commands

### Building
```bash
# Project setup (planned tech stack)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .  # Will use pyproject.toml once created
```

### Dependencies Installation
```bash
# Core dependencies (as per plan)
pip install playwright openai rich
playwright install chromium
```

### Testing
```bash
# Unit tests (once implemented)
pytest tests/

# Integration tests
pytest tests/integration/

# Run specific test scenarios
python -m src.main --scenario test_scenarios/checkout_flow.json
```

### Linting and Code Quality
```bash
# Pre-commit hooks (to be configured)
pre-commit install
pre-commit run --all-files

# Code formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Development Commands
```bash
# Run in development mode with verbose logging
python -m src.main --debug --verbose

# Generate test execution report
python -m src.monitoring.reporter --input data/test_run_123.jsonl --output reports/
```

## High-Level Architecture

### Core Components

1. **Multi-Agent System**
   - **Test Planner Agent**: Analyzes requirements/PRDs and creates structured test plans
   - **Test Runner Agent**: Orchestrates test execution and decides next steps
   - **Action Agent**: Converts screenshots and instructions to grid-based coordinates
   - **Evaluator Agent**: Assesses screenshot results against expected outcomes

2. **Browser Automation Layer**
   - Playwright-Python wrapper for Chromium
   - Adaptive grid overlay system (initially 60×60 with refinement) for reliable cross-application interaction
   - WebSocket/CDP communication for native browser control

3. **Orchestration Framework**
   - Agent coordination and communication
   - State management for test execution
   - Workflow orchestration between agents

4. **Core Infrastructure**
   - Error handling and recovery mechanisms
   - Security layer (rate limiting, sensitive data protection)
   - Monitoring and reporting system
   - Configuration management

### Architecture Flow

```
Human Input (Requirements/PRD)
    ↓
Test Planner Agent
    ↓
Test Runner Agent ←→ [Coordination Loop]
    ↓                      ↑
Action Agent              |
    ↓                      |
Browser Driver            |
    ↓                      |
Evaluator Agent ----------
    ↓
Test Report Generation
```

### Key Design Principles

1. **Agent Specialization**: Each AI agent has a focused role and expertise domain
2. **Coordinated Intelligence**: Agents collaborate while maintaining independent decision-making
3. **Fail-Fast with Recovery**: Agent-level validation with cross-agent verification
4. **Observable by Default**: Every agent communication and decision is traceable
5. **Human-Interpretable**: All outputs (plans, actions, results) are understandable
6. **Modular & Extensible**: Easy to add new agent types or replace existing ones

### Technology Stack

- **Language**: Python 3.10+
- **Browser Automation**: Playwright-Python (Chromium)
- **AI Models**: OpenAI GPT-4o mini (4 instances for multi-agent system)
- **Grid System**: Adaptive grid overlay with coordinate mapping and refinement
- **Logging**: JSONL format with rich console output
- **Packaging**: setuptools with pyproject.toml

