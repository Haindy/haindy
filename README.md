# HAINDY - Autonomous AI Testing Agent

AI-driven autonomous testing agent — Transform requirements into comprehensive test scenarios with coordinated multi-agent AI workflow

## Overview

HAINDY is an autonomous AI testing agent that uses a multi-agent architecture to accept high-level requirements and autonomously execute testing workflows on web applications. The system employs four specialized AI agents coordinating together to plan, execute, and evaluate tests.

## Key Features

- **Multi-Agent Architecture**: Four specialized AI agents (Test Planner, Test Runner, Action Agent, Evaluator)
- **DOM-Free Visual Interaction**: Adaptive grid-based interaction with automatic refinement
- **Just-In-Time Scripted Automation**: Records successful actions for faster replay
- **Hierarchical Validation**: Multi-layer validation to prevent hallucinations
- **Detailed Execution Journaling**: Comprehensive logging in structured natural language

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fkeegan/haindy.git
cd haindy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

4. Install Playwright browsers:
```bash
playwright install chromium
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

### Environment Configuration

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Interactive Mode
```bash
python -m src.main --requirements
```
Opens an interactive prompt where you can paste or type multi-line test requirements.

### File-based Testing
```bash
python -m src.main --plan requirements.md
```
Pass any document file (PRD, design doc, etc.) directly to the AI to extract test requirements.

### Run Existing Test
```bash
python -m src.main --json-test-plan test_scenarios/login_test.json
```
Run a pre-defined test scenario from a JSON file.

### Berserk Mode
```bash
python -m src.main --berserk --plan requirements.pdf
```
Fully autonomous mode - attempts to complete all tests without human intervention.

### Utility Commands
```bash
python -m src.main --test-api    # Test OpenAI API connection
python -m src.main --version     # Show version (0.1.0)
python -m src.main --help        # Display help with examples
```

### Development

Run tests:
```bash
pytest
```

Run linting and formatting:
```bash
black src/ tests/
isort src/ tests/
mypy src/
```

### Project Structure

```
haindy/
├── src/
│   ├── agents/         # AI agent implementations
│   ├── orchestration/  # Multi-agent coordination
│   ├── browser/        # Browser automation layer
│   ├── grid/          # Adaptive grid system
│   ├── models/        # AI model interfaces
│   ├── core/          # Core abstractions
│   ├── error_handling/# Error recovery mechanisms
│   ├── security/      # Security and rate limiting
│   ├── monitoring/    # Logging and reporting
│   └── config/        # Configuration management
├── tests/             # Unit and integration tests
├── test_scenarios/    # Example test scenarios
├── data/             # Runtime data storage
├── reports/          # Test execution reports
└── docs/             # Documentation
```

## Contributing

See [HAINDY_PLAN.md](docs/HAINDY_PLAN.md) for the detailed implementation plan and architecture.