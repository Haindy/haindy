# HAINDY - Autonomous AI Testing Agent

AI-driven autonomous testing agent — Transform requirements into comprehensive test scenarios with coordinated multi-agent AI workflow

## Overview

HAINDY is an autonomous AI testing agent that uses a multi-agent architecture to accept high-level requirements and execute desktop-first testing workflows. The system uses a dedicated situational setup stage plus specialized agents that plan, execute, and evaluate tests.

## Key Features

- **Multi-Agent Architecture**: Scope Triage, Situational, Test Planner, Test Runner, and Action Agent
- **DOM-Free Visual Interaction**: Adaptive grid-based interaction with automatic refinement
- **Desktop + Computer Use**: Optional OS-level driver with OpenAI and Google provider support
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
pip install -r requirements.lock
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

5. Install desktop browser tooling:
```bash
playwright install chromium
```

### Environment Configuration

Copy `.env.example` and update values:
```bash
cp .env.example .env
```

Core model settings:
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL`, `OPENAI_REQUEST_TIMEOUT_SECONDS`
- `OPENAI_MAX_RETRIES`

Optional per-agent overrides use the `HAINDY_<AGENT>_*` namespace:
- `HAINDY_SCOPE_TRIAGE_*`
- `HAINDY_TEST_PLANNER_*`
- `HAINDY_TEST_RUNNER_*`
- `HAINDY_ACTION_AGENT_*`
- `HAINDY_SITUATIONAL_AGENT_*`

Supported `*_REASONING_LEVEL` values: `none`, `minimal`, `low`, `medium`,
`high`, and `xhigh`.

`*_TEMPERATURE`, `*_MODEL`, and `*_MODALITIES` on an agent override the built-in defaults when set.

Computer Use options use either OpenAI or Google provider:
- `HAINDY_ACTIONS_USE_COMPUTER_TOOL`
- `HAINDY_COMPUTER_USE_MODEL` (OpenAI)
- `CU_PROVIDER` (`openai` or `google`, default `google`)
- `GOOGLE_CU_MODEL` and Vertex vars (`VERTEX_API_KEY`, `VERTEX_PROJECT`, `VERTEX_LOCATION`)
- `CU_SAFETY_POLICY`
- `HAINDY_ACTIONS_COMPUTER_TOOL_*` controls tool behavior

Other groups are in `.env.example`: grid, execution, recording, desktop, storage,
logging, security, and rate limiting.

Desktop automation requires OS-level dependencies (`ffmpeg`, `xrandr`, `/dev/uinput`,
`xclip`). See `docs/RUNBOOK.md` for setup details.

## Usage

### Running Tests

#### Desktop-First Execution
```bash
# Both files are required:
# --plan: test requirements
# --context: runtime setup/context details (target type, URL/app launch, constraints)
python -m src.main --plan requirements.md --context execution_context.txt

# Short form
python -m src.main -p requirements.md --context execution_context.txt

# With debug output for detailed logging
python -m src.main --plan requirements.md --context execution_context.txt --debug

# Emit structured JSON logs (suitable for automation)
python -m src.main --plan requirements.md --context execution_context.txt --verbose
```

#### Desktop + Computer Use Quickstart
```bash
export HAINDY_ACTIONS_USE_COMPUTER_TOOL=true
export CU_PROVIDER=google
python -m src.main --plan test_scenarios/wikipedia_search_simple.txt --context test_scenarios/wikipedia_search_simple.txt
```

### Execution Options

#### Berserk Mode
```bash
# Fully autonomous mode - no confirmations
python -m src.main --berserk --plan requirements.pdf --context execution_context.txt
```

### Output Options

#### Report Formats
```bash
# HTML report (default)
python -m src.main --plan requirements.md --context execution_context.txt --format html

# JSON report
python -m src.main --plan requirements.md --context execution_context.txt --format json

# Markdown report
python -m src.main --plan requirements.md --context execution_context.txt --format markdown

# Custom output directory
python -m src.main --plan requirements.md --context execution_context.txt --output custom_reports/
```

### Advanced Options

#### Timeouts and Limits
```bash
# Set execution timeout (default: 7200 seconds)
python -m src.main --plan requirements.md --context execution_context.txt --timeout 3600

# Set maximum steps (default: 50)
python -m src.main --plan requirements.md --context execution_context.txt --max-steps 100

# Force desktop recording on or off for a single run
python -m src.main --plan requirements.md --context execution_context.txt --record
python -m src.main --plan requirements.md --context execution_context.txt --no-record
```

### Utility Commands
```bash
# Test OpenAI API connection
python -m src.main --test-api

# Show version information
python -m src.main --version

# Display help with all options
python -m src.main --help
```

### Example Requirement Files

The repository includes sample requirement documents in `test_scenarios/`:

- `wikipedia_search_simple.txt` - simple requirement prompt for a Wikipedia flow
- `aubilities-bundles-fmc-admin.md` - longer-form scope document example

### Reports and Output

Test execution generates:
- **HTML Report**: Detailed test report with screenshots and AI conversations
- **Debug Logs**: Timestamped directory with all screenshots and interactions
- **Model Logs + Trace**: `data/model_logs/model_calls.jsonl` and `data/traces/<run_id>.json`
- **Caches**: task plan, execution replay, and desktop coordinate caches under `data/`
- **Console Output**: Real-time progress and summary

Reports are saved to:
- Default: `reports/YYYYMMDD_HHMMSS/`
- Custom: Use `--output` to specify directory

Example report path:
```
reports/20250709_231324/test_report_9c16faf8-8ecf-415d-82e8-9bf894f3e0df_20250709_211551.html
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
├── test_scenarios/    # Example requirements documents
├── data/             # Runtime data storage
├── reports/          # Test execution reports
└── docs/             # Documentation
```

## Contributing

See [HAINDY_PLAN.md](docs/HAINDY_PLAN.md) for the detailed implementation plan and architecture.
