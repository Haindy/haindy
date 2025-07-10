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

### Running Test Scenarios

#### Run Existing Test Scenario
```bash
# Run a pre-defined test scenario from JSON file
python -m src.main --json-test-plan test_scenarios/wikipedia_search.json

# Short form
python -m src.main -j test_scenarios/wikipedia_search.json

# With debug output for detailed logging
python -m src.main --json-test-plan test_scenarios/wikipedia_search.json --debug
```

#### Interactive Mode
```bash
# Enter test requirements interactively
python -m src.main --requirements

# Short form
python -m src.main -r
```
Opens an interactive prompt where you can paste or type multi-line test requirements.

#### Document-based Testing
```bash
# Extract test requirements from a document (PRD, design doc, etc.)
python -m src.main --plan requirements.md

# Short form
python -m src.main -p requirements.md

# With URL specified in command
python -m src.main --plan requirements.md --url https://example.com
```

### Execution Options

#### Berserk Mode
```bash
# Fully autonomous mode - no confirmations
python -m src.main --berserk --plan requirements.pdf

# With existing test scenario
python -m src.main --berserk -j test_scenarios/login_test.json
```

#### Plan-Only Mode
```bash
# Generate test plan without executing
python -m src.main --plan-only -j test_scenarios/wikipedia_search.json
```

#### Browser Options
```bash
# Browser runs in headless mode by default
python -m src.main -j test_scenarios/login_test.json

# Force headless mode explicitly
python -m src.main -j test_scenarios/login_test.json --headless

# Note: To show browser window, you need to modify the .env file or settings:
# Add to .env: BROWSER_HEADLESS=false
# Or modify src/config/settings.py temporarily
```

### Output Options

#### Report Formats
```bash
# HTML report (default)
python -m src.main -j test_scenarios/login_test.json --format html

# JSON report
python -m src.main -j test_scenarios/login_test.json --format json

# Markdown report
python -m src.main -j test_scenarios/login_test.json --format markdown

# Custom output directory
python -m src.main -j test_scenarios/login_test.json --output custom_reports/
```

### Advanced Options

#### Timeouts and Limits
```bash
# Set execution timeout (default: 300 seconds)
python -m src.main -j test_scenarios/complex_test.json --timeout 600

# Set maximum steps (default: 50)
python -m src.main -j test_scenarios/long_test.json --max-steps 100
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

### Example Test Scenarios

The repository includes example test scenarios in the `test_scenarios/` directory:

- `wikipedia_search.json` - Search functionality test on Wikipedia
- `checkout_flow.json` - E-commerce checkout process (example)
- `login_test.json` - User authentication flow (example)

### Reports and Output

Test execution generates:
- **HTML Report**: Detailed test report with screenshots and AI conversations
- **Debug Logs**: Timestamped directory with all screenshots and interactions
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
├── test_scenarios/    # Example test scenarios
├── data/             # Runtime data storage
├── reports/          # Test execution reports
└── docs/             # Documentation
```

## Contributing

See [HAINDY_PLAN.md](docs/HAINDY_PLAN.md) for the detailed implementation plan and architecture.