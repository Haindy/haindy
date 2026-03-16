# HAINDY Development Runbook

This document contains instructions for running tests, demos, and common development tasks.

## Environment Setup

### Initial Setup

#### Shared Steps

```bash
# Clone the repository
git clone https://github.com/Haindy/haindy.git
cd haindy

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.lock
pip install -e .[dev]
```

#### macOS

- Base installation is supported.
- Developer checks and tests are supported.
- The `desktop` automation backend is not supported yet because the current
  implementation depends on Linux/X11 input and capture tooling.
- Use macOS for development, CI-like local verification, and `--mobile` ADB runs.
- No Playwright browser runtime installation is required.

#### Linux

- Base installation is supported.
- The `desktop` automation backend is supported on Linux/X11 after installing
  the runtime dependencies below.
- Use Linux for full desktop automation runs.
- No Playwright browser runtime installation is required.

### Environment Variables

Create a `.env` file in the project root:

```text
HAINDY_OPENAI_API_KEY=your-api-key-here
HAINDY_COMPUTER_USE_MODEL=gpt-5.4
HAINDY_CU_PROVIDER=google
HAINDY_GOOGLE_CU_MODEL=gemini-2.5-computer-use-preview-10-2025
HAINDY_ANTHROPIC_API_KEY=your-anthropic-api-key
HAINDY_ANTHROPIC_CU_MODEL=claude-sonnet-4-6
HAINDY_ANTHROPIC_CU_BETA=computer-use-2025-11-24
HAINDY_ANTHROPIC_CU_MAX_TOKENS=16384
HAINDY_VERTEX_API_KEY=your-vertex-api-key
HAINDY_VERTEX_PROJECT=your-vertex-project
HAINDY_VERTEX_LOCATION=us-central1
HAINDY_CU_SAFETY_POLICY=auto_approve
HAINDY_AUTOMATION_BACKEND=desktop
```

Platform guidance:

- macOS: leave `HAINDY_AUTOMATION_BACKEND` unset unless you are intentionally
  using `mobile_adb`; the current `desktop` backend is Linux/X11-only.
- Linux desktop runs: set `HAINDY_AUTOMATION_BACKEND=desktop`.

### Desktop Automation Dependencies (Linux/X11)

Desktop automation uses OS-level input and screen capture. Install:

- `ffmpeg` (x11grab) for screenshots
- `xrandr` (x11-xserver-utils) for resolution switching
- `/dev/uinput` permissions for soft keyboard/mouse
- `xclip` for clipboard read/write
- `gdbus` (glib2.0) for optional GNOME screen recording

Example (Ubuntu/Debian):

```bash
sudo apt-get install ffmpeg x11-xserver-utils xclip libglib2.0-bin
sudo modprobe uinput
sudo usermod -aG input $USER
```

Log out/in after updating group membership.

macOS note:

- None of these Linux/X11 desktop dependencies apply on macOS.
- A macOS-native desktop backend has not been implemented yet.

### Desktop Driver Configuration

Use `HAINDY_AUTOMATION_BACKEND=desktop` to enable OS-level control.
Common overrides:

```text
HAINDY_AUTOMATION_BACKEND=desktop
HAINDY_DESKTOP_RESOLUTION=1440,900
HAINDY_DESKTOP_KEYBOARD_LAYOUT=us
HAINDY_DESKTOP_KEYBOARD_SCANCODES=true
HAINDY_DESKTOP_ENABLE_RESOLUTION_SWITCH=true
HAINDY_DESKTOP_SCREENSHOT_DIR=data/screenshots/desktop
HAINDY_DESKTOP_COORDINATE_CACHE_PATH=data/desktop_cache/coordinates.json
```

Use this section only on Linux/X11. It does not apply to macOS yet.

### Mobile ADB Backend

Use CLI `--mobile` for a hard mobile override per run. Optional defaults:

```text
HAINDY_AUTOMATION_BACKEND=desktop
HAINDY_MOBILE_DEFAULT_ADB_SERIAL=
HAINDY_MOBILE_SCREENSHOT_DIR=data/screenshots/mobile
HAINDY_MOBILE_COORDINATE_CACHE_PATH=data/mobile_cache/coordinates.json
HAINDY_MOBILE_ADB_TIMEOUT_SECONDS=15.0
```

For mobile context, provide either:

- `adb_serial` + `app_package` (optional `app_activity`), or
- `adb_commands` that discover/select the device and open the app.

### Caching and Trace Artifacts

Backported caches and logs are stored under `data/` by default:

- `data/task_plan_cache.json`
- `data/planning_cache.json`
- `data/situational_cache.json`
- `data/execution_replay_cache.json`
- `data/desktop_cache/coordinates.json`
- `data/model_logs/model_calls.jsonl`
- `data/traces/<run_id>.json`

Execution replay behavior:
- Replay is attempted for non-loop steps when `HAINDY_ENABLE_EXECUTION_REPLAY_CACHE=true`.
- Replay entries are recorded from successful execution actions (driver actions only; validation is always live).
- Validation-only step decompositions (`assert`, `skip_navigation`, `wait`, `screenshot`) are not persisted in replay cache.
- Replayed macro actions enforce a minimum 2-second stabilization wait between actions.
- Replay validation can request additional model-directed settling waits (replay-only) with a per-step cap of 30 seconds before cache invalidation/fallback.
- Replay keys include a plan fingerprint; when plan content changes, old replay entries are ignored automatically.
- Legacy `can_be_replayed` step flags are accepted in payloads but ignored by runtime replay gating.

Control replay caching with:

```text
HAINDY_ENABLE_PLANNING_CACHE=true
HAINDY_ENABLE_SITUATIONAL_CACHE=true
HAINDY_ENABLE_EXECUTION_REPLAY_CACHE=true
HAINDY_TASK_PLAN_CACHE_PATH=data/task_plan_cache.json
HAINDY_PLANNING_CACHE_PATH=data/planning_cache.json
HAINDY_SITUATIONAL_CACHE_PATH=data/situational_cache.json
HAINDY_EXECUTION_REPLAY_CACHE_PATH=data/execution_replay_cache.json
HAINDY_MODEL_LOG_PATH=data/model_logs/model_calls.jsonl
```

## Running Tests

### All Tests
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing
```

### Test Specific Phases
```bash
# Phase 1: Core Foundation
python -m pytest tests/test_base_agent.py tests/test_types.py tests/test_interfaces.py -v

# Phase 3: Test Planner Agent
python -m pytest tests/test_planner_agent.py -v

# Phase 4: Action Agent
python -m pytest tests/test_action_agent.py -v

# Phase 5: Evaluator Agent
python -m pytest tests/test_evaluator_agent.py -v

# Phase 6: Test Runner Agent
python -m pytest tests/test_test_runner.py -v

# Phase 7: Agent Coordination
python -m pytest tests/test_communication.py tests/test_state_manager.py tests/test_coordinator.py -v

# Phase 8: Execution Journaling
python -m pytest tests/test_journal_models.py tests/test_journal_manager.py tests/test_pattern_matcher.py tests/test_script_recorder.py -v
```

### Running Specific Test Files
```bash
# Run a single test file
python -m pytest tests/test_base_agent.py -v

# Run tests matching a pattern
python -m pytest -k "test_agent" -v

# Run tests with specific markers (once we add them)
python -m pytest -m "unit" -v
```

## Running Demo Scripts

### Phase-Specific Demos

```bash
# Activate virtual environment first
source venv/bin/activate

# Phase 3: Test Planner Demo
python examples/planner_demo.py

# Phase 4: Action Agent Demo
python examples/action_demo.py

# Phase 5: Evaluator Demo
python examples/evaluator_demo.py

# Phase 6: Test Runner Demo
python examples/runner_demo.py

# Phase 7: Orchestration Demo
python examples/orchestration_demo.py

# Phase 8: Journaling Demo
python examples/journaling_demo.py
```

### Running Tests

#### Document-based Testing
```bash
# Extract requirements from document
python -m src.main --plan requirements.md
python -m src.main -p requirements.md

# With debug output
python -m src.main --plan requirements.md --debug
```

#### Execution Options
```bash
# Berserk mode (fully autonomous)
python -m src.main --berserk --plan requirements.md

# Plan-only mode (generate plan without executing)
python -m src.main --plan-only --plan test_scenarios/wikipedia_search_simple.txt

# Browser runs headless by default
# To show browser, set BROWSER_HEADLESS=false in .env file

# Custom timeout (default: 7200 seconds)
python -m src.main --plan requirements.md --timeout 3600

# Custom output directory
python -m src.main --plan requirements.md --output custom_reports/

# Google Computer Use example
export HAINDY_CU_PROVIDER=google
export HAINDY_AUTOMATION_BACKEND=desktop
python -m src.main --plan test_scenarios/wikipedia_search_simple.txt

# OpenAI Computer Use example
export HAINDY_CU_PROVIDER=openai
export HAINDY_COMPUTER_USE_MODEL=gpt-5.4
export HAINDY_AUTOMATION_BACKEND=desktop
python -m src.main --plan test_scenarios/wikipedia_search_simple.txt

# Anthropic Computer Use example
export HAINDY_CU_PROVIDER=anthropic
export HAINDY_ANTHROPIC_API_KEY=your-anthropic-api-key
export HAINDY_AUTOMATION_BACKEND=desktop
python -m src.main --plan test_scenarios/wikipedia_search_simple.txt
```

## Code Quality Checks

### Linting
```bash
# Run ruff linter
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

### Type Checking
```bash
# Run mypy
mypy src/

# Run on specific module
mypy src/agents/
```

### Code Formatting
```bash
# Format with black
black src/ tests/

# Check without modifying
black --check src/ tests/

# Sort imports with isort
isort src/ tests/
```

## Development Workflow

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files src/agents/base_agent.py
```

### Creating a New Branch
```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/phase-X-description

# Make changes and test
# ... edit files ...
python -m pytest tests/test_new_feature.py -v

# Commit changes
git add -A
git commit -m "Implement Phase X: Description"

# Push branch
git push -u origin feature/phase-X-description
```

### Creating a Pull Request
```bash
# Ensure tests pass
python -m pytest --cov=src

# Ensure code quality
black src/ tests/
ruff check src/ tests/
mypy src/

# Push changes
git push

# Create PR via GitHub CLI
gh pr create --title "Phase X: Description" --body "Summary of changes..."
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall package: `pip install -e .[dev]`

2. **Test Failures**
   - Check environment variables are set
   - Verify backend prerequisites for your chosen mode:
     Linux desktop runs need the Linux/X11 tools below, and `--mobile` runs
     need a working `adb` installation and device connection

3. **Coverage Below Threshold**
   - Run specific test file with coverage: `python -m pytest tests/test_file.py --cov=src.module`
   - Add missing tests for uncovered code

4. **Type Checking Errors**
   - Ensure all type annotations are present
   - Use `# type: ignore` sparingly for third-party issues

### Debug Mode
```bash
# Run tests with debugging output
python -m pytest -vvs tests/test_file.py

# Run with pdb on failure
python -m pytest --pdb tests/test_file.py

# Run specific test function
python -m pytest tests/test_file.py::TestClass::test_function -v
```

## Performance Testing

### Running Performance Benchmarks
```bash
# Once implemented, run performance tests
python -m pytest tests/performance/ -v

# Profile test execution
python -m cProfile -s cumulative examples/runner_demo.py
```

## Documentation

### Building Documentation (Future)
```bash
# Install docs dependencies
pip install sphinx sphinx-autodoc-typehints

# Build HTML docs
cd docs
make html

# View docs
open _build/html/index.html
```

## Release Process (Future)

### Creating a Release
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md

# Create release commit
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.1.0"

# Tag release
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push tag
git push origin v0.1.0

# Build distribution
python -m build

# Upload to PyPI (when ready)
python -m twine upload dist/*
```

## Monitoring and Logs

### Log Locations
- Application logs: `logs/haindy.log` (Note: Directory may need to be created manually)
- Test execution journals: `data/journals/` (Note: Not yet implemented)
- Screenshots: `data/screenshots/` (Note: Currently empty, screenshots embedded in reports)
- Test reports: `reports/` (HTML reports with embedded screenshots and step details)

### Log Levels
```bash
# Set log level via environment variable
export HAINDY_LOG_LEVEL=DEBUG

# Or via command line (once implemented)
python -m src.main --log-level DEBUG
```

### Debugging Test Executions

#### Understanding Test Output
When running tests, the console output shows:
- JSON-formatted log messages with timestamps
- Test execution progress indicators
- Summary of test results including:
  - Total steps planned
  - Steps completed
  - Steps failed
  - Success rate
  - Bug reports for failed steps

#### Accessing Test Details
1. **HTML Reports**: The most comprehensive source of test execution details
   ```bash
   # Reports are saved with timestamp and UUID
   ls -la reports/test_report_*.html
   
   # Open the latest report
   xdg-open reports/test_report_*_$(date +%Y%m%d)*.html  # Linux
   open reports/test_report_*_$(date +%Y%m%d)*.html      # macOS
   ```

2. **Console Output**: Run with `--debug` flag for verbose output
   ```bash
   python -m src.main --plan requirements.md --debug
   ```

3. **Test Execution Flow**:
   - Step 1 is typically navigation (uses navigation workflow)
   - Subsequent steps use click, type, or assert workflows
   - Each step shows success/failure in the summary
   - Failed steps include detailed error messages

#### Common Issues
- **"Field required" errors**: Usually indicates a mismatch between expected and actual result formats
- **"Could not locate element"**: The AI couldn't find the target element in the screenshot
- **"Click had no effect"**: The click was executed but no UI changes were detected

## Notes

- Always work in a virtual environment
- Run tests before committing changes
- Keep test coverage above 60%
- Follow the coding standards in CLAUDE.md
- Update this runbook when adding new commands or workflows
