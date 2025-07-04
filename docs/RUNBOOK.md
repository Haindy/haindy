# HAINDY Development Runbook

This document contains instructions for running tests, demos, and common development tasks.

## Environment Setup

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/fkeegan/haindy.git
cd haindy

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Install Playwright browsers
playwright install chromium
```

### Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
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

# Phase 2: Browser & Grid
python -m pytest tests/test_grid_overlay.py tests/test_grid_coordinator.py tests/test_browser_driver.py -v

# Phase 3: Test Planner Agent
python -m pytest tests/test_planner_agent.py -v

# Phase 4: Action Agent
python -m pytest tests/test_action_agent.py -v

# Phase 5: Evaluator Agent
python -m pytest tests/test_evaluator_agent.py -v

# Phase 6: Test Runner Agent
python -m pytest tests/test_runner_agent.py -v

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

# Phase 2: Grid System Demo
python examples/grid_demo.py

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

### Interactive Testing
```bash
# Run the main entry point (once implemented)
python -m src.main

# With debug mode
python -m src.main --debug

# With custom config
python -m src.main --config custom_config.json
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
   - Ensure Playwright browsers are installed: `playwright install chromium`

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
- Application logs: `logs/haindy.log`
- Test execution journals: `data/journals/`
- Screenshots: `data/screenshots/`
- Test reports: `reports/`

### Log Levels
```bash
# Set log level via environment variable
export HAINDY_LOG_LEVEL=DEBUG

# Or via command line (once implemented)
python -m src.main --log-level DEBUG
```

## Notes

- Always work in a virtual environment
- Run tests before committing changes
- Keep test coverage above 60%
- Follow the coding standards in CLAUDE.md
- Update this runbook when adding new commands or workflows