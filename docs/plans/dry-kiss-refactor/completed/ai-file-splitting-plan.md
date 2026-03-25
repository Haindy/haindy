# DRY/KISS File-Splitting Refactor

## Summary
- Create branch `codex/dry-kiss-file-splitting` from `main`.
- Refactor the three oversized production files and the two oversized regression files while preserving behavior and public imports.
- Keep the work behavior-preserving: no prompt contract, cache format, or report schema changes unless required to keep existing tests passing.

## Public Interfaces
- Keep `from haindy.agents.computer_use import ComputerUseSession, ComputerUseSessionResult, ComputerUseExecutionError` unchanged.
- Keep `from haindy.agents.test_runner import TestRunner` unchanged.
- Keep `EnhancedReporter.generate_report(test_state, output_dir, action_storage=None)` unchanged.

## Implementation Changes
1. Split `src/monitoring/enhanced_reporter.py` into a thin writer, a template-data helper module, and a Jinja template file.
2. Split `src/agents/test_runner.py` into orchestration plus step-processing, verification, and summary collaborators while preserving private wrapper methods used by tests.
3. Split `src/agents/computer_use/session.py` into a small facade plus typed helpers, provider-specific runners, action execution logic, and shared normalization utilities while preserving private wrapper methods used by tests.
4. Split `tests/test_computer_use_session.py` and `tests/test_test_runner.py` so the test layout mirrors the new module boundaries.
5. Add focused `EnhancedReporter` tests because that file currently has no dedicated coverage.

## Validation
- Run targeted tests while refactoring:
  - `.venv/bin/pytest tests/test_enhanced_reporter.py -q`
  - `.venv/bin/pytest tests/test_test_runner_*.py -q`
  - `.venv/bin/pytest tests/test_computer_use_session_*.py -q`
- Run required repo validation before finishing:
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format .`
  - `.venv/bin/mypy src`
  - `.venv/bin/pytest`

## Defaults
- Keep current public module paths intact even if implementation moves to helper modules.
- Prefer small internal collaborators over new compatibility shims.
- Keep no file in scope above roughly 900 lines after the refactor.
