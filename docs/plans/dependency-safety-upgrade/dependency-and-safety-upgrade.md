# Dependency & Safety Upgrade Plan

Purpose: Outline the work required to modernize third-party dependencies, adopt the latest OpenAI Computer Use features, and reinforce safety controls without introducing manual review checkpoints.

## 1. Update OpenAI SDK

- **Target version**: Bump `openai` from `^1.x` to the latest `^2.x` release (current: `2.3.0`).
- **Tasks**
  - Adjust `pyproject.toml` dependency range to allow the 2.x line (e.g., `openai>=2.3.0` or `^2.3`).
  - Regenerate the lockfile / `poetry.lock` (if applicable) and verify that the resolved version matches the target.
  - Review breaking changes between 1.x and 2.x (notably Responses API helpers, client initialization, and request payload schemas) and update our wrappers accordingly.
- **Verification**
  - Run the existing test suite (`pytest`, `pytest -m "not slow"`, and smoke test the CLI scenarios).
  - Execute `ruff`, `black`, `isort`, and `mypy` to ensure API changes don’t break linting or typing.

## 2. Refresh Remaining Dependencies

- **Goal**: Align critical libraries and dev tools with their latest stable releases to keep parity with upstream security and tooling fixes.
- **Scope**
  - Update automation/tooling packages (`playwright`, `pytest`, `pytest-cov`, `ruff`, `black`, `mypy`, `pre-commit`, etc.).
  - Bring foundational packages (`aiofiles`, `anyio`, `requests`, `typing_extensions`, etc.) up to current versions unless blocked by compatibility constraints.
- **Process**
  - Increment versions in `pyproject.toml` (or rely on flexible specifiers where possible).
  - Regenerate lockfiles and reinstall the environment.
  - Run the full lint/test suite to surface incompatibilities early.
  - Capture any intentional holds (e.g., if a new major version introduces regressions) in release notes or this document.

## 3. Action Agent Enhancements (Computer Use Safety)

- **3.a Contextual metadata for Computer Use**
  - Extend the action execution payload to always pass `current_url` and a stable `safety_identifier` on each `computer_call_output`.
  - Ensure the values are routed through the existing `test_context` metadata so they’re accessible whenever the loop sends follow-up screenshots.

- **3.b Enforced “observe mode” for assertions**
  - Introduce an assertion-only execution mode that denies state-changing actions (`keypress`, `type`, `click`, etc.) while still allowing screenshots and observations.
  - Enforce allowlists/blocklists in the browser driver (e.g., restrict navigation URLs, deny dangerous domains) and log violations.
  - Wire the runner so assertion steps automatically switch the action agent into observe mode; emit a structured error if the Computer Use model attempts prohibited actions.

- **3.c Safety check surfacing (modified)**
  - Honor OpenAI `pending_safety_check` responses by logging and aborting the current action with a descriptive failure, but do **not** require human confirmation.
  - Propagate the failure back to the test runner so it can decide whether to retry, mark the step failed, or continue—keeping the system fully autonomous while still respecting safety signals.

## 4. Validation & Rollout

- Run integration scenarios (`python -m src.main --plan …`) to confirm no regressions in end-to-end flows after the upgrades.
- Capture before/after dependency inventory and document notable changes (e.g., new safeguards, altered behavior) in `docs/` or `GRID_TEST_PROGRESS.md` per repo guidelines.
- Prepare a follow-up ticket/PR checklist documenting:
  - Updated dependency versions and changelog links.
  - Computer Use loop changes (observe mode, metadata flow).
  - Any remaining gaps or future opportunities (e.g., optional human-in-the-loop modes for higher-stakes suites).

## 5. Risks & Mitigations

- **Dependency incompatibilities**: Mitigate via staged upgrades (core SDK first, then tooling) and comprehensive test coverage.
- **OpenAI SDK changes**: Validate payload shapes against the 2.x schema; add unit tests for the API bridge to catch regressions.
- **Observe-mode false positives**: Implement clear logging and metrics to monitor when the Computer Use model attempts blocked actions so we can fine-tune prompts if needed.

