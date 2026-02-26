# Ruff Modernization Backlog Plan

Date: 2026-02-24

## Objective
- Eliminate the current repo-wide Ruff backlog and make `ruff check .` a reliable quality gate for day-to-day development.
- Keep behavior stable while prioritizing safe, reviewable, mostly mechanical fixes first.

## Baseline Snapshot (2026-02-24)
- Command: `python -m ruff check . --statistics`
- Result: `5263` total violations, `4734` auto-fixable.
- Top rule buckets:
  - `W293` blank-line-with-whitespace: `3315`
  - `UP006` non-pep585-annotation: `689`
  - `UP045` non-pep604-annotation-optional: `541`
  - `F401` unused-import: `157`
  - `UP035` deprecated-import: `146`
  - `W291` trailing-whitespace: `122`
  - `I001` unsorted-imports: `73`
- High-volume files (sample):
  - `src/agents/action_agent.py` (`623`)
  - `src/agents/test_runner.py` (`265`)
  - `src/agents/computer_use/session.py` (`151`)

## Scope
- In scope:
  - `src/**/*.py`
  - `tests/**/*.py`
  - Python files under docs/examples that are treated as code by Ruff.
- Out of scope:
  - Rule-set expansion beyond current `select` list.
  - Functional refactors unrelated to lint correctness.

## Execution Plan

### Phase 1: Stabilize Ruff Configuration
1. Move deprecated top-level Ruff settings in `pyproject.toml` to the current `tool.ruff.lint.*` keys.
2. Confirm local tooling parity:
   - `pyproject.toml` Ruff config
   - `.pre-commit-config.yaml` Ruff hook version (already updated)
3. Run `ruff check . --statistics` again to confirm baseline only changes by config migration.

Exit criteria:
- Ruff deprecation warning about top-level settings is gone.
- Baseline counts are captured in this file (or a follow-up update).

### Phase 2: Mechanical Auto-Fix Pass (Safe Fixes)
1. Run auto-fix in batches to keep diffs reviewable:
   - `ruff check src --fix`
   - `ruff check tests --fix`
   - `ruff check docs --fix` (Python files only)
2. Re-run `ruff check . --statistics`.
3. Run targeted regression checks after each batch:
   - `pytest -m "not slow"`

Primary rules expected to drop heavily:
- `W293`, `W291`, `W292`, `I001`, `F401`, many `UP*`.

Exit criteria:
- All safe auto-fixable categories reduced to near-zero.
- Test suite remains green.

### Phase 3: Type Annotation Modernization Cleanup
1. Resolve remaining `UP006`, `UP045`, `UP035`, `UP037`, `UP007` issues that were not safely auto-fixed.
2. Keep changes mechanical:
   - `List/Dict/Set/Tuple` -> `list/dict/set/tuple`
   - `Optional[T]` -> `T | None`
   - remove deprecated typing imports
3. Prioritize high-volume modules first:
   - `src/agents/action_agent.py`
   - `src/agents/test_runner.py`
   - `src/models/openai_client.py`

Exit criteria:
- No remaining pyupgrade violations in actively maintained runtime paths.
- `pytest -m "not slow"` still green.

### Phase 4: Manual Non-Autofix Rules
1. Address non-auto-fix rule buckets intentionally:
   - `E402`, `E722`, `B007`, `B904`, `B006`, `C401`, `B905`, `E731`, etc.
2. Fix file-by-file with behavior-preserving edits and focused tests where risk exists.
3. For any intentional exception, document and use narrow `# noqa` with justification.

Exit criteria:
- `ruff check .` returns zero errors.
- Any remaining `noqa` entries are minimal and documented.

### Phase 5: Enforce and Prevent Regressions
1. Keep Ruff enabled in pre-commit and ensure contributors run it locally.
2. Add/confirm CI step:
   - `ruff check .`
   - `pytest -m "not slow"`
3. Add a short note in contributor docs/runbook with the canonical lint command.

Exit criteria:
- CI fails on lint regressions.
- New PRs do not reintroduce backlog categories.

## Delivery Strategy
- Use incremental commits by phase (not one mega-commit).
- Suggested commit sequence:
  1. Ruff config migration
  2. Mechanical auto-fix (`src`)
  3. Mechanical auto-fix (`tests` + docs python files)
  4. Manual non-autofix cleanup
  5. Final gate + docs touchups

## Risks and Mitigations
- Risk: Large formatting churn obscures logic changes.
  - Mitigation: Isolate mechanical commits from manual fixes.
- Risk: Behavior change in high-complexity files.
  - Mitigation: Run `pytest -m "not slow"` after each phase and keep fixes local.
- Risk: Team friction from broad style edits.
  - Mitigation: Announce phased rollout and keep commit messages explicit about mechanical scope.

## Done Definition
- `ruff check .` passes with no errors.
- `pytest -m "not slow"` passes after final cleanup.
- Ruff configuration deprecation warnings are resolved.
- Backlog tracking in this file is updated with final counts and completion date.

## Completion Snapshot (2026-02-24)
- `ruff check .` now returns zero errors after full mechanical and manual cleanup.
- Violations before remediation: `5263` total.
- Violations after completion: `0` total.
- Remaining command checks executed:
  - `ruff check . --statistics`
  - `ruff check .` (no errors)
