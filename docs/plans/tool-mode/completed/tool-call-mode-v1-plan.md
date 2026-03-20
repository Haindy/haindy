# Tool Call Mode V1 Plan

## Summary

Implement the full V1 tool-call surface in one pass: a dedicated session daemon, direct `act` and `test` commands, session lifecycle and variable management, a stable JSON contract, and repo-level documentation. V2 work such as `explore`, live-screen situational assessment, `session prune`, and desktop `--url` startup stays out of scope.

## Scope

- In scope: `session new|list|status|close|set|unset|vars`, `act`, `test`, daemon lifecycle, Unix-socket IPC, session filesystem layout, V1 skill, docs, and validation coverage
- Out of scope: `explore`, desktop startup ownership, persistent session variables, live-screen situational assessment, and V2 cleanup commands

## Decision Summary

- Build tool-call mode as a separate runtime under `src/tool_call_mode/` instead of forcing daemon/session concerns into the batch pipeline
- Reuse the existing `DesktopController`, `MobileController`, `ActionAgent`, `TestPlannerAgent`, and `TestRunner` wherever possible
- Bypass `WorkflowCoordinator`, scope triage, and situational gating for tool-call mode
- Keep `HAINDY_AUTOMATION_BACKEND` as the shared backend env contract and add `HAINDY_HOME` for session state
- Store session variables in memory only for V1 and redact secret values in responses

## Implementation Steps

1. Split CLI routing at `src/main.py`.
Self-check: tool-call commands must branch before the legacy parser so standard mode stays stable and tool mode can own its JSON-only contract.

2. Add a dedicated tool-call package for models, paths, variables, IPC, logging, daemon, runtime, and CLI dispatch.
Self-check: keep the new surface DRY by centralizing envelope construction, session paths, and command-name normalization in shared helpers instead of duplicating them across client and daemon code.

3. Implement a persistent per-session daemon that owns one controller and one warm agent stack for the session lifetime.
Self-check: session startup must be deterministic and cheap to reuse, with readiness signaling, stale-session cleanup, idle timeout handling, and single-command concurrency enforcement.

4. Reuse the agent layer instead of inventing a second execution engine.
Self-check: `act` should flow through `ActionAgent`, `session status` should use observe-only action behavior, and `test` should use a tool-mode planning path plus a safely resettable `TestRunner`.

5. Translate internal results into the V1 public JSON contract.
Self-check: every command must emit exactly one JSON object with stable top-level fields, correct exit codes, absolute screenshot paths when available, and consistent `exit_reason` mapping.

6. Implement session variables and secrecy rules.
Self-check: interpolation must stay intentionally simple (`$NAME`, `$$`, unknown names unchanged), and secret values must not land in `session.json`, stdout responses, or session logs after Haindy receives them.

7. Update repo-facing contracts.
Self-check: README, `.env.example`, `docs/RUNBOOK.md`, and the bundled skill must all describe the same V1 surface, explicitly call out deferred features, and avoid introducing extra wrapper-specific behavior.

8. Validate with targeted and repo-wide checks.
Self-check: cover parser behavior, dead-session handling, session listings, variable semantics, and main-entrypoint routing before running lint, format, mypy, and pytest across the repo.

## Main Risks

- Parser regressions: tool-call usage errors must stay JSON-only without breaking legacy CLI behavior
- Result translation drift: internal agent/test statuses do not map 1:1 to the public envelope and need explicit normalization
- Session lifecycle leaks: readiness, idle timeout, and force-close paths can leave stale socket or PID state behind if not handled carefully
- Documentation drift: backend/env semantics are shared contracts and must be updated together

## Verification

- Unit coverage for parser routing, dead-session detection, session listing, variable interpolation, and JSON usage errors
- Type and lint validation on the new tool-call package and entrypoint wiring
- Full repo checks: `ruff check .`, `ruff format .`, `mypy src`, and `pytest`
