# Phase 11b — CLI Improvements

**Status**: Completed

## Summary

Phase 11b introduced the modern CLI shape centered on document-driven input.
The runtime now expects requirements to come from files and supports utility/automation flags around that flow.

## Supported Commands

### Core Input

```bash
python -m haindy.main --plan <file>
python -m haindy.main -p <file>
```

- Reads requirements text directly from the provided file.
- Works for requirement docs, PRDs, and prompt files.
- URL can be included in requirements or supplied via `--url`.

### Utility Commands

```bash
python -m haindy.main --test-api
python -m haindy.main --version
python -m haindy.main --help
```

### Execution Modes

```bash
python -m haindy.main --berserk --plan requirements.md
python -m haindy.main --plan-only --plan requirements.md
```

## Key Implementation Notes

- `argparse` parser with a mutually exclusive input/utility group.
- File-based ingestion as the single test-input path.
- Clear usage examples in help output.
- Runtime overrides for debug/logging/output/timeout behavior.

## Outcomes

- Reduced CLI surface area for better maintainability.
- Aligned docs and tests with a single ingestion path.
- Preserved automation-friendly flags (`--verbose`, `--debug`, `--output`, `--format`, `--timeout`, `--max-steps`).
