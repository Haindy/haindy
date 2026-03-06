# Docs Guidelines

## Map

- `RUNBOOK.md`: environment and operational notes.
- `design/`: architecture and design docs.
- `issues/`: audits, backlogs, and investigation notes.
- `phases/`: phase-by-phase implementation records.
- `plans/`: topic-based plan folders.
- `plans/<topic>/completed/`: implemented plans kept as historical reference.
- Active plan files may exist at `plans/<topic>/` when current planning work is in progress.

## Rules

- Prefer current code, `RUNBOOK.md`, and phase docs over old plans.
- Treat files under `completed/` as historical reference only; they may not fully match the current codebase.
- When a plan is implemented, move it into that topic's `completed/` folder instead of overwriting its history.
