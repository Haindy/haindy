# Docs Guidelines

## Map

- `RUNBOOK.md`: environment and operational notes.
- `design/`: architecture and design docs.
- `issues/`: audits, backlogs, and investigation notes.
- `phases/`: phase-by-phase implementation records.
- `plans/`: topic-based plan folders.
- `plans/<topic>/`: active or draft plans for that topic.
- `plans/<topic>/completed/`: implemented plans kept as historical reference.

## Rules

- Read the active plan in `plans/<topic>/` first when doing plan-driven work.
- Treat files under `completed/` as historical reference only; they may not fully match the current codebase.
- When a plan is implemented, move it into that topic's `completed/` folder instead of overwriting its history.
