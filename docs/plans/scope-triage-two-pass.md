# Scope Triage and Two-Pass Test Planning Workflow

## Overview
- We need the test planner to honor the exact scope provided by users, especially when requirements bundles contain both in-scope and out-of-scope material.
- Misinterpretations (e.g., assuming an Admin detail view when only an edit modal exists) cause false blockers and wasted execution turns.
- The solution is a two-pass pipeline that first normalizes the input and only then generates a test plan from the curated scope.

## Objectives
1. Prevent the planner from inventing functionality that was not explicitly approved.
2. Surface contradictions or missing information before any execution starts.
3. Produce tighter, more actionable test plans that align with the clarified scope.

## Pass 1 — Scope Triage Agent
- **Inputs**: Raw user/PRD text, plus optional context (environment URL, credentials).
- **Outputs**:
  - `in_scope`: Plain-language summary of features, flows, and constraints that are explicitly allowed.
  - `explicit_exclusions`: Items the user ruled out (e.g., “ONLY the admin part”).
  - `ambiguous_points`: Requirements mentioned but not clearly scoped; keep neutral language.
  - `blocking_questions`: Contradictions or missing details that make planning unsafe.
- **Behavior Rules**:
  - Treat direct scope statements as authoritative. When conflicts appear, log them instead of guessing.
  - If `blocking_questions` is non-empty, halt the pipeline and print the questions in a readable block—do not proceed to Pass 2.
  - Otherwise emit a concise, ready-for-planner brief combining `in_scope` and `explicit_exclusions`.

## Pass 2 — Test Planner Agent
- **Inputs**: Pass 1 brief plus a reminder that no new functionality may be inferred.
- **Behavior Rules**:
  - Generate test cases strictly from `in_scope`.
  - Acknowledge exclusions so that multi-field actions stay distinct and out-of-scope views are ignored.
  - If `ambiguous_points` remain but are non-blocking, call them out in the final plan notes for human follow-up.

## Implementation Notes
- **Prompt Design**:
  - Create a dedicated triage system prompt emphasizing scope extraction, contradiction detection, and “do not plan if unsure.”
  - Update the planner prompt to reference the Pass 1 brief (e.g., “Use the curated scope below; do not introduce functionality beyond it.”)
- **Control Flow**:
  - Add orchestrator logic that runs Pass 1, checks for blocking questions, and decides whether to continue.
  - Provide a consistent formatted output when the pipeline stops after Pass 1 so users can answer the questions and retry.
- **Testing**:
  - Craft fixtures covering: clear scope (Admin-only), conflicting scope (Admin vs. FMC), and missing prerequisites (no URL).
  - Ensure regression tests confirm the planner never references out-of-scope features.

## Open Questions
- Should Pass 1 attempt light auto-resolution (e.g., drop FMC sections automatically when Admin-only is declared), or always flag them for confirmation?
- How do we surface `ambiguous_points` to action agents later—inline in plans, or as side metadata?
- Do we need a user-facing CLI switch to skip Pass 1 for trusted inputs?
