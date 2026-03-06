# Plan: Step Interpreter Visual Context & Navigation Handling

## Objectives
- Ground the Test Runner’s step interpretation in the actual UI by attaching screenshots and richer scenario context.
- Allow the interpreter to short-circuit redundant navigation via a new `skip_navigation` action type and make the Action Agent respect it.
- Preserve deterministic behavior across test cases by sharing prior-step/test-case details.

## Work Breakdown

1. **Initial Screenshot Capture & Cache**
   - Hook into the executor setup to wait for the browser launch URL to stabilize (respect the existing stabilization wait).
   - Capture an “initial screenshot” once per test run, persist it via the debug logger, and cache the path/bytes for downstream use.
   - Expose cached screenshots through the Test Runner so subsequent steps can reference the latest available image without re-capturing.

2. **Per-Step Screenshot Context**
   - Before calling the interpreter, determine the screenshot to use: initial screenshot for the first step, otherwise the latest “after” screenshot from the previous executed action/step.
   - Embed screenshot metadata in the interpretation request (e.g., path plus note that the Action Agent already sees the raw image) and retain the link in the step log.
   - Ensure fallback messaging (“no previous test cases or steps”) when no screenshot/history exists.

3. **Richer Test Case & Sequence Context**
   - Extend the interpretation prompt to include:
     - Entire current test case definition (title, step list, expected results).
     - Current step index, previous step summary/result, and next step summary.
     - When starting a new test case (not the first overall), provide the preceding test case title and its final step.
   - Serialize this data in a structured block to keep the model aware of “where” (via screenshot) and “when” (via ordering).

4. **Schema Update with `skip_navigation`**
   - Add `skip_navigation` to the allowed action types in the interpreter schema and propagate through:
     - `ActionType` enum / pydantic models.
     - Test Runner action handling and reporting.
     - Documentation of action semantics.

5. **Action Agent Short-Circuit Logic**
   - Detect `skip_navigation` actions in the Action Agent and return a successful `EnhancedActionResult` without invoking Computer Use.
   - Record the decision in logs/reports so observers see that navigation was intentionally skipped.

6. **Prompt & Documentation Alignment**
   - Update the interpreter prompt template to describe the new `skip_navigation` behavior and screenshot usage expectations.
   - Refresh design docs (e.g., `docs/design/AGENT_ARCHITECTURE_DESIGN.md`) to note the revised responsibility split.

7. **Validation & Follow-Up**
   - Add or update tests (unit/integration) that simulate first-step navigation and confirm `skip_navigation` is emitted when the screenshot already matches the expected state.
   - Leave full end-to-end scenario execution to the user for manual observation; note any recommended scenario in the handoff but do not run the tool here.
   - Document any TODOs for future optimization (e.g., prompt size management) if discovered.

## Risks & Mitigations
- **Prompt Size Growth:** Keep data structured and leverage markdown headings; monitor token counts but proceed per current cost tolerance.
- **Screenshot Staleness:** Always refresh cache after actions complete; include safeguards when previous steps fail and no “after” image exists.
- **Enum/Schema Drift:** Ensure all consumers (serialization, reports, analytics) recognize the new action type to avoid runtime errors.
