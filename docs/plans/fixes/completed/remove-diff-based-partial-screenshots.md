# Remove Diff-Based Partial Screenshot Cropping

## Problem

The current visual-state pipeline still allows persisted screenshots to inherit a crop that was selected for provider follow-up efficiency.
That model-facing crop is still influenced by the changed region of the screen.

The issue is narrower than this plan originally assumed:

- Cartography gating already prevents some unknown-target cases from using partial screenshots.
- But when a partial follow-up frame is selected, the same cropped bytes are still persisted as action "after" screenshots, promoted to step screenshots, and embedded in reports.

That is not acceptable for report artifacts because text entry and other small visual changes can produce crops that show only a fragment of a control instead of the full UI element.

## Goal

Change persisted screenshot behavior so it follows this rule:

- If the intended UI element has a known bounding box, crop around that element with a small fixed context margin.
- If the intended UI element does not have a known bounding box, use a full-screen screenshot.

The changed-region heuristic may still be used for model follow-up optimization, but it must no longer influence the crop that is persisted for reports, bug evidence, or step screenshots.

## Non-Goals

- Do not redesign cartography.
- Do not change how action coordinates are interpreted.
- Do not optimize token usage in a way that reintroduces diff-driven crops.

## Proposed Changes

1. Separate model-visible follow-up frames from persisted artifact frames.
2. Remove changed-region bounds as an input to persisted screenshot cropping.
3. Keep element-aware partial screenshots only when the target element is explicitly known and trusted by session cartography.
4. When the target element is unknown, absent, or cannot be matched confidently, persist a full-screen capture instead of a partial crop.
5. Preserve enough context margin around known elements so the crop shows the whole control and nearby UI context.
6. Update report and evidence assumptions so persisted partial screenshots are element-based only, never diff-based.

## Expected Behavior After Change

- Typing into a field produces either:
  - an element-based crop showing the full field, or
  - a full-screen screenshot.
- Provider follow-up optimization can still use local diff plus target information internally, but report artifacts do not expose that diff-driven crop directly.
- Tiny text-only crops no longer appear in reports.
- Cropped screenshots become more stable and easier to read across runs.

## Validation

Add or update tests to cover:

- known trusted target -> persisted partial screenshot shows the whole element
- missing or untrusted target -> persisted full-screen screenshot is used
- model-visible patch selection can remain diff-driven without affecting persisted artifacts
- typing actions do not create text-only persisted crops
- report artifacts render valid screenshots for both partial and full-screen cases

## Acceptance Criteria

- No persisted report or step screenshot is cropped solely from changed pixels.
- Every persisted partial screenshot is traceable to a known trusted target element.
- If no trusted target element is known, the stored artifact is full-screen.
- Existing diff-driven follow-up patches can continue to support provider efficiency without leaking into report artifacts.

## Risks

- More full-screen screenshots may increase artifact size and token usage.
- If element discovery is weak in some flows, the system may fall back to full-screen more often than desired.

## Rollout

1. Introduce a dedicated artifact frame alongside the model-visible follow-up frame.
2. Update tests.
3. Run report-generation validation on a representative mobile run.
4. Review a sample of report screenshots manually to confirm readability.
