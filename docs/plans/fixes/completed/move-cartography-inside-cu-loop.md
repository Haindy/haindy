# Move Cartography Inside the CU Loop

## Problem

Cartography is currently generated as a separate non-CU model call.
That has three costs:

- extra latency before or during action execution
- extra token usage outside the active CU conversation
- weaker session coherence because the localization state is produced out-of-band instead of inside the same reasoning loop that is executing the action

This is the opposite of the intended design.

## Goal

Move cartography into the active Computer Use session so it becomes an internal part of the same action conversation rather than a separate side call.

The desired model is:

- one action
- one CU session
- cartography generated and maintained inside that session
- cropped follow-up screenshots only when the session already has a valid target map

## Desired Behavior

1. The CU session starts from the normal action prompt.
2. When the session needs localization support, it performs that as an in-session reasoning step.
3. The resulting target map remains part of the action-session state.
4. Follow-up turns can reuse that session-local target map without issuing a separate model request.
5. If the map becomes stale, the CU session refreshes it internally instead of calling a separate external cartography pass.

## Non-Goals

- Do not redesign coordinate remapping.
- Do not change report generation in this fix.
- Do not introduce a second parallel “mapping subsystem” outside the CU session.

## Why This Change

- It reduces extra request overhead.
- It keeps localization and action execution in one conversation.
- It matches the intended architecture: the model should carry the visual grounding inside the active action loop.
- It avoids paying separate model-call cost for something that is really part of the same reasoning task.

## Proposed Design

### 1. Treat Cartography as Session State

Replace the current out-of-band cartography request model with session-owned state.
The CU session should be able to hold:

- the current full-screen reference snapshot
- the current target map derived from that reference
- metadata describing when that map was last refreshed

### 2. Build the Map Through In-Session Structured Output

When a full-screen reference snapshot is needed, the CU session should ask for localization as part of the same conversation.
This can be done by introducing an explicit in-session phase or protocol such as:

- localize target on current screen
- store target bounds in session state
- continue normal execution

The exact transport differs by provider, but the architecture should be the same.

### 3. Reuse Map Across the Full Action Session

The map should live for the lifetime of the action session unless invalidated.
Normal follow-up turns should rely on session state rather than repeating localization.

### 4. Refresh Only on Clear Triggers

Refresh cartography only when one of these happens:

- navigation or major screen transition
- the current target is no longer trustworthy
- the map is missing
- the session has exceeded a configured refresh interval
- the action explicitly changes to a different target that requires a new full-screen reference

### 5. Keep Partial Screenshots Dependent on Session Map Availability

Partial screenshots should only be used when the current action session already has a valid target map.
If the session does not have one, use a full-screen reference until in-session cartography is established.

## Implementation Areas

### Session Lifecycle

Update the Computer Use session layer so cartography becomes part of the per-action session state rather than an external helper call.

### Provider Adapters

Each provider implementation will need a way to:

- request localization in-session
- parse structured localization output
- persist that state across follow-up turns in the same action session

The provider-specific mechanics may differ, but the session contract should be provider-neutral.

### Visual-State Planner

The visual-state planner should stop depending on an external cartography generator call.
Instead, it should consume session-owned localization state and make decisions based on:

- map available -> allow element-based partial screenshots
- map missing -> use full-screen reference

### Logging and Debugging

Logging should distinguish:

- full-screen reference refresh
- in-session cartography refresh
- map reuse
- map invalidation

This is important so we can verify that cartography is actually staying inside the CU loop and is not being regenerated unnecessarily.

## Risks

- Provider APIs may make structured in-session localization easier in some backends than others.
- If the in-session localization protocol is poorly designed, it could confuse the execution loop instead of helping it.
- If refresh rules are too aggressive, we will still remap too often and lose the intended benefit.
- If refresh rules are too lax, cropped follow-up screenshots may rely on stale localization.

## Validation

Add tests for:

- one action session creates localization without a separate out-of-band cartography request
- localization state survives across multiple follow-up turns in the same action
- navigation or equivalent major transition invalidates or refreshes the map
- partial screenshots are used only when session-local localization is present
- provider-neutral session behavior remains consistent across OpenAI and Google paths

## Acceptance Criteria

- Cartography is no longer generated through a separate non-CU model call during normal action execution.
- Localization state is created and reused inside the active CU session.
- Repeated turns in one action do not remap unless a real refresh trigger occurs.
- Token and latency overhead from cartography is reduced relative to the current design.
- The design remains provider-neutral at the session contract level.

## Rollout

1. Define the provider-neutral session contract for in-loop cartography.
2. Implement the session-state plumbing.
3. Update one provider path end-to-end.
4. Add tracing to confirm map reuse and refresh behavior.
5. Extend the approach to the second provider path.
6. Remove the old out-of-band cartography path once parity is confirmed.
