# User Path Docs — Research / Opinion

Status: exploratory, not a spec. Captures a conversation about closing a gap
we keep hitting when coding agents use HAINDY against a real app.

## Problem

The HAINDY skill does a good job teaching coding agents how to *drive HAINDY*.
The remaining gap is different: coding agents have the app's code in context,
but **code does not equal functionality**. To actually test or exercise the
app, the agent has to infer how a human *uses* it — either by exploring the
running app or by reading code and deducing the intended flow. Both are slow,
token-heavy, and repeated by every agent that touches the project.

The knowledge of how to use the app already exists in the team's heads. It is
simply not written down in a form an agent can consume cheaply.

## Proposal

A new, lightweight document type — a **User Path Doc** (UPD). Not a spec,
not a design doc, not a test plan. Purpose: give a coding agent the minimum
instruction a human would give another human to perform a use case.

Name rationale: the document is organized around *paths* (sequential steps a
user takes to accomplish a goal), and "user paths" reads naturally as a noun
without the "user manual" confusion of implying the audience is an end user.
The audience is agents acting *as* users.

Not a spec. Not a test plan. No assertions. No expectations. Just: "to do X,
here is the sequence of actions a user performs."

### Shape

Plain markdown. Use cases at the top. Each use case has one or more named
paths. Steps are numbered, terse, imperative.

```
Use case: Sign up for a new Coach user

Path 1 — Email sign up:
Starts from: logged-out landing screen
1. Tap Sign Up                          <-- takes you to the sign-up screen
2. Make sure the email radio is selected
3. Enter email and password
4. Check the ToS checkbox
5. Make sure the User Type dropdown is set to "Coach"
6. Tap Create User                      <-- takes you to the code-validation screen
```

Path reuse is explicit and terse:

```
Path 2 — Phone number sign up:
Same as Path 1 except:
1. The phone radio button should be selected
2. A valid phone number is entered instead of email
```

### Design principles

- **High level, human-to-human voice.** The least instruction a person gives
  another person to perform the task. If it reads like a spec, it is too
  formal.
- **No assertions.** This is a capability map, not a test. Negatives are
  tested by antagonizing the happy path; they do not need their own document.
- **Happy paths only.** Exhaustive coverage is a test plan's job.
- **Annotate state transitions inline, not every step.** An arrow goes on a
  step that changes screen/state; steps that do not change state get nothing.
  Density stays low, signal stays high.
- **Entry state at the top of each path.** "Starts from: …" so the path is
  composable with no extra parsing.
- **No formalization beyond markdown.** No screen IDs, no cross-reference
  syntax, no schema. The value is that it reads like a human wrote it for
  another human.

## Why annotations matter more than they look

The inline state-transition arrows are not just reading aids. Because every
path declares its entry state and annotates its exit state, the collection of
paths is effectively a state graph written in prose. That gives us
**composition for free**: "how do I get to screen X to test something there?"
is answered by finding a path whose exit is X and running it first. The
preconditions problem — the thing that actually burns agent tokens today —
collapses into "grep for the path that ends where you need to start."

This is the main argument for the format. Without the transition annotations,
it is just a prettier test plan.

## Tradeoffs and open questions

### Staleness

This is the honest weak point. Code is kept honest by compilers; tests are
kept honest by failures. A UPD rots silently — the UI changes, a
button renames, a flow reorders, and nobody notices until an agent follows
stale steps and wastes a run.

Position: **maintenance is on the humans on the team**, same as design docs,
architecture notes, or API references. No format choice removes that
responsibility.

Possible mitigation, not a prerequisite: an optional HAINDY skill that, given
an existing UPD, can run through it and flag steps that no longer match
reality. Closes the loop nicely — HAINDY already has the machinery to drive a
UI and observe state — but it is a convenience, not the core proposal.

### Authoring

Two modes, both valid:

1. **Human-authored.** Someone who knows the app writes the UPD once.
   Cheap, high-signal, drifts over time.
2. **HAINDY-assisted.** An exploration run produces a draft UPD that a
   human reviews and trims. Closes the "re-discovery every time" loop, but
   needs a review step — unreviewed agent output is exactly the kind of
   low-signal noise the UPD is meant to avoid.

The human review step is non-negotiable in either mode. The value of this
document is that a human vouched for it.

### Scope boundary

Happy paths only. Negatives, edge cases, and error recovery are out of scope
on purpose — they belong in test plans or the test agent's exploration. The
UPD answers "how do I use this?", not "what are all the ways this can
break?"

## What this is not

- Not a replacement for design docs or specs. Those describe *what the system
  is*; the UPD describes *how a user drives it*.
- Not a replacement for test plans. Tests assert; the UPD does not.
- Not a runbook. Runbooks are for operators; the UPD is for end-user flows.
- Not auto-generated ground truth. It is a vouched-for artifact, kept honest
  by the humans who own the feature.

## Next steps (if we pursue this)

- Pick a location convention (`docs/user-paths/` in the target app's repo,
  probably, not HAINDY's).
- Write one real UPD for a real app as a shape test — the format either
  holds up or it reveals missing pieces.
- Decide whether the optional HAINDY "UPD check" skill is worth building,
  and when.
- Decide whether the HAINDY skill should teach coding agents to *look for* a
  UPD before exploring.
