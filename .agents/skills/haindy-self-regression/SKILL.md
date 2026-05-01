---
name: haindy-self-regression
description: Use before release-facing, provider, runtime, or surface changes to manually regression-test HAINDY through its own public tool-call commands across configured providers and available surfaces.
metadata:
  short-description: Run HAINDY through its own public commands
---

# HAINDY Self-Regression

This is a repo-local development skill. It is not bundled for HAINDY users and must stay under `.agents/skills/haindy-self-regression/`.

Run it manually before release-facing, provider, runtime, or surface changes. It is not required for every small edit or docs-only commit.

## Boundary

Use only HAINDY commands and workflows:

```bash
haindy doctor
haindy auth status
haindy provider list
haindy provider set-computer-use <provider>
haindy session new --desktop
haindy session new --android [--android-serial <SERIAL>] [--android-app <PACKAGE>]
haindy session new --ios [--ios-udid <UDID>] [--ios-app <BUNDLE_ID>]
haindy session status --session <SESSION_ID>
haindy session list
haindy session close --session <SESSION_ID>
haindy session close --session <SESSION_ID> --force
haindy explore "..." --session <SESSION_ID>
haindy explore-status --session <SESSION_ID>
haindy act "..." --session <SESSION_ID>
```

Do not use ad-hoc scripts, Python helpers, direct SDK probes, raw `adb` or `idb`, X11/macOS/Windows automation commands, browser automation, DOM inspection, Playwright, Selenium, or any other bypass around HAINDY itself.

## Preflight

1. Install the branch build in the local virtual environment.
2. Run:

```bash
haindy doctor
haindy auth status
haindy provider list
haindy session list
```

3. Record the original active computer-use provider from `haindy provider list`. Restore this provider at the end with `haindy provider set-computer-use <provider>`.
4. Keep a list of every opened session ID. Close every session before finishing, using `haindy session close --session <SESSION_ID> --force` if normal close does not work.

## Provider Order

Iterate configured providers in this exact order:

1. `google`
2. `openai`
3. `anthropic`

Skip a provider only when `haindy auth status` or `haindy provider list` shows that it is not configured. Record the skip in the report.

For each configured provider:

1. Run `haindy provider set-computer-use <provider>`.
2. Start a desktop session with `haindy session new --desktop`.
3. Run `haindy explore "Inspect the current desktop and report the visible application or shell state without changing settings." --session <SESSION_ID>`.
4. Poll `haindy explore-status --session <SESSION_ID>` until terminal.
5. Run `haindy act "Move focus to a harmless visible area without opening applications or changing settings." --session <SESSION_ID>`.
6. Run `haindy session status --session <SESSION_ID>`.
7. Close the session.

Desktop `explore` and `act` must run in the same session for each provider.

## Mobile Surfaces

Use only HAINDY session commands against already booted or connected targets.

- Android: try `haindy session new --android` only when HAINDY can connect to an already available target. If HAINDY reports no target, skip Android with the exact HAINDY message.
- iOS: try `haindy session new --ios` only when HAINDY can connect to an already available target. If HAINDY reports no target, skip iOS with the exact HAINDY message.

Do not launch emulators, simulators, desktop apps, browsers, or device tooling outside HAINDY.

For each available mobile surface, repeat the provider loop with that surface's `session new` command, then run an `explore`, poll `explore-status`, run an `act`, run `session status`, and close the session.

## Report

End with a concise table:

| Provider | Surface | Session ID | Commands | Result | Skipped Items | Failures | Reproduction Notes |
|---|---|---|---|---|---|---|---|

Include the original provider, whether it was restored, any sessions force-closed, and the exact command/output clue needed to reproduce failures or skips.
