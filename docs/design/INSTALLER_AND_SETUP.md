# Installer and Setup Design

## Overview

HAINDY ships a guided setup experience to reduce first-run friction. Two
commands cover the full lifecycle:

- `haindy setup` — interactive wizard that walks a new user through every
  required configuration step
- `haindy doctor` — standalone check that validates the environment at any time

## First-Run Detection

- Every HAINDY invocation checks for `~/.haindy/setup_complete` before doing
  any real work.
- If the marker is missing, HAINDY exits with code 1 and prints instructions
  to run `haindy setup`.
- The following commands bypass first-run detection and always run regardless
  of whether setup has been completed:
  - `--help`, `-h`
  - `--version`
  - `haindy setup`
  - `haindy doctor`

## `haindy setup` Wizard

The wizard is an 8-step interactive flow built with Rich:

1. Welcome and prerequisites overview
2. OS dependency check (platform-specific)
3. Credential configuration — delegates to the existing `--auth login` flow for
   each provider
4. Settings file creation (`~/.haindy/settings.json`) with sensible defaults
5. AI CLI detection — scans PATH for `claude`, `codex`, and `opencode`
6. Skill installation — offers to install setup and operator skills for each
   detected CLI (see Skill Installation Targets below)
7. Run `haindy doctor` to validate the completed setup
8. Write `~/.haindy/setup_complete` marker only after doctor passes

Additional behaviour:

- Ctrl+C exits gracefully; partial setup is preserved so the wizard can be
  resumed later
- `--non-interactive` flag skips all prompts, performs only checks, and exits
  with code 0 on success or 1 on any failure

## Skill Installation Targets

| CLI | Binary | Setup skill path | Operator skill path |
|-----|--------|-----------------|---------------------|
| Claude Code | `claude` | `~/.claude/skills/haindy-setup/` | `~/.claude/skills/haindy/` |
| OpenAI Codex | `codex` | `~/.agents/skills/haindy-setup/` | `~/.agents/skills/haindy/` |
| OpenCode | `opencode` | `~/.config/opencode/skills/haindy-setup/` | `~/.config/opencode/skills/haindy/` |

## Doctor Check Matrix

| Check | macOS | Linux | Required |
|-------|-------|-------|----------|
| Python >= 3.10 | yes | yes | yes |
| haindy package | yes | yes | yes |
| OpenAI credentials | yes | yes | yes |
| Anthropic credentials | yes | yes | yes |
| Google credentials | yes | yes | yes |
| pynput/mss | yes | N/A | yes |
| Accessibility permission | yes | N/A | yes |
| Screen Recording permission | yes | N/A | yes |
| ffmpeg | N/A | yes | yes |
| xdotool | N/A | yes | yes |
| xclip | N/A | yes | yes |
| /dev/uinput | N/A | yes | yes |
| DISPLAY | N/A | yes | yes |
| adb | optional | optional | no |
| idb-companion | optional | N/A | no |

## Non-Interactive / CI Behavior

- Triggered by `--non-interactive` flag or when stdin is not a tty.
- Skips all prompts, only performs checks and prints results.
- Exit code 0 if all required components are present, 1 otherwise.
- Suitable for use in CI pre-flight steps and Docker image validation.
