# Packaging & Installer Plan

## Overview

This plan covers everything needed to make `pip install haindy` work cleanly,
followed by a first-run setup experience that rivals Flutter's onboarding quality.

---

## Part 1 — Fix Packaging

### 1.1 Rename `src/` to `haindy/`

The package is currently structured as `src/` and imported internally as
`src.something`. When installed via pip, Python sees the top-level package name
as `src`, which conflicts with any other project using the same layout. The
package must be renamed so it installs as `haindy`.

Steps:
- Rename the `src/` directory to `haindy/`
- Update all internal imports from `src.X` to `haindy.X`
- Update the entry point in `pyproject.toml`:
  ```toml
  haindy = "haindy.main:main"
  ```
- Update `[tool.setuptools.packages.find]` to include `haindy*`
- Update `CLAUDE.md` references to the old `src/` path table
- Update any `python -m src.main` references in docs to `python -m haindy.main`

### 1.2 Fix GitHub URLs

`pyproject.toml` currently points to `fkeegan/haindy`. Update to the org:

```toml
[project.urls]
Homepage = "https://github.com/Haindy/haindy"
Repository = "https://github.com/Haindy/haindy"
Issues = "https://github.com/Haindy/haindy/issues"
```

### 1.3 Loosen dependency pins

Published packages must not pin exact versions (`==`) in `[project.dependencies]`
because this creates conflicts in users' environments. Use minimum versions (`>=`)
or compatible release (`~=`) instead.

The `requirements.lock` file keeps exact pins for reproducible development installs
and is unaffected by this change.

Example:
```toml
# Before
"openai==2.23.0"

# After
"openai>=2.23.0"
```

Apply to all entries in `[project.dependencies]`. Keep `[project.optional-dependencies]`
(dev tools) exactly pinned since those are not published constraints.

### 1.4 Add classifiers and trove metadata

Add to `pyproject.toml`:
```toml
classifiers = [
    ...
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Topic :: Software Development :: Testing :: Acceptance",
]
```

---

## Part 2 — First-Run Wizard & `haindy setup`

### 2.1 First-run detection

On every invocation of `haindy` (any command), check whether setup has been
completed. Detection heuristic: absence of `~/.haindy/setup_complete` marker
file (written at the end of a successful setup).

If not set up, exit 1 with:

```
Haindy is not set up yet. Run:

  haindy setup

Or, if you have Claude Code, Codex, or OpenCode installed, install the setup skill:

  haindy setup --install-skill

Then run /haindy-setup inside your AI coding tool.
```

Exceptions — these commands bypass the gate and always work:
- `haindy --help` / `haindy -h`
- `haindy --version`
- `haindy setup`
- `haindy doctor`

### 2.2 `haindy setup` command

New top-level command. Runs an interactive wizard via `rich` (already a
dependency). Wizard flow:

```
Step 1: Welcome + brief explanation of what Haindy needs
Step 2: Detect environment (OS, installed tools, existing config)
Step 3: Offer skill install if AI CLI detected (see 2.3)
Step 4: Dependency check (see 2.4)
Step 5: Credential setup (delegates to existing --auth login flow)
Step 6: Credential setup (delegates to existing --auth login flow)
Step 7: Run doctor automatically (see 2.5)
Step 8: Write ~/.haindy/setup_complete marker only if doctor passes
```

The wizard must be skippable at any point with Ctrl+C, leaving a partial
setup that the user can resume with `haindy setup`.

Non-interactive mode: `haindy setup --non-interactive` (or when stdin is not a
tty) skips prompts and only performs checks, printing results. Good for CI.

### 2.3 Skill install detection and offer

During setup, detect which AI CLIs are installed by probing `PATH`:

| CLI | Binary to probe | Global skill path |
|-----|----------------|-------------------|
| Claude Code | `claude` | `~/.claude/skills/haindy-setup/` |
| OpenAI Codex | `codex` | `~/.agents/skills/haindy-setup/` |
| OpenCode | `opencode` | `~/.config/opencode/skills/haindy-setup/` |

For each detected CLI, prompt:

```
We found Claude Code on your system.
Install the Haindy setup skill to Claude Code? [Y/n]
```

If yes:
1. Copy `.agents/skills/haindy-setup/SKILL.md` (bundled with the package)
   to the target path, creating directories as needed.
2. Print:
   ```
   Skill installed. Open Claude Code and run:
     /haindy-setup
   ```

The skill files are bundled with the pip package under `haindy/skills/` and
declared in `pyproject.toml` package data.

Also install the `haindy` operator skill (`.agents/skills/haindy/SKILL.md`)
to the same paths alongside `haindy-setup` — both are always installed together.

If no CLI is detected, skip this step silently — do not make the user feel
like they are missing out.

### 2.4 Dependency check (per-OS)

Do NOT run package managers automatically. Inspect the environment and tell
the user what to install if anything is missing.

Desktop is the **mandatory** backend — the wizard ensures at least one desktop
backend is fully configured. Android and iOS are **optional** and only checked
if the user opts in during setup or explicitly runs `haindy doctor --all`.

The wizard is OS-aware: macOS users never see Linux instructions and vice versa.
iOS setup is only offered on macOS (it is not supported on Linux).

#### macOS (desktop — mandatory)

Check:
- `pynput` and `mss` — already installed via pip, confirmed via `importlib.util.find_spec`
- Accessibility permission — detect via `osascript -e 'tell application "System Events" to get name of first process'`;
  non-zero exit means not granted; print System Settings path
- Screen Recording permission — detect by capturing a 1x1 screenshot with `mss`
  and checking if the result is solid black; print System Settings path if so

#### Linux (desktop — mandatory)

Check:
- `ffmpeg` — `shutil.which("ffmpeg")`
- `xdotool` — `shutil.which("xdotool")`
- `xclip` — `shutil.which("xclip")`
- `/dev/uinput` write access — `os.access('/dev/uinput', os.W_OK)`
- `DISPLAY` — `os.environ.get("DISPLAY")`

For each missing item, print the install command for the detected distro
(detect via `/etc/os-release`), but do NOT run it.

#### Android (optional, any OS)

Only checked if user opts in during wizard or runs `haindy doctor --android`:
- `adb` — `shutil.which("adb")`; if missing, print install instructions per OS
- Run `adb devices` and display output so user can confirm device is listed

#### iOS (optional, macOS only)

Only offered on macOS. Only checked if user opts in:
- `idb-companion` — `shutil.which("idb_companion")`; if missing: `brew install idb-companion`
- `fb-idb` Python package — `importlib.util.find_spec("idb")`

### 2.5 `haindy doctor` command

Standalone subcommand that can be called at any time:

```bash
haindy doctor
```

Runs all checks from 2.4 and prints a table:

```
Component               Status    Notes
--------------------    -------   ---------------------------
Python >= 3.10          OK        3.14.0
pip package             OK        haindy 0.1.0
OpenAI credentials      OK
Anthropic credentials   MISSING   run: haindy --auth login anthropic
Accessibility (macOS)   OK
Screen Recording        OK
ffmpeg                  N/A       not on Linux
adb                     MISSING   install: brew install android-platform-tools
idb-companion           N/A       not on iOS backend
```

Exit code 0 if all required components are present, 1 if anything required
is missing. Optional components that are absent print as `N/A` not `MISSING`.

### 2.6 Skill bundling in the pip package

The two existing skill directories need to be included in the pip package:

```
haindy/
  skills/
    haindy-setup/
      SKILL.md
    haindy/
      SKILL.md
```

`pyproject.toml` package data:
```toml
[tool.setuptools.package-data]
"haindy.skills" = ["**/*.md"]
"haindy.monitoring" = ["templates/*.j2"]
```

At runtime, locate bundled skills via:
```python
from importlib.resources import files
skill_path = files("haindy.skills").joinpath("haindy-setup/SKILL.md")
```

---

## Part 3 — PyPI Publication

### 3.1 Build and test locally

```bash
pip install build twine
python -m build
pip install dist/haindy-0.1.0-py3-none-any.whl --force-reinstall
haindy --version
haindy doctor
```

### 3.2 Publish to TestPyPI first

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ haindy
```

Verify the install flow works end-to-end on a clean environment.

### 3.3 Publish to PyPI

```bash
twine upload dist/*
```

---

## Part 4 — Documentation Updates

### 4.1 Update `docs/RUNBOOK.md`

- Replace venv-based setup instructions with `pip install haindy` as the
  primary install path
- Keep venv instructions under a "Development setup" section
- Add `haindy setup` and `haindy doctor` to the setup section

### 4.2 Update `docs/plans/phases/completed/PHASE_19_PACKAGING_DOCUMENTATION.md`

Mark as superseded by this plan. Add a note pointing here.

### 4.3 Update `CLAUDE.md`

- Update the source path table (`src/` -> `haindy/`)
- Update `python -m src.main` references
- Add `haindy setup` and `haindy doctor` to the Commands section

### 4.4 New doc: `docs/design/INSTALLER_AND_SETUP.md`

Describes the setup experience design decisions:
- First-run detection approach
- Skill install targets per CLI tool
- Doctor check matrix
- Non-interactive / CI behavior

---

## Implementation Order

1. Rename `src/` -> `haindy/` and fix all imports (Part 1.1) — highest risk, do first
2. Fix `pyproject.toml` (URLs, dep pins, package data) (Parts 1.2, 1.3, 1.4, 2.6)
3. Implement `haindy doctor` (Part 2.5) — self-contained, good to test early
4. Implement `haindy setup` wizard (Part 2.2, 2.3, 2.4)
5. Implement first-run detection (Part 2.1)
6. Update docs (Part 4)
7. Build, test locally, publish (Part 3)

---

## Design Decisions (resolved 2026-03-24)

- Dependency installs are instructed, not automated (Flutter model)
- Skill install bridges the user to their own AI tool; haindy does not invoke
  the AI CLI itself
- Launch with Claude Code, Codex, and OpenCode skill targets (all share agentskills.io format)
- Both skills (`haindy-setup` and `haindy`) installed together, always
- `setup_complete` marker written only after doctor passes
- First-run gate is a hard block (exit 1); only `--help`, `--version`, `setup`, `doctor` bypass it
- Desktop is the mandatory backend; Android and iOS are optional and OS-gated
  (iOS offered on macOS only; Linux-specific checks never shown to macOS users)
- `haindy setup` runs doctor automatically at the end; `--auth` is the separate
  path for credential updates post-setup
- Version: 0.1.0
