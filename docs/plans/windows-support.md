# Windows Support for HAINDY

## Status

- **Branch:** `feat/windows-support`
- **Milestone 1 (Linux prep):** COMPLETE — all changes committed on the branch.
- **Milestone 2 (Windows VM):** NOT STARTED — resume here in the Windows VM.
- **OpenAI CUA `environment="windows"` verification:** DONE (accepted by API).

### Picking up in a fresh session on Windows

If you're reading this in a new Claude Code session inside a Windows VM, do
this to orient:

1. Confirm you're on the `feat/windows-support` branch:
   `git rev-parse --abbrev-ref HEAD`.
2. Skim the "Milestone 1 summary" section below — that's what's already on the
   branch.
3. Start at "Milestone 2 — Windows VM" and work through the checkbox list in
   order. The stubs at `haindy/windows/*.py` are where the real code goes;
   mirror `haindy/macos/*.py` file-for-file.
4. `haindy/cli/doctor.py` and `docs/RUNBOOK.md` both have a Windows section to
   fill in.
5. Run the verification commands at the end of the M2 section before opening
   PR #2.

## Context

HAINDY's desktop automation today runs on Linux (X11/uinput, xdotool, xclip,
ffmpeg x11grab, GNOME screencast) and macOS (pynput + mss). The Android backend
is cross-platform via ADB.

The next priority is Windows support: let users run HAINDY on Windows both
(a) pointing at an Android device over ADB (mostly free — ADB is
cross-platform) and (b) driving the Windows desktop via computer-use. The
desktop half is the real work: a new `haindy/windows/` driver mirroring the
macOS pattern, plus renaming the misleadingly-named `haindy/desktop/` module to
`haindy/linux/` to make the per-OS layout obvious.

`HAINDY_AUTOMATION_BACKEND=desktop` stays polymorphic — users keep setting
`desktop` and HAINDY picks the right driver by `sys.platform`. No config churn.

## Milestone 1 — Linux prep (PR #1) — COMPLETE

Done on a Linux workstation. The repo still runs green on Linux after each
step; Windows driver stubs raise `NotImplementedError`.

- [x] Relocate shared modules out of `haindy/desktop/` into `haindy/core/`:
  - `haindy/desktop/cache.py` → `haindy/core/coordinate_cache.py`
  - `haindy/desktop/execution_replay.py` → `haindy/core/driver_actions.py`
- [x] Rename `haindy/desktop/` → `haindy/linux/` via `git mv` (class names
  stay; `DesktopDriver`/`DesktopController` describe what they are, not where
  they live).
- [x] Settings: rename `desktop_coordinate_cache_path` →
  `linux_coordinate_cache_path`; add `windows_coordinate_cache_path` with
  default `data/windows_cache/coordinates.json`. Legacy JSON key
  `desktop.coordinate_cache_path` mapped for back-compat.
- [x] `haindy/runtime/environment.py`: `sys.platform == "win32"` branches in
  `coordinate_cache_attribute` and `openai_computer_environment`; added
  `"windows"` / `"win"` / `"win32"` → `"desktop"` aliases.
- [x] Platform dispatch: `WindowsController` wired into `haindy/main.py` and
  `haindy/tool_call_mode/runtime.py`. `ScreenRecorder` instantiation gated on
  `sys.platform == "linux"`.
- [x] Scaffold `haindy/windows/` — `__init__.py`, `controller.py`, `driver.py`,
  `input_handler.py`, `screen_capture.py`. All `start()` methods raise
  `NotImplementedError("Windows driver lands in Milestone 2")`.
- [x] `pyproject.toml`: widened `pynput` and `mss` markers to
  `sys_platform == 'darwin' or sys_platform == 'win32'`.
- [x] `haindy/cli/doctor.py`: Windows branch added (pynput import, mss import,
  Credential Manager keyring backend, `LongPathsEnabled` registry check).
- [x] `haindy/agents/computer_use/support_mixin.py`: desktop system prompt
  branched on host OS (linux / darwin / win32).
- [x] `.gitattributes` at repo root (LF for text, CRLF for
  `.bat`/`.cmd`/`.ps1`, binary markers for images and video).
- [x] This document checked in under `docs/plans/windows-support.md`.

OpenAI CUA `environment="windows"` verified via the REPL call below — keep
returning `"windows"` from `openai_computer_environment`:

```python
from openai import OpenAI
OpenAI().responses.create(
    model="computer-use-preview",
    tools=[{"type":"computer_use_preview","display_width":1920,"display_height":1080,"environment":"windows"}],
    input="say hi",
    truncation="auto",
)
```

## Milestone 2 — Windows VM (PR #2) — PENDING

Done inside a Windows VM on the same branch.

### VM setup

- [ ] Install Python 3.11+ (matches `pyproject.toml` lower bound).
- [ ] `git config --global core.autocrlf false` (defense-in-depth;
  `.gitattributes` already handles this).
- [ ] Install ADB (`winget install Google.PlatformTools`).
- [ ] Enable long paths:
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`.
- [ ] Clone repo, checkout `feat/windows-support`, `python -m venv .venv`,
  `.venv\Scripts\pip install -r requirements.lock`,
  `.venv\Scripts\pip install -e ".[dev]"`.
- [ ] `.venv\Scripts\haindy setup` → `.venv\Scripts\haindy auth login openai`.
  Verify Windows Credential Manager backend via
  `.venv\Scripts\haindy doctor`.

### Implementation (macOS parity)

- [ ] `haindy/windows/driver.py`: flesh out `WindowsDriver(AutomationDriver)`.
  Same lifecycle as `MacOSDriver`. `start()` initializes
  `WindowsScreenCapture` + `WindowsInputHandler`, records
  `_pixel_width/_pixel_height` from the primary display. Implement the full
  `AutomationDriver` interface (click, drag, type, key, scroll, screenshot,
  clipboard, viewport).
- [ ] `haindy/windows/input_handler.py`: `pynput.mouse.Controller` +
  `pynput.keyboard.Controller`. Mirror `haindy/macos/input_handler.py`, except
  the primary modifier is `Key.ctrl` (not `Key.cmd`) and `"meta"` maps to
  `Key.cmd` (the Windows key).
- [ ] `haindy/windows/screen_capture.py`: `mss.mss()` for capture; return PNG
  bytes. Parse PNG header for dimensions — reuse `_parse_png_size` from
  `haindy/macos/driver.py` (lift to `haindy/core/` if both drivers need it).
- [ ] `haindy/windows/controller.py`: thin wrapper owning a `WindowsDriver`
  (mirror `haindy/macos/controller.py`). The M1 stub already passes
  `desktop_screenshot_dir` + `windows_coordinate_cache_path`; add
  `windows_screenshot_dir` / `windows_keyboard_layout` / clipboard settings
  fields if the macOS driver relies on them.
- [ ] Clipboard: match whichever approach `haindy/macos/driver.py` uses
  (pbpaste/pbcopy on mac — on Windows use `pyperclip` or pynput's hooks).
- [ ] `haindy/cli/doctor.py`: replace M1 stub rows with real checks; also add
  a Windows version check (10+).
- [ ] `haindy/tool_call_mode/cli.py`: add an early `sys.platform == "win32"`
  guard that exits with "tool-call mode not yet supported on Windows".
- [ ] `haindy/tool_call_mode/paths.py` line ~216: add
  `hasattr(signal, "SIGKILL")` guard defensively.
- [ ] `docs/RUNBOOK.md`: add a Windows section covering venv activation path
  (`.venv\Scripts\` vs `.venv/bin/`), UAC elevation caveat for pynput, keyring
  backend expectations.
- [ ] `tests/test_windows_driver.py` new file, guarded with
  `@pytest.mark.skipif(sys.platform != "win32")`, mirroring
  `tests/test_macos_driver.py`.

### Explicit non-goals for M2

- **No resolution manager.** Linux needs xrandr to downshift for token
  savings; Windows drives at native resolution.
- **No screen recorder.** GNOME-specific; Windows-native recording is M3+.
- **No tool-call mode daemon on Windows.** `haindy/tool_call_mode/daemon.py`
  and `ipc.py` use Unix domain sockets. Gate CLI entry instead of porting.

### Verification

- [ ] `.venv\Scripts\ruff check .` and `.venv\Scripts\ruff format --check .`
  clean.
- [ ] `.venv\Scripts\mypy haindy` clean.
- [ ] `.venv\Scripts\pytest` — all previously-passing tests still pass (plus
  new `tests/test_windows_driver.py`).
- [ ] `.venv\Scripts\haindy doctor` — all rows green on a clean VM.
- [ ] `.venv\Scripts\haindy test-api` — OpenAI / Google / Anthropic
  connectivity.
- [ ] `.venv\Scripts\haindy run --plan <fixture> --context <fixture>` — full
  scenario against Notepad or Calculator (built-in apps, no third-party
  prerequisites). Screenshots land in `data/screenshots/`, coordinate cache
  lands at `data/windows_cache/coordinates.json`.
- [ ] Android path: `.venv\Scripts\haindy run --mobile --plan ... --context ...`
  with a USB-connected (or Hyper-V WSA) device — verify ADB dispatch works on
  Windows.

## Windows gotchas covered

| Gotcha | Resolution |
|---|---|
| Line endings: test fixtures / jsonlines corrupted by autocrlf | `.gitattributes` — `*.jsonl eol=lf`, `*.json eol=lf` |
| `.bat`/`.cmd`/`.ps1` need CRLF | `.gitattributes` |
| Unix domain sockets in tool-call mode | Guarded off on Windows (M2 non-goal) |
| `signal.SIGKILL` missing on Windows | `hasattr` guard in `tool_call_mode/paths.py` (M2) |
| `signal.SIGHUP` missing on Windows | Already `getattr(signal, name, None)`-safe |
| Venv activation path (`.venv\Scripts\` vs `.venv/bin/`) | `docs/RUNBOOK.md` Windows section (M2) |
| Keyring backend (Windows Credential Manager) | Doctor check verifies `WinVaultKeyring` |
| Long paths (>260 chars) in `haindy_home/sessions/<uuid>/screenshots/` | Doctor warning + registry-key instruction |
| Rich/ANSI on legacy `cmd.exe` | Rich auto-detects; Windows Terminal recommended |
| GNOME screencast only works on Linux | `ScreenRecorder` instantiation gated on `sys.platform == "linux"` (M1) |

## Open assumptions to verify during M2

1. Google Gemini CU on Windows desktop — `ENVIRONMENT_UNSPECIFIED` is returned
   for all desktops today; should Just Work. Smoke-test during M2.
2. Anthropic CU on Windows desktop — tool config is OS-agnostic (only
   `display_width_px`/`display_height_px`); should Just Work. Smoke-test
   during M2.
3. pynput key-event reliability on Windows under UAC-elevated target apps —
   pynput's `SendInput` backend may not deliver events to elevated windows
   unless HAINDY itself runs elevated. Document in M2 RUNBOOK section.
