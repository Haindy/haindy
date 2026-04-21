# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-21

### Added

- Windows desktop automation via `haindy/windows/` using `pynput` for input and `mss` for screen capture
- Windows tool-call mode support using a detached daemon plus TCP loopback transport instead of Unix domain sockets
- Windows-specific runtime settings, screenshot and coordinate-cache paths, doctor checks, and RUNBOOK coverage
- New Windows driver test coverage in `tests/test_windows_driver.py`

### Changed

- Split desktop host implementations into explicit `haindy/linux/`, `haindy/macos/`, and `haindy/windows/` modules
- Moved shared driver logic into `haindy/core/` so desktop-platform behavior is clearer and easier to extend
- Updated desktop environment detection, OpenAI computer-use environment mapping, and host-specific system prompts for Windows
- Expanded packaging metadata and dependency markers so Windows installs receive the correct desktop dependencies

### Fixed

- Setup wizard execution inside an already-running asyncio event loop
- Windows process liveness checks for tool-call daemon management
- Windows console-script detection in smoke tests when `pip` installs `haindy.exe`

## [0.1.0] - 2026-03-26

Initial public release.

### Added

- **Tool-call mode**: JSON CLI for coding agents to drive computer-use sessions interactively (`session`, `act`, `test`, `screenshot` commands)
- **Batch mode**: autonomous test planning and execution from requirements files (`haindy run`)
- **Multi-provider computer use**: OpenAI, Google Gemini, and Anthropic Claude as interchangeable computer-use backends
- **Cross-platform automation**: Linux/X11 (uinput), macOS (pynput + mss), Android (ADB), iOS (idb)
- **Agent pipeline**: scope triage, test planner, situational agent, action agent, test runner
- **Setup wizard**: `haindy setup` for interactive first-run configuration, `haindy doctor` for environment verification
- **Credential management**: system keychain storage via `haindy auth`, with encrypted file fallback
- **Provider management**: `haindy provider set` and `haindy provider set-computer-use` for switching AI providers
- **Settings file**: `~/.haindy/settings.json` for persistent non-secret configuration
- **Bundled skills**: auto-installed for Claude Code, Codex CLI, and OpenCode during setup
- **Caching**: plan cache, situational cache, and execution replay cache
- **Reporting**: HTML test reports with screenshots, JSONL execution logs, model call tracing
