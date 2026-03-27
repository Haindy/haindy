# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
