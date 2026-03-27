# Contributing to HAINDY

Thanks for your interest in contributing. This guide covers the development workflow and expectations for pull requests.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e ".[dev]"
```

## Code quality

All PRs must pass these checks:

```bash
ruff check .          # lint
ruff format --check . # format check
mypy haindy           # type check
pytest                # tests
```

Run `ruff format .` to auto-format before committing.

## Style guidelines

- Python 3.10+ -- use modern syntax (`X | Y` unions, `match` statements where appropriate)
- `ruff` handles formatting and import sorting -- no need to run `black` or `isort` separately
- Type annotations on all public functions
- No emojis in code, comments, commit messages, or PR descriptions

## Pull requests

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all checks pass locally
4. Open a PR against `main` with a description of what changed and why

Keep PRs focused on a single concern. If you find an unrelated issue while working, open a separate PR for it.

## Reporting bugs

Open an issue using the bug report template. Include:
- Steps to reproduce
- Expected vs actual behavior
- Platform (Linux, macOS, Android, iOS) and Python version
- Provider in use (OpenAI, Google, Anthropic)

## Feature requests

Open an issue using the feature request template. Describe the use case and why existing functionality doesn't cover it.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
