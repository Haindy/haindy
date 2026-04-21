---
name: haindy-release
description: Use when preparing, validating, or publishing a HAINDY release from this repository, bumping package versions, updating the changelog or release-facing docs, or verifying the tag-driven PyPI and GitHub release flow. Covers required local validation, release commit and tag steps, and HAINDY-specific version guardrails such as keeping `haindy version` wired to installed package metadata instead of a hardcoded string.
metadata:
  short-description: Validate and publish HAINDY releases
---

# HAINDY Release Workflow

Use this skill for repo-local release work only.

## Release surfaces

- `pyproject.toml`: package version source of truth
- `CHANGELOG.md`: add the new release entry
- `README.md`, `docs/RUNBOOK.md`, `.env.example`: update when user-visible behavior, supported platforms, setup, or config changed
- `haindy/main.py`: `show_version()` must read installed package metadata, not a hardcoded string
- `haindy/tool_call_mode/cli.py`: keep the `version` parser entry wired if CLI/parser work changes
- `tests/test_main.py`: keep regression coverage for the version command
- `.github/workflows/release.yml`: tag-driven release flow (`v*`)

## Workflow

1. Inspect the current release state.

```bash
git status --short
git tag --sort=-v:refname | head
rg -n "version =|Version:|package_version|haindy version|release" \
  pyproject.toml haindy tests README.md CHANGELOG.md .github/workflows/release.yml
```

2. Choose the next version with SemVer.

- New supported platforms or major feature milestones usually justify a minor bump.
- Keep the change targeted; do not backfill unrelated old changelog history unless asked.

3. Update release files together.

- Bump `pyproject.toml`
- Add the changelog entry
- Refresh release-facing docs if the change altered supported platforms, setup, or behavior
- If CLI or parser code changed, verify both the parser path and the rendered `haindy version` output

4. Refresh the local editable environment before validating.

```bash
source .venv/bin/activate
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

5. Run the required checks before any release commit or tag.

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy haindy
.venv/bin/pytest
```

Recommended artifact check:

```bash
.venv/bin/pip install build
.venv/bin/python -m build
```

6. Verify the installed CLI reports the same version as package metadata.

```bash
.venv/bin/python -c "import importlib.metadata as m; print(m.version('haindy'))"
.venv/bin/haindy version
```

If those disagree, fix the CLI before releasing.

7. Commit and publish.

```bash
git add <release-files>
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

8. Watch the release workflow and confirm the public outputs.

```bash
gh run list --workflow release.yml --limit 3
gh run watch <run-id> --exit-status
gh release view vX.Y.Z --json url,name,tagName,publishedAt,isDraft
python3 - <<'PY'
import json, urllib.request
with urllib.request.urlopen('https://pypi.org/pypi/haindy/json', timeout=20) as r:
    print(json.load(r)["info"]["version"])
PY
```

## Guardrails

- Do not trust a green editable checkout alone after `pyproject.toml` changes; reinstall so metadata and entry points refresh.
- Do not hardcode version strings in CLI output. `haindy version` must read installed package metadata.
- If `haindy version` breaks, check both:
  - parser registration in `haindy/tool_call_mode/cli.py`
  - dispatch and output in `haindy/main.py`
- When release work changes shared contracts, update the companion surfaces together:
  - backend semantics: `haindy/runtime/environment.py`, `haindy/config/settings.py`, `.env.example`, `README.md`, relevant tests
  - runtime/config surface: `haindy/config/settings.py`, `.env.example`, `README.md`, `docs/RUNBOOK.md`, relevant tests
- Mention workflow warnings or annotations after a successful release if they indicate maintenance work the team should schedule.

## Quick checks

- `git status --short` is clean after the release commit and tag push
- `git ls-remote --tags origin refs/tags/vX.Y.Z refs/tags/vX.Y.Z^{}` shows both the tag object and peeled commit
- `gh release view vX.Y.Z` succeeds
- PyPI reports `X.Y.Z`
