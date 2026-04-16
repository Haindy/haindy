"""Pre-filled GitHub issue URLs for user feedback and bug reports.

No telemetry, no outbound requests. Generates a URL that the user (or the
human behind an AI coding agent) can click to land on github.com with an issue
body pre-populated with run context. The user still reviews and edits before
submitting.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import urlencode


def _detect_version() -> str:
    try:
        return version("haindy")
    except PackageNotFoundError:
        return "unknown"


_HAINDY_VERSION = _detect_version()

ISSUE_URL_BASE = "https://github.com/Haindy/haindy/issues/new"
ISSUE_LABELS = "user-feedback"
OPT_OUT_ENV_VAR = "HAINDY_NO_FEEDBACK_URL"

# GitHub tolerates ~8KB URLs. Leave headroom for encoding overhead and base URL.
MAX_BODY_CHARS = 6000
MAX_ERROR_SNIPPET_CHARS = 2000


def feedback_enabled() -> bool:
    """Return ``False`` when the user has opted out via env var."""

    return os.environ.get(OPT_OUT_ENV_VAR, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len("\n... [truncated]")] + "\n... [truncated]"


def _render_body(
    *,
    command: str,
    run_id: str | None,
    exit_reason: str | None,
    error: str | None,
) -> str:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    error_block = _truncate(error.strip(), MAX_ERROR_SNIPPET_CHARS) if error else "_(none)_"

    body = (
        "<!-- Pre-filled by HAINDY. Edit freely before submitting. -->\n"
        "\n"
        f"**Command:** `haindy {command}`\n"
        f"**HAINDY version:** {_HAINDY_VERSION}\n"
        f"**Platform:** {_platform.platform()}\n"
        f"**Python:** {python_version}\n"
        f"**Run ID:** {run_id or 'n/a'}\n"
        f"**Exit reason:** {exit_reason or 'n/a'}\n"
        "\n"
        "### What happened\n"
        f"{error_block}\n"
        "\n"
        "### What I expected\n"
        "\n"
        "\n"
        "### Steps to reproduce\n"
    )
    return _truncate(body, MAX_BODY_CHARS)


def _render_title(command: str, exit_reason: str | None) -> str:
    tail = exit_reason or "issue"
    return f"[haindy {_HAINDY_VERSION}] {command}: {tail}"


def build_issue_url(
    *,
    command: str,
    run_id: str | None = None,
    exit_reason: str | None = None,
    error: str | None = None,
) -> str | None:
    """Build a pre-filled GitHub issue URL for the given run context.

    Returns ``None`` when the user has opted out via ``HAINDY_NO_FEEDBACK_URL``.
    """

    if not feedback_enabled():
        return None

    params = {
        "title": _render_title(command, exit_reason),
        "body": _render_body(
            command=command,
            run_id=run_id,
            exit_reason=exit_reason,
            error=error,
        ),
        "labels": ISSUE_LABELS,
    }
    return f"{ISSUE_URL_BASE}?{urlencode(params)}"


def console_feedback_hint(url: str) -> str:
    """Return a Rich-formatted one-liner pointing users at the issue URL."""

    return f"[dim]Hit a bug or have feedback? Open a pre-filled issue:[/dim] [link]{url}[/link]"
