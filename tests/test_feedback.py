"""Tests for the feedback URL helper."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from haindy import feedback
from haindy.feedback import (
    ISSUE_URL_BASE,
    MAX_ERROR_SNIPPET_CHARS,
    OPT_OUT_ENV_VAR,
    build_issue_url,
    console_feedback_hint,
    feedback_enabled,
)


def test_build_issue_url_contains_expected_fields() -> None:
    url = build_issue_url(
        command="act",
        run_id="run-123",
        exit_reason="element_not_found",
        error="Could not locate the Submit button",
    )
    assert url is not None
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == ISSUE_URL_BASE

    params = parse_qs(parsed.query)
    title = params["title"][0]
    body = params["body"][0]
    labels = params["labels"][0]

    assert "act" in title
    assert "element_not_found" in title
    assert labels == "user-feedback"

    assert "**Command:** `haindy act`" in body
    assert "**Run ID:** run-123" in body
    assert "**Exit reason:** element_not_found" in body
    assert "Could not locate the Submit button" in body
    assert "**HAINDY version:**" in body
    assert "**Platform:**" in body
    assert "**Python:**" in body


def test_build_issue_url_none_context() -> None:
    url = build_issue_url(command="run")
    assert url is not None
    body = parse_qs(urlparse(url).query)["body"][0]
    assert "**Run ID:** n/a" in body
    assert "**Exit reason:** n/a" in body
    assert "_(none)_" in body


def test_build_issue_url_truncates_long_error() -> None:
    huge = "x" * 20000
    url = build_issue_url(
        command="explore",
        run_id="r",
        exit_reason="stuck",
        error=huge,
    )
    assert url is not None
    body = parse_qs(urlparse(url).query)["body"][0]
    assert "[truncated]" in body
    assert len(body) < feedback.MAX_BODY_CHARS + 50
    assert len(url) < 9000  # well under GitHub's ~8KB URL limit + scheme/path


def test_error_snippet_truncation_threshold() -> None:
    exact = "y" * (MAX_ERROR_SNIPPET_CHARS + 500)
    url = build_issue_url(command="run", error=exact)
    assert url is not None
    body = parse_qs(urlparse(url).query)["body"][0]
    assert "[truncated]" in body


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_opt_out_env_var_disables_url(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(OPT_OUT_ENV_VAR, value)
    assert feedback_enabled() is False
    assert build_issue_url(command="run") is None


def test_opt_out_default_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPT_OUT_ENV_VAR, raising=False)
    assert feedback_enabled() is True
    assert build_issue_url(command="run") is not None


def test_console_feedback_hint_contains_url() -> None:
    url = "https://example.com/issues/new"
    hint = console_feedback_hint(url)
    assert url in hint
    assert "feedback" in hint.lower() or "bug" in hint.lower()
