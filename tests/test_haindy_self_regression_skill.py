"""Tests for the repo-local HAINDY self-regression skill."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / ".agents" / "skills" / "haindy-self-regression" / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_self_regression_skill_exists_only_under_agents() -> None:
    assert SKILL_PATH.is_file()
    assert not (
        ROOT / "haindy" / "skills" / "haindy-self-regression" / "SKILL.md"
    ).exists()


def test_self_regression_skill_enforces_haindy_only_boundary() -> None:
    text = _skill_text()

    for command in [
        "haindy doctor",
        "haindy auth status",
        "haindy provider list",
        "haindy provider set-computer-use <provider>",
        "haindy session new --desktop",
        "haindy session status --session <SESSION_ID>",
        "haindy session list",
        "haindy session close --session <SESSION_ID>",
        'haindy explore "',
        "haindy explore-status --session <SESSION_ID>",
        'haindy act "',
    ]:
        assert command in text

    for forbidden in [
        "ad-hoc scripts",
        "Python helpers",
        "direct SDK probes",
        "raw `adb` or `idb`",
        "X11/macOS/Windows automation commands",
        "browser automation",
        "DOM inspection",
        "Playwright",
        "Selenium",
    ]:
        assert forbidden in text


def test_self_regression_skill_documents_provider_order_and_restore() -> None:
    text = _skill_text()

    assert text.index("1. `google`") < text.index("2. `openai`")
    assert text.index("2. `openai`") < text.index("3. `anthropic`")
    assert "Record the original active computer-use provider" in text
    assert "Restore this provider at the end" in text


def test_self_regression_skill_requires_session_cleanup_and_report() -> None:
    text = _skill_text()

    assert "Keep a list of every opened session ID" in text
    assert "Close every session before finishing" in text
    assert "session close --session <SESSION_ID> --force" in text
    assert (
        "| Provider | Surface | Session ID | Commands | Result | Skipped Items | Failures | Reproduction Notes |"
        in text
    )
    assert "sessions force-closed" in text


def test_self_regression_skill_does_not_introduce_helper_scripts() -> None:
    skill_dir = SKILL_PATH.parent
    paths = [path for path in skill_dir.rglob("*") if path.is_file()]

    assert paths == [SKILL_PATH]
