"""Tests for local OpenAI Codex OAuth helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.auth.codex_oauth import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_REDIRECT_URI,
    CODEX_OAUTH_SCOPE,
    CODEX_RESPONSES_BASE_URL,
    CODEX_SYSTEM_INSTRUCTIONS,
    CodexOAuthClient,
    CodexOAuthToken,
    derive_account_label,
    extract_chatgpt_account_id,
)
from src.auth.manager import OpenAIAuthManager
from src.auth.store import LocalEncryptedAuthStore, StoredCodexOAuthCredentials


def _jwt_with_claims(claims: dict[str, object]) -> str:
    import base64
    import json

    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode("ascii")
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True).encode("utf-8"))
        .rstrip(b"=")
        .decode("ascii")
    )
    return f"{header}.{payload}."


def test_local_encrypted_auth_store_round_trip(tmp_path: Path) -> None:
    store = LocalEncryptedAuthStore(
        store_path=tmp_path / "state" / "codex.enc",
        key_path=tmp_path / "state" / "codex.key",
    )
    credentials = StoredCodexOAuthCredentials(
        access_token="access-token",
        refresh_token="refresh-token",
        id_token="id-token",
        account_label="user@example.com",
        expires_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    store.set_codex_oauth_credentials(credentials)

    loaded = store.get_codex_oauth_credentials()
    assert loaded == credentials
    encrypted_text = (tmp_path / "state" / "codex.enc").read_text(encoding="utf-8")
    assert "access-token" not in encrypted_text
    assert "refresh-token" not in encrypted_text


def test_local_encrypted_auth_store_clear(tmp_path: Path) -> None:
    store = LocalEncryptedAuthStore(
        store_path=tmp_path / "codex.enc",
        key_path=tmp_path / "codex.key",
    )
    store.set_codex_oauth_credentials(
        StoredCodexOAuthCredentials(
            access_token="a",
            refresh_token="b",
            id_token="c",
            account_label="user@example.com",
            expires_at=datetime.now(timezone.utc),
        )
    )

    store.clear_codex_oauth_credentials()

    assert store.get_codex_oauth_credentials() is None
    assert not (tmp_path / "codex.enc").exists()


def test_codex_oauth_build_authorize_url_and_parse_redirect() -> None:
    pkce = CodexOAuthClient.generate_pkce()
    authorize_url = CodexOAuthClient.build_authorize_url(
        pkce.state, pkce.code_challenge
    )

    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(authorize_url)
    params = parse_qs(parsed.query)
    assert parsed.netloc == "auth.openai.com"
    assert params["client_id"] == [CODEX_OAUTH_CLIENT_ID]
    assert params["redirect_uri"] == [CODEX_OAUTH_REDIRECT_URI]
    assert params["scope"] == [CODEX_OAUTH_SCOPE]
    assert params["codex_cli_simplified_flow"] == ["true"]

    code, state = CodexOAuthClient.parse_redirect_url(
        "http://localhost:1455/auth/callback?code=abc123&state=state-1"
    )
    assert code == "abc123"
    assert state == "state-1"


def test_extract_chatgpt_account_id_and_account_label() -> None:
    jwt_token = _jwt_with_claims(
        {
            "email": "user@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123",
            },
        }
    )

    assert extract_chatgpt_account_id(jwt_token) == "acct_123"
    assert derive_account_label(jwt_token) == "user@example.com"


@pytest.mark.asyncio
async def test_auth_manager_prefers_oauth_and_refreshes_expiring_tokens() -> None:
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    stored = StoredCodexOAuthCredentials(
        access_token="",
        refresh_token="refresh-token",
        id_token=_jwt_with_claims({"email": "fresh@example.com"}),
        account_label="",
        expires_at=now,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.saved: StoredCodexOAuthCredentials | None = None

        def get_codex_oauth_credentials(self) -> StoredCodexOAuthCredentials | None:
            return self.saved or stored

        def set_codex_oauth_credentials(
            self, credentials: StoredCodexOAuthCredentials | None
        ) -> None:
            self.saved = credentials

        def clear_codex_oauth_credentials(self) -> None:
            self.saved = None

    class FakeOAuthClient:
        async def refresh_access_token(self, refresh_token: str) -> CodexOAuthToken:
            assert refresh_token == "refresh-token"
            return CodexOAuthToken(
                access_token="new-access",
                refresh_token="new-refresh",
                id_token=_jwt_with_claims({"email": "fresh@example.com"}),
                expires_at=now + timedelta(hours=1),
            )

    manager = OpenAIAuthManager(
        settings=SimpleNamespace(openai_api_key="fallback-key"),
        store=FakeStore(),
        oauth_client=FakeOAuthClient(),
        now=lambda: now,
    )

    resolved = await manager.resolve_openai_auth()

    assert resolved.mode == "codex_oauth"
    assert resolved.token == "new-access"
    assert resolved.base_url == CODEX_RESPONSES_BASE_URL
    assert resolved.default_headers["OpenAI-Beta"] == "responses=experimental"
    assert resolved.account_label == "fresh@example.com"


def test_auth_manager_status_falls_back_to_api_key_when_no_oauth() -> None:
    manager = OpenAIAuthManager(
        settings=SimpleNamespace(openai_api_key="sk-test"),
        store=SimpleNamespace(
            get_codex_oauth_credentials=lambda: None,
            clear_codex_oauth_credentials=lambda: None,
        ),
        oauth_client=CodexOAuthClient(),
    )

    status = manager.get_status()

    assert status.active_mode == "api_key"
    assert status.oauth_connected is False
    assert status.api_key_available is True


def test_codex_system_instructions_constant_is_non_empty() -> None:
    assert CODEX_SYSTEM_INSTRUCTIONS == "You are Codex."
