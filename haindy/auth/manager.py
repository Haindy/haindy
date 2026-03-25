"""Resolution logic for OpenAI API key vs Codex OAuth auth modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from haindy.config.settings import Settings, get_settings

from .codex_oauth import (
    CODEX_RESPONSES_BASE_URL,
    CodexOAuthClient,
    CodexOAuthError,
    derive_account_label,
    extract_chatgpt_account_id,
)
from .store import (
    EncryptedJsonFileStore,
    StoredCodexOAuthCredentials,
    _make_oauth_store,
)

_REFRESH_LEAD_TIME = timedelta(seconds=60)


@dataclass(slots=True)
class ResolvedOpenAIAuth:
    """Resolved auth bundle for a non-CU OpenAI request path."""

    mode: str
    token: str
    base_url: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    account_label: str | None = None
    expires_at: datetime | None = None


@dataclass(slots=True)
class OpenAIAuthStatus:
    """Inspectable auth status for CLI presentation."""

    active_mode: str
    oauth_connected: bool
    oauth_account_label: str | None
    oauth_expires_at: datetime | None
    oauth_expired: bool
    api_key_available: bool


class OpenAIAuthManager:
    """Resolve and persist the non-CU OpenAI auth mode."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: EncryptedJsonFileStore | None = None,
        oauth_client: CodexOAuthClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or _make_oauth_store()
        self._oauth_client = oauth_client or CodexOAuthClient()
        self._now = now or (lambda: datetime.now(timezone.utc))

    async def resolve_openai_auth(
        self, api_key_override: str | None = None
    ) -> ResolvedOpenAIAuth:
        """Resolve the active auth mode for non-CU OpenAI requests."""
        credentials = self._get_oauth_credentials()
        if credentials and credentials.refresh_token.strip():
            credentials = await self._refresh_if_needed(credentials)
            headers: dict[str, str] = {
                "OpenAI-Beta": "responses=experimental",
                "originator": "pi",
            }
            account_id = extract_chatgpt_account_id(credentials.id_token)
            if not account_id:
                account_id = extract_chatgpt_account_id(credentials.access_token)
            if account_id:
                headers["chatgpt-account-id"] = account_id
            return ResolvedOpenAIAuth(
                mode="codex_oauth",
                token=credentials.access_token,
                base_url=CODEX_RESPONSES_BASE_URL,
                default_headers=headers,
                account_label=credentials.account_label or None,
                expires_at=credentials.expires_at,
            )

        api_key = str(api_key_override or self._settings.openai_api_key or "").strip()
        if api_key:
            return ResolvedOpenAIAuth(mode="api_key", token=api_key)

        raise ValueError(
            "OpenAI API key not provided. Set HAINDY_OPENAI_API_KEY or login via "
            "--codex-auth login."
        )

    def get_status(self, api_key_override: str | None = None) -> OpenAIAuthStatus:
        """Return the persisted auth status without mutating stored credentials."""
        credentials = self._get_oauth_credentials()
        now = self._now()
        oauth_connected = bool(credentials and credentials.refresh_token.strip())
        oauth_expires_at = credentials.expires_at if credentials else None
        oauth_expired = bool(
            oauth_expires_at and oauth_expires_at.astimezone(timezone.utc) <= now
        )
        api_key_available = bool(
            str(api_key_override or self._settings.openai_api_key or "").strip()
        )
        active_mode = (
            "codex_oauth"
            if oauth_connected
            else ("api_key" if api_key_available else "none")
        )
        return OpenAIAuthStatus(
            active_mode=active_mode,
            oauth_connected=oauth_connected,
            oauth_account_label=credentials.account_label if credentials else None,
            oauth_expires_at=oauth_expires_at,
            oauth_expired=oauth_expired,
            api_key_available=api_key_available,
        )

    def save_oauth_token_bundle(self, token: Any) -> StoredCodexOAuthCredentials:
        """Persist an OAuth token response and return the stored credentials."""
        account_label = derive_account_label(token.id_token, token.access_token)
        credentials = StoredCodexOAuthCredentials(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            id_token=token.id_token,
            account_label=account_label,
            expires_at=token.expires_at.astimezone(timezone.utc),
        )
        self._set_oauth_credentials(credentials)
        return credentials

    def clear_oauth_credentials(self) -> None:
        """Delete persisted Codex OAuth credentials."""
        for key in (
            "access_token",
            "refresh_token",
            "id_token",
            "account_label",
            "expires_at",
        ):
            self._store.delete(key)

    async def _refresh_if_needed(
        self, credentials: StoredCodexOAuthCredentials
    ) -> StoredCodexOAuthCredentials:
        if credentials.access_token.strip() and not self._token_expiring_soon(
            credentials.expires_at
        ):
            return credentials

        try:
            token = await self._oauth_client.refresh_access_token(
                credentials.refresh_token
            )
        except CodexOAuthError as exc:
            raise ValueError(
                "Stored Codex OAuth credentials are no longer usable. Re-run "
                "--codex-auth login or --codex-auth logout."
            ) from exc

        refreshed = self.save_oauth_token_bundle(token)
        if not refreshed.access_token.strip():
            raise ValueError(
                "Stored Codex OAuth credentials did not yield an access token. "
                "Re-run --codex-auth login."
            )
        return refreshed

    def _get_oauth_credentials(self) -> StoredCodexOAuthCredentials | None:
        data = self._store.get_all()
        refresh_token = data.get("refresh_token", "")
        if not refresh_token:
            return None
        expires_at_raw = data.get("expires_at", "")
        try:
            expires_at = datetime.fromisoformat(expires_at_raw)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None
        return StoredCodexOAuthCredentials(
            access_token=data.get("access_token", ""),
            refresh_token=refresh_token,
            id_token=data.get("id_token", ""),
            account_label=data.get("account_label", ""),
            expires_at=expires_at.astimezone(timezone.utc),
        )

    def _set_oauth_credentials(self, credentials: StoredCodexOAuthCredentials) -> None:
        json_dict = credentials.to_json_dict()
        for key, value in json_dict.items():
            self._store.set(key, str(value))

    def _token_expiring_soon(self, expires_at: datetime) -> bool:
        deadline = self._now() + _REFRESH_LEAD_TIME
        normalized = expires_at.astimezone(timezone.utc)
        return normalized <= deadline
