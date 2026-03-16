"""OpenAI Codex OAuth helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

CODEX_OAUTH_AUTH_BASE_URL = "https://auth.openai.com"
CODEX_OAUTH_AUTHORIZE_PATH = "/oauth/authorize"
CODEX_OAUTH_TOKEN_PATH = "/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
CODEX_OAUTH_SCOPE = "openid profile email offline_access"
CODEX_RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_SYSTEM_INSTRUCTIONS = "You are Codex."


class CodexOAuthError(RuntimeError):
    """Base error for Codex OAuth operations."""


class CodexOAuthUnauthorizedError(CodexOAuthError):
    """Raised when the OAuth exchange or refresh is rejected."""


@dataclass(slots=True)
class CodexPKCEValues:
    """PKCE values required for the OAuth authorization code flow."""

    state: str
    code_verifier: str
    code_challenge: str


@dataclass(slots=True)
class CodexOAuthToken:
    """Token bundle returned by the Codex OAuth server."""

    access_token: str
    refresh_token: str
    id_token: str
    expires_at: datetime


class CodexOAuthClient:
    """Perform OAuth exchange and refresh requests against OpenAI auth."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._http_client = http_client
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_pkce() -> CodexPKCEValues:
        """Generate PKCE state, verifier, and challenge."""
        verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("ascii")
        state = base64.urlsafe_b64encode(os.urandom(24)).rstrip(b"=").decode("ascii")
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return CodexPKCEValues(
            state=state,
            code_verifier=verifier,
            code_challenge=challenge,
        )

    @staticmethod
    def build_authorize_url(state: str, code_challenge: str) -> str:
        """Build the browser authorization URL."""
        query = urlencode(
            {
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "redirect_uri": CODEX_OAUTH_REDIRECT_URI,
                "scope": CODEX_OAUTH_SCOPE,
                "response_type": "code",
                "code_challenge_method": "S256",
                "code_challenge": code_challenge.strip(),
                "state": state.strip(),
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "originator": "pi",
            }
        )
        return f"{CODEX_OAUTH_AUTH_BASE_URL}{CODEX_OAUTH_AUTHORIZE_PATH}?{query}"

    @staticmethod
    def parse_redirect_url(redirect_url: str) -> tuple[str, str]:
        """Extract the authorization code and state from a redirect URL."""
        parsed = urlparse(redirect_url.strip())
        params = parse_qs(parsed.query)
        oauth_error = params.get("error", [""])[0].strip()
        if oauth_error:
            description = params.get("error_description", [""])[0].strip()
            if description:
                raise CodexOAuthError(f"OAuth error: {oauth_error}: {description}")
            raise CodexOAuthError(f"OAuth error: {oauth_error}")

        code = params.get("code", [""])[0].strip()
        state = params.get("state", [""])[0].strip()
        if not code:
            raise CodexOAuthError("Missing code in redirect URL")
        if not state:
            raise CodexOAuthError("Missing state in redirect URL")
        return code, state

    async def exchange_authorization_code(
        self, code: str, code_verifier: str
    ) -> CodexOAuthToken:
        """Exchange an authorization code for access and refresh tokens."""
        return await self._post_token(
            {
                "grant_type": "authorization_code",
                "code": code.strip(),
                "redirect_uri": CODEX_OAUTH_REDIRECT_URI,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier.strip(),
            }
        )

    async def refresh_access_token(self, refresh_token: str) -> CodexOAuthToken:
        """Refresh an existing access token."""
        return await self._post_token(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token.strip(),
                "client_id": CODEX_OAUTH_CLIENT_ID,
            }
        )

    async def _post_token(self, form_data: dict[str, str]) -> CodexOAuthToken:
        url = f"{CODEX_OAUTH_AUTH_BASE_URL}{CODEX_OAUTH_TOKEN_PATH}"
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await client.post(
                url,
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise CodexOAuthError(f"OAuth request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code in {400, 401, 403}:
            raise CodexOAuthUnauthorizedError("OAuth token exchange was rejected")
        if response.status_code >= 500:
            raise CodexOAuthError("OpenAI OAuth is currently unavailable")
        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text.strip()
            raise CodexOAuthError(
                f"OAuth token exchange failed: {response.status_code} - {detail}"
            )

        payload = response.json()
        access_token = str(payload.get("access_token", "")).strip()
        if not access_token:
            raise CodexOAuthError("OAuth token response missing access_token")
        expires_in = int(payload.get("expires_in", 0) or 0)
        expires_at = self._now()
        if expires_in > 0:
            expires_at = expires_at + timedelta(seconds=expires_in)
        return CodexOAuthToken(
            access_token=access_token,
            refresh_token=str(payload.get("refresh_token", "")).strip(),
            id_token=str(payload.get("id_token", "")).strip(),
            expires_at=expires_at.astimezone(timezone.utc),
        )


def extract_chatgpt_account_id(jwt_token: str) -> str:
    """Extract the ChatGPT account id claim from a JWT if present."""
    claims = _decode_jwt_claims(jwt_token)
    if not claims:
        return ""

    for key in (
        "https://api.openai.com/auth.chatgpt_account_id",
        "chatgpt_account_id",
    ):
        account_id = _stringify_claim(claims.get(key))
        if account_id:
            return account_id

    auth_claim = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        account_id = _stringify_claim(auth_claim.get("chatgpt_account_id"))
        if account_id:
            return account_id
    return ""


def derive_account_label(id_token: str, access_token: str = "") -> str:
    """Build a human-friendly account label from token claims."""
    claims = _decode_jwt_claims(id_token) or _decode_jwt_claims(access_token)
    if not claims:
        return ""

    for key in ("email", "preferred_username", "name"):
        label = _stringify_claim(claims.get(key))
        if label:
            return label

    return extract_chatgpt_account_id(id_token) or extract_chatgpt_account_id(
        access_token
    )


def _decode_jwt_claims(jwt_token: str) -> dict[str, Any]:
    token = jwt_token.strip()
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    encoded = parts[1]
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(encoded + padding)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _stringify_claim(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text
