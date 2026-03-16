"""Local encrypted storage for OAuth credentials."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .paths import get_codex_oauth_key_path, get_codex_oauth_store_path

_STORE_SCHEMA_VERSION = 1


@dataclass(slots=True)
class StoredCodexOAuthCredentials:
    """Persisted Codex OAuth credential bundle."""

    access_token: str
    refresh_token: str
    id_token: str
    account_label: str
    expires_at: datetime

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize credentials to a JSON-compatible dict."""
        payload = asdict(self)
        payload["expires_at"] = self.expires_at.astimezone(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> StoredCodexOAuthCredentials:
        """Deserialize credentials from a JSON-compatible dict."""
        expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return cls(
            access_token=str(payload.get("access_token", "")),
            refresh_token=str(payload.get("refresh_token", "")),
            id_token=str(payload.get("id_token", "")),
            account_label=str(payload.get("account_label", "")),
            expires_at=expires_at.astimezone(timezone.utc),
        )


class LocalEncryptedAuthStore:
    """Persist OAuth credentials in an AES-GCM encrypted local file."""

    def __init__(
        self,
        store_path: Path | None = None,
        key_path: Path | None = None,
    ) -> None:
        self._store_path = store_path or get_codex_oauth_store_path()
        self._key_path = key_path or get_codex_oauth_key_path()

    def get_codex_oauth_credentials(self) -> StoredCodexOAuthCredentials | None:
        """Load stored Codex OAuth credentials if present."""
        if not self._store_path.exists():
            return None

        payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0) or 0) != _STORE_SCHEMA_VERSION:
            raise ValueError("Unsupported auth store schema version")

        nonce = base64.b64decode(str(payload["nonce"]))
        ciphertext = base64.b64decode(str(payload["ciphertext"]))
        plaintext = AESGCM(self._load_or_create_key()).decrypt(nonce, ciphertext, None)
        decoded = json.loads(plaintext.decode("utf-8"))
        return StoredCodexOAuthCredentials.from_json_dict(decoded)

    def set_codex_oauth_credentials(
        self, credentials: StoredCodexOAuthCredentials | None
    ) -> None:
        """Persist or clear Codex OAuth credentials."""
        self._ensure_parent_dir()
        if credentials is None:
            self.clear_codex_oauth_credentials()
            return

        plaintext = json.dumps(credentials.to_json_dict(), sort_keys=True).encode(
            "utf-8"
        )
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._load_or_create_key()).encrypt(nonce, plaintext, None)
        payload = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        self._store_path.write_text(json.dumps(payload), encoding="utf-8")
        self._chmod_private(self._store_path)

    def clear_codex_oauth_credentials(self) -> None:
        """Delete the encrypted credential blob if it exists."""
        if self._store_path.exists():
            self._store_path.unlink()

    def _load_or_create_key(self) -> bytes:
        self._ensure_parent_dir()
        if self._key_path.exists():
            encoded = self._key_path.read_text(encoding="utf-8").strip()
            key = base64.b64decode(encoded)
            if len(key) != 32:
                raise ValueError("Invalid auth store encryption key")
            return key

        key = os.urandom(32)
        self._key_path.write_text(
            base64.b64encode(key).decode("ascii"),
            encoding="utf-8",
        )
        self._chmod_private(self._key_path)
        return key

    def _ensure_parent_dir(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._store_path.parent, 0o700)
        except OSError:
            return

    @staticmethod
    def _chmod_private(path: Path) -> None:
        try:
            os.chmod(path, 0o600)
        except OSError:
            return
