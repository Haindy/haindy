"""AES-GCM encrypted key-value store for local credential persistence."""

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


class EncryptedJsonFileStore:
    """Persist string key-value pairs in an AES-GCM encrypted local file."""

    def __init__(
        self,
        store_path: Path,
        key_path: Path,
    ) -> None:
        self._store_path = store_path
        self._key_path = key_path

    def get(self, key: str) -> str | None:
        """Return the value for *key*, or None if not present."""
        return self._load().get(key)

    def get_all(self) -> dict[str, str]:
        """Return a copy of all stored key-value pairs."""
        return dict(self._load())

    def set(self, key: str, value: str) -> None:
        """Persist *value* under *key*."""
        data = self._load()
        data[key] = value
        self._save(data)

    def delete(self, key: str) -> None:
        """Remove *key* from the store (no-op if absent)."""
        data = self._load()
        if key in data:
            data.pop(key)
            self._save(data)

    def _load(self) -> dict[str, str]:
        if not self._store_path.exists():
            return {}
        raw = self._store_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        nonce = base64.b64decode(str(payload["nonce"]))
        ciphertext = base64.b64decode(str(payload["ciphertext"]))
        plaintext = AESGCM(self._load_or_create_key()).decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))

    def _save(self, data: dict[str, str]) -> None:
        self._ensure_parent_dir()
        plaintext = json.dumps(data, sort_keys=True).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = AESGCM(self._load_or_create_key()).encrypt(nonce, plaintext, None)
        payload = {
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        self._store_path.write_text(json.dumps(payload), encoding="utf-8")
        self._chmod_private(self._store_path)

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


def _make_oauth_store(
    store_path: Path | None = None,
    key_path: Path | None = None,
) -> EncryptedJsonFileStore:
    """Return an EncryptedJsonFileStore configured for OAuth credential storage."""
    return EncryptedJsonFileStore(
        store_path=store_path or get_codex_oauth_store_path(),
        key_path=key_path or get_codex_oauth_key_path(),
    )
