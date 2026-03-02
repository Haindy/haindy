import json
from pathlib import Path

from src.runtime.situational_cache import (
    SITUATIONAL_CACHE_VERSION,
    SituationalCache,
    build_situational_cache_key_payload,
    hash_situational_cache_key,
)


def test_situational_cache_store_lookup_and_overwrite(tmp_path: Path) -> None:
    cache_path = tmp_path / "situational_cache.json"
    cache = SituationalCache(cache_path)

    key_hash = "same-key"
    payload_v1 = {
        "target_type": "desktop_app",
        "sufficient": True,
        "missing_items": [],
        "setup": {"app_name": "Calculator"},
        "entry_actions": [],
        "notes": [],
    }
    payload_v2 = {
        "target_type": "web",
        "sufficient": True,
        "missing_items": [],
        "setup": {"web_url": "https://example.com"},
        "entry_actions": [],
        "notes": [],
    }

    cache.store(key_hash=key_hash, assessment_payload=payload_v1)
    cache.store(key_hash=key_hash, assessment_payload=payload_v2)

    hit = cache.lookup(key_hash)
    assert hit is not None
    assert hit.cache_version == SITUATIONAL_CACHE_VERSION
    assert hit.assessment_payload == payload_v2


def test_situational_cache_invalidate_removes_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "situational_cache.json"
    cache = SituationalCache(cache_path)

    key_a = "key-a"
    key_b = "key-b"
    payload = {"target_type": "desktop_app", "sufficient": True}

    cache.store(key_hash=key_a, assessment_payload=payload)
    cache.store(key_hash=key_b, assessment_payload=payload)
    cache.invalidate(key_a)

    assert cache.lookup(key_a) is None
    assert cache.lookup(key_b) is not None


def test_situational_cache_malformed_json_is_treated_as_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "situational_cache.json"
    cache_path.write_text("{not-valid-json", encoding="utf-8")
    cache = SituationalCache(cache_path)

    assert cache.lookup("unknown") is None
    cache.store(
        key_hash="known",
        assessment_payload={"target_type": "desktop_app", "sufficient": True},
    )
    assert cache.lookup("known") is not None


def test_situational_cache_lookup_ignores_version_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "situational_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "cache_version": SITUATIONAL_CACHE_VERSION + 1,
                    "key_hash": "versioned-key",
                    "cached_at_epoch_seconds": 1,
                    "assessment_payload": {
                        "target_type": "desktop_app",
                        "sufficient": True,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    cache = SituationalCache(cache_path)

    assert cache.lookup("versioned-key") is None


def test_situational_cache_key_hash_is_stable_and_sensitive() -> None:
    payload_a = build_situational_cache_key_payload(
        requirements="Test login flow",
        context_text="app_name: Calculator",
    )
    payload_b = build_situational_cache_key_payload(
        requirements="Test login flow",
        context_text="app_name: Calculator",
    )
    payload_c = build_situational_cache_key_payload(
        requirements="Test login flow changed",
        context_text="app_name: Calculator",
    )

    assert hash_situational_cache_key(payload_a) == hash_situational_cache_key(
        payload_b
    )
    assert hash_situational_cache_key(payload_a) != hash_situational_cache_key(
        payload_c
    )
