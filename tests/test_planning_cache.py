import json
from pathlib import Path

from haindy.runtime.planning_cache import (
    PLANNING_CACHE_VERSION,
    PlanningCache,
    build_planning_cache_key_payload,
    hash_planning_cache_key,
)


def test_planning_cache_store_lookup_and_invalidate(tmp_path: Path) -> None:
    cache_path = tmp_path / "planning_cache.json"
    cache = PlanningCache(cache_path)

    key_payload = build_planning_cache_key_payload(
        requirements="Validate checkout flow",
        context={"browser": "chromium", "locale": "en-US"},
        provider="openai",
        triage_model="gpt-5",
        planner_model="gpt-5",
    )
    key_hash = hash_planning_cache_key(key_payload)
    triage_payload = {"in_scope": "Checkout", "blocking_questions": []}
    plan_payload = {"name": "Checkout smoke", "steps": [{"name": "Open cart"}]}

    cache.store(
        key_hash=key_hash,
        triage_payload=triage_payload,
        test_plan_payload=plan_payload,
        has_blockers=False,
    )

    hit = cache.lookup(key_hash)
    assert hit is not None
    assert hit.cache_version == PLANNING_CACHE_VERSION
    assert hit.key_hash == key_hash
    assert hit.triage_payload == triage_payload
    assert hit.test_plan_payload == plan_payload
    assert hit.has_blockers is False

    cache.invalidate(key_hash)
    assert cache.lookup(key_hash) is None


def test_planning_cache_malformed_json_is_treated_as_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "planning_cache.json"
    cache_path.write_text("{not-valid-json", encoding="utf-8")
    cache = PlanningCache(cache_path)

    assert cache.lookup("unknown") is None

    cache.store(
        key_hash="known",
        triage_payload={"in_scope": "Login"},
        test_plan_payload={"name": "Login smoke"},
        has_blockers=False,
    )
    assert cache.lookup("known") is not None


def test_planning_cache_lookup_ignores_version_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "planning_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "cache_version": PLANNING_CACHE_VERSION + 1,
                    "key_hash": "versioned-key",
                    "cached_at_epoch_seconds": 1,
                    "triage_payload": {"in_scope": "Checkout"},
                    "test_plan_payload": {"name": "plan"},
                    "has_blockers": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    cache = PlanningCache(cache_path)

    assert cache.lookup("versioned-key") is None


def test_planning_cache_key_hash_is_stable_and_sensitive() -> None:
    payload_a = build_planning_cache_key_payload(
        requirements="A",
        context={"z": 1, "a": 2},
        provider="openai",
    )
    payload_b = build_planning_cache_key_payload(
        requirements="A",
        context={"a": 2, "z": 1},
        provider="openai",
    )
    payload_c = build_planning_cache_key_payload(
        requirements="B",
        context={"a": 2, "z": 1},
        provider="openai",
    )

    assert hash_planning_cache_key(payload_a) == hash_planning_cache_key(payload_b)
    assert hash_planning_cache_key(payload_a) != hash_planning_cache_key(payload_c)


def test_planning_cache_blocker_entry_stores_null_plan_payload(tmp_path: Path) -> None:
    cache_path = tmp_path / "planning_cache.json"
    cache = PlanningCache(cache_path)

    key_hash = hash_planning_cache_key(
        build_planning_cache_key_payload(
            requirements="A",
            context="B",
            provider="openai",
        )
    )
    cache.store(
        key_hash=key_hash,
        triage_payload={"blocking_questions": ["Need credentials"]},
        test_plan_payload=None,
        has_blockers=True,
    )

    hit = cache.lookup(key_hash)
    assert hit is not None
    assert hit.has_blockers is True
    assert hit.test_plan_payload is None

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert raw[-1]["test_plan_payload"] is None
