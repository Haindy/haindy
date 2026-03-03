import json
from pathlib import Path

from src.runtime.execution_replay_cache import (
    EXECUTION_REPLAY_CACHE_VERSION,
    ExecutionReplayCache,
    ExecutionReplayCacheKey,
)


def test_execution_replay_cache_store_lookup_newest_wins(tmp_path: Path) -> None:
    cache_path = tmp_path / "execution_replay_cache.json"
    cache = ExecutionReplayCache(cache_path)
    key = ExecutionReplayCacheKey(
        scenario="scenario",
        step="step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
    )

    cache.store(
        key,
        [{"type": "click", "x": 1, "y": 2, "button": "left", "click_count": 1}],
    )
    cache.store(
        key,
        [{"type": "click", "x": 9, "y": 8, "button": "left", "click_count": 1}],
    )

    entry = cache.lookup(key)
    assert entry is not None
    assert entry.actions == [
        {"type": "click", "x": 9, "y": 8, "button": "left", "click_count": 1}
    ]

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) == 2


def test_execution_replay_cache_invalidate_removes_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "execution_replay_cache.json"
    cache = ExecutionReplayCache(cache_path)
    key_a = ExecutionReplayCacheKey(
        scenario="scenario",
        step="step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
    )
    key_b = ExecutionReplayCacheKey(
        scenario="scenario",
        step="other_step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
    )

    cache.store(
        key_a,
        [{"type": "click", "x": 1, "y": 2, "button": "left", "click_count": 1}],
    )
    cache.store(
        key_b,
        [{"type": "click", "x": 3, "y": 4, "button": "left", "click_count": 1}],
    )
    cache.invalidate(key_a)

    assert cache.lookup(key_a) is None
    assert cache.lookup(key_b) is not None


def test_execution_replay_cache_lookup_miss_when_plan_fingerprint_differs(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "execution_replay_cache.json"
    cache = ExecutionReplayCache(cache_path)
    stored_key = ExecutionReplayCacheKey(
        scenario="scenario",
        step="step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
        plan_fingerprint="fingerprint-a",
    )

    cache.store(
        stored_key,
        [{"type": "click", "x": 1, "y": 2, "button": "left", "click_count": 1}],
    )

    miss_key = ExecutionReplayCacheKey(
        scenario="scenario",
        step="step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
        plan_fingerprint="fingerprint-b",
    )
    assert cache.lookup(miss_key) is None
    assert cache.lookup(stored_key) is not None


def test_execution_replay_cache_legacy_entry_without_plan_fingerprint_compatible(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "execution_replay_cache.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "cache_version": EXECUTION_REPLAY_CACHE_VERSION,
                    "key": {
                        "scenario_name": "scenario",
                        "step_name": "step",
                        "environment": "desktop",
                        "viewport_resolution": [1920, 1080],
                        "keyboard_layout": "us",
                    },
                    "recorded_at_epoch_seconds": 123,
                    "actions": [
                        {
                            "type": "click",
                            "x": 7,
                            "y": 8,
                            "button": "left",
                            "click_count": 1,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    cache = ExecutionReplayCache(cache_path)
    empty_fingerprint_key = ExecutionReplayCacheKey(
        scenario="scenario",
        step="step",
        environment="desktop",
        resolution=(1920, 1080),
        keyboard_layout="us",
        plan_fingerprint="",
    )

    entry = cache.lookup(empty_fingerprint_key)
    assert entry is not None
    assert entry.key.plan_fingerprint == ""
    assert entry.actions == [
        {"type": "click", "x": 7, "y": 8, "button": "left", "click_count": 1}
    ]
