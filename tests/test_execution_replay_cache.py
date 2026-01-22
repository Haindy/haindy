import json
from pathlib import Path

from src.runtime.execution_replay_cache import (
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
