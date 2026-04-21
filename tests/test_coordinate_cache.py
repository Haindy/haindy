from pathlib import Path

from haindy.core.coordinate_cache import CoordinateCache


def test_coordinate_cache_add_lookup_and_invalidate(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    cache = CoordinateCache(cache_path)

    cache.add("target", "click", 10, 20, (1920, 1080))
    hit = cache.lookup("target", "click", (1920, 1080))
    assert hit is not None
    assert hit.x == 10
    assert hit.y == 20

    cache.invalidate("target", "click", (1920, 1080))
    assert cache.lookup("target", "click", (1920, 1080)) is None
