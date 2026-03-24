from pathlib import Path

from haindy.runtime.task_cache import TaskPlanCache


def test_task_plan_cache_store_lookup_and_invalidate(tmp_path: Path) -> None:
    cache_path = tmp_path / "task_plan_cache.json"
    cache = TaskPlanCache(cache_path)

    context = {"url": "https://example.com", "step": 1}
    actions = [{"type": "click", "target": "login"}]
    cache.store("step_one", context, actions)

    assert cache.lookup("step_one", context) == actions
    assert cache.lookup("step_one", {"url": "https://other.com"}) is None

    updated_actions = [{"type": "click", "target": "logout"}]
    cache.store("step_one", context, updated_actions)
    assert cache.lookup("step_one", context) == updated_actions

    cache.invalidate("step_one", context)
    assert cache.lookup("step_one", context) is None
