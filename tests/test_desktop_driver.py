from pathlib import Path

import pytest

from haindy.desktop.driver import DesktopDriver


class DummyVirtualInput:
    def __init__(self, *args, **kwargs) -> None:
        self.clicks: list[tuple[int, int, str, int]] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.scrolls: list[tuple[int, int]] = []

    async def click(
        self, x: int, y: int, button: str = "left", click_count: int = 1
    ) -> None:
        self.clicks.append((x, y, button, click_count))

    async def type_text(self, text: str) -> None:
        self.typed.append(text)

    async def press_key(self, key: str) -> None:
        self.keys.append(key)

    async def move(self, x: int, y: int, steps: int = 1) -> None:  # pragma: no cover
        return

    async def drag(self, start, end, steps: int = 1) -> None:  # pragma: no cover
        return

    async def scroll(self, x: int = 0, y: int = 0) -> None:
        self.scrolls.append((x, y))


@pytest.mark.asyncio
async def test_desktop_driver_smoke(monkeypatch, tmp_path: Path) -> None:
    dummy_input = DummyVirtualInput()

    def _virtual_input_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        return dummy_input

    monkeypatch.setattr("haindy.desktop.driver.VirtualInput", _virtual_input_factory)

    driver = DesktopDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
        prefer_resolution=(800, 600),
        enable_resolution_switch=False,
    )
    monkeypatch.setattr(driver.resolution_manager, "maybe_downshift", lambda: None)
    monkeypatch.setattr(driver.resolution_manager, "viewport_size", lambda: (800, 600))
    monkeypatch.setattr(driver.resolution_manager, "restore", lambda: None)

    await driver.start()
    await driver.click(10, 20)
    await driver.type_text("hello")
    await driver.press_key("enter")
    await driver.scroll_by_pixels(x=0, y=120)
    await driver.stop()

    assert dummy_input.clicks == [(10, 20, "left", 1)]
    assert dummy_input.typed == ["hello"]
    assert dummy_input.keys == ["enter"]
    assert dummy_input.scrolls == [(0, 120)]


@pytest.mark.asyncio
async def test_desktop_driver_start_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    created = {"count": 0}
    dummy_input = DummyVirtualInput()

    def _virtual_input_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        created["count"] += 1
        return dummy_input

    monkeypatch.setattr("haindy.desktop.driver.VirtualInput", _virtual_input_factory)

    driver = DesktopDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
        prefer_resolution=(800, 600),
        enable_resolution_switch=False,
    )
    monkeypatch.setattr(driver.resolution_manager, "maybe_downshift", lambda: None)
    monkeypatch.setattr(driver.resolution_manager, "viewport_size", lambda: (800, 600))
    monkeypatch.setattr(driver.resolution_manager, "restore", lambda: None)

    await driver.start()
    await driver.start()
    await driver.click(1, 2)
    await driver.stop()

    assert created["count"] == 1


@pytest.mark.asyncio
async def test_desktop_driver_scroll_rejects_invalid_direction(
    monkeypatch, tmp_path: Path
) -> None:
    dummy_input = DummyVirtualInput()

    def _virtual_input_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        return dummy_input

    monkeypatch.setattr("haindy.desktop.driver.VirtualInput", _virtual_input_factory)
    driver = DesktopDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
        prefer_resolution=(800, 600),
        enable_resolution_switch=False,
    )
    monkeypatch.setattr(driver.resolution_manager, "maybe_downshift", lambda: None)
    monkeypatch.setattr(driver.resolution_manager, "viewport_size", lambda: (800, 600))
    monkeypatch.setattr(driver.resolution_manager, "restore", lambda: None)

    await driver.start()
    with pytest.raises(ValueError):
        await driver.scroll("diagonal", 100)
    await driver.stop()


@pytest.mark.asyncio
async def test_desktop_driver_restores_resolution_when_start_fails(
    monkeypatch, tmp_path: Path
) -> None:
    driver = DesktopDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
        prefer_resolution=(800, 600),
        enable_resolution_switch=True,
    )

    calls: list[str] = []
    monkeypatch.setattr(
        driver.resolution_manager,
        "maybe_downshift",
        lambda: calls.append("downshift"),
    )
    monkeypatch.setattr(
        driver.resolution_manager,
        "restore",
        lambda: calls.append("restore"),
    )
    monkeypatch.setattr(driver.resolution_manager, "viewport_size", lambda: (800, 600))

    def _raise_virtual_input(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("virtual input failed")

    monkeypatch.setattr("haindy.desktop.driver.VirtualInput", _raise_virtual_input)

    with pytest.raises(RuntimeError, match="virtual input failed"):
        await driver.start()

    assert calls == ["downshift", "restore"]
    assert driver.virtual_input is None
