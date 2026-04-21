"""Unit tests for WindowsDriver and WindowsInputHandler using stubs."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

if sys.platform != "win32":
    pytest.skip("Windows only", allow_module_level=True)

from haindy.windows.driver import WindowsDriver, _parse_png_size
from haindy.windows.input_handler import WindowsInputHandler

# ---------------------------------------------------------------------------
# Minimal PNG factory helpers
# ---------------------------------------------------------------------------


def _make_png(width: int, height: int) -> bytes:
    """Create a minimal valid PNG of given dimensions (black RGB image)."""

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + b"\x00" * (width * 3)
    idat = chunk(b"IDAT", zlib.compress(raw_rows))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


TINY_2X2_PNG = _make_png(2, 2)
TINY_4X4_PNG = _make_png(4, 4)


# ---------------------------------------------------------------------------
# Stub pynput controllers (no real input injected)
# ---------------------------------------------------------------------------


class StubMouseController:
    def __init__(self) -> None:
        self._position: tuple[int, int] = (0, 0)
        self.positions: list[tuple[int, int]] = []
        self.clicks: list[tuple[Any, int]] = []
        self.pressed: list[Any] = []
        self.released: list[Any] = []
        self.scrolls: list[tuple[float, float]] = []

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._position = value
        self.positions.append(value)

    def click(self, button: Any, count: int = 1) -> None:
        self.clicks.append((button, count))

    def press(self, button: Any) -> None:
        self.pressed.append(button)

    def release(self, button: Any) -> None:
        self.released.append(button)

    def scroll(self, dx: float, dy: float) -> None:
        self.scrolls.append((dx, dy))


class StubKeyboardController:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.pressed: list[Any] = []
        self.released: list[Any] = []

    def type(self, text: str) -> None:
        self.typed.append(text)

    def press(self, key: Any) -> None:
        self.pressed.append(key)

    def release(self, key: Any) -> None:
        self.released.append(key)


# ---------------------------------------------------------------------------
# Stub screen capture (returns fixed PNG; no mss needed)
# ---------------------------------------------------------------------------


class StubScreenCapture:
    """Return a fixed 4x4 PNG (pixel space) with logical size 2x2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.capture_calls: list[str] = []

    def get_logical_size(self) -> tuple[int, int]:
        return (2, 2)

    def capture(self, label: str = "screenshot") -> tuple[bytes, str]:
        self.capture_calls.append(label)
        return TINY_4X4_PNG, "/fake/path/screenshot.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WindowsDriver:
    monkeypatch.setattr("haindy.windows.driver.WindowsScreenCapture", StubScreenCapture)
    monkeypatch.setattr(
        "haindy.windows.driver.WindowsInputHandler", _make_stub_input_class()
    )
    return WindowsDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
    )


def _make_stub_input_class() -> type:
    """Return a WindowsInputHandler replacement that records calls."""

    class _StubInputHandler:
        def __init__(
            self,
            logical_size: tuple[int, int],
            scale_x: float,
            scale_y: float,
            **kwargs: Any,
        ) -> None:
            self.logical_size = logical_size
            self.scale_x = scale_x
            self.scale_y = scale_y
            self.clicks: list[tuple[int, int, str, int]] = []
            self.typed: list[str] = []
            self.keys: list[str] = []
            self.scrolls: list[tuple[int, int]] = []

        async def click(
            self,
            x: int,
            y: int,
            button: str = "left",
            click_count: int = 1,
            **kwargs: Any,
        ) -> None:
            self.clicks.append((x, y, button, click_count))

        async def type_text(self, text: str) -> None:
            self.typed.append(text)

        async def press_key(self, key: str) -> None:
            self.keys.append(key)

        async def move(self, x: int, y: int) -> None:
            pass

        async def drag(
            self, start: tuple[int, int], end: tuple[int, int], **kwargs: Any
        ) -> None:
            pass

        async def scroll(self, x: int = 0, y: int = 0) -> None:
            self.scrolls.append((x, y))

    return _StubInputHandler


# ---------------------------------------------------------------------------
# _parse_png_size unit tests
# ---------------------------------------------------------------------------


def test_parse_png_size_2x2() -> None:
    assert _parse_png_size(TINY_2X2_PNG) == (2, 2)


def test_parse_png_size_4x4() -> None:
    assert _parse_png_size(TINY_4X4_PNG) == (4, 4)


def test_parse_png_size_rejects_too_short() -> None:
    with pytest.raises(ValueError, match="too short"):
        _parse_png_size(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5)


def test_parse_png_size_rejects_non_png() -> None:
    with pytest.raises(ValueError, match="Not a PNG"):
        _parse_png_size(b"\xff\xd8\xff\xe0" + b"\x00" * 20)


# ---------------------------------------------------------------------------
# WindowsDriver unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_detects_scale_factor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """scale_x / scale_y == 2.0 when logical=2x2 and screenshot pixels=4x4."""
    StubInput = _make_stub_input_class()
    monkeypatch.setattr("haindy.windows.driver.WindowsScreenCapture", StubScreenCapture)
    monkeypatch.setattr("haindy.windows.driver.WindowsInputHandler", StubInput)

    driver = WindowsDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
    )
    await driver.start()

    assert driver.input_handler is not None
    assert driver.input_handler.scale_x == 2.0
    assert driver.input_handler.scale_y == 2.0
    assert driver.input_handler.logical_size == (2, 2)


@pytest.mark.asyncio
async def test_start_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling start() twice does not re-create the input handler."""
    created: list[int] = []

    class CountingInput(_make_stub_input_class()):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created.append(1)

    monkeypatch.setattr("haindy.windows.driver.WindowsScreenCapture", StubScreenCapture)
    monkeypatch.setattr("haindy.windows.driver.WindowsInputHandler", CountingInput)

    driver = WindowsDriver(
        screenshot_dir=tmp_path / "shots",
        cache_path=tmp_path / "coords.json",
    )
    await driver.start()
    await driver.start()

    assert len(created) == 1


@pytest.mark.asyncio
async def test_click_delegates_to_input_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    await driver.click(10, 20)
    assert driver.input_handler is not None
    assert driver.input_handler.clicks == [(10, 20, "left", 1)]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_type_text_delegates_to_input_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    await driver.type_text("hello world")
    assert driver.input_handler is not None
    assert driver.input_handler.typed == ["hello world"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_press_key_delegates_to_input_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    await driver.press_key("ctrl+c")
    assert driver.input_handler is not None
    assert driver.input_handler.keys == ["ctrl+c"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_scroll_by_pixels_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    await driver.scroll_by_pixels(x=0, y=120)
    assert driver.input_handler is not None
    assert driver.input_handler.scrolls == [(0, 120)]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_screenshot_returns_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    result = await driver.screenshot()
    assert isinstance(result, bytes)
    assert result == TINY_4X4_PNG


@pytest.mark.asyncio
async def test_get_viewport_size_returns_pixel_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    w, h = await driver.get_viewport_size()
    assert w == 4
    assert h == 4


@pytest.mark.asyncio
async def test_scroll_rejects_invalid_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    with pytest.raises(ValueError, match="Invalid scroll direction"):
        await driver.scroll("diagonal", 100)


@pytest.mark.asyncio
async def test_stop_clears_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = _make_driver(monkeypatch, tmp_path)
    await driver.start()
    assert driver._started is True
    assert driver.input_handler is not None

    await driver.stop()
    assert driver._started is False
    assert driver.input_handler is None


# ---------------------------------------------------------------------------
# WindowsInputHandler coordinate scaling unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_handler_scales_click_coordinates() -> None:
    """Click at pixel (100, 100) on 2x display should inject logical (50, 50)."""
    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=2.0,
        scale_y=2.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
    )

    await handler.click(100, 100)

    assert stub_mouse.positions[-1] == (50, 50)
    assert len(stub_mouse.clicks) == 1


@pytest.mark.asyncio
async def test_input_handler_no_scale_on_non_hidpi() -> None:
    """On non-HiDPI (scale=1.0) coordinates pass through unchanged."""
    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1920, 1080),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
    )

    await handler.click(300, 400)

    assert stub_mouse.positions[-1] == (300, 400)


@pytest.mark.asyncio
async def test_input_handler_type_text() -> None:
    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
        key_delay_ms=0,
    )

    await handler.type_text("abc")

    assert stub_keyboard.typed == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_input_handler_ctrl_c_uses_ctrl_key() -> None:
    """On Windows 'cmd+c' should press ctrl (not Key.cmd) then 'c'."""
    from pynput.keyboard import Key

    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
        key_delay_ms=0,
    )

    await handler.press_key("cmd+c")

    # On Windows, 'cmd' maps to Key.ctrl (the primary action modifier)
    assert Key.ctrl in stub_keyboard.pressed
    assert Key.ctrl in stub_keyboard.released
    assert "c" in stub_keyboard.pressed
    assert "c" in stub_keyboard.released


@pytest.mark.asyncio
async def test_input_handler_win_key_uses_cmd() -> None:
    """On Windows 'win' / 'meta' should map to Key.cmd (the Windows key)."""
    from pynput.keyboard import Key

    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
        key_delay_ms=0,
    )

    await handler.press_key("win+r")

    assert Key.cmd in stub_keyboard.pressed
    assert Key.cmd in stub_keyboard.released


@pytest.mark.asyncio
async def test_input_handler_press_special_key() -> None:
    """Pressing 'enter' should press and release Key.enter."""
    from pynput.keyboard import Key

    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
        key_delay_ms=0,
    )

    await handler.press_key("enter")

    assert Key.enter in stub_keyboard.pressed
    assert Key.enter in stub_keyboard.released


@pytest.mark.asyncio
async def test_input_handler_scroll_direction() -> None:
    """Scroll down (positive y) should produce negative pynput dy (scroll down)."""
    stub_mouse = StubMouseController()
    stub_keyboard = StubKeyboardController()

    handler = WindowsInputHandler(
        logical_size=(1280, 800),
        scale_x=1.0,
        scale_y=1.0,
        mouse_controller=stub_mouse,
        keyboard_controller=stub_keyboard,
    )

    await handler.scroll(x=0, y=120)

    assert len(stub_mouse.scrolls) == 1
    _dx, dy = stub_mouse.scrolls[0]
    assert dy < 0  # negative = scroll down
