from __future__ import annotations

import pytest

from haindy.desktop.virtual_input import VirtualInput


class _DummyProcess:
    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr


@pytest.mark.asyncio
async def test_virtual_input_falls_back_to_xdotool_when_uinput_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    async def _fake_create_subprocess_exec(*cmd, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(tuple(str(part) for part in cmd))
        return _DummyProcess()

    monkeypatch.setattr("haindy.desktop.virtual_input._EVDEV_AVAILABLE", False)
    monkeypatch.setattr(
        "haindy.desktop.virtual_input.shutil.which",
        lambda name: "/usr/bin/xdotool" if name == "xdotool" else None,
    )
    monkeypatch.setattr(
        "haindy.desktop.virtual_input.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    virtual_input = VirtualInput(viewport=(1280, 720))
    await virtual_input.click(10, 20, button="right", click_count=2)
    await virtual_input.type_text("abc")
    await virtual_input.press_key("ctrl+l")
    await virtual_input.scroll(y=240)

    assert virtual_input._ui is None
    assert virtual_input._xdotool_binary == "/usr/bin/xdotool"
    assert commands == [
        ("/usr/bin/xdotool", "mousemove", "--sync", "10", "20"),
        ("/usr/bin/xdotool", "click", "--repeat", "2", "3"),
        (
            "/usr/bin/xdotool",
            "type",
            "--delay",
            "12",
            "--clearmodifiers",
            "--",
            "abc",
        ),
        ("/usr/bin/xdotool", "key", "--clearmodifiers", "ctrl+l"),
        ("/usr/bin/xdotool", "click", "--repeat", "2", "5"),
    ]


def test_virtual_input_raises_without_uinput_or_xdotool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("haindy.desktop.virtual_input._EVDEV_AVAILABLE", False)
    monkeypatch.setattr("haindy.desktop.virtual_input.shutil.which", lambda name: None)

    with pytest.raises(
        RuntimeError, match="desktop input backend requires Linux evdev or xdotool"
    ):
        VirtualInput(viewport=(1280, 720))
