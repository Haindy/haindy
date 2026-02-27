import asyncio
from collections.abc import Sequence

import pytest

from src.mobile.adb_client import ADBClient, ADBCommandResult
from src.mobile.driver import MobileDriver


def _png_bytes(width: int, height: int) -> bytes:
    ihdr_payload = (
        width.to_bytes(4, byteorder="big")
        + height.to_bytes(4, byteorder="big")
        + b"\x08\x02\x00\x00\x00"
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + len(ihdr_payload).to_bytes(4, byteorder="big")
        + b"IHDR"
        + ihdr_payload
        + b"\x00\x00\x00\x00"
    )


class StubADBClient:
    def __init__(
        self,
        wm_size_output: str = "Physical size: 1080x2400\n",
        screenshot_bytes: bytes | None = None,
    ) -> None:
        self.serial: str | None = None
        self.timeout_seconds = 15.0
        self.commands: list[tuple[str, ...]] = []
        self._wm_size_output = wm_size_output.encode("utf-8")
        self._screenshot_bytes = screenshot_bytes or _png_bytes(540, 1200)

    async def resolve_serial(self, preferred_serial: str | None = None) -> str:
        return preferred_serial or "device-123"

    async def run_adb(
        self,
        *args: str,
        timeout_seconds: float | None = None,
        check: bool = True,
        serial: str | None = None,
    ) -> ADBCommandResult:
        del timeout_seconds, check, serial
        command = tuple(args)
        self.commands.append(command)
        if command == ("get-state",):
            return ADBCommandResult(("adb", *command), 0, b"device\n", b"")
        if command == ("shell", "wm", "size"):
            return ADBCommandResult(("adb", *command), 0, self._wm_size_output, b"")
        if command == ("exec-out", "screencap", "-p"):
            return ADBCommandResult(("adb", *command), 0, self._screenshot_bytes, b"")
        return ADBCommandResult(("adb", *command), 0, b"", b"")

    async def run_command(
        self,
        command: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> ADBCommandResult:
        del timeout_seconds
        normalized = tuple(command)
        self.commands.append(normalized)
        if normalized and normalized[0] != "adb":
            raise ValueError("Only adb commands are allowed.")
        return ADBCommandResult(normalized, 0, b"", b"")


@pytest.mark.asyncio
async def test_adb_client_rejects_non_adb_command(monkeypatch) -> None:
    called = {"value": False}

    async def _unexpected_subprocess(*args: Sequence[str], **kwargs: object) -> None:
        del args, kwargs
        called["value"] = True
        raise AssertionError("subprocess should not run for non-adb commands")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_subprocess)
    client = ADBClient()
    with pytest.raises(ValueError, match="Only adb commands are allowed"):
        await client.run_command(["python3", "-V"])
    assert called["value"] is False


@pytest.mark.asyncio
async def test_mobile_driver_prefers_override_viewport_size() -> None:
    stub = StubADBClient(
        wm_size_output="Physical size: 1080x2400\nOverride size: 720x1600\n"
    )
    driver = MobileDriver(adb_client=stub)
    assert await driver.get_viewport_size() == (720, 1600)


@pytest.mark.asyncio
async def test_mobile_driver_scales_click_coordinates_from_screenshot_space() -> None:
    stub = StubADBClient(
        wm_size_output="Physical size: 1080x2400\n",
        screenshot_bytes=_png_bytes(540, 1200),
    )
    driver = MobileDriver(adb_client=stub)
    await driver.start()
    await driver.screenshot()
    await driver.click(270, 600)

    assert ("shell", "input", "tap", "540", "1200") in stub.commands


@pytest.mark.asyncio
async def test_mobile_driver_type_and_press_key_mapping() -> None:
    stub = StubADBClient()
    driver = MobileDriver(adb_client=stub)
    await driver.start()

    await driver.type_text("hello world&ok")
    await driver.press_key("ctrl+l")

    assert ("shell", "input", "text", "hello%sworld\\&ok") in stub.commands
    assert ("shell", "input", "keyevent", "113", "40") in stub.commands

    with pytest.raises(ValueError, match="Unsupported Android key"):
        await driver.press_key("unknown_key")


@pytest.mark.asyncio
async def test_mobile_driver_scroll_rejects_invalid_direction() -> None:
    stub = StubADBClient()
    driver = MobileDriver(adb_client=stub)
    await driver.start()
    with pytest.raises(ValueError, match="Invalid scroll direction"):
        await driver.scroll("diagonal", 100)


@pytest.mark.asyncio
async def test_mobile_driver_move_mouse_is_noop_and_capture_compatible() -> None:
    stub = StubADBClient()
    driver = MobileDriver(adb_client=stub)
    await driver.start()

    command_count_before = len(stub.commands)
    driver.start_capture()
    await driver.move_mouse(10, 20, steps=3)
    captured_calls = driver.stop_capture()

    assert len(stub.commands) == command_count_before
    assert captured_calls == [
        {"method": "move_mouse", "params": {"x": 10, "y": 20, "steps": 3, "noop": True}}
    ]


@pytest.mark.asyncio
async def test_mobile_driver_launch_app_and_run_commands() -> None:
    stub = StubADBClient()
    driver = MobileDriver(adb_client=stub)
    await driver.configure_target(
        adb_serial="emulator-5554",
        app_package="com.example.app",
        app_activity=".MainActivity",
    )
    await driver.start()

    await driver.launch_app("com.example.app", ".MainActivity")
    await driver.run_adb_commands(["adb devices"])

    assert (
        "shell",
        "am",
        "start",
        "-n",
        "com.example.app/.MainActivity",
    ) in stub.commands
    assert ("adb", "devices") in stub.commands
