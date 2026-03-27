import asyncio
import json
import sys
from collections.abc import Sequence

import pytest

from haindy.mobile.idb_client import IDBClient, IDBClientProtocol, IDBCommandResult
from haindy.mobile.ios_driver import IOSDriver


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


def _describe_json(width_points: int, height_points: int) -> bytes:
    data = {
        "udid": "test-udid",
        "name": "Test Device",
        "state": "Booted",
        "screen": {
            "width": width_points * 3,
            "height": height_points * 3,
            "density": 3.0,
            "width_points": width_points,
            "height_points": height_points,
        },
    }
    return json.dumps(data).encode("utf-8")


class StubIDBClient(IDBClientProtocol):
    def __init__(
        self,
        describe_output: bytes | None = None,
        screenshot_path_bytes: bytes | None = None,
    ) -> None:
        self.udid: str | None = "test-udid"
        self.timeout_seconds = 15.0
        self.commands: list[tuple[str, ...]] = []
        self._describe_output = describe_output or _describe_json(390, 844)
        # screenshot_path_bytes is written to the path arg when 'screenshot' is called
        self._screenshot_bytes = screenshot_path_bytes or _png_bytes(1170, 2532)

    async def resolve_udid(self, preferred_udid: str | None = None) -> str:
        return preferred_udid or "test-udid"

    async def run_idb(
        self,
        *args: str,
        timeout_seconds: float | None = None,
        check: bool = True,
        udid: str | None = None,
    ) -> IDBCommandResult:
        del timeout_seconds, check, udid
        command = tuple(args)
        self.commands.append(command)

        if command == ("describe",):
            return IDBCommandResult(("idb", *command), 0, self._describe_output, b"")

        if len(command) >= 2 and command[0] == "screenshot":
            # Write PNG bytes to the provided path
            output_path = command[1]
            with open(output_path, "wb") as fh:
                fh.write(self._screenshot_bytes)
            return IDBCommandResult(("idb", *command), 0, b"", b"")

        return IDBCommandResult(("idb", *command), 0, b"", b"")

    async def run_command(
        self,
        command: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> IDBCommandResult:
        del timeout_seconds
        normalized = tuple(command)
        self.commands.append(normalized)
        if normalized and normalized[0] != "idb":
            raise ValueError("Only idb commands are allowed.")
        return IDBCommandResult(normalized, 0, b"", b"")


@pytest.mark.asyncio
async def test_idb_client_rejects_non_idb_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    async def _unexpected_subprocess(*args: Sequence[str], **kwargs: object) -> None:
        del args, kwargs
        called["value"] = True
        raise AssertionError("subprocess should not run for non-idb commands")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_subprocess)
    client = IDBClient()
    with pytest.raises(ValueError, match="Only idb commands are allowed"):
        await client.run_command(["python3", "-V"])
    assert called["value"] is False


@pytest.mark.asyncio
async def test_ios_driver_get_viewport_size_from_describe() -> None:
    stub = StubIDBClient(describe_output=_describe_json(390, 844))
    driver = IOSDriver(idb_client=stub)
    await driver.start()
    assert await driver.get_viewport_size() == (390, 844)


@pytest.mark.asyncio
async def test_ios_driver_scales_click_coordinates_from_screenshot_space() -> None:
    # Device logical: 390x844, screenshot physical: 1170x2532 (3x Retina)
    stub = StubIDBClient(
        describe_output=_describe_json(390, 844),
        screenshot_path_bytes=_png_bytes(1170, 2532),
    )
    driver = IOSDriver(idb_client=stub)
    await driver.start()
    await driver.screenshot()
    # Click at pixel (585, 1266) in screenshot space => should map to (195, 422) in points
    await driver.click(585, 1266)

    tap_commands = [c for c in stub.commands if c and c[0] == "ui" and c[1] == "tap"]
    assert len(tap_commands) == 1
    assert tap_commands[0] == ("ui", "tap", "195", "422")


@pytest.mark.asyncio
async def test_ios_driver_type_text() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    await driver.type_text("hello world")
    assert ("ui", "text", "hello world") in stub.commands


@pytest.mark.asyncio
async def test_ios_driver_press_key_mapping() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    await driver.press_key("enter")
    await driver.press_key("backspace")
    await driver.press_key("cmd+a")

    assert ("ui", "key-sequence", "40") in stub.commands
    assert ("ui", "key-sequence", "42") in stub.commands
    assert ("ui", "key-sequence", "227", "4") in stub.commands


@pytest.mark.asyncio
async def test_ios_driver_press_key_letter_digit_mapping() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    await driver.press_key("a")
    await driver.press_key("z")
    await driver.press_key("1")
    await driver.press_key("0")

    assert ("ui", "key-sequence", "4") in stub.commands  # a
    assert ("ui", "key-sequence", "29") in stub.commands  # z
    assert ("ui", "key-sequence", "30") in stub.commands  # 1
    assert ("ui", "key-sequence", "39") in stub.commands  # 0


@pytest.mark.asyncio
async def test_ios_driver_press_key_arrow_aliases() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    await driver.press_key("arrowup")
    await driver.press_key("arrowdown")
    await driver.press_key("arrowleft")
    await driver.press_key("arrowright")

    assert ("ui", "key-sequence", "82") in stub.commands  # up
    assert ("ui", "key-sequence", "81") in stub.commands  # down
    assert ("ui", "key-sequence", "80") in stub.commands  # left
    assert ("ui", "key-sequence", "79") in stub.commands  # right


@pytest.mark.asyncio
async def test_ios_driver_press_key_unknown_raises() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    with pytest.raises(ValueError, match="Unsupported iOS key"):
        await driver.press_key("unknown_key_xyz")


@pytest.mark.asyncio
async def test_ios_driver_scroll_rejects_invalid_direction() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    with pytest.raises(ValueError, match="Invalid scroll direction"):
        await driver.scroll("diagonal", 100)


@pytest.mark.asyncio
async def test_ios_driver_scroll_produces_swipe() -> None:
    stub = StubIDBClient(describe_output=_describe_json(390, 844))
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    await driver.scroll("down", 200)

    swipe_commands = [
        c for c in stub.commands if c and c[0] == "ui" and c[1] == "swipe"
    ]
    assert len(swipe_commands) == 1
    # Command: ("ui", "swipe", start_x, start_y, end_x, end_y, "--duration", secs)
    # Scrolling down means finger moves up: start_y > end_y
    start_y = int(swipe_commands[0][3])
    end_y = int(swipe_commands[0][5])
    assert start_y > end_y


@pytest.mark.asyncio
async def test_ios_driver_move_mouse_is_noop_and_capture_compatible() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
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
async def test_ios_driver_screenshot_validates_png() -> None:
    stub = StubIDBClient(screenshot_path_bytes=_png_bytes(1170, 2532))
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    data = await driver.screenshot()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert driver._last_screenshot_size == (1170, 2532)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only: simctl fallback requires xcrun")
@pytest.mark.asyncio
async def test_ios_driver_screenshot_invalid_png_raises() -> None:
    stub = StubIDBClient(screenshot_path_bytes=b"not a png")
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    with pytest.raises(RuntimeError, match="valid PNG"):
        await driver.screenshot()


@pytest.mark.asyncio
async def test_ios_driver_get_page_url_and_title() -> None:
    stub = StubIDBClient()
    driver = IOSDriver(idb_client=stub)
    await driver.start()

    assert await driver.get_page_url() == ""
    assert await driver.get_page_title() == "iOS Device"


@pytest.mark.asyncio
async def test_idb_client_parse_targets_text() -> None:
    text = (
        "iPhone 14 Pro | 12345678-1234-1234-1234-123456789012 | Booted | simulator | iOS 17.0\n"
        "My iPhone | 00008030-0012-3456-7890-ABCDEF012345 | Connected | device | iOS 16.0\n"
    )
    from haindy.mobile.idb_client import IDBClient

    client = IDBClient()
    targets = client._parse_targets_text(text)
    assert len(targets) == 2
    assert targets[0]["udid"] == "12345678-1234-1234-1234-123456789012"
    assert targets[0]["state"] == "Booted"
    assert targets[1]["udid"] == "00008030-0012-3456-7890-ABCDEF012345"
    assert targets[1]["state"] == "Connected"
