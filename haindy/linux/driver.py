"""Desktop driver implementing the browser driver interface using OS-level input."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import logging
import shutil
import subprocess
from asyncio import subprocess as aio_subprocess
from collections.abc import Iterable
from pathlib import Path

from haindy.core.coordinate_cache import CoordinateCache
from haindy.core.interfaces import AutomationDriver
from haindy.linux.resolution_manager import ResolutionManager
from haindy.linux.screen_capture import ScreenCapture
from haindy.linux.virtual_input import VirtualInput

logger = logging.getLogger(__name__)

SUPPORTED_KEYBOARD_LAYOUTS = frozenset({"us", "es"})
DEFAULT_KEYBOARD_LAYOUT = "us"
AUTO_KEYBOARD_LAYOUT = "auto"


def _first_xkb_layout(layouts: str) -> str | None:
    """Return the first layout from an XKB layout list such as ``es,us``."""
    for layout in layouts.split(","):
        normalized = layout.strip().lower()
        if normalized:
            return normalized
    return None


def _layout_from_setxkbmap_output(output: str) -> str | None:
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "layout":
            return _first_xkb_layout(value)
    return None


def _layout_from_localectl_output(output: str) -> str | None:
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "x11 layout":
            return _first_xkb_layout(value)
    return None


def _gsettings_current_index(output: str) -> int | None:
    stripped = output.strip()
    if not stripped:
        return None
    token = stripped.rsplit(maxsplit=1)[-1]
    if token.isdigit():
        return int(token)
    return None


def _layout_from_gsettings_outputs(
    current_output: str, sources_output: str
) -> str | None:
    current_index = _gsettings_current_index(current_output)
    if current_index is None:
        return None

    try:
        sources = ast.literal_eval(sources_output.strip())
    except (SyntaxError, ValueError):
        return None

    if not isinstance(sources, list) or not 0 <= current_index < len(sources):
        return None

    source = sources[current_index]
    if (
        not isinstance(source, tuple)
        or len(source) < 2
        or str(source[0]).lower() != "xkb"
    ):
        return None
    return _first_xkb_layout(str(source[1]))


def _run_layout_command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _detect_desktop_keyboard_layout() -> str | None:
    """Best-effort XKB layout detection for Linux desktop sessions."""
    gsettings_current = _run_layout_command(
        ["gsettings", "get", "org.gnome.desktop.input-sources", "current"]
    )
    gsettings_sources = _run_layout_command(
        ["gsettings", "get", "org.gnome.desktop.input-sources", "sources"]
    )
    if gsettings_current and gsettings_sources:
        layout = _layout_from_gsettings_outputs(gsettings_current, gsettings_sources)
        if layout:
            return layout

    setxkbmap_output = _run_layout_command(["setxkbmap", "-query"])
    if setxkbmap_output:
        layout = _layout_from_setxkbmap_output(setxkbmap_output)
        if layout:
            return layout

    localectl_output = _run_layout_command(["localectl", "status"])
    if localectl_output:
        return _layout_from_localectl_output(localectl_output)

    return None


def _resolve_keyboard_layout(layout: str) -> str:
    """Resolve configured keyboard layout to a concrete VirtualInput layout."""
    normalized = (layout or AUTO_KEYBOARD_LAYOUT).strip().lower()
    if normalized in SUPPORTED_KEYBOARD_LAYOUTS:
        return normalized
    if normalized and normalized != AUTO_KEYBOARD_LAYOUT:
        logger.warning(
            "Unsupported desktop keyboard layout configured; falling back to US",
            extra={"keyboard_layout": layout},
        )
        return DEFAULT_KEYBOARD_LAYOUT

    detected = _detect_desktop_keyboard_layout()
    if detected in SUPPORTED_KEYBOARD_LAYOUTS:
        logger.info(
            "Auto-detected desktop keyboard layout",
            extra={"keyboard_layout": detected},
        )
        return detected

    if detected:
        logger.warning(
            "Unsupported desktop keyboard layout detected; falling back to US",
            extra={"keyboard_layout": detected},
        )
    else:
        logger.warning(
            "Could not auto-detect desktop keyboard layout; falling back to US"
        )
    return DEFAULT_KEYBOARD_LAYOUT


class DesktopDriver(AutomationDriver):
    """OS-level driver that controls an existing desktop session."""

    def __init__(
        self,
        screenshot_dir: Path,
        cache_path: Path,
        prefer_resolution: tuple[int, int] = (1920, 1080),
        enable_resolution_switch: bool = False,
        display: str | None = None,
        keyboard_layout: str = AUTO_KEYBOARD_LAYOUT,
        keyboard_emit_scancodes: bool = True,
        keyboard_key_delay_ms: int = 12,
        clipboard_timeout_seconds: float = 3.0,
        clipboard_hold_seconds: float = 15.0,
        max_screenshots: int | None = None,
    ) -> None:
        self.resolution_manager = ResolutionManager(
            preferred_width=prefer_resolution[0],
            preferred_height=prefer_resolution[1],
            enable_switch=enable_resolution_switch,
        )
        self.screen_capture = ScreenCapture(
            resolution_manager=self.resolution_manager,
            screenshot_dir=screenshot_dir,
            display=display,
            max_screenshots=max_screenshots,
        )
        self.keyboard_layout = keyboard_layout
        self.keyboard_emit_scancodes = keyboard_emit_scancodes
        self.keyboard_key_delay_ms = keyboard_key_delay_ms
        self.coordinate_cache = CoordinateCache(cache_path)
        self.virtual_input: VirtualInput | None = None
        self._started = False
        self._clipboard_timeout_seconds = max(float(clipboard_timeout_seconds), 0.5)
        self._clipboard_hold_seconds = max(float(clipboard_hold_seconds), 0.5)
        self._clipboard_owner: aio_subprocess.Process | None = None
        self._clipboard_owner_reaper: asyncio.Task[None] | None = None
        self._capturing = False
        self._captured_calls: list[dict[str, object]] = []

    async def start(self) -> None:
        """Initialize resolution and virtual input device."""
        if self._started and self.virtual_input is not None:
            return
        self.resolution_manager.maybe_downshift()
        try:
            if self.virtual_input is None:
                viewport = self.resolution_manager.viewport_size()
                keyboard_layout = _resolve_keyboard_layout(self.keyboard_layout)
                self.virtual_input = VirtualInput(
                    viewport=viewport,
                    keyboard_layout=keyboard_layout,
                    emit_scancodes=self.keyboard_emit_scancodes,
                    key_delay_ms=self.keyboard_key_delay_ms,
                )
        except Exception:
            # If startup fails after a resolution switch, restore immediately so the
            # host display is not left in the downshifted mode.
            with contextlib.suppress(Exception):
                self.resolution_manager.restore()
            self.virtual_input = None
            self._started = False
            raise
        self._started = True

    async def stop(self) -> None:
        """Restore resolution if changed."""
        if not self._started and self.virtual_input is None:
            return
        await self._cleanup_clipboard_owner()
        self.resolution_manager.restore()
        self.virtual_input = None
        self._started = False

    async def __aenter__(self) -> DesktopDriver:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()

    async def navigate(self, url: str) -> None:
        """Best-effort navigation by typing into the address bar."""
        await self._ensure_ready()
        await self.press_key("ctrl+l")
        await self.type_text(url)
        await self.press_key("enter")

    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        if modifiers:
            await virtual_input.click(
                x, y, button=button, click_count=click_count, modifiers=modifiers
            )
        else:
            await virtual_input.click(x, y, button=button, click_count=click_count)
        call_info: dict[str, object] = {
            "x": x,
            "y": y,
            "button": button,
            "click_count": click_count,
        }
        if modifiers:
            call_info["modifiers"] = modifiers
        self._capture_call("click", call_info)

    async def move_mouse(self, x: int, y: int, steps: int = 1) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        await virtual_input.move(x, y, steps=steps)
        self._capture_call("move_mouse", {"x": x, "y": y, "steps": steps})

    async def drag_mouse(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 1,
    ) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        await virtual_input.drag((start_x, start_y), (end_x, end_y), steps=steps)
        self._capture_call(
            "drag_mouse",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "steps": steps,
            },
        )

    async def type_text(self, text: str) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        await virtual_input.type_text(text)
        self._capture_call("type_text", {"length": len(text)})

    async def press_key(self, key: str | Iterable[str]) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        await virtual_input.press_key(key)
        self._capture_call(
            "press_key",
            {"key": list(key) if isinstance(key, (list, tuple, set)) else key},
        )

    async def scroll(
        self,
        direction: str,
        amount: int,
        origin: tuple[int, int] | None = None,
    ) -> None:
        normalized_direction = str(direction or "").strip().lower()
        if normalized_direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Invalid scroll direction: {direction!r}")
        magnitude = abs(int(amount))
        delta = magnitude if normalized_direction in {"down", "right"} else -magnitude
        await self.scroll_by_pixels(
            y=delta if normalized_direction in {"up", "down"} else 0,
            x=delta if normalized_direction in {"left", "right"} else 0,
        )

    async def scroll_by_pixels(
        self,
        x: int = 0,
        y: int = 0,
        smooth: bool = True,
        origin: tuple[int, int] | None = None,
    ) -> None:
        await self._ensure_ready()
        virtual_input = self.virtual_input
        assert virtual_input is not None
        await virtual_input.scroll(x=x, y=y)
        self._capture_call("scroll_by_pixels", {"x": x, "y": y, "smooth": smooth})

    async def screenshot(self) -> bytes:
        bytes_, _ = await self._capture("desktop")
        self._capture_call("screenshot", {"label": "desktop"})
        return bytes_

    async def wait(self, milliseconds: int) -> None:
        await asyncio.sleep(milliseconds / 1000.0)
        self._capture_call("wait", {"milliseconds": milliseconds})

    async def get_viewport_size(self) -> tuple[int, int]:
        return self.resolution_manager.viewport_size()

    async def get_page_url(self) -> str:
        return ""

    async def get_page_title(self) -> str:
        return "Desktop Session"

    async def save_screenshot(self, path: Path) -> None:
        bytes_, _ = await self._capture("save")
        path.write_bytes(bytes_)
        self._capture_call("save_screenshot", {"path": str(path)})

    async def wait_for_load_state(self, state: str = "networkidle") -> None:
        return

    async def capture(self, label: str) -> tuple[bytes, str]:
        """Capture a screenshot with the given label."""
        bytes_, path = await self._capture(label)
        self._capture_call("capture", {"label": label, "path": path})
        return bytes_, path

    def start_capture(self) -> None:
        """Begin collecting automation call traces."""
        self._capturing = True
        self._captured_calls = []

    def stop_capture(self) -> list[dict[str, object]]:
        """Stop trace collection and return captured calls."""
        calls = self._captured_calls.copy()
        self._capturing = False
        self._captured_calls = []
        return calls

    async def read_clipboard(self) -> str:
        """Return the current clipboard contents (requires xclip)."""
        cmd = self._clipboard_read_command()
        if not cmd:
            raise RuntimeError("Unable to read clipboard: xclip not available.")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._clipboard_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Clipboard read timed out.") from exc
        if proc.returncode != 0:
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"Clipboard read failed: {stderr_text.strip()}")
        return (stdout or b"").decode("utf-8", errors="ignore").strip()

    async def write_clipboard(self, text: str) -> None:
        """Set the clipboard contents (requires xclip)."""
        cmd = self._clipboard_write_command()
        if not cmd:
            raise RuntimeError("Unable to write clipboard: xclip not available.")
        await self._cleanup_clipboard_owner()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.DEVNULL,
            stderr=aio_subprocess.DEVNULL,
        )
        if not proc.stdin:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Clipboard write failed: unable to open stdin.")
        proc.stdin.write(text.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        # xclip may legitimately stay alive to serve the selection (X11 ownership model).
        # Do not block waiting for exit; instead keep it around briefly and reap it.
        self._clipboard_owner = proc
        self._clipboard_owner_reaper = asyncio.create_task(
            self._reap_clipboard_owner(proc)
        )
        try:
            await asyncio.wait_for(
                proc.wait(), timeout=min(self._clipboard_timeout_seconds, 0.25)
            )
        except asyncio.TimeoutError:
            return
        if proc.returncode != 0:
            _, stderr = await proc.communicate()
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"Clipboard write failed: {stderr_text.strip()}")

    async def _capture(self, label: str) -> tuple[bytes, str]:
        await self._ensure_ready()
        return self.screen_capture.capture(label)

    def _capture_call(self, method: str, params: dict[str, object]) -> None:
        if not self._capturing:
            return
        self._captured_calls.append({"method": method, "params": params})

    async def _ensure_ready(self) -> None:
        if not self._started:
            await self.start()
        if not self.virtual_input:
            viewport = self.resolution_manager.viewport_size()
            self.virtual_input = VirtualInput(
                viewport=viewport,
                keyboard_layout=self.keyboard_layout,
                emit_scancodes=self.keyboard_emit_scancodes,
                key_delay_ms=self.keyboard_key_delay_ms,
            )

    @staticmethod
    def _clipboard_read_command() -> list[str] | None:
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard", "-o"]
        return None

    @staticmethod
    def _clipboard_write_command() -> list[str] | None:
        if shutil.which("xclip"):
            # xclip defaults to backgrounding itself; force foreground so the agent can
            # manage its lifetime and avoid orphaned clipboard owner processes.
            return ["xclip", "-quiet", "-selection", "clipboard"]
        return None

    async def _cleanup_clipboard_owner(self) -> None:
        task = self._clipboard_owner_reaper
        self._clipboard_owner_reaper = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        proc = self._clipboard_owner
        self._clipboard_owner = None
        if not proc or proc.returncode is not None:
            return
        proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    async def _reap_clipboard_owner(self, proc: aio_subprocess.Process) -> None:
        try:
            await asyncio.sleep(self._clipboard_hold_seconds)
            if proc.returncode is None:
                proc.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                if proc.returncode is None:
                    proc.kill()
            if proc.returncode is None:
                await proc.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to reap clipboard owner process.")
