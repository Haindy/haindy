"""System dependency checker for haindy doctor."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

_console = Console()


def _ok(notes: str = "") -> tuple[Text, str]:
    return Text("OK", style="bold green"), notes


def _missing(notes: str = "") -> tuple[Text, str]:
    return Text("MISSING", style="bold red"), notes


def _na(notes: str = "") -> tuple[Text, str]:
    return Text("N/A", style="dim"), notes


def _check_python_version() -> tuple[Text, str]:
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi >= (3, 10):
        return _ok(version_str)
    return _missing(f"{version_str} (need >= 3.10)")


def _check_haindy_installed() -> tuple[Text, str]:
    try:
        version = importlib.metadata.version("haindy")
        return _ok(version)
    except importlib.metadata.PackageNotFoundError:
        return _missing("run: pip install haindy")


def _check_api_key(provider: str) -> tuple[Text, str]:
    try:
        from haindy.auth.credentials import get_api_key

        if get_api_key(provider):
            return _ok("")
        return _missing(f"run: haindy --auth login {provider}")
    except Exception:
        return _missing(f"run: haindy --auth login {provider}")


def _check_codex_oauth() -> tuple[Text, str]:
    try:
        from haindy.auth.manager import OpenAIAuthManager

        status = OpenAIAuthManager().get_status()
        if status.oauth_connected:
            label = status.oauth_account_label or ""
            if status.oauth_expired:
                return _missing(
                    f"token expired ({label}) — run: haindy --auth login openai-codex"
                )
            return _ok(label)
        return _missing("run: haindy --auth login openai-codex")
    except Exception:
        return _missing("run: haindy --auth login openai-codex")


def _check_macos_pynput() -> tuple[Text, str]:
    if importlib.util.find_spec("pynput") is not None:
        return _ok("")
    return _missing("pip install pynput")


def _check_macos_mss() -> tuple[Text, str]:
    if importlib.util.find_spec("mss") is not None:
        return _ok("")
    return _missing("pip install mss")


def _check_macos_accessibility() -> tuple[Text, str]:
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of first process',
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _ok("")
        return _missing("Grant in System Settings > Privacy & Security > Accessibility")
    except Exception:
        return _missing("Grant in System Settings > Privacy & Security > Accessibility")


def _check_macos_screen_recording() -> tuple[Text, str]:
    if importlib.util.find_spec("mss") is None:
        return _na("mss not installed")
    try:
        import mss

        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 1, "height": 1}
            screenshot = sct.grab(monitor)
            pixel = screenshot.pixel(0, 0)
            if pixel == (0, 0, 0):
                return _missing(
                    "Grant in System Settings > Privacy & Security > Screen Recording"
                )
        return _ok("")
    except Exception:
        return _missing(
            "Grant in System Settings > Privacy & Security > Screen Recording"
        )


def _check_linux_which(tool: str, install_hint: str = "") -> tuple[Text, str]:
    if shutil.which(tool):
        return _ok("")
    notes = install_hint if install_hint else f"install {tool}"
    return _missing(notes)


def _check_linux_uinput() -> tuple[Text, str]:
    if os.access("/dev/uinput", os.W_OK):
        return _ok("")
    return _missing("sudo usermod -aG input $USER && reboot")


def _check_linux_display() -> tuple[Text, str]:
    display = os.environ.get("DISPLAY")
    if display:
        return _ok(display)
    return _missing("DISPLAY env var not set")


def run_doctor(
    include_android: bool = False,
    include_ios: bool = False,
) -> int:
    """Run system dependency checks and print a rich table.

    Returns 0 if all required components are present, 1 if any required
    component is missing.
    """
    table = Table(title="Haindy System Check", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Notes")

    any_missing = False

    def _add(label: str, status: Text, notes: str, required: bool = True) -> None:
        nonlocal any_missing
        table.add_row(label, status, notes)
        if required and status.plain == "MISSING":
            any_missing = True

    status, notes = _check_python_version()
    _add("Python >= 3.10", status, notes)

    status, notes = _check_haindy_installed()
    _add("haindy package", status, notes)

    status, notes = _check_api_key("openai")
    _add("OpenAI credentials", status, notes)

    status, notes = _check_api_key("anthropic")
    _add("Anthropic credentials", status, notes)

    status, notes = _check_api_key("vertex")
    _add("Google credentials", status, notes)

    status, notes = _check_codex_oauth()
    _add("OpenAI Codex (OAuth)", status, notes)

    if sys.platform == "darwin":
        status, notes = _check_macos_pynput()
        _add("pynput", status, notes)

        status, notes = _check_macos_mss()
        _add("mss", status, notes)

        status, notes = _check_macos_accessibility()
        _add("Accessibility permission", status, notes)

        status, notes = _check_macos_screen_recording()
        _add("Screen Recording permission", status, notes)

    if sys.platform == "linux":
        status, notes = _check_linux_which(
            "ffmpeg", "apt install ffmpeg / brew install ffmpeg"
        )
        _add("ffmpeg", status, notes)

        status, notes = _check_linux_which("xdotool")
        _add("xdotool", status, notes)

        status, notes = _check_linux_which("xclip")
        _add("xclip", status, notes)

        status, notes = _check_linux_uinput()
        _add("/dev/uinput access", status, notes)

        status, notes = _check_linux_display()
        _add("DISPLAY", status, notes)

    if include_android:
        if shutil.which("adb"):
            _add("adb", *_ok(), required=False)
        else:
            _add(
                "adb",
                *_missing("Install Android SDK platform-tools"),
                required=False,
            )
    else:
        _add("adb (Android)", *_na("pass --include-android to check"), required=False)

    if include_ios:
        if sys.platform == "darwin":
            if shutil.which("idb_companion"):
                _add("idb-companion", *_ok(), required=False)
            else:
                _add(
                    "idb-companion",
                    *_missing("brew install facebook/fb/idb-companion"),
                    required=False,
                )
            if importlib.util.find_spec("idb") is not None:
                _add("fb-idb Python package", *_ok(), required=False)
            else:
                _add(
                    "fb-idb Python package",
                    *_missing("pip install fb-idb"),
                    required=False,
                )
        else:
            _add("idb-companion", *_na("macOS only"), required=False)
            _add("fb-idb Python package", *_na("macOS only"), required=False)
    else:
        _add("idb-companion (iOS)", *_na("pass --include-ios to check"), required=False)

    _console.print(table)

    return 1 if any_missing else 0
