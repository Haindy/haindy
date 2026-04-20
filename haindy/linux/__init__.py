"""Desktop automation helpers for OS-level control."""

from haindy.linux.controller import DesktopController
from haindy.linux.driver import DesktopDriver
from haindy.linux.resolution_manager import DisplayMode, ResolutionManager
from haindy.linux.screen_capture import ScreenCapture
from haindy.linux.screen_recorder import ScreenRecorder, ScreenRecorderError
from haindy.linux.virtual_input import VirtualInput

__all__ = [
    "DesktopController",
    "DesktopDriver",
    "DisplayMode",
    "ResolutionManager",
    "ScreenCapture",
    "ScreenRecorder",
    "ScreenRecorderError",
    "VirtualInput",
]
