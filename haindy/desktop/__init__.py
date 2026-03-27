"""Desktop automation helpers for OS-level control."""

from haindy.desktop.cache import CachedCoordinate, CoordinateCache
from haindy.desktop.controller import DesktopController
from haindy.desktop.driver import DesktopDriver
from haindy.desktop.execution_replay import (
    DriverActionError,
    normalize_driver_action,
    normalize_driver_actions,
    replay_driver_actions,
)
from haindy.desktop.resolution_manager import DisplayMode, ResolutionManager
from haindy.desktop.screen_capture import ScreenCapture
from haindy.desktop.screen_recorder import ScreenRecorder, ScreenRecorderError
from haindy.desktop.virtual_input import VirtualInput

__all__ = [
    "CachedCoordinate",
    "CoordinateCache",
    "DesktopController",
    "DesktopDriver",
    "DisplayMode",
    "DriverActionError",
    "normalize_driver_action",
    "normalize_driver_actions",
    "replay_driver_actions",
    "ResolutionManager",
    "ScreenCapture",
    "ScreenRecorder",
    "ScreenRecorderError",
    "VirtualInput",
]
