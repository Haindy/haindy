"""Desktop automation helpers for OS-level control."""

from src.desktop.cache import CoordinateCache, CachedCoordinate
from src.desktop.controller import DesktopController
from src.desktop.driver import DesktopDriver
from src.desktop.execution_replay import (
    DriverActionError,
    normalize_driver_action,
    normalize_driver_actions,
    replay_driver_actions,
)
from src.desktop.resolution_manager import DisplayMode, ResolutionManager
from src.desktop.screen_capture import ScreenCapture
from src.desktop.screen_recorder import ScreenRecorder, ScreenRecorderError
from src.desktop.virtual_input import VirtualInput

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
