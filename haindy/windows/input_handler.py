"""Windows input handler stub.

Real implementation (pynput-based, mirroring macOS input handler with the
Windows key mapped to ``Key.cmd``) lands in Milestone 2.
"""

from __future__ import annotations


class WindowsInputHandler:
    """Windows keyboard/mouse input handler (stub)."""

    def __init__(
        self,
        logical_size: tuple[int, int],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        keyboard_layout: str = "us",
        key_delay_ms: int = 12,
    ) -> None:
        self.logical_size = logical_size
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.keyboard_layout = keyboard_layout
        self.key_delay_ms = key_delay_ms
        raise NotImplementedError("Windows input handler lands in Milestone 2")
