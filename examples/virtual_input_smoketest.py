"""Manual smoke test for the VirtualInput mouse device."""

from __future__ import annotations

import argparse
import asyncio
import math
import time
import sys

from src.desktop.resolution_manager import ResolutionManager
from src.desktop.virtual_input import VirtualInput


async def main() -> int:
    parser = argparse.ArgumentParser(description="Spin the virtual pointer in a circle to verify uinput.")
    parser.add_argument("--duration", type=float, default=20.0, help="How long to run in seconds (default: 20s).")
    parser.add_argument("--steps-per-loop", type=int, default=90, help="Points per circle (default: 90).")
    parser.add_argument("--step-delay", type=float, default=0.02, help="Delay between moves in seconds (default: 0.02).")
    args = parser.parse_args()

    manager = ResolutionManager(enable_switch=False)
    mode = manager.detect_current_mode()
    viewport = (mode.width, mode.height)

    try:
        virtual_input = VirtualInput(viewport=viewport)
    except Exception as exc:  # pragma: no cover - manual smoke helper
        print(f"Failed to create virtual input device: {exc}")
        return 1

    center_x, center_y = mode.width // 2, mode.height // 2
    radius = min(mode.width, mode.height) // 6
    steps_per_loop = max(args.steps_per_loop, 12)
    step_delay = max(args.step_delay, 0.001)

    print(f"Viewport detected: {mode.width}x{mode.height}")
    print(f"Moving pointer in a visible circle near the center for ~{args.duration:.1f}s. Press Ctrl+C to stop early.")

    end_time = time.monotonic() + max(args.duration, 1.0)
    loops = 0
    while time.monotonic() < end_time:
        for step in range(steps_per_loop):
            angle = (2 * math.pi * step) / steps_per_loop
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            await virtual_input.move(x, y)
            await asyncio.sleep(step_delay)
        loops += 1
        await asyncio.sleep(0.25)

    print(f"Done after {loops} loops.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(1)
