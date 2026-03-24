# macOS Desktop Automation Backend

## Context

haindy currently supports Linux/X11 as its only desktop automation backend, using evdev/uinput for input injection, ffmpeg/ImageMagick for screenshots, and xrandr for resolution management. The `desktop` automation backend hard-codes Linux assumptions throughout `src/desktop/`.

This plan adds macOS as a first-class desktop backend at feature parity with Linux. The approach keeps the user-visible backend name `desktop` and adds OS-level dispatch at driver instantiation time, so nothing changes for users — the right driver is selected automatically based on `sys.platform`. Screen resolution switching is intentionally omitted for macOS; native (Retina) resolution is used as-is.

---

## Tool Stack (macOS)

| Concern | Tool | Why |
|---|---|---|
| Mouse/keyboard input | `pynput` (pip) | Native macOS Quartz CGEvent under the hood; clean high-level API; pip-installable; actively maintained (v1.8.1, Mar 2025) |
| Screenshots | `mss` (pip) | 15-30x faster than `screencapture` CLI; returns bytes directly (no disk I/O); exposes both logical (point) and pixel dimensions; handles Retina automatically |
| Clipboard | `pbcopy` / `pbpaste` (built-in) | Same pattern as Linux `xclip`; zero extra dependencies |
| Display geometry | `mss` monitor metadata | `mss.monitors[N]` gives logical dimensions; screenshot pixel size / logical size = scale factor |

**Permissions required at runtime (documented in RUNBOOK.md):**
- **Accessibility** — required for keyboard injection; grant in System Settings -> Privacy & Security -> Accessibility
- **Screen Recording** — required for `mss` screenshot capture; grant in System Settings -> Privacy & Security -> Screen Recording

---

## Retina Coordinate Scaling

macOS Retina displays present two coordinate spaces:

- **Logical points** — what pynput and macOS APIs accept (e.g., 1280x800 on a 13" MBP)
- **Physical pixels** — what `mss` screenshots produce (e.g., 2560x1600 = 2x in each axis)

The computer-use AI model receives screenshots at full native (pixel) resolution and returns click coordinates in that same space. Before injecting with pynput, coordinates must be divided by the scale factor.

Detection flow (on `MacOSDriver.start()`):
1. Call `mss.mss().monitors[1]` to get logical `width` x `height`
2. Capture one screenshot; read PNG header dimensions for pixel `width` x `height`
3. `scale_x = pixel_width / logical_width` (typically 2.0 on Retina, 1.0 on non-Retina)
4. Store `scale_x`, `scale_y` on the input handler
5. All coordinate injection: `pynput_x = int(screenshot_x / scale_x)`

`get_viewport_size()` returns **pixel** dimensions (matching what the model sees in the screenshot), consistent with how `IOSDriver` handles Retina on iOS.

---

## New Files

```
src/macos/
    __init__.py               exports: MacOSDriver, MacOSController
    driver.py                 MacOSDriver(AutomationDriver)
    controller.py             MacOSController  (mirrors DesktopController)
    input_handler.py          MacOSInputHandler  (mirrors VirtualInput)
    screen_capture.py         MacOSScreenCapture  (mirrors ScreenCapture)

tests/
    test_macos_driver.py      unit tests with stub input handler
```

---

## Modified Files

### `src/main.py`

In `_create_coordinator_stack()`, replace the `else` branch with platform detection:

```python
else:
    import sys
    if sys.platform == "darwin":
        from haindy.macos.controller import MacOSController
        automation_controller = MacOSController()
    else:
        automation_controller = DesktopController()
```

### `src/runtime/environment.py`

1. Add `"macos"` alias to `_RUNTIME_ENV_ALIASES` -> `"desktop"`
2. Fix `openai_computer_environment` to return `"mac"` on darwin
3. Fix `coordinate_cache_attribute` for macOS -> `"macos_coordinate_cache_path"`

### `src/config/settings.py`

Add macOS-specific fields:
- `macos_screenshot_dir`, `macos_coordinate_cache_path`
- `macos_keyboard_layout`, `macos_keyboard_key_delay_ms`
- `macos_clipboard_timeout_seconds`, `macos_clipboard_hold_seconds`

### `src/config/settings_file.py`

Add JSON mappings under `"macos"` section.

### `pyproject.toml`

Add darwin-conditional deps:
```toml
"pynput>=1.7.7; sys_platform == 'darwin'",
"mss>=9.0.2; sys_platform == 'darwin'",
```

---

## Verification

**Stop before executing any live session test.** Confirm with the user before running any command that injects real input events.

Automated tests:
```bash
.venv/bin/pytest tests/test_macos_driver.py -v
```

No new slow tests; all macOS tests use stubs and run without a display or pynput installed.
