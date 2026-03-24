# iOS Support via idb

## Context

HAINDY supports Android via ADB as its mobile automation backend. This plan adds iOS support using the same principle: the CU loop takes screenshots, the AI decides on coordinates and actions, and the driver executes them. No programmatic UI introspection — pure visual computer-use.

The implementation mirrors the Android ADB architecture exactly, using `idb` (Meta's iOS Development Bridge) as the iOS equivalent of ADB. It supports both real devices and Xcode simulators via UDID.

## Approach

Use `idb` CLI tool as a subprocess driver — the same pattern as `ADBClient` wrapping `adb`. New backend name: `mobile_ios`. New CLI flag: `--ios`. The three-layer structure mirrors Android: `IDBClient` (subprocess) → `IOSDriver` (AutomationDriver impl) → `IOSController` (async wrapper).

### idb command mapping

| Operation | Command |
|-----------|---------|
| Screenshot | `idb screenshot /tmp/screen.png --udid UDID` → read bytes |
| Tap | `idb ui tap X Y --udid UDID` |
| Swipe/drag | `idb ui swipe X1 Y1 X2 Y2 --duration SECS --udid UDID` |
| Type text | `idb type TEXT --udid UDID` |
| Key press | `idb key-sequence CODE1 [CODE2...] --udid UDID` |
| List devices | `idb list-targets --json` |
| Describe device | `idb describe --udid UDID` |
| Launch app | `idb launch BUNDLE_ID --udid UDID` |
| Terminate app | `idb terminate BUNDLE_ID --udid UDID` |
| Open URL | `idb open URL --udid UDID` |

### HID keycode table

| Key | Code |
|-----|------|
| a–z | 4–29 |
| 1–9, 0 | 30–38, 39 |
| Return/Enter | 40 |
| Escape | 41 |
| Backspace/Delete | 42 |
| Tab | 43 |
| Space | 44 |
| Up Arrow | 82 |
| Down Arrow | 81 |
| Left Arrow | 80 |
| Right Arrow | 79 |
| Home | 74 |
| End | 77 |
| Page Up | 75 |
| Page Down | 78 |
| Left Ctrl | 224 |
| Left Shift | 225 |
| Left Alt/Option | 226 |
| Left Cmd | 227 |

### Prerequisites (macOS only)

- `brew install idb-companion`
- `pip install fb-idb`
- Developer mode enabled on real devices (Settings > Privacy & Security > Developer Mode, iOS 16+)
- Device paired with Mac (trust prompt on first connect)

## Files Created

| File | Purpose |
|------|---------|
| `src/mobile/idb_client.py` | IDB subprocess client (mirrors `adb_client.py`) |
| `src/mobile/ios_driver.py` | `IOSDriver` implementing `AutomationDriver` |
| `src/mobile/ios_controller.py` | `IOSController` async wrapper |
| `tests/test_ios_driver.py` | Unit tests (no device required) |

## Files Modified

| File | Change |
|------|--------|
| `src/mobile/__init__.py` | Export `IOSController`, `IOSDriver`, `IDBClient` |
| `src/runtime/environment.py` | Add `mobile_ios` backend + aliases |
| `src/config/settings.py` | Add iOS settings fields |
| `src/config/settings_file.py` | Add `_JSON_TO_FIELD` mappings |
| `.env.example` | Add iOS env vars |
| `src/main.py` | Add `--ios` CLI flag |
| `docs/RUNBOOK.md` | Add iOS prerequisites section |
| `pyproject.toml` | Add `fb-idb` optional dependency |

## New Settings

| Field | Env var | Default |
|-------|---------|---------|
| `ios_screenshot_dir` | `HAINDY_IOS_SCREENSHOT_DIR` | `data/screenshots/ios` |
| `ios_coordinate_cache_path` | `HAINDY_IOS_COORDINATE_CACHE_PATH` | `data/ios_cache/coordinates.json` |
| `ios_default_device_udid` | `HAINDY_IOS_DEFAULT_DEVICE_UDID` | `""` |
| `ios_idb_timeout_seconds` | `HAINDY_IOS_IDB_TIMEOUT_SECONDS` | `15.0` |

## Coordinate Mapping

iOS screenshots are taken at physical pixel resolution (Retina/high-DPI). `idb ui tap` uses logical points. The `_map_point_to_device()` method scales screenshot-space coordinates to device point-space using the same math as the Android driver, with `get_viewport_size()` returning logical dimensions from `idb describe`.

## Verification

1. `idb list-targets` shows connected device/simulator
2. `haindy --ios --plan <plan> --context <context>` initializes without error
3. `HAINDY_AUTOMATION_BACKEND=mobile_ios haindy --plan <plan> --context <context>` also works
4. `pytest tests/test_ios_driver.py` passes (no device required)
5. `ruff check src/mobile/ios_driver.py src/mobile/idb_client.py src/mobile/ios_controller.py`
6. `mypy haindy/mobile/ios_driver.py`
