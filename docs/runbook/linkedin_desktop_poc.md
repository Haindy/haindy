# LinkedIn Desktop POC Runbook

## Prerequisites
- Ubuntu desktop with a single active display.
- `/dev/uinput` available to the current user (load `uinput` module; adjust udev/group perms as needed).
- `ffmpeg` and `xrandr` installed for screenshots and optional resolution switching.
- Firefox window already open and signed in to LinkedIn; leave it visible/normal (not minimized).
- `OPENAI_API_KEY` configured with access to `gpt-5.1`, `gpt-5.1-mini`, and `computer-use-preview`.

## Suggested Resolution
- Preferred: 1920x1080. Enable temporary downshift by setting `HAINDY_DESKTOP_RES_SWITCH=1` if you want the tool to change resolution automatically (single-display only). Otherwise, keep your current mode.

## Running the POC
1) Install deps: `pip install -e ".[dev]"` (ensures `evdev` and updated `openai` are present).
2) Ensure uinput: `sudo modprobe uinput` and confirm `/dev/uinput` is writable by your user.
3) Confirm single monitor: `xrandr --listmonitors` should show one display.
4) Open Firefox, sign in to LinkedIn, leave the window visible.
5) Execute the scenario (desktop mode is default):  
   `python -m src.main -j test_scenarios/linkedin_desktop_poc.json --timeout 900`

## What to Expect
- The agent will skip Playwright and operate via desktop screenshots and uinput.
- It will try cached coordinates from `data/desktop_cache/linkedin.json` when available; otherwise it will invoke the computer-use tool.
- Screenshots land in `debug_screenshots/<run_id>/` and `debug_screenshots/desktop/`.
- Coordinate cache is append-only at `data/desktop_cache/linkedin.json`.

## Troubleshooting
- **No input happening:** verify `/dev/uinput` permissions and that the process can create uinput devices.
- **Screenshots empty/black:** ensure DISPLAY is :0 and ffmpeg is installed; window must be visible on the primary display.
- **Resolution mismatch:** disable downshift (`HAINDY_DESKTOP_RES_SWITCH=0`) or manually set 1920x1080.
- **Model/tool errors:** confirm API key access to `computer-use-preview` and `gpt-5.1` models.
