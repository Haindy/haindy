#!/usr/bin/env bash
# install_haindy_in_vm.sh — install HAINDY into a freshly-reset OSWorld VM.
#
# Synchronous install script intended to run inside the OSWorld guest. Total
# runtime ~5-10 min (apt + pip + numpy/cryptography/evdev compile).
#
# Caller responsibilities:
#   * Upload this script into the VM (e.g. via /setup/upload or by piping
#     contents through /execute), then invoke it.
#   * Wrap invocation with the background+poll pattern because OSWorld's
#     /execute endpoint has a hardcoded 120s subprocess timeout. /run_bash_script
#     is broken in the current VM image (`_append_event` undefined).
#
# Inputs:
#   SUDO_PASS env var, defaults to OSWorld's stock "password".
#
# Side effects in the VM (under user `user`):
#   * apt installs: build-essential, python3.10-dev, xdotool, xclip
#   * pip user-installs: haindy and its deps (~250 MB site-packages)
#   * /dev/uinput becomes world-rw (chmod a+rw)
#   * pip upgraded to latest (was 22.0.2 stock; needs >=24 for evdev metadata)
#
# Verification:
#   `which haindy` returns a path under ~/.local/bin
#   exit code 0
#
# Credentials: NOT handled here. Caller must upload `~/.osworld-secrets/keys.sh`
# (shell-safe `export KEY=value` lines) and source it in subsequent commands.
# See phase4 runner / docs for the keys.env -> keys.sh shlex.quote pipeline.

set -euo pipefail

SUDO_PASS="${SUDO_PASS:-password}"

echo "[install_haindy] [1/5] apt deps (build tools + xdotool/xclip)..."
echo "${SUDO_PASS}" | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    build-essential python3.10-dev xdotool xclip 2>&1 | tail -8

echo "[install_haindy] [2/5] chmod /dev/uinput..."
echo "${SUDO_PASS}" | sudo -S chmod a+rw /dev/uinput
ls -l /dev/uinput

echo "[install_haindy] [3/5] upgrade pip (stock 22.0.2 has metadata bug)..."
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade pip 2>&1 | tail -3

echo "[install_haindy] [4/5] pip install --user haindy..."
pip install --user haindy 2>&1 | tail -5

echo "[install_haindy] [5/5] verify..."
which haindy
ls -la "$(which haindy)"

echo "[install_haindy] OK"
