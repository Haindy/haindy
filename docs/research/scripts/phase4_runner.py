"""Phase 4 runner — install HAINDY into a fresh OSWorld VM and validate it
can run, before any credential setup or session test.

Steps:
    A. boot fresh VM
    B. pip install --user haindy (background + poll, since pip+deps may exceed 120s)
    C. chmod a+rw /dev/uinput (needs sudo password)
    D. haindy doctor (capture output — will likely complain about missing
       creds, but should otherwise be green)

Then keeps the VM alive for follow-up commands via /tmp/phase4_cmd:
    doctor    - re-run `haindy doctor`
    probe     - check key system bins (ffmpeg, xrandr, xdpyinfo, scrot)
    session   - haindy session new --desktop ; haindy session status ; haindy session close
                (will fail without creds but tells us how it fails)
    close     - shutdown
"""

import json
import sys
import time
from pathlib import Path

import requests

CMD_FILE = Path("/tmp/phase4_cmd")
RESULT_FILE = Path("/tmp/phase4_result")
TASK_PATH = Path(
    "/home/fkeegan/src/osworld/evaluation_examples/examples/os/"
    "5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57.json"
)
SUDO_PASS = "password"

HAINDY_INSTALL_BG = (
    f': > /tmp/haindy_install.log; '
    f'nohup bash -c \''
    f'set -e; '
    f'echo "[1/4] apt deps (build tools for evdev + xdotool/xclip for haindy)..."; '
    f'echo "{SUDO_PASS}" | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential python3.10-dev xdotool xclip 2>&1 | tail -8; '
    f'echo "[2/4] pip upgrade..."; '
    f'export PATH=$HOME/.local/bin:$PATH; '
    f'pip install --user --upgrade pip 2>&1 | tail -5; '
    f'echo "[3/4] pip install haindy..."; '
    f'pip install --user haindy 2>&1 | tail -20; '
    f'echo "[4/4] verify..."; '
    f'which haindy && ls -la "$(which haindy)"; '
    f'echo "EXIT=0"\' '
    f'>> /tmp/haindy_install.log 2>&1 </dev/null & '
    f'echo "started pid=$!"'
)

CHMOD_UINPUT = (
    f'echo "{SUDO_PASS}" | sudo -S chmod a+rw /dev/uinput && '
    f'ls -l /dev/uinput'
)

# Standard PATH-setting prefix for in-VM commands
PATH_PREFIX = "export PATH=$HOME/.local/bin:$PATH; export DISPLAY=:0; "


def write_result(text: str) -> None:
    RESULT_FILE.write_text(text + "\n")
    print(f"[result] {text}", flush=True)


def vm_exec(port: int, command: str, shell: bool = True, timeout: int = 30) -> dict:
    return requests.post(
        f"http://localhost:{port}/execute",
        json={"shell": shell, "command": command},
        timeout=timeout,
    ).json()


def show(label: str, r: dict, body_chars: int = 600) -> None:
    print(
        f"[{label}] rc={r.get('returncode')} status={r.get('status')}\n"
        f"  out: {(r.get('output') or '').strip()[:body_chars]}\n"
        f"  err: {(r.get('error') or '').strip()[:300]}",
        flush=True,
    )


def main() -> None:
    CMD_FILE.unlink(missing_ok=True)
    RESULT_FILE.unlink(missing_ok=True)

    sys.path.insert(0, "/home/fkeegan/src/osworld")
    from desktop_env.desktop_env import DesktopEnv

    task = json.loads(TASK_PATH.read_text())
    print(f"[setup] booting fresh OSWorld VM...", flush=True)
    env = DesktopEnv(
        provider_name="docker",
        os_type="Ubuntu",
        action_space="pyautogui",
        headless=True,
    )
    env.reset(task_config=task)
    port = (
        getattr(env, "server_port", None)
        or getattr(getattr(env, "controller", None), "server_port", None)
        or 5000
    )
    print(f"[setup] VM ready on port {port}", flush=True)
    Path("/tmp/vm-haindy-port").write_text(str(port))

    print("[install] step B: launching apt-deps + pip upgrade + haindy install in background...", flush=True)
    r = vm_exec(port, HAINDY_INSTALL_BG, timeout=30)
    show("step B (launch)", r)

    print("[install] step C: polling for haindy to appear (max ~10 min)...", flush=True)
    deadline = time.time() + 600
    haindy_path = ""
    poll_cmd = (
        PATH_PREFIX
        + "tail -1 /tmp/haindy_install.log; which haindy 2>/dev/null"
    )
    while time.time() < deadline:
        time.sleep(8)
        check = vm_exec(port, poll_cmd, timeout=15)
        out = (check.get("output") or "").strip()
        if "/haindy" in out and "EXIT=0" in out:
            haindy_path = out
            break
        if "EXIT=" in out and "EXIT=0" not in out:
            print(f"[install] pip install exited non-zero: {out}", flush=True)
            break

    if not haindy_path:
        tail = vm_exec(port, "tail -50 /tmp/haindy_install.log", timeout=15).get(
            "output", ""
        )
        write_result(f"install FAILED — haindy not found. log tail:\n{tail}")
        env.close()
        return

    print(f"[install] haindy installed:\n{haindy_path}", flush=True)

    print("[install] step D: chmod /dev/uinput...", flush=True)
    r = vm_exec(port, CHMOD_UINPUT, timeout=15)
    show("chmod uinput", r)

    print("[install] step E: haindy doctor...", flush=True)
    r = vm_exec(port, PATH_PREFIX + "haindy doctor 2>&1", timeout=60)
    show("haindy doctor", r, body_chars=2000)

    write_result(f"haindy installed and doctor-checked (port={port})")

    print(
        f"\n[ready] VM is alive. Wrote port to /tmp/vm-haindy-port. "
        f"Write to {CMD_FILE}: doctor | probe | session | close",
        flush=True,
    )

    while True:
        if CMD_FILE.exists():
            cmd = CMD_FILE.read_text().strip()
            CMD_FILE.unlink()
            print(f"[cmd] {cmd}", flush=True)

            if cmd == "doctor":
                r = vm_exec(port, PATH_PREFIX + "haindy doctor 2>&1", timeout=60)
                show("doctor", r, body_chars=2000)
                write_result(f"doctor rc={r.get('returncode')}")
            elif cmd == "probe":
                r = vm_exec(
                    port,
                    "for b in ffmpeg xrandr xdpyinfo scrot xdotool wmctrl "
                    "fluxbox python3-tk; do printf '%-15s ' \"$b\"; "
                    "command -v $b 2>/dev/null || echo MISSING; done",
                    timeout=15,
                )
                show("probe", r, body_chars=1500)
                write_result(f"probe rc={r.get('returncode')}")
            elif cmd == "session":
                r = vm_exec(
                    port,
                    PATH_PREFIX
                    + "haindy session new --desktop 2>&1 | head -40",
                    timeout=60,
                )
                show("session new", r, body_chars=2000)
                write_result(
                    f"session new rc={r.get('returncode')} "
                    f"output={(r.get('output') or '').strip()[-400:]}"
                )
            elif cmd == "close":
                write_result("closing")
                break
            else:
                write_result(f"unknown command: {cmd}")

        time.sleep(0.5)

    Path("/tmp/vm-haindy-port").unlink(missing_ok=True)
    env.close()
    print("[setup] closed", flush=True)


if __name__ == "__main__":
    main()
