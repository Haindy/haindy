"""Phase 3 recon — boot a fresh OSWorld VM and probe what we need to know
to build the HAINDY install snippet.

Reports: Python version, glibc, /dev/uinput presence + permissions,
X session details (DISPLAY, XAUTHORITY), user account groups, package
manager state, and what HAINDY's runtime would need to acquire.
Closes the env when done so we don't tie up the VM.
"""

import json
import sys
from pathlib import Path

import requests

TASK_PATH = Path(
    "/home/fkeegan/src/osworld/evaluation_examples/examples/os/"
    "5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57.json"
)


def vm_run(port: int, command: str, shell: bool = True, timeout: int = 60) -> dict:
    r = requests.post(
        f"http://localhost:{port}/execute",
        json={"shell": shell, "command": command},
        timeout=timeout,
    )
    return r.json()


def show(label: str, result: dict) -> None:
    rc = result.get("returncode")
    out = (result.get("output") or "").rstrip()
    err = (result.get("error") or "").rstrip()
    print(f"\n=== {label} (rc={rc}) ===")
    if out:
        print(out)
    if err:
        print(f"[stderr] {err}")


def main() -> None:
    sys.path.insert(0, "/home/fkeegan/src/osworld")
    from desktop_env.desktop_env import DesktopEnv

    task = json.loads(TASK_PATH.read_text())
    print(f"[setup] Booting fresh OSWorld VM for recon...", flush=True)
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

    probes: list[tuple[str, str]] = [
        ("OS release", "cat /etc/os-release | head -5"),
        ("Kernel", "uname -a"),
        ("Glibc", "ldd --version | head -1"),
        ("System Python", "which python3 && python3 --version"),
        ("Pip available", "which pip3 || which pip; (pip3 --version 2>&1) || (pip --version 2>&1)"),
        ("Active user", "id -un && id"),
        ("Whoami / groups", "whoami; groups"),
        ("X session env", "echo DISPLAY=$DISPLAY; echo XAUTHORITY=$XAUTHORITY; ls -l $XAUTHORITY 2>/dev/null"),
        ("xdpyinfo (X reachability)", "DISPLAY=:0 xdpyinfo 2>&1 | head -8"),
        ("/dev/uinput", "ls -l /dev/uinput 2>&1; lsmod 2>/dev/null | grep -i uinput"),
        ("Display managers", "ps -eo user,pid,comm | grep -iE 'xorg|gnome|gdm|xfce' | head -10"),
        ("Network out (PyPI)", "curl -sSI https://pypi.org/simple/ -o /dev/null -w 'HTTP %{http_code}\\n'"),
        ("Network out (Google)", "curl -sSI https://generativelanguage.googleapis.com -o /dev/null -w 'HTTP %{http_code}\\n'"),
        ("Free disk", "df -h / | tail -1"),
        ("Free memory", "free -h | head -2"),
    ]

    for label, cmd in probes:
        show(label, vm_run(port, cmd))

    env.close()
    print("\n[setup] Closed.", flush=True)


if __name__ == "__main__":
    main()
