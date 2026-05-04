"""Phase 3 runner — install Codex CLI into a fresh OSWorld VM, then keep
the VM alive so the user can run `codex login` via noVNC.

NOTE: OSWorld's /run_bash_script endpoint is broken in this VM image
(`_append_event` is undefined). We use /execute (120s timeout per call)
chunked, with the slow `npm install -g @openai/codex` launched in the
background and polled for completion.

Polls control commands at /tmp/phase3_cmd:
    version   - re-check `codex --version`
    smoke     - run a tiny non-interactive Codex prompt (auth must already
                be done by the user via the noVNC terminal)
    close     - clean shutdown
"""

import json
import sys
import time
from pathlib import Path

import requests

CMD_FILE = Path("/tmp/phase3_cmd")
RESULT_FILE = Path("/tmp/phase3_result")
TASK_PATH = Path(
    "/home/fkeegan/src/osworld/evaluation_examples/examples/os/"
    "5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57.json"
)
SUDO_PASS = "password"  # OSWorld VM default; needed in Phase 4 for /dev/uinput

# Install Node into the user's home — no sudo, no apt, no NodeSource.
# ~/node-bin/bin will be on PATH and npm prefix, so `npm install -g` is also
# user-scoped.
NODE_VERSION = "v20.18.1"
NODE_TARBALL = f"node-{NODE_VERSION}-linux-x64.tar.xz"
NODE_URL = f"https://nodejs.org/dist/{NODE_VERSION}/{NODE_TARBALL}"

INSTALL_NODE = (
    f'set -e; '
    f'export PATH=$HOME/node-bin/bin:$PATH; '
    f'echo "[1/4] download node {NODE_VERSION}..."; '
    f'curl -fsSLo /tmp/node.tar.xz {NODE_URL}; '
    f'echo "[2/4] extract to ~/node-bin..."; '
    f'mkdir -p ~/node-bin && tar -xJf /tmp/node.tar.xz -C ~/node-bin --strip-components=1; '
    f'echo "[3/4] verify (PATH-aware so npm shebang works)..."; '
    f'node --version; '
    f'npm --version; '
    f'echo "[4/4] persist PATH in .bashrc and .profile (idempotent)..."; '
    f'grep -q node-bin ~/.bashrc 2>/dev/null || echo \'export PATH=$HOME/node-bin/bin:$PATH\' >> ~/.bashrc; '
    f'grep -q node-bin ~/.profile 2>/dev/null || echo \'export PATH=$HOME/node-bin/bin:$PATH\' >> ~/.profile; '
    f'echo done'
)

# Launch npm install in background; poll for /usr-local-style codex shim.
NPM_INSTALL_CODEX_BG = (
    f': > /tmp/codex_install.log; '
    f'nohup bash -c \'export PATH=$HOME/node-bin/bin:$PATH; '
    f'npm install -g @openai/codex >> /tmp/codex_install.log 2>&1; '
    f'echo "EXIT=$?" >> /tmp/codex_install.log\' </dev/null >/dev/null 2>&1 & '
    f'echo "started pid=$!"'
)


def write_result(text: str) -> None:
    RESULT_FILE.write_text(text + "\n")
    print(f"[result] {text}", flush=True)


def vm_exec(port: int, command: str, shell: bool = True, timeout: int = 30) -> dict:
    return requests.post(
        f"http://localhost:{port}/execute",
        json={"shell": shell, "command": command},
        timeout=timeout,
    ).json()


def show(label: str, r: dict) -> None:
    print(
        f"[{label}] rc={r.get('returncode')} status={r.get('status')}\n"
        f"  out: {(r.get('output') or '').strip()[:600]}\n"
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

    print("[install] step A: install Node into ~/node-bin (no sudo)...", flush=True)
    r = vm_exec(port, INSTALL_NODE, timeout=120)
    show("step A", r)
    if r.get("returncode") != 0:
        write_result(f"install FAILED at step A rc={r.get('returncode')}")
        env.close()
        return

    print("[install] step B: launching npm install -g @openai/codex in background...", flush=True)
    r = vm_exec(port, NPM_INSTALL_CODEX_BG, timeout=30)
    show("step B (launch)", r)

    print("[install] step C: polling for codex to appear (max ~5 min)...", flush=True)
    deadline = time.time() + 300
    codex_path = ""
    poll_cmd = (
        "export PATH=$HOME/node-bin/bin:$PATH; "
        "tail -1 /tmp/codex_install.log; which codex; codex --version 2>/dev/null"
    )
    while time.time() < deadline:
        time.sleep(8)
        check = vm_exec(port, poll_cmd, timeout=15)
        out = (check.get("output") or "").strip()
        if "/codex" in out and ("codex-cli" in out or "0." in out):
            codex_path = out
            break
        if "EXIT=" in out and "EXIT=0" not in out:
            print(f"[install] npm install exited non-zero: {out}", flush=True)
            break

    if not codex_path:
        tail = vm_exec(port, "tail -50 /tmp/codex_install.log", timeout=15).get("output", "")
        write_result(f"install FAILED — codex not found. log tail:\n{tail}")
        env.close()
        return

    write_result(f"codex installed: {codex_path} (port={port})")

    print(
        f"\n[ready] VM is alive on noVNC http://localhost:8006/ — open a terminal "
        f"and run `codex login` to authenticate.",
        flush=True,
    )
    print(
        f"[ready] Then write to {CMD_FILE}: version | smoke | close",
        flush=True,
    )

    while True:
        if CMD_FILE.exists():
            cmd = CMD_FILE.read_text().strip()
            CMD_FILE.unlink()
            print(f"[cmd] {cmd}", flush=True)

            if cmd == "version":
                v = vm_exec(
                    port,
                    "export PATH=$HOME/node-bin/bin:$PATH; codex --version 2>&1",
                ).get("output", "").strip()
                write_result(f"codex --version: {v}")
            elif cmd == "smoke":
                r = vm_exec(
                    port,
                    "export PATH=$HOME/node-bin/bin:$PATH; "
                    "codex exec 'reply with the single word ok' 2>&1 | tail -20",
                    timeout=120,
                )
                write_result(
                    f"smoke rc={r.get('returncode')} output={(r.get('output') or '').strip()[-400:]}"
                )
            elif cmd == "close":
                write_result("closing")
                break
            else:
                write_result(f"unknown command: {cmd}")

        time.sleep(0.5)

    env.close()
    print("[setup] closed", flush=True)


if __name__ == "__main__":
    main()
