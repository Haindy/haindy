"""Phase 5 runner — full end-to-end OSWorld trash recovery task.

Boots fresh OSWorld VM, installs HAINDY, uploads creds, invokes Codex on the
host with the wrapper prompt (Codex drives the VM through vm-haindy), then
calls env.evaluate() to score. Captures all artifacts under
~/osworld-runs/<run_id>/.

Phase 5 deliberately does this synchronously and in one script so we can see
the whole flow when iterating.
"""

import base64
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---- paths ---------------------------------------------------------------

OSWORLD_REPO = Path("/home/fkeegan/src/osworld")
TASK_PATH = (
    OSWORLD_REPO
    / "evaluation_examples/examples/os/5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57.json"
)
SCRIPTS_DIR = Path("/home/fkeegan/src/haindy/haindy/docs/research/scripts")
INSTALL_SH = SCRIPTS_DIR / "install_haindy_in_vm.sh"
PROMPT_PATH = SCRIPTS_DIR / "phase5_codex_prompt.md"
HOST_KEYS_ENV = Path.home() / ".osworld-secrets" / "keys.env"
RUNS_ROOT = Path.home() / "osworld-runs"

CODEX_TIMEOUT_S = 600  # 10 min hard wall-clock for Codex
INSTALL_TIMEOUT_S = 600  # 10 min for haindy install in VM


def vm_exec(port: int, command: str, timeout: int = 30) -> dict:
    return requests.post(
        f"http://localhost:{port}/execute",
        json={"shell": True, "command": command},
        timeout=timeout,
    ).json()


def upload_install_script_to_vm(port: int) -> None:
    b64 = base64.b64encode(INSTALL_SH.read_bytes()).decode()
    cmd = (
        f"echo '{b64}' | base64 -d > /tmp/install_haindy_in_vm.sh && "
        "chmod +x /tmp/install_haindy_in_vm.sh && "
        "ls -la /tmp/install_haindy_in_vm.sh"
    )
    r = vm_exec(port, cmd, timeout=15)
    if r.get("returncode") != 0:
        raise RuntimeError(f"failed to upload install script: {r}")


def launch_install_in_vm(port: int) -> None:
    cmd = (
        ": > /tmp/haindy_install.log; "
        "nohup bash -c 'bash /tmp/install_haindy_in_vm.sh "
        ">> /tmp/haindy_install.log 2>&1; echo \"EXIT=$?\" "
        ">> /tmp/haindy_install.log' </dev/null >/dev/null 2>&1 & "
        "echo started pid=$!"
    )
    r = vm_exec(port, cmd, timeout=15)
    if r.get("returncode") != 0:
        raise RuntimeError(f"failed to launch install: {r}")


def poll_install(port: int, deadline: float) -> str:
    while time.time() < deadline:
        time.sleep(8)
        check = vm_exec(
            port,
            "tail -1 /tmp/haindy_install.log; "
            "test -x $HOME/.local/bin/haindy && echo HAINDY_OK",
            timeout=15,
        )
        out = (check.get("output") or "").strip()
        if "HAINDY_OK" in out and "EXIT=0" in out:
            return out
        if "EXIT=" in out and "EXIT=0" not in out:
            tail = vm_exec(port, "tail -50 /tmp/haindy_install.log", timeout=15)
            raise RuntimeError(
                f"install exited non-zero. log tail:\n{tail.get('output')}"
            )
    tail = vm_exec(port, "tail -50 /tmp/haindy_install.log", timeout=15)
    raise RuntimeError(f"install timeout. log tail:\n{tail.get('output')}")


def upload_keys(port: int) -> None:
    """Read host keys.env, shlex-quote values, write keys.sh in VM."""
    quoted_lines = []
    for line in HOST_KEYS_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        quoted_lines.append(f"export {key.strip()}={shlex.quote(val)}")
    safe = ("\n".join(quoted_lines) + "\n").encode()
    b64 = base64.b64encode(safe).decode()
    cmd = (
        "mkdir -p ~/.osworld-secrets && "
        f"echo '{b64}' | base64 -d > ~/.osworld-secrets/keys.sh && "
        "chmod 600 ~/.osworld-secrets/keys.sh"
    )
    r = vm_exec(port, cmd, timeout=15)
    if r.get("returncode") != 0:
        raise RuntimeError(f"failed to upload keys: {r}")


def run_codex(prompt: str, log_path: Path) -> tuple[int, str, float]:
    """Invoke codex on the host with our wrapper prompt. Returns
    (exit_code, captured_output, wall_seconds)."""
    start = time.time()
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-s",
        "danger-full-access",
        prompt,
    ]
    print(f"[codex] launching with timeout {CODEX_TIMEOUT_S}s...", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_S,
        )
        out = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (
            "\n--- STDERR ---\n" + e.stderr if e.stderr else ""
        ) + f"\n--- TIMEOUT after {CODEX_TIMEOUT_S}s ---"
        rc = -1
        timed_out = True
    elapsed = time.time() - start
    log_path.write_text(out)
    print(
        f"[codex] {'TIMED OUT' if timed_out else f'exited rc={rc}'} "
        f"after {elapsed:.1f}s",
        flush=True,
    )
    return rc, out, elapsed


def main() -> None:
    if not HOST_KEYS_ENV.exists():
        sys.exit(f"missing {HOST_KEYS_ENV} — see Phase 4 doc")
    if not INSTALL_SH.exists():
        sys.exit(f"missing {INSTALL_SH}")
    if not PROMPT_PATH.exists():
        sys.exit(f"missing {PROMPT_PATH}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"{run_id}-trash"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] artifacts: {run_dir}", flush=True)

    sys.path.insert(0, str(OSWORLD_REPO))
    from desktop_env.desktop_env import DesktopEnv

    task = json.loads(TASK_PATH.read_text())
    print(f"[run] task: {task['id']} — {task['instruction']}", flush=True)

    print("[run] booting fresh OSWorld VM...", flush=True)
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
    print(f"[run] VM ready on port {port}", flush=True)
    Path("/tmp/vm-haindy-port").write_text(str(port))

    try:
        # Sanity: initial evaluate should be 0 (file is in trash)
        initial = env.evaluate()
        print(f"[run] initial evaluate(): {initial}", flush=True)
        if initial != 0.0:
            raise RuntimeError(f"unexpected initial score {initial}; expected 0.0")

        # Install HAINDY
        print("[run] uploading install script...", flush=True)
        upload_install_script_to_vm(port)
        print("[run] launching install (background, polling)...", flush=True)
        launch_install_in_vm(port)
        deadline = time.time() + INSTALL_TIMEOUT_S
        poll_install(port, deadline)
        print("[run] haindy installed", flush=True)

        # Upload credentials
        print("[run] uploading keys...", flush=True)
        upload_keys(port)
        print("[run] keys uploaded", flush=True)

        # Quick wrapper sanity check
        from subprocess import run as _run
        wrapper_check = _run(
            ["vm-haindy", "session", "new", "--desktop"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "VM_HAINDY_PORT": str(port)},
        )
        if wrapper_check.returncode != 0:
            raise RuntimeError(
                f"wrapper sanity check failed: {wrapper_check.stdout} "
                f"{wrapper_check.stderr}"
            )
        sanity = json.loads(wrapper_check.stdout.strip().splitlines()[-1])
        sid = sanity["session_id"]
        print(f"[run] wrapper sanity: session {sid} ok, closing it", flush=True)
        _run(
            ["vm-haindy", "session", "close", "--session", sid],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "VM_HAINDY_PORT": str(port)},
        )

        # Run codex
        prompt = PROMPT_PATH.read_text()
        rc, codex_out, codex_secs = run_codex(prompt, run_dir / "codex.log")

        # Final evaluate
        print("[run] running env.evaluate()...", flush=True)
        score = env.evaluate()
        print(f"[run] FINAL SCORE: {score}", flush=True)

        # Save summary
        summary = {
            "run_id": run_id,
            "task_id": task["id"],
            "initial_score": initial,
            "final_score": score,
            "codex_rc": rc,
            "codex_wall_seconds": codex_secs,
            "codex_timed_out": rc == -1,
            "vm_port": port,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[run] summary:\n{json.dumps(summary, indent=2)}", flush=True)

    finally:
        Path("/tmp/vm-haindy-port").unlink(missing_ok=True)
        try:
            env.close()
            print("[run] env closed", flush=True)
        except Exception as e:
            print(f"[run] env close error: {e}", flush=True)


if __name__ == "__main__":
    main()
