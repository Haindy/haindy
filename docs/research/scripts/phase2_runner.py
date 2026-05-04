"""Phase 2 manual harness for the trash recovery task.

Holds an OSWorld DesktopEnv alive and accepts commands via a control file.
Lets us evaluate, reset, and close on demand while doing manual VM
manipulations (docker exec, VNC) in between.

Commands (write to /tmp/phase2_cmd):
    evaluate  - run env.evaluate() and write score to /tmp/phase2_result
    reset     - re-run the task config (re-download + re-trash) then evaluate
    close     - clean shutdown
"""

import json
import sys
import time
from pathlib import Path

CMD_FILE = Path("/tmp/phase2_cmd")
RESULT_FILE = Path("/tmp/phase2_result")
TASK_PATH = Path(
    "/home/fkeegan/src/osworld/evaluation_examples/examples/os/"
    "5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57.json"
)


def write_result(text: str) -> None:
    RESULT_FILE.write_text(text + "\n")
    print(f"[result] {text}", flush=True)


def main() -> None:
    CMD_FILE.unlink(missing_ok=True)
    RESULT_FILE.unlink(missing_ok=True)

    sys.path.insert(0, "/home/fkeegan/src/osworld")
    from desktop_env.desktop_env import DesktopEnv

    task = json.loads(TASK_PATH.read_text())
    print(f"[setup] Task: {task['id']}", flush=True)
    print(f"[setup] Instruction: {task['instruction']}", flush=True)

    env = DesktopEnv(
        provider_name="docker",
        os_type="Ubuntu",
        action_space="pyautogui",
        headless=True,
    )
    print("[setup] Resetting env (downloads poster, gio trashes it)...", flush=True)
    env.reset(task_config=task)

    info = {}
    provider = getattr(env, "controller", None)
    for attr in ("server_port", "vnc_port", "chromium_port", "vlc_port"):
        info[attr] = getattr(env, attr, None) or getattr(provider, attr, None)
    container = getattr(env, "container", None)
    if container is not None:
        info["container_name"] = getattr(container, "name", None)
    print(f"[setup] VM connection info: {info}", flush=True)

    score = env.evaluate()
    print(f"[setup] Initial evaluate(): {score}", flush=True)
    write_result(f"initial score={score} info={info}")

    print(
        f"\n[ready] Write commands to {CMD_FILE} (evaluate | reset | close).",
        flush=True,
    )

    while True:
        if CMD_FILE.exists():
            cmd = CMD_FILE.read_text().strip()
            CMD_FILE.unlink()
            print(f"[cmd] {cmd}", flush=True)

            if cmd == "evaluate":
                write_result(f"score={env.evaluate()}")
            elif cmd == "reset":
                env.reset(task_config=task)
                write_result(f"reset done, post-reset score={env.evaluate()}")
            elif cmd == "close":
                write_result("closing")
                break
            else:
                write_result(f"unknown command: {cmd}")

        time.sleep(0.5)

    env.close()
    print("[setup] Closed.", flush=True)


if __name__ == "__main__":
    main()
