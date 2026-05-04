# OSWorld First Task — Trash Recovery

## Goal

Run a single OSWorld task end-to-end on the local machine, three times, using **host-side Codex CLI** (already installed and OAuth'd) as the reasoning agent and **HAINDY in the OSWorld VM** (Gemini CU) as the desktop driver. A small `vm-haindy` wrapper on the host proxies HAINDY commands from Codex into the VM via OSWorld's `/execute` HTTP endpoint, so Codex sees haindy as if it ran locally.

```
Host (your machine)                          OSWorld VM (Ubuntu guest)
─────────────────────                        ─────────────────────────
Codex CLI (host install, OAuth'd)            HAINDY (per-task pip install)
   ↓                                            ↑
~/.agents/skills/vm-haindy/SKILL.md             X session, /dev/uinput
   ↓ tells Codex to call vm-haindy
docs/research/scripts/vm-haindy                 daemon survives /execute calls
   ↓ POST http://localhost:<port>/execute
   └──────────────────────────────────────── HAINDY runs the goal here
```

Learn how the OSWorld + HAINDY pieces fit together. No automation beyond what's needed for one task.

## Task

`5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57` — "I have wrongly deleted a poster of party night. Could you help me recover it from the Trash?"

- Snapshot: `os` (Ubuntu)
- Setup: downloads `poster_party_night.webp` to Desktop, then `gio trash`s it
- Evaluator: shell check that the file is back at `/home/user/Desktop/poster_party_night.webp`

## Phase 0 — Host prerequisites

- [x] `/dev/kvm` exists, `fkeegan` is in both `kvm` and `libvirt` groups
- [x] 664 GB free on `/` — well over the 80 GB target
- [x] Docker 29.4.1 installed; `docker run hello-world` succeeds
- [x] OSWorld checkout location: `~/src/osworld/` (matches existing `~/src/<project>/` pattern alongside `haindy`, `keenbench`, etc.)

## Phase 1 — OSWorld local install + smoke test

- [x] Cloned to `/home/fkeegan/src/osworld/` (HEAD `b87586c` on `main`)
- [x] Created venv with Python 3.10.20 via `uv venv --python 3.10` (system Python is 3.12, OSWorld pins are 3.10-era). Installed all 250 deps via `uv pip install -r requirements.txt`. Venv weighs 6.6 GB (torch + ~2 GB CUDA wheels are bulk; can prune later since we won't use OSWorld's baseline ML agents).
- [x] Ran `python quickstart.py --provider_name docker`. First-boot downloaded the 11.4 GB Ubuntu VM disk image into a Docker volume.
- [x] Quickstart hit all four expected markers: "Starting OSWorld environment…" → "Environment reset complete!" → "Action executed successfully!" → "Environment closed." Exit 0.
- [x] Disk footprint after first run: root disk +41 GB (206 → 247 GB used). Docker volume holding the VM image = 34.4 GB. OSWorld Docker image itself is only 359 MB. Pre-existing 66 GB of unrelated Docker build cache is reclaimable via `docker builder prune` if needed.

## Phase 2 — Validate the task manually (no agent)

Goal: prove the task setup + evaluator work before adding any AI.

- [x] Wrote `docs/research/scripts/phase2_runner.py` — holds env alive in background, takes commands via `/tmp/phase2_cmd` (`evaluate` / `reset` / `close`), writes results to `/tmp/phase2_result`. Loads task `5ea617a3-...json`, inits Docker DesktopEnv, resets, prints VM ports, then polls.
- [x] Initial `evaluate()` after reset returned **0.0** (file is in Trash, evaluator's `[ -f ... ]` check fails as expected)
- [x] Restored via shell — used HTTP `POST /execute` against the VM's control API on `localhost:5000` (no SSH on the Docker provider; this is the equivalent channel). `mv ~/.local/share/Trash/files/poster_party_night.webp ~/Desktop/ && rm ~/.local/share/Trash/info/poster_party_night.webp.trashinfo` → re-evaluate returned **1.0**
- [x] Reset to put the file back in Trash, then restored via the Files GUI inside the VM (accessed via noVNC at `http://localhost:8006/`) — re-evaluate returned **1.0**
- [x] VM access details documented below (replaces the original SSH item — the Docker provider doesn't expose SSH).

### VM access (Docker provider)

All on `localhost`, mapped from inside the container:

| Port | Purpose | How to use |
|------|---------|------------|
| 5000 | OSWorld control HTTP API | `curl -s http://localhost:5000/execute -H 'Content-Type: application/json' -d '{"shell":true,"command":"..."}'` — runs arbitrary shell in the guest, returns stdout/stderr/returncode JSON. This is what `vm_command_line` evaluators use. |
| 8006 | noVNC web client | Open `http://localhost:8006/` in a browser for a live GUI view + interaction (mouse/keyboard) |
| 9222 | Chromium DevTools | Browser automation hook |
| 8080 | VLC | Media app hook |

Other useful endpoints on 5000: `/screenshot` (GET), `/run_python` (POST), `/file` (POST), `/run_bash_script` (POST), `/terminal` (GET). Full list in `desktop_env/server/main.py`.

### VM environment reference (recon, captured 2026-04-27)

Run via `docs/research/scripts/phase3_recon.py` against a freshly-reset task. Ground truth for the install snippets in Phase 3 and 4.

| Probe | Finding |
|-------|---------|
| OS | Ubuntu 22.04.3 LTS, kernel 6.5 |
| Python | `/usr/bin/python3` = Python 3.10.12 |
| Pip | `pip3` = 22.0.2 |
| User | `user` (uid 1000), groups: `user adm cdrom sudo dip plugdev lpadmin lxd sambashare` — has sudo |
| X | `DISPLAY=:0` reachable, X.Org running under GDM3, `XAUTHORITY` empty but X is open enough that `xdpyinfo` works without it |
| `/dev/uinput` | Exists, perms `crw------- root root` — needs `sudo chmod a+rw /dev/uinput` per session for HAINDY's desktop backend |
| Network | PyPI HTTP 200, `generativelanguage.googleapis.com` reachable |
| Resources | `/` has 8.1 GB free, RAM 3.8 GB total / 2.4 GB available |
| Port allocator | Each fresh VM gets a unique server_port (5000, 5001, …) — install snippets / harness must read it dynamically, not hardcode |

## Phase 3 — Host-side `vm-haindy` wrapper + skill

Codex stays on the host. We give it a transparent proxy command (`vm-haindy`) that forwards args to HAINDY running in the VM, and a slightly tweaked skill that points at the wrapper. Codex never knows there's a VM in the middle.

- [x] Wrapper at `docs/research/scripts/vm-haindy` — bash, ~50 lines. Reads VM port from `$VM_HAINDY_PORT` or `/tmp/vm-haindy-port`. Sources `~/.osworld-secrets/keys.sh` in the VM (per the install snippet from Phase 4), exports `PATH=$HOME/.local/bin:$PATH`, `DISPLAY=:0`, `HAINDY_CU_PROVIDER=google`, `HAINDY_AUTOMATION_BACKEND=desktop`, then runs `haindy <args>`. Unwraps `/execute` JSON to re-emit HAINDY's actual stdout/stderr/exit code.
- [x] Symlinked to `~/.local/bin/vm-haindy` so Codex finds it on PATH.
- [x] Skill at `~/.agents/skills/vm-haindy/SKILL.md` — `sed 's/\bhaindy\b/vm-haindy/g'` of HAINDY's own `SKILL.md`, plus a preamble explaining the wrapper. Description updated to call out OSWorld VM context.
- [x] Wrapper smoke test from host: `vm-haindy session new --desktop` + `vm-haindy act "click on the desktop background" --session <id>` + `vm-haindy session close --session <id>` — all returned the same JSON envelope shape as direct `/execute` curls. Gemini CU completed an action in ~9s through the wrapper.
- [x] Codex smoke test: `codex exec --skip-git-repo-check -s danger-full-access "Use the vm-haindy skill to start a new desktop session, take one screenshot, and close the session..."` — Codex read the skill, ran 3 `vm-haindy` calls in sequence, reported session_id and screenshot_path. ~39k tokens.

### Codex sandbox caveat

Codex's default Linux sandbox (bubblewrap) failed on this machine: `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. Likely needs `kernel.unprivileged_userns_clone=1` or similar sysctl. Workaround for now: `-s danger-full-access`. Fine for our use case (Codex is already constrained to `vm-haindy`-only via the skill prompt; the VM provides the real isolation), but worth fixing properly later — possibly via a `~/.codex/config.toml` profile so the flag isn't on every invocation.

## Phase 4 — HAINDY install procedure (in VM, per-task)

Per-task pip install of HAINDY into the freshly-reset OSWorld guest. Driven via `/execute` on the dynamically-allocated server port. Install state itself is disposable — what we keep is the snippet.

Validated end-to-end: HAINDY runs in the OSWorld VM, drives the desktop, completes a Gemini CU action.

- [x] Install snippet authored in `docs/research/scripts/phase4_runner.py` (`HAINDY_INSTALL_BG`). One background bash chain: apt build deps + xdotool/xclip → upgrade pip → `pip install --user haindy`. Polls `/tmp/haindy_install.log` for completion; uses `which haindy` as the success signal (NOT `haindy --version` — there is no such flag).
- [x] Snippet runs cleanly against a freshly-reset VM. ~5–10 min total (apt + pip + numpy/cryptography/evdev compile).
- [x] `haindy doctor` shows everything green **except credentials** (Doctor uses `get_api_key()` which checks the keychain only, NOT env vars — see `haindy/cli/doctor.py:55` and `haindy/auth/credentials.py:29`. Env vars work at runtime; Doctor's "MISSING" is misleading for our setup.)
- [x] `haindy session new --desktop` → returns success JSON envelope, screenshot_path, session_id; daemon spawns and persists across /execute calls
- [x] `haindy act "click on the desktop background" --session <id>` → Gemini CU takes screenshot, decides target, clicks. ~9s for one action. Status: success.
- [x] `haindy session close --session <id>` → clean shutdown, action count returned.
- [x] Extracted to `docs/research/scripts/install_haindy_in_vm.sh` — self-contained synchronous bash script. Caller wraps with background+poll because of /execute's 120s timeout.

### Credential injection (env vars, no keychain)

Host-side `~/.osworld-secrets/keys.env` holds two `KEY=VALUE` lines (`HAINDY_OPENAI_API_KEY`, `HAINDY_VERTEX_API_KEY`). Inline Python helper reads it, applies `shlex.quote` to each value (one of Federico's keys had a `>` that breaks dash sourcing), base64-encodes, POSTs to `/execute` to write `~/.osworld-secrets/keys.sh` inside the VM. Each subsequent haindy invocation runs `. ~/.osworld-secrets/keys.sh` as part of its command prefix.

The full prefix used in /execute calls:
```
. ~/.osworld-secrets/keys.sh; \
  export PATH=$HOME/.local/bin:$PATH; \
  export DISPLAY=:0; \
  export HAINDY_CU_PROVIDER=google; \
  export HAINDY_AUTOMATION_BACKEND=desktop; \
  haindy <args>
```

### Findings worth keeping

- **Pip 22.0.2 in VM is too old** — has a metadata extraction bug that fails on `evdev` sdist with "produced metadata for project name unknown". Must upgrade pip (`pip install --user --upgrade pip`) before installing haindy.
- **`evdev` is a C extension** with no prebuilt wheels. Needs `build-essential` + `python3.10-dev` to compile.
- **HAINDY's desktop backend needs `xdotool` + `xclip`** beyond the obvious deps. Doctor's "Automation backend MISSING" is the symptom.
- **`/dev/uinput` is `crw------- root root`** — must `chmod a+rw` per VM boot. Sudo password is `password`.
- **Doctor doesn't reflect env-var credentials.** Trust the actual command, not doctor's cred lines.
- **`/run_bash_script` endpoint is broken** in this VM image (`_append_event` undefined in `desktop_env/server/main.py`). Use chunked `/execute` (120s timeout) and background+poll for slow operations.
- **Shell-special chars in keys.** OpenAI/etc. keys can contain `>`, `$`, `&`, etc. Always `shlex.quote` values before writing them to a shell-sourced file.
- **`pip --user` writes to `~/.local/bin`** — must be on PATH explicitly in non-login shells (GNOME Terminal is a login shell so `~/.profile` adds it; `/bin/sh -c` from /execute is not, hence the explicit `export PATH=$HOME/.local/bin:$PATH` in every command).

## Phase 5 — Single end-to-end run

- [x] Wrapper prompt at `docs/research/scripts/phase5_codex_prompt.md` — OSWorld instruction + vm-haindy-only rule + "do not verify via shell" + action budget guidance.
- [x] Harness at `docs/research/scripts/phase5_runner.py` — synchronous flow: reset env → write port to `/tmp/vm-haindy-port` → upload + run install script in VM (background+poll) → upload keys → wrapper sanity check → `codex exec` on host with 600s timeout → `env.evaluate()` → save `summary.json` to `~/osworld-runs/<run_id>-trash/`.
- [x] **First run: PASS, score 1.0**, run id `20260430T113125Z`, codex wall 183.8s, codex rc=0.
- [x] Codex's actual flow: 4 `vm-haindy act` calls (Files icon → Trash → select poster → Restore) interleaved with 4 `session status` checks. 1 attempted `explore` failed with a CLI dispatcher bug (see Findings below); Codex adapted to `act`-only and still scored 1.0.

### Findings from the first real run

- **`vm-haindy explore "..." --session <id> --timeout 180` returned `haindy: error: argument COMMAND: invalid choice: 'explore'`** (top-level CLI choices were `version, doctor, setup, test-api, auth, config, provider, run`). But `vm-haindy session new --desktop`, `act`, `session status`, `session close` all worked. So the dispatcher recognises `session` as a tool-call command but not `explore` in this code path. Worth investigating in `haindy/main.py` / `tool_call_mode/cli.py` — possibly a missing entry in the tool-call command set or a check on daemon presence. Not blocking for the trash task; matters more for tasks that need the multi-step awareness loop.
- **Step-by-step `act` worked great for this task.** 4 actions to recover a file from Trash via Nautilus is roughly the human floor. Codex's prompt-driven verification cadence (status check between acts) is a useful pattern.
- **Total tokens / cost**: codex used the sandbox bypass (`-s danger-full-access`); didn't extract token count from logs yet — to do for the next runs.
- **HAINDY journal in VM** is at `/home/user/.haindy/sessions/<session_id>/`. We don't pull it back yet; future improvement.

## Phase 6 — Three runs + observations

**3-for-3 pass on the trash recovery task.** All artifacts under `~/osworld-runs/<utc-timestamp>-trash/`.

| Run | UTC ID | Score | Codex wall (s) | Codex `act` calls |
|-----|--------|-------|---------------:|------------------:|
| 1   | 20260430T113125Z | 1.0 | 183.8 | 4 |
| 2   | 20260430T113920Z | 1.0 | 173.1 | 4 |
| 3   | 20260430T114335Z | 1.0 | 203.0 | 5 |

Mean codex wall ~187s (~30s range). Action counts within ±1.

### Observations

- **Codex's GUI strategy was identical across all three runs**: open Files (left dock), click Trash sidebar, select the poster, click Restore. With one extra disambiguation click in run 3 (`act count = 5`).
- **The `vm-haindy explore` CLI dispatcher bug appeared in all three runs** — Codex tried explore each time, got "invalid choice", then fell back cleanly to step-by-step `act`. The fallback is robust but the bug is real and worth fixing in HAINDY's CLI dispatcher (Federico already has a branch for explore work — note follow-up there).
- **Determinism was high.** Each run got a fresh VM snapshot, pip-reinstalled HAINDY, fresh codex session. The agent strategy didn't drift; the wall time variance came from Gemini CU latency on individual `act` calls.
- **Wrapper sanity check at run start was useful** — caught no failures across the 3 runs, but it's a cheap pre-flight that would have surfaced env-var or daemon problems before sinking ~5 min into a doomed Codex call.
- **What we still aren't capturing:** Codex token counts (parseable from the tail of codex.log: `tokens used N`), full HAINDY journal from inside the VM, screenshots beyond what's saved by HAINDY's session daemon. Improvements for the next benchmark expansion.

### Decision: what changes before going broader

- File the `vm-haindy explore` dispatcher bug as a follow-up (Federico's existing explore branch).
- Add token-count parsing to the Phase 5 runner so the summary.json captures it.
- Pull HAINDY's journal back from VM into the run dir before `env.close()` (so the artifacts are self-contained on host).
- Resolve the open question (where the harness lives — keep it under `docs/research/scripts/` for now is fine).

## Decisions

- **Codex must route through HAINDY only.** No shell fallback for app interactions. The wrapper prompt enforces this. We're evaluating the CU stack; if Codex bypasses it, the run is meaningless.
- **HAINDY caches cleared manually between runs.** No automation needed for the first experiment — `rm -rf` the relevant `~/.haindy` subdirectories before each run.
- **Done-vs-stalled handled with a hard wall-clock timeout** (~10 min for the trash task). On timeout, kill Codex and call `env.evaluate()` regardless. OSWorld's evaluator is the source of truth — Codex's self-report is ignored. Secondary guard: cap Codex's max turns so it exits on its own first when possible.
- **Codex stays on host; HAINDY runs in VM; `vm-haindy` wrapper bridges them.** Earlier plan was Codex-in-VM (forced npm + Node + per-task `codex login`). We pivoted because Codex is already installed and OAuth'd on the host — putting it in the VM was duplicating a working setup and adding fragile install steps. The wrapper is a 40-line bash script (`docs/research/scripts/vm-haindy`) that proxies haindy commands into the VM via /execute and unwraps the response so output is identical to running haindy locally.
- **HAINDY installed per-task in VM, not baked into the VM image.** OSWorld restores from a clean snapshot on every `env.reset()`, so a one-time guest install would be wiped. For the POC the per-task install snippet pip-installs haindy fresh each run. Slower per-task and noisier in trajectory logs, but cheap to iterate. Revisit a custom Docker image with HAINDY baked in once we expand beyond this single task.
- **VM credentials via `HAINDY_*` env vars per task, not the keychain.** The skill explicitly documents env vars as the highest-priority source and the supported CI/CD path. Avoids interactive OAuth in the VM and keychain portability questions. The host is the source of truth for credentials; the per-task setup writes them into the VM's env.
- **OSWorld VM image has a bug: `/run_bash_script` endpoint is broken** (`name '_append_event' is not defined` in `desktop_env/server/main.py:1735`). Use chunked `/execute` calls (120s subprocess timeout each) and background+poll for anything longer.

## Open questions

- [ ] Where to keep the harness script — in this repo under `docs/research/`, or separate?

## Deferred

- Migrate OSWorld checkout, VM image cache, and Docker data root from the root disk (`nvme1n1`, ~664 GB free at start, now ~622 GB after first VM image pull) to the unused 1 TB SSD at `/data` (`nvme0n1`, ~870 GB free). POC stays on the root disk; revisit once we know we want to keep running OSWorld and disk pressure becomes real. Each OSWorld task snapshot likely adds another ~10–30 GB volume, so this gets urgent if we expand beyond a handful of tasks.
- Prune OSWorld's `requirements.txt` to drop the ML-agent-only deps (torch, transformers, ~2 GB CUDA wheels) since we're using Codex+HAINDY, not OSWorld's baseline agents. Saves ~4 GB of venv. Not blocking for the POC.
