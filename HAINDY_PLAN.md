# Vision-Grid Browser Agent — High-Level Implementation Plan

## 1  MVP Goals

| # | Goal | Success criterion |
|---|------|-------------------|
| 1 | Run end-to-end flows on a single website using grid-based clicks. | ≥ 90 % of "happy-path" scenarios finish without human help. |
| 2 | Support the basic actions `click`, `scroll`, `type`, `wait`. | All actions invocable via prompt. |
| 3 | Keep one full loop (screenshot → model → action) ≤ 1 s on reference HW (GPU T4). | Measured latency ≤ 1000 ms. |
| 4 | Persist logs for later debugging and fine-tuning. | PNG + prompt + model reply saved to disk. |

---

## 2  Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Orchestrator | **Python 3.10+** | Best ecosystem for AI & local installs. |
| Browser driver | **Playwright-Python (Chromium)** | WebSocket/CDP, native screenshots & absolute mouse clicks. |
| AI engine | Vision-LLM (e.g. **ShowUI-2B** quantized) + **GridGPT** 60 × 60 overlay | GUI-grounding specialists, small enough for local GPU. |
| Initial datasets | MiniWob++, Mind2Web | Baseline for testing and optional fine-tune. |
| Packaging | `setuptools` + `pyproject.toml` | Simple `pip` distribution. |
| Logs / metrics | JSONL + PNG + [`rich`](https://github.com/Textualize/rich) console | Human-readable and machine-parsable. |

---

## 3  Suggested Folder Layout

```
vision-grid-agent/
│
├─ src/
│   ├─ agent/              # high-level control loop
│   │   ├─ **init**.py
│   │   ├─ controller.py   # Playwright wrapper (click, scroll, screenshot)
│   │   ├─ planner.py      # prompt building & action parsing
│   │   └─ runner.py       # step-by-step loop
│   ├─ grid/
│   │   ├─ overlay.py      # 60 × 60 grid overlay utilities
│   │   └─ utils.py
│   ├─ models/
│   │   └─ showui\_runner.py
│   └─ config/
│       └─ settings.py     # grid size, timeouts, paths
│
├─ tests/                  # validation flows
├─ data/                   # captured screenshots & logs
├─ docs/                   # this file and extra docs
└─ pyproject.toml
```

---

## 4  Development Phases

| Phase | Tasks | ETA |
|-------|-------|-----|
| **0 — Prep** | Repo scaffold, `pre-commit`, CI smoke test. | 1 day |
| **1 — Playwright wrapper** | Launch Chromium, implement `screenshot`, `click_cell`, `scroll`. | 2-3 days |
| **2 — Grid overlay** | Generate 60×60 grid, map ↔ cell helpers. | 1 day |
| **3 — VLM integration** | Load quantized ShowUI-2B, `infer_action`, robust parsing. | 3-4 days |
| **4 — Agent loop** | Implement screenshot → VLM → exec cycle, step/time guards, logging. | 2 days |
| **5 — Testing & tuning** | 5 demo flows, measure success & latency, tweak grid/timeouts. | 1-2 days |
| **6 — Packaging & docs** | `python -m build`, README, release v0.1.0. | 1 day |

_Total: roughly **10–14 working days**._

---

## 5  Design Principles

1. **Separation of concerns** – agent ≠ driver ≠ model.  
2. **Fail-fast with rollback** – validate after every action; re-plan if no progress.  
3. **Explicit config** – grid size, timeouts, model, viewport in `settings.py` / env vars.  
4. **Logs first** – every step must be reproducible.  
5. **No blockers for scale-out** – headless Docker ready for future grid farm.

---

## 6  Post-MVP Roadmap

| Milestone | Description |
|-----------|-------------|
| **v0.2** | Support multiple tabs, text entry (`type`), basic form handling. |
| **v0.3** | Hybrid mode: fall back to unique DOM/ARIA selectors when available. |
| **v0.4** | Expose a REST service for remote job execution & screenshot uploads. |
| **v0.5** | Fine-tune on industrial logs; evaluate robustness across resolutions. |

---

## 7  Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Browser update breaks CDP API. | Pin Playwright ≥ 1.45; daily CI smoke tests. |
| High latency on older GPUs. | int8 quantization, option to offload inference to GPU server. |
| Infinite loops on endless animations. | Global `step_limit`, watchdog 120 s. |
| Miss-clicks in dense UIs. | Increase grid (72×72) or switch to mask when model confidence high. |

---

## 8  Immediate Next Steps

1. Add this `PLAN.md` to `docs/`.  
2. Create `conda`/`venv` with Python 3.10.  
3. `pip install playwright && playwright install chromium`.  
4. Build minimal wrapper; verify fixed click and screenshot.  
5. Integrate basic GridGPT overlay before wiring the VLM. 