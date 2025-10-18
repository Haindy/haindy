# BaaA System Architecture (Conceptual)

> **Scope:** Conceptual architecture for a Business‑as‑an‑Agent (BaaA) that discovers, bids, executes, and delivers black‑box QA work using Haindy as the execution core. Aligned with the **BAAA_ROADMAP.md** Bootstrap → Phases 3–5.

## 1) Purpose & Principles

**Purpose.** Define the components, responsibilities, and flows of an agentic system that operates in public gig marketplaces while remaining local‑first, compliant, and reputation‑safe.

**Guiding Principles**
- **Local & Sandbox‑First:** All execution runs on operator‑controlled machines/VMs.
- **Human‑Gateable:** Every autonomous action can be paused or routed for approval.
- **Compliance by Design:** Rate‑limits, timing jitter, and ToS‑aware behaviors.
- **Traceable & Auditable:** Every decision/action leaves artifacts and logs.
- **Modular & Replaceable:** Agents are decoupled via typed messages.
- **Incremental Trust:** Autonomy increases only after meeting success thresholds.

**Non‑Goals (for now)**
- Multi‑tenant cloud SaaS
- Real‑time multi‑persona scaling beyond one primary freelancer identity
- Full UX audits (beyond black‑box functional testing & bug reproduction)

---

## 2) High‑Level Component Map

```
Business Layer (Supervisor + Policy)
  ├─ Orchestrator (Job Lifecycle Controller)
  │   ├─ GigScoutAgent (Discovery & Ranking)
  │   ├─ BidAgent / Proposal Composer (Response & Pricing)
  │   ├─ CommunicationAgent (Messages, Delivery, Handoffs)
  │   └─ QAExecutionAgent (Haindy Core Interface)
  ├─ Safety & Compliance Layer (Rate/ToS/Persona Guardrails)
  └─ Telemetry & Reporting (Observability, Earnings, Trust)

Shared Services
  ├─ Data Store (Artifacts, runs, proposals, message logs)
  ├─ Config Registry (Profiles, allowlists, thresholds)
  └─ Secrets Vault (API keys, credentials)
```

**Notes**
- The **Orchestrator** owns lifecycle state machines; agents remain stateless where possible.
- **Haindy** is invoked by QAExecutionAgent and returns structured artifacts for delivery.

---

## 3) Agent Responsibilities

### 3.1 Orchestrator (Job Lifecycle Controller)
**Role:** Central state machine from discovery → bidding → execution → delivery → closure. Enforces automation level (P1 alerts, P2 confirm‑send, P3 autonomous w/ safety stops).

**Key Functions**
- Maintain job queue and status (NEW, QUALIFIED, BID_SENT, WON, RUNNING, DELIVERED, CLOSED, BLOCKED)
- Route tasks to agents with typed messages
- Apply policy: rate limits, timing jitter, persona hours
- Persist lifecycle events for audit/rollback

### 3.2 GigScoutAgent (Discovery & Ranking)
**Role:** Find relevant gigs, normalize listings, score by fit/ROI, and emit **JobLead** events.

**Inputs:** Marketplace feeds (API or scraping), local keyword/profile prefs

**Outputs:** `JobLead{platform, job_id, title, budget, tags, description, confidence, freshness}`

**Core Behaviors**
- Relevance classification (QA/black‑box/bug reproduction/website testing)
- Freshness priority & dedupe
- Confidence estimation; thresholding for P2/P3
- Back‑off and jitter to avoid ToS risks

### 3.3 BidAgent / Proposal Composer
**Role:** Generate tailored proposals with pricing options and attachments.

**Inputs:** `JobLead`, pricing policy, portfolio snippets, deliverable templates

**Outputs:** `BidDraft{cover_letter, price_model, attachments, confidence}`

**Core Behaviors**
- Template selection by job archetype
- Price band suggestion (fixed/hourly/mixed) with rationale hints
- P1: draft‑only → human edits; P2: one‑click send; P3: autonomous submit if confidence ≥ threshold

### 3.4 CommunicationAgent (Messaging & Delivery)
**Role:** Manage client‑facing communication across the job lifecycle.

**Inputs:** lifecycle events, artifacts, policy (tone, SLA, availability window)

**Outputs:** `MessageDraft` / `MessageSent`, `DeliveryPackage`

**Core Behaviors**
- Pre‑engagement: clarifying questions (drafts) and ETA framing
- Delivery: attach reports, screenshots, defect summaries
- Post‑delivery: follow‑ups, revision requests, closure notes, review requests
- P1 draft; P2 confirm‑send; P3 auto with safety stops on ambiguity

### 3.5 QAExecutionAgent (Haindy Core Interface)
**Role:** Translate job requirements into Haindy runs and return standardized artifacts.

**Inputs:** `ExecutionOrder{url(s), requirements, scenarios, environment hints}`

**Outputs:** `ExecutionResult{report.md/html, screenshots/, action_logs.json, defects.json, summary.md}`

**Core Behaviors**
- Prepare run folders and environment
- Invoke Haindy modes (plan/execute) with policy (allowlists, observe‑only steps where required)
- Collect and normalize outputs into **DeliveryPackage** schema
- Map failures to actionable client‑facing summaries

### 3.6 Safety & Compliance Layer
**Role:** Cross‑cutting policies to keep accounts safe and sustainable.

**Functions**
- Rate limiting, randomized delays, active hours windows
- Platform‑specific caps (daily bids, message frequency)
- Content filters (no PII leakage, no external links if disallowed)
- Persona guardrails (tone, honesty, light AI disclosure if needed)
- Watchdog triggers (lockdown on repeated errors or warnings)

---

## 4) Data Model (Conceptual)

- **JobLead**: platform, job_id, title, description, budget, client_signals, tags[], freshness, confidence
- **BidDraft / BidSent**: job_id, cover_letter, price_model{type, amount, rationale}, attachments[], confidence
- **ExecutionOrder**: job_id, scope, URLs, test_type, constraints, timelines
- **ExecutionResult**: run_id, pass/fail summary, artifacts[], defects[], coverage_notes
- **DeliveryPackage**: job_id, summary.md, report.html/md, screenshots.zip, defects.json, effort_notes
- **LifecycleEvent**: ts, actor, state_from→state_to, message_id, notes
- **Policy**: thresholds, rate limits, timing windows, trust levels, ToS constraints

Storage is local‑first (filesystem + lightweight DB). Artifacts kept under `/runs/<job_id>/<run_id>/…` with symlinked `latest/` for operator convenience.

---

## 5) Lifecycle Flows

### 5.1 Discovery → Qualification
1. **GigScoutAgent** polls/receives jobs → normalize → score relevance and ROI
2. Orchestrator filters by confidence & policy → queue as **QUALIFIED**
3. P1: Alert operator with `JobLead`; P2: show one‑click **Compose Draft**; P3: proceed to BidAgent automatically

### 5.2 Bid Composition → Submission
1. **BidAgent** selects template & price band → produces `BidDraft`
2. P1: operator edits/sends in marketplace UI
3. P2: one‑click **Send Bid** via local action
4. P3: auto‑submit (API/automation) with rate limiting and timing jitter

### 5.3 Job Won → Execution
1. Orchestrator creates `ExecutionOrder` from client requirements
2. **QAExecutionAgent** runs Haindy, collects artifacts → `ExecutionResult`
3. Policy evaluates safety/quality thresholds; escalate if ambiguous or blocked

### 5.4 Delivery → Closure
1. **CommunicationAgent** builds `DeliveryPackage` and message
2. P1: draft review; P2: confirm‑send; P3: auto‑send + schedule follow‑up
3. Orchestrator marks job **DELIVERED**, tracks revisions, then **CLOSED**

---

## 6) Automation Levels & Trust Gates

- **Phase 1 (P1) – Alerts‑Only:** Agents produce drafts/alerts; human executes.
- **Phase 2 (P2) – Confirm‑to‑Act:** Single‑click sends; guarded automation.
- **Phase 3 (P3) – Autonomous:** Automatic actions when confidence ≥ thresholds.

**Trust Gates (per function)**
- Discovery: precision ≥ X% on relevance over N leads
- Bidding: win‑rate uplift or acceptance rate ≥ Y% with no ToS flags
- Execution: defect/coverage quality meets client satisfaction on M jobs
- Delivery: zero content policy violations across K sends

Thresholds live in **Policy** and are adjustable without code changes.

---

## 7) Observability & Telemetry

- **Event Log**: append‑only lifecycle events with correlation IDs
- **Artifacts Index**: run_id → paths for reports, screenshots, logs
- **Outcome Metrics**: leads → bids → wins → deliveries; acceptance & rating
- **Trust Metrics**: per‑agent confidence curves and error classes
- **Watchdog**: detects repeated failures; triggers lockdown & operator alert

Outputs summarized in a local HTML/CLI dashboard (no external services required).

---

## 8) Security & Compliance

- Local secret storage (OS keychain or file‑based vault) with minimal scope
- Strict marketplace ToS adherence; human review for edge behaviors
- Rate limits + exponential back‑off; randomized intervals
- Persona hygiene: consistent tone, no deceptive claims, optional light AI mention
- Auditability: immutable logs for defense if disputes occur

---

## 9) Deployment & Runtime Model

- **Runtime Modes:**
  - Dev (simulated feeds, dry‑run delivery)
  - Supervised (P1/P2, interactive prompts)
  - Autonomous (P3, safety‑stops enabled)

- **Topologies:**
  - Single node (laptop/desktop)
  - Multi‑node (controller + workers: VMs or physical machines)

- **CLI First:** All agents invokable via CLI; optional thin TUI for operator comfort.

---

## 10) Risks & Mitigations

- **Marketplace counter‑automation measures** → conservative rate limits, randomization, manual interleaving in P1/P2
- **False positives in relevance/bidding** → human review in P1, trust gates before P3
- **Client dissatisfaction** → standard deliverables, clear revision terms, rapid response templates
- **Operational drift** → monthly self‑audit checklist, sample run reviews

---

## 11) Roadmap Alignment (from BAAA_ROADMAP.md)

- **Stage A (Bootstrap)** → Tracks 1–4 map to GigScout/Bid drafts, manual triggers, artifact templates
- **Phase 3** → Enable auto‑bidding/delivery with Safety Stops; watchdog active
- **Phase 4** → Learning loops, pricing optimization, multi‑node experiments
- **Phase 5** → Continuous autonomous loop with periodic audits

---

## 12) Open Questions

1. Upwork API vs. headless automation for bid submission (capabilities, limits)?
2. Confidence scoring methodology: rules, heuristics, or lightweight model?
3. Standard deliverable set per job archetype (minimal viable + premium tiers)?
4. Operator UX: CLI only or minimal TUI (ncurses) for faster triage?
5. Multi‑persona roadmap and when to incorporate (post Phase 4?)

---

## 13) Trackable Checklist (Living)

- [ ] Orchestrator state machine defined (concept)
- [ ] Message schemas v0 for JobLead, BidDraft, ExecutionOrder, DeliveryPackage
- [ ] Artifact templates (summary.md, report.html/md, defects.json)
- [ ] GigScout v0 returns normalized leads
- [ ] Proposal Composer v0 produces usable drafts
- [ ] Manual one‑click job trigger in Haindy
- [ ] Safety & Compliance policy file v0
- [ ] Event Log + Artifacts Index implemented (local)
- [ ] Watchdog behaviors defined
- [ ] P2 confirm‑to‑act path validated on 3 internal tests
- [ ] P3 autonomous path validated on low‑risk gigs

