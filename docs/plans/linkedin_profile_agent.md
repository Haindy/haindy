# LinkedIn Profile Data Entry Agent — Implementation Plan

## Goals
- Long-lived desktop agent that listens in Slack for LinkedIn profile URLs and performs a deterministic scrape + delivery workflow.
- Operate via OS-level computer-vision/soft-input (no DOM/Playwright), reusing the POC desktop substrate.
- Minimize re-interpretation by caching LinkedIn UI coordinates and reusing them broadly (no cap), invalidating only on failure.
- Post results to the backend using a fixed JSON schema (example to be provided) and acknowledge in Slack with reactions.

## Assumptions & Constraints
- Environment: Ubuntu, single monitor, 1920x1080 preferred. Firefox already open with an authenticated LinkedIn session; Slack desktop app/window already open to `#data-entry-agent-test`.
- Input model: One task at a time (single pointer/keyboard). Agent alternates focus between Slack (intake/ack) and Firefox (scrape).
- Slack protocol: Messages in `#data-entry-agent-test`. A new unprocessed task is a message containing a LinkedIn profile URL without the ⏳ (ack) or ✅ (done) reaction. Agent adds ⏳ when claimed and ✅ on success.
- Backend: POST payload must match the provided example JSON (to be embedded in the prompt). GET validation is TBD.
- Persistence: Coordinate/action caches are unbounded (cache every successful action coordinate); invalidate only the failing entry when an action fails and retry live. Reports/artifacts should remain light and capped (keep only the most recent handful), while still capturing the ~9 key step screenshots per run.
- `money_type_id` is always `USD` (backend-enforced). `photoUrl` is the LinkedIn profile image presigned URL obtained via right-click/copy image URL. The sample payload below is trimmed to two experience entries purely to keep the example short; real payloads should include all scraped experience rows.

## High-Level Flow
1) Intake loop:
   - Bring Slack forward; scan the channel timeline for the newest message with a LinkedIn profile URL lacking ⏳/✅.
   - Add ⏳ reaction to claim; record message metadata for later ✅.
   - Capture minimal intake screenshot for traceability.
2) Profile processing (Firefox):
   - Bring Firefox LinkedIn window frontmost; confirm session.
   - Follow predefined action plan to gather:
     - Contact info (open/close modal).
     - Skills (scroll, expand).
     - Education (scroll, expand).
     - Experience (scroll, expand).
     - Profile image presigned URL (open avatar, right-click copy image URL).
   - Populate JSON payload per provided example (schema embedded in prompt); include cached coordinates reuse where possible.
3) Backend I/O:
   - POST candidate JSON to backend client.
   - Optional: stub hook for future GET verification (TBD).
4) Completion:
   - Return to Slack, add ✅ reaction to the originating message.
   - Persist task-level screenshots (intake + major steps) and append coordinate cache updates (no truncation; invalidate only on failure).

## Architecture / Component Plan
- **Plan definition**: New task plan JSON under `test_scenarios/` (desktop_mode=true) that encodes the fixed workflow (no dynamic planner). Steps map 1:1 to the flow above.
- **Operator/Runner**: Reuse `TestRunner` in desktop mode as the operator; preload with the static plan and disable test-planner calls.
- **Action Agent**: Reuse existing desktop action agent (computer-use) with coordinate cache lookup-first; cache is keyed by label/action/resolution (e.g., “contact info link” → x/y). On failure, invalidate only that entry and retry live. Do not truncate successful entries.
- **Slack client**: Lightweight poller for `#data-entry-agent-test` (RTM or Web API). Responsibilities: find unclaimed messages, add reactions, fetch message text/URL, and post final reactions. Runs outside the computer-use loop; only focus-switching is done by the action agent.
- **Backend client**: Thin wrapper around the POST endpoint; embeds the example JSON template in prompts and validation; placeholder hook for future GET verification. Enforce `money_type_id="USD"`. Use the full scraped experience list in real payloads; the sample is trimmed only to reduce prompt size.
- **Focus management**: Simple toggles between Slack window and Firefox LinkedIn window; keep POC resolution/downshift behavior.
- **Caching & retention**:
  - Coordinate cache: append-only; cache every successful label/action/resolution coordinate. Include screenshot hash when available. Invalidate only failing entries.
  - Task evidence and reports: store only the small set of step screenshots (intake + major scrape steps). Keep reports/artifacts light and capped to the most recent handful of tasks.

## Stepwise Tasks
1) Author fixed scenario file (e.g., `test_scenarios/linkedin_profile_agent.json`) with desktop_mode enabled and the exact scrape/post/ack steps.
2) Add plan doc to prompts: embed backend JSON example (once provided) into the payload-construction prompt.
3) Implement Slack intake loop:
   - Channel filter and unprocessed-message detection (no ⏳/✅).
   - Add ⏳ on claim, return message metadata for execution context.
4) Implement focus-switch helpers for Slack<->Firefox (reuse computer-use prompts; window hints for both apps).
5) Wire operator path:
   - Skip planner; load static plan and run through TestRunner desktop path.
   - Inject Slack message context (profile URL, message ID) into the run.
6) Backend client:
   - POST payload builder using scraped fields.
   - Leave GET validation stubbed/TBD.
7) Coordinate cache hygiene: ensure resolution is part of the key; on failure invalidate the specific entry and retry without cache; allow manual reset flag; do not truncate successful entries.
8) Evidence handling: ensure only key step screenshots are saved; cap reports/artifacts to the recent handful of tasks; tag runs with candidate identifier and Slack message ID.
9) Runbook: add a short runbook covering Slack token/config, Firefox/LinkedIn prep, and how to launch the agent loop.

## Open Items / To Clarify Later
- Exact backend POST example JSON (will be embedded once provided).
- Confirmation GET contract and success criteria.
- Slack auth method (bot token vs. user token) and polling interval/limits.
- Any non-default keyboard layouts or accessibility settings that could affect coordinates. 

## Backend Payload Template (to embed in prompts)
The payload must follow this structure; `money_type_id` stays `USD`, `photoUrl` is the LinkedIn presigned image URL. The example is trimmed to two experience entries only to keep the prompt short; actual payloads should include all scraped experience rows.

```json
{
  "data": {
    "fullName": "Alex Doe",
    "photoUrl": "https://media.licdn.com/dms/image/C4D03AQExample?e=9999999999&v=beta&t=FAKE_SIGNED_TOKEN",
    "linkedin": "https://www.linkedin.com/in/alex-doe/",
    "email": "alex.doe@example.com",
    "job_title": "Backend & Trading Bots Developer • Data Engineering & Automation • AI-Powered Development",
    "location": "Valencia, Valencian Community, Spain",
    "about": "Backend & Trading Bots Developer specialized in Data Engineering, Automation, and AI-Powered Development.",
    "work_type_preference_id": 1,
    "money_type_id": "USD",
    "sector": "Information Technology",
    "work_experience_years": 4,
    "skills": [
      "Backend Development",
      "Trading Bots",
      "Data Engineering",
      "Automation",
      "AI-Powered Development",
      "Git",
      "Django",
      "Blockchain",
      "QA Engineering"
    ],
    "education": [
      {
        "institution": "Tech University",
        "title": "Telecommunications Engineering",
        "from": "2012",
        "to": "2018",
        "description": "Version control, software development and related skills"
      },
      {
        "institution": "Technical Institute Berlin",
        "title": "Engineering Exchange",
        "from": "2017",
        "to": "2017",
        "description": "Analysis and engineering coursework"
      }
    ],
    "experience": [
      {
        "organization": "GyS Crypto",
        "organizationLogoUrl": "https://logo.example/gys_crypto.png",
        "position": "Developer",
        "type": "Full-time",
        "from": "Jun 2023",
        "to": "Nov 2025",
        "duration": "2 yrs 6 mos",
        "description": "Designed a proprietary web trading platform and high-frequency trading bots.",
        "location": "Valencia, Spain - On-site",
        "skills": ["Information Technology", "Git"]
      },
      {
        "organization": "Entelgy",
        "organizationLogoUrl": "https://logo.example/entelgy.png",
        "position": "Advanced-TV QA Engineer",
        "type": "Full-time",
        "from": "May 2022",
        "to": "Jun 2023",
        "duration": "1 yr 2 mos",
        "description": "Built a dynamic URL generator for advanced TV platforms and automated load testing.",
        "location": "Valencia, Spain - Remote",
        "skills": ["Information Technology", "Git"]
      }
    ]
  }
}
```
