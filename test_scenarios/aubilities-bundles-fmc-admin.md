# Context

I need you to test a new feature in my admin app called Group Session Bundles. 

I will give you the PRD but you must take as scope of this testing ONLY the admin part. 

The admin is running locally and I'll provide you with this temporary user: 
username: federico@aubilities.com
password: Test1234!!

You will need to sign in first by going to http://localhost:3001 and entering the credentials, then clicking the sign in button.

There are no bundles already created in this environment that you can reuse for testing, you will have to create your own. You can reuse existing group sessions though.

Once you log in you'll get to the dashboard, and to access the bundles and group sessions sections you have to use the side menu, which is expandable. You need to hover over in order to expand it.

Unfortunately I don't have any screenshots or exact copys of the lables, buttons, etc you you will need to keep the tests generic and figure out what is what as you look at the screenshots. 

Don't test localisation for now, just assume it will all be in english. 

Select just one specialism, any one will do. 

Don't test image upload, it would take too much time.

The usual amount of sessions for a bundle is 3, don't go higher than that. 

Here's the PRD:

# Session Bundles PRD

## 1. Background & Rationale
- Aubilities delivers B2B neurodiversity-focused coaching to employees of our client companies. Pricing, billing, and entitlements are handled in client contracts and remain out of product scope.
- Individual group sessions cannot adequately cover multi-session curricula (e.g., Executive Function coaching). Program managers need a way to present multi-part offerings without forcing employees to stitch together ad-hoc bookings.
- Session bundles allow us to package related group sessions into a single course that employees can browse, understand, and RSVP to in one action, while still tracking attendance per underlying session.

## 2. Goals & Non-Goals
**Goals**
- Offer curated multi-session courses that surface as single cards in FMC user-facing experiences.
- Preserve per-session metadata (date, time, calendar invites) while managing capacity and RSVP at the bundle level.
- Enable the central admin team to create, update, and manage bundles from existing or newly created group sessions.
- Maintain parity with current translation workflows across supported locales (EN, ES, IT, DE, FR, PL).

**Non-Goals**
- Displaying pricing, payment, or invoicing flows.
- Allowing employees to RSVP to individual sessions from a bundle independently.
- Providing bundle progress tracking, coaching analytics, or new notification cadences (future roadmap items).
- Granting coaches self-service tools for bundle management.

## 3. Success Criteria & Signals
- ≥1 bundle per flagship program launched by program managers in Admin within the first release cycle.
- Employees can RSVP to a bundle in ≤3 clicks from recommendation surfaces, with no drop-off compared to standalone group sessions.
- Calendar invitations, Nylas sync, and email confirmations continue to work for each underlying session when bundle RSVPs fire.
- Bundled sessions stop appearing as standalone recommendations or listings once attached to a bundle.

## 4. In-Scope vs. Out-of-Scope
| In Scope | Out of Scope |
| --- | --- |
| Showing bundles as single recommendation cards in FMC frontend | Pricing, discounts, or subscription handling |
| Bundle detail view with nested session mini-cards | Waitlist management or partial enrollment |
| RSVP/cancel flows that enroll/unenroll all sessions together | Bundle-level reminder cadence or progress dashboards |
| Admin bundle CRUD, session association, image upload, capacity controls | Coach-facing management tooling |
| Data model updates and translation job wiring | Migration scripts to auto-bundle legacy sessions (manual curation only) |

## 5. Primary Users & Personas
- **Employee (End User)**: Uses FMC frontend to discover group coaching; needs clear explanation of course structure without extra pricing context.
- **Program Manager / Operations Admin**: Uses Admin portal to configure bundles, attach existing sessions, manage capacity, and ensure localized content.
- **Internal Systems**: 
  - Bundle Matching Service (backend) recommends bundles alongside standalone sessions.
  - Journey RSVP service ensures Nylas calendar invites and attendance tracking work per session.

## 6. Use Cases
| ID | Persona | Scenario | Outcome |
| --- | --- | --- | --- |
| UC-1 | Employee | Views recommended coaching options and sees a bundle card summarizing a multi-session course. | Employee understands course scope without seeing individual sessions separately. |
| UC-2 | Employee | Opens bundle detail page to read full description, review schedule, and confirm capacity. | Employee evaluates fit and decides whether to RSVP. |
| UC-3 | Employee | RSVPs to a bundle; backend enrolls them in all sessions, sends calendar invites, and increments capacity. | Employee receives confirmation for each session while perceiving a single booking action. |
| UC-4 | Employee | Cancels a bundle RSVP even after first session has started. | All session RSVPs are removed, invites canceled, capacity decremented. |
| UC-5 | Admin | Creates a new bundle by selecting 3 existing sessions, assigning a coach, and uploading an image. | Bundle becomes available for recommendation once validations pass. |
| UC-6 | Admin | Updates bundle capacity after assessing demand. | Bundle participant limits adjust immediately while preserving RSVP integrity. |
| UC-7 | Admin | Removes a session from a bundle with existing RSVPs. | System warns of impact; admin decides whether to proceed (policy TBD). |

## 7. User Experience Overview
### 7.1 Discovery (FMC Frontend)
- Recommendation endpoints return bundles as part of the existing `/journeys/{id}/recommended` flow.
- Each bundle appears as a single card:
  - Title, description excerpt, coach, participant capacity (current/max), specialisms chips.
  - Bundled sessions do **not** appear as standalone cards or search results.
- Filters continue to operate on bundle specialisms; UI should gracefully handle cases where only bundles match selected filters.

### 7.2 Bundle Detail (FMC Frontend)
- Single page per bundle accessible from the recommendation card and shareable deep link.
- Content:
  - Hero section with bundle image (if available), full description, capacity state, start/end dates (derived from earliest/latest session timestamps).
  - Specialisms list.
  - Session list rendered as collapsible mini-cards ordered chronologically (session timestamp ascending) showing date, time, duration, and coach.
- RSVP button reflects state:
  - `Join bundle` (default).
  - `Already registered` when RSVP status is `booked`.
  - Disabled with status message when bundle is full or all sessions expired.

### 7.3 RSVP & Cancellation (FMC Frontend → Backend)
- RSVP action calls `/coaching/bundles/{bundle_id}/rsvp` with journey/user context; backend:
  - Validates capacity and that first session has not expired (based on `first_session_timestamp`).
  - Registers the user for each session using existing group session RSVP flow.
  - Increments bundle participant count atomically.
  - Triggers per-session calendar invites and existing email confirmations.
- Cancellation uses `/coaching/bundles/{bundle_id}/rsvp` (DELETE); backend:
  - Cancels all underlying session RSVPs (calendar removal + notifications).
  - Decrements bundle participant count.
- Employees can cancel at any time, including after the course starts; no pricing implications need to surface.

## 8. Admin Experience
### 8.1 Bundle Management (Admin Frontend)
- Bundles table lists title, specialisms, session count, capacity (current/max), availability status (available / full / started / expired), and translation status.
- CRUD form allows:
  - Title, description, specialisms (min 1), max participants.
  - Assigning a coach (single coach shared by all sessions).
  - Selecting 2–20 sessions from existing catalog; validation prevents sessions already attached to another bundle.
  - Chronological ordering enforced automatically based on session timestamps (UI displays order and derived course duration).
  - Uploading/replacing a bundle image (S3-backed, same constraints as session images).
- Editing supports updating metadata, swapping sessions (while respecting min 2 sessions), or capacity adjustments.
- Deleting a bundle detaches associated sessions (sessions revert to standalone visibility).

### 8.2 Session Association Rules
- Sessions may be created independently and later attached to a bundle; bundles do not auto-create sessions.
- When a session is attached:
  - Session metadata gains `bundle_id`, `bundle_title`, `bundle_position`, `is_bundled`.
  - Session is hidden from standalone recommendations/lists.
- Removing a session updates metadata accordingly; UI must block removal if it would drop below 2 sessions.

### 8.3 Localization Workflow
- Admin translations leverage existing job queue:
  - On creation, backend kicks off translation job unless `job_id` provided.
  - Admin UI surfaces translation in-progress badge for supported locales (EN, ES, IT, DE, FR, PL).
  - Manual overrides follow the same translation editor patterns as sessions.

## 9. Data Model & System Interactions
### 9.1 DynamoDB Entities
- `SessionBundle` (new table): stores bundle metadata, ordered `session_ids`, capacity counts, computed timestamps, optional image key, translation job state.
- `GroupSession` (existing): extended with `bundle_id`, `bundle_position`, `bundle_title`, `is_bundled`.
- `Journey.bundle_rsvps` array records bundle-level RSVP status, mirroring per-session booking.

### 9.2 Services & APIs
- `BundleService` handles validation, CRUD, session linking, capacity updates, timestamp recomputation, translation job orchestration.
- `BundleMatchingService` integrates with journey recommendations, filtering out expired/started bundles based on `all_sessions_expired` and capacity.
- `bundleRoutes` provide REST endpoints consumed by Admin and FMC frontend:
  - `/coaching/bundles` (CRUD, filtering, image upload).
  - `/coaching/bundles/rsvps` (user RSVP lookup).
  - `/coaching/bundles/{bundle_id}/rsvp` (POST/DELETE for RSVP management).
  - `/coaching/bundles/{bundle_id}/participants` (capacity adjustments, Admin only).
- RSVP operations re-use `journeyService` to guarantee Nylas calendar invites and cancellations align with existing session flows.

### 9.3 Visibility Logic
- Bundles are visible when `!all_sessions_expired`; availability for RSVP determined by `current_participants < max_participants` and first session in the future.
- Sessions marked `is_bundled` are excluded from standalone session queries to avoid duplicate exposure.

## 10. Technical Considerations
- **Concurrency**: Use DynamoDB conditional updates for participant counts to avoid race conditions during concurrent RSVPs or cancellations.
- **Ordering**: `bundle_position` stored on sessions to persist display order; recompute after session additions/removals.
- **Translations**: Continue using translation job queue; ensure locale fallbacks (EN default) when translations incomplete.
- **Assets**: Bundle image upload/delete follows established presigned URL utilities with size/type validations.
- **Access Control**: Admin routes remain API-key protected; RSVP routes require authenticated employee tokens (JWT).
- **Error Handling**: Provide descriptive HTTP 400 errors for capacity/full, expired, or invalid session combinations.
- **Telemetry**: Extend existing logging (`logger` utilities) to capture bundle recommendation hits, RSVP attempts, and capacity thresholds.

## 11. Edge Cases & Failure States
- Sessions removed from a bundle while employees are registered: system should keep their individual RSVPs intact; admin must confirm before removal (soft validation).
- Calendar invite failures: follow existing retry/alert pathways; bundle RSVP should surface partial errors if some sessions fail to enqueue invites.
- Translation job failures: log and surface status in Admin for manual follow-up; bundle remains usable with source language copy.
- Bundle deletion with active RSVPs: block deletion until all employees are canceled or provide a soft warning with manual override (decision pending implementation details).

## 12. Rollout Strategy
- Launch via feature flag controlling bundle recommendation surface in FMC frontend to allow backend/Admin readiness first.
- Seed pilot bundles manually through Admin; verify content across locales before exposing to all employees.
- Monitor capacity metrics and RSVP error rates post-launch; iterate on matching thresholds if bundles dominate recommendations.

## 13. Analytics & Reporting
- Track bundle card impressions vs. detail view clicks vs. RSVPs.
- Monitor cancellation rates per bundle to validate flexibility requirements.
- Report on average participants per bundle and fill rate to inform program planning.