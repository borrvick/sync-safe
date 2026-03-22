# Sync-Safe 2.0 — Migration Master Plan
## From Streamlit/HuggingFace → Django/Modal/Fly.io

---

# PART 1: WHAT TO PROVIDE BEFORE STARTING

Before any agent begins work, the following must be ready. Do not start
the migration until this checklist is complete.

## Accounts to Create
- [ ] Fly.io account (fly.io) — for Django app hosting
- [ ] Neon account (neon.tech) — for Postgres database
- [ ] Upstash account (upstash.com) — for Redis (Celery task queue)
- [ ] Modal account (modal.com) — for GPU ML worker
- [ ] GitHub OAuth App — Client ID + Secret (github.com/settings/developers)
- [ ] Google OAuth App — Client ID + Secret (console.cloud.google.com)
- [ ] Vercel account (vercel.com) — for Next.js frontend

## Values to Gather from Existing App
- [ ] Last.fm API key (currently in .env or st.secrets)
- [ ] Any other API keys currently in use
- [ ] HuggingFace Space URL (the current live app URL)
- [ ] Current allin1 model name (from ModelParams in core/config.py)
- [ ] Current Whisper model size (from config)

## Decisions to Make Upfront
- [ ] Frontend choice: Next.js (recommended) OR Django templates + HTMX
- [ ] Auth provider: django-allauth (recommended) OR Supabase Auth
- [ ] Pricing model: free only / freemium / paid tiers (affects user model design)
- [ ] Will HuggingFace Space stay live during migration? (recommended: yes)

## Codebase to Provide
- [ ] Full existing repo access (current music-sync-app)
- [ ] All files in services/ (forensics.py, analysis.py, compliance.py,
      nlp.py, ingestion.py, discovery.py, legal.py)
- [ ] All files in core/ (models.py, config.py, protocols.py, exceptions.py)
- [ ] All files in tests/ including fixtures/
- [ ] CLAUDE.md

---

# PART 2: AGENT TEAM DEFINITION

Each agent has a defined role, scope, and veto power. Run them in the
order listed. No agent should write code outside its defined scope.

---

## AGENT 1 — Senior Architect
**Trigger:** Run first, before any code is written.
**Scope:** Design review only. No code output.
**Responsibilities:**
- Review existing services/ and core/ and produce a service dependency map
- Define the Django app structure (which Django apps to create)
- Define the Modal worker API contract (inputs, outputs, error shapes)
- Define the data model schema (all tables, relationships, indexes)
- Define the async job flow (request → Celery → Modal → webhook → DB)
- Identify any SOLID violations in the migration plan before they are built
- Produce a written Architecture Decision Record (ADR) that all other
  agents must follow

**Output required before proceeding:** Written ADR document covering:
  1. Django app layout
  2. All database models with field types
  3. Modal endpoint signatures
  4. API endpoint list with HTTP methods and auth requirements
  5. Any risks or blockers identified

**Prompt to use:**
```
You are a Senior Software Architect with 15 years experience in Django,
Python microservices, and ML infrastructure. Your job is design review
ONLY — produce no implementation code.

Review the attached codebase (services/, core/, tests/, CLAUDE.md) and
produce a full Architecture Decision Record for migrating this Streamlit
app to Django + Modal + Fly.io + Neon Postgres + Next.js.

The ADR must include:
1. Django project layout — list every Django app, what it owns, what it
   does NOT own
2. Full database schema — every table, every field with type, every
   foreign key, every index needed for performance
3. Modal worker contract — function signatures for each ML service
   (forensics, analysis, NLP, compliance) with typed inputs/outputs
4. Full API endpoint list — path, HTTP method, auth required, request
   body, response shape
5. Async job flow diagram — how a user submits audio, how the job
   reaches Modal, how results come back, how the frontend knows it's done
6. Security surface area — every place user input touches the system,
   every auth boundary, every place secrets are used
7. Risk register — anything in the current codebase that will be hard
   to migrate, any tech debt that should be resolved now vs later

Format as a structured markdown document. Do not write any Django code,
Modal code, or configuration files. This is a design document only.
Flag anything that violates SOLID principles or introduces security risk.
```

---

## AGENT 2 — Infrastructure Engineer
**Trigger:** Run after Architect ADR is approved.
**Scope:** All infrastructure config. No application logic.
**Responsibilities:**
- Scaffold the Django project skeleton (settings, urls, wsgi, asgi)
- Configure Fly.io (fly.toml, Dockerfile, .dockerignore)
- Configure Neon Postgres connection (DATABASE_URL, connection pooling)
- Configure Upstash Redis connection (CELERY_BROKER_URL)
- Configure Modal project (modal app, secrets, GPU type)
- Set up GitHub Actions CI/CD pipeline
- Configure environment variable management (.env.example, no secrets
  in code, all secrets via environment)
- Configure Django settings for dev/staging/production split

**Hard rules:**
- Never hardcode secrets. All secrets via environment variables.
- Use django-environ or pydantic-settings for config — no raw os.getenv
  scattered through the codebase
- Dockerfile must be production-grade (non-root user, minimal image,
  health check endpoint)
- All infrastructure must be reproducible from code (no manual clicks)

**Prompt to use:**
```
You are a Senior Infrastructure Engineer specializing in Django
deployments, containerization, and cloud-native Python applications.

Using the Architecture Decision Record provided, scaffold the complete
infrastructure for this Django application. The target stack is:
- Django 5.x application hosted on Fly.io
- Neon serverless Postgres (DATABASE_URL provided)
- Upstash Redis for Celery broker (REDIS_URL provided)
- Modal for GPU ML worker (separate — do not configure here)
- GitHub Actions for CI/CD

Produce:
1. Django project skeleton with settings split:
   - base.py (shared settings)
   - development.py (DEBUG=True, console email backend)
   - production.py (security headers, HTTPS-only, production DB)
2. fly.toml — configured for Django with health check, auto-scaling off
   on free tier, correct memory/CPU for the app
3. Dockerfile — multi-stage build, non-root user, minimal final image,
   gunicorn server, health check at /health/
4. .github/workflows/deploy.yml — run tests, then deploy to Fly.io on
   push to main
5. requirements/base.txt, requirements/dev.txt, requirements/prod.txt
6. .env.example — every required environment variable documented,
   no actual values
7. core/config.py replacement — pydantic-settings BaseSettings class
   reading from environment, replaces the existing CONSTANTS pattern

Security requirements:
- ALLOWED_HOSTS must be locked down
- CSRF protection must be enabled
- SECRET_KEY must come from environment, never hardcoded
- Database connection must use SSL in production
- Static files must be served via whitenoise or CDN, not Django in prod

Do not write any Django models, views, or business logic. Infrastructure
only.
```

---

## AGENT 3 — Security Auditor
**Trigger:** Run after every other agent completes a feature. This agent
reviews, never builds.
**Scope:** Security review across all code produced in the migration.
**Responsibilities:**
- Review every authentication flow for vulnerabilities
- Review every file upload handler for path traversal, malware upload,
  size limit bypass
- Review every database query for SQL injection risk (even with ORM)
- Review CORS configuration
- Review session management
- Review rate limiting coverage
- Review secret handling
- Produce a security report with PASS/FAIL/WARN per item
- Block any feature from merging that has a FAIL

**Checklist the auditor must verify for every PR:**
```
AUTH
[ ] All views require login except explicitly public ones
[ ] JWT tokens expire appropriately
[ ] Password reset tokens are single-use and time-limited
[ ] OAuth state parameter is validated (CSRF on OAuth flow)
[ ] Session fixation is prevented

INPUT VALIDATION
[ ] All file uploads validate MIME type server-side (not just extension)
[ ] All file uploads enforce max size server-side
[ ] Audio files are processed in isolated temp directories with cleanup
[ ] No user input is passed to shell commands without shlex.quote()
[ ] All URL parameters are validated before DB queries

DATABASE
[ ] No raw SQL queries — ORM only, or parameterized queries
[ ] Users can only access their own data (row-level filtering on every query)
[ ] No sensitive data (raw audio bytes, full API keys) stored in DB

API
[ ] Rate limiting on all endpoints (especially /analyze — GPU is expensive)
[ ] CORS whitelist is explicit, not wildcard in production
[ ] API keys are never returned in responses
[ ] Error messages do not leak stack traces in production

SECRETS
[ ] No secrets in code, git history, or logs
[ ] All API keys scoped to minimum required permissions
[ ] Database password rotatable without app downtime
```

**Prompt to use:**
```
You are a Senior Application Security Engineer specializing in Django
web applications and Python ML pipelines. Your role is security review
ONLY — you do not write features.

Review the code provided against OWASP Top 10 and Django security
best practices. Use the checklist above as your minimum bar.

For every finding produce:
  SEVERITY: CRITICAL / HIGH / MEDIUM / LOW / INFO
  LOCATION: file:line
  FINDING: what the vulnerability is
  EXPLOIT: how it could be abused
  FIX: exact code change required

CRITICAL and HIGH findings must be fixed before the feature merges.
MEDIUM findings must have a GitHub issue created.
LOW and INFO are advisory.

Do not approve any feature that has a CRITICAL or HIGH finding unresolved.
Be especially rigorous on: file upload handling, auth flows, and any
place user-controlled data touches the filesystem or a shell command.
```

---

## AGENT 4 — Django Backend Engineer
**Trigger:** Run after Infrastructure is complete and ADR is approved.
**Scope:** Django models, views, DRF serializers, Celery tasks, URL routing.
**Responsibilities:**
- Port core/models.py to Django ORM models
- Write all database migrations
- Build DRF serializers and viewsets
- Set up Celery tasks for async ML job dispatch
- Write webhook handler for Modal job completion callbacks
- Implement row-level data isolation (users only see their own analyses)
- Port existing pure functions from services/ as Django service layer

**Hard rules:**
- Every model must have created_at, updated_at timestamps
- Every queryset that returns user data must filter by request.user
- No business logic in views — views call service functions only
- All Celery tasks must be idempotent
- All file handling must use try/finally to clean up temp files

**Django apps to create (from ADR):**
```
sync_safe/          ← Django project root
├── accounts/       ← User model, auth, profile
├── analyses/       ← Analysis jobs, results, labels
├── billing/        ← Tiers, usage limits (stub for now)
└── core/           ← Shared utilities, base models, exceptions
```

**Prompt to use:**
```
You are a Senior Django Engineer. Using the Architecture Decision Record
and the existing codebase (attached), implement the Django backend.

Existing codebase context:
- core/models.py defines: AudioBuffer, AnalysisResult, ForensicsResult,
  StructureResult, ComplianceReport, TranscriptSegment, Section
- services/ contains pure Python ML service classes
- tests/ contains pytest tests including fixture-based regression tests
- The app currently has NO database and NO auth

Your implementation must:
1. Create Django ORM models that map to the existing Pydantic models —
   preserve all field names so JSON serialization stays compatible
2. Write django migrations for all models
3. Build DRF viewsets with proper permission classes:
   - POST /api/analyses/ — submit new analysis job (authenticated)
   - GET /api/analyses/ — list user's own analyses only
   - GET /api/analyses/{id}/ — retrieve single result (owner only)
   - PATCH /api/analyses/{id}/label/ — set category label
   - GET /health/ — public health check for Fly.io
4. Write a Celery task `dispatch_analysis(analysis_id)` that:
   - Loads the Analysis object
   - Calls Modal worker via HTTP
   - Updates Analysis status (pending → processing → complete/failed)
   - Stores result JSON on completion
5. Port the existing services/ as a Django service layer — these are
   NOT called directly in production (Modal does that), but are used
   in tests and local dev without Modal
6. Write unit tests for every model method and serializer

Constraints from CLAUDE.md:
- No bare except clauses — use domain-specific exceptions
- All temp files cleaned up with try/finally
- No magic numbers — constants in settings
- Constructor injection only, no global state
- Type hints on every function signature

Do not build auth (that's a separate agent). Assume request.user is
always available and authenticated.
```

---

## AGENT 5 — Modal ML Worker Engineer
**Trigger:** Run in parallel with Django Backend Agent.
**Scope:** Modal app that wraps existing services/.
**Responsibilities:**
- Port services/forensics.py → Modal function
- Port services/analysis.py → Modal function
- Port services/nlp.py → Modal function
- Port services/compliance.py → Modal function
- Port services/ingestion.py → Modal function (for URL ingestion)
- Implement job completion webhook back to Django
- Handle GPU selection (A10G for allin1/Whisper, CPU for others)
- Implement proper error handling and retry logic

**Hard rules:**
- Each Modal function must be fully self-contained (no shared GPU state)
- Audio bytes must never be logged or stored by Modal
- Each function must accept a job_id and POST results back to Django
  webhook on completion or failure
- All existing @spaces.GPU patterns must be preserved as Modal equivalents

**Prompt to use:**
```
You are a senior Python engineer specializing in Modal.com deployments
and ML inference pipelines.

Port the following existing services to Modal functions. The existing
code is attached — preserve all business logic exactly, only change
the infrastructure wrapper.

Existing services to port:
- services/forensics.py → @app.function(cpu=2, memory=4096)
- services/analysis.py → @app.function(gpu="A10G", timeout=600)
- services/nlp.py → @app.function(gpu="A10G", timeout=300)
- services/compliance.py → @app.function(cpu=2, memory=2048)
- services/ingestion.py (URL download only) → @app.function(cpu=1)

Each function must:
1. Accept: job_id (str), audio_bytes (bytes), config (dict)
2. Run the corresponding service
3. POST results back to {DJANGO_BASE_URL}/api/webhooks/analysis-complete/
   with body: {job_id, status: "complete"|"failed", result: {...}, error: "..."}
4. Never raise unhandled exceptions — always POST failure webhook instead

The webhook endpoint on Django will be authenticated with a shared
MODAL_WEBHOOK_SECRET that must be passed as a header.

Preserve these behaviors from the existing code:
- MPS → CPU fallback retry in analysis.py _analyse_structure
- All try/finally temp file cleanup
- ModelInferenceError with context dict for logging
- The CAPTURE_OUTPUTS env gate must be REMOVED (Modal is not local dev)

Write a modal_app/main.py that registers all functions and a
modal_app/client.py that Django uses to trigger jobs.
```

---

## AGENT 6 — Auth Engineer
**Trigger:** Run after Django Backend skeleton is complete.
**Scope:** Auth only — login, register, OAuth, sessions, permissions.
**Responsibilities:**
- django-allauth setup with email + Google + GitHub providers
- Custom User model (AbstractUser) with extra fields
- Email verification flow
- Password reset flow
- JWT token auth for API (via djangorestframework-simplejwt)
- Login/register API endpoints for Next.js frontend
- Rate limiting on auth endpoints

**Hard rules:**
- Custom User model must be defined BEFORE first migration runs —
  this cannot be changed later without a full DB reset
- Passwords must never be logged, stored plain, or returned in responses
- OAuth tokens must not be stored (only use them to get user identity)
- All auth endpoints must have rate limiting (django-ratelimit or similar)
- Email enumeration must be prevented (same response whether email
  exists or not on forgot password)

**Prompt to use:**
```
You are a senior Django security engineer specializing in authentication
systems. Implement the complete auth layer for this Django application.

Requirements:
1. Custom User model in accounts/models.py extending AbstractUser:
   - id: UUID (not integer — harder to enumerate)
   - email: unique, required (login identifier)
   - username: optional display name
   - tier: CharField choices=['free','pro'] default='free'
   - created_at, updated_at timestamps

2. django-allauth configuration:
   - Email/password registration with email verification required
   - Google OAuth2 provider
   - GitHub OAuth2 provider
   - ACCOUNT_EMAIL_REQUIRED = True
   - ACCOUNT_USERNAME_REQUIRED = False

3. DRF + simplejwt for API auth:
   - POST /api/auth/token/ — email + password → access + refresh tokens
   - POST /api/auth/token/refresh/ — refresh → new access token
   - POST /api/auth/register/ — create account + send verification email
   - POST /api/auth/password/reset/ — initiate password reset
   - GET /api/auth/me/ — current user profile

4. Rate limiting:
   - /api/auth/token/ — 5 attempts per minute per IP
   - /api/auth/register/ — 3 per hour per IP
   - /api/auth/password/reset/ — 3 per hour per IP

5. Security requirements:
   - Password reset tokens expire in 1 hour
   - Email verification tokens expire in 24 hours
   - Prevent email enumeration on all auth endpoints
   - Log auth events (login, logout, failed attempt, password reset)
     to the existing PipelineLogger pattern — no PII in logs

Write full tests for every auth endpoint including failure cases,
rate limit behavior, and token expiry.
```

---

## AGENT 7 — Frontend Engineer (Next.js)
**Trigger:** Run after Auth and Backend API are complete and tested.
**Scope:** Next.js frontend only.
**Responsibilities:**
- Port all Streamlit UI to Next.js React components
- Auth pages (login, register, OAuth callback, forgot password)
- Dashboard (list of user's past analyses)
- Upload/analyze flow with real-time progress
- Report page (port from ui/pages/report.py)
- Audio player with timestamp navigation
- Responsive layout

**Hard rules:**
- Never store JWT tokens in localStorage — use httpOnly cookies or
  memory + refresh token rotation
- All API calls go through a single apiClient module
- No hardcoded API URLs — use NEXT_PUBLIC_API_URL env var
- Audio bytes are never sent to Next.js — stay on Django/Modal side

**Prompt to use:**
```
You are a senior React/Next.js engineer. Port the existing Streamlit
application UI to a Next.js 14 app with App Router.

Existing UI reference: ui/pages/report.py and ui/pages/loading.py
are attached. Reproduce the same information architecture and all
displayed data — not the exact visual style, but all the data fields,
sections, and interactivity.

Pages to build:
1. /login — email/password form + Google + GitHub OAuth buttons
2. /register — registration form with email verification messaging
3. /dashboard — list of user's past analyses, verdict badge, timestamp
4. /analyze — file upload OR YouTube URL input, submit button,
   real-time progress indicator (poll GET /api/analyses/{id}/ for status)
5. /report/[id] — full analysis report (port of report.py):
   - Audio player with timestamp click navigation
   - Track Overview (BPM, key, sections, metadata)
   - Authenticity Audit (verdict, flags, scores)
   - Sync Readiness Checks (compliance flags)
   - Discovery & Licensing section
   - Lyrics & Content Audit table

Technical requirements:
- Auth: JWT stored in memory, refresh token in httpOnly cookie
- API calls: single src/lib/apiClient.ts with interceptor for token refresh
- Real-time progress: poll /api/analyses/{id}/ every 2s while status
  is "pending" or "processing", stop on "complete" or "failed"
- Audio player: use native HTML5 audio element with a ref,
  seek on timestamp click — do not use a Streamlit workaround
- Responsive: works on mobile and desktop

Do not build any backend code. Consume the DRF API only.
Reference the existing CLAUDE.md for business logic context.
```

---

## AGENT 8 — Test & Parity Validator
**Trigger:** Run after each Epic is complete.
**Scope:** Testing only. No features.
**Responsibilities:**
- Port all existing pytest tests to the new stack
- Write API integration tests
- Write auth flow tests
- Validate that all 10 forensics regression fixtures still produce
  the same verdicts in the new stack
- Validate that API responses match the shape expected by the frontend
- Produce a parity report: for each existing Streamlit feature, confirm
  it works in the new app

**Prompt to use:**
```
You are a senior QA engineer and test automation specialist.

Your job is to validate that the Django migration is feature-complete
and that no regression has been introduced.

Existing tests to port (attached):
- tests/test_forensics.py — 46 tests including regression fixtures
- tests/test_compliance.py — 44 tests
- tests/conftest.py — fixture loaders

Tasks:
1. Port all existing pytest tests to work against the new Django app.
   The service layer (services/) is unchanged — tests that test pure
   functions should still pass without modification.

2. Write Django test cases for every API endpoint:
   - Auth required → returns 401 without token
   - Wrong user → returns 403 (not 404, to prevent enumeration)
   - Valid request → returns correct shape
   - Invalid input → returns 400 with descriptive error

3. Write an end-to-end parity checklist:
   For each of the following Streamlit features, write a test or
   describe the manual verification step:
   - File upload → analysis completes → verdict displayed
   - YouTube URL → analysis completes → verdict displayed
   - Timestamp click → audio player seeks to position
   - Category label saved and persists on reload
   - Past analyses visible in dashboard
   - User can only see their own analyses

4. Run the existing 10-fixture regression suite and confirm all verdicts
   match _EXPECTED_VERDICTS in test_forensics.py. If any have changed,
   flag immediately — do not update the expected values without review.

Report format: markdown table with columns:
  Feature | Test Type | Status (PASS/FAIL/MANUAL) | Notes
```

---

# PART 3: EPICS AND STORIES

## Epic 1 — Foundation & Infrastructure
**Goal:** Deployable skeleton with no features but production-ready config.

| Story | Agent | Priority |
|---|---|---|
| Django project scaffold with settings split | Infra | P0 |
| Fly.io Dockerfile + fly.toml | Infra | P0 |
| Neon Postgres connection + health check | Infra | P0 |
| Upstash Redis + Celery worker config | Infra | P0 |
| GitHub Actions CI (test + deploy) | Infra | P0 |
| Environment variable management (.env.example) | Infra | P0 |
| pydantic-settings config replacing CONSTANTS | Infra | P0 |
| Architecture Decision Record | Architect | P0 |

---

## Epic 2 — Auth & User Accounts
**Goal:** Users can register, log in, and their data is isolated.

| Story | Agent | Priority |
|---|---|---|
| Custom User model (UUID, email, tier) | Auth | P0 |
| Email/password register + verify | Auth | P0 |
| Login → JWT access + refresh tokens | Auth | P0 |
| Google OAuth2 | Auth | P1 |
| GitHub OAuth2 | Auth | P1 |
| Password reset flow | Auth | P1 |
| Rate limiting on all auth endpoints | Auth | P0 |
| Auth security audit | Security | P0 |

---

## Epic 3 — Data Models & Storage
**Goal:** AnalysisResult and all related data persists in Postgres.

| Story | Agent | Priority |
|---|---|---|
| Analysis model (maps to AnalysisResult) | Backend | P0 |
| ForensicsResult, StructureResult sub-models | Backend | P0 |
| TrackLabel model (replaces labels.json) | Backend | P0 |
| Row-level isolation (user → own analyses only) | Backend | P0 |
| Database migrations | Backend | P0 |
| DRF serializers for all models | Backend | P0 |

---

## Epic 4 — ML Worker (Modal)
**Goal:** All GPU processing runs on Modal, results return via webhook.

| Story | Agent | Priority |
|---|---|---|
| Modal project setup + secrets config | Modal | P0 |
| Forensics function (CPU) | Modal | P0 |
| Analysis function (GPU — allin1) | Modal | P0 |
| NLP/Whisper function (GPU) | Modal | P0 |
| Compliance function (CPU) | Modal | P0 |
| Ingestion function (URL download) | Modal | P1 |
| Webhook POST back to Django on complete/fail | Modal | P0 |
| Django webhook receiver + Celery status update | Backend | P0 |
| MPS → CPU fallback preserved from analysis.py | Modal | P0 |

---

## Epic 5 — API Layer
**Goal:** Full REST API consumed by Next.js frontend.

| Story | Agent | Priority |
|---|---|---|
| POST /api/analyses/ — submit job | Backend | P0 |
| GET /api/analyses/ — list own results | Backend | P0 |
| GET /api/analyses/{id}/ — poll for status | Backend | P0 |
| PATCH /api/analyses/{id}/label/ — set category | Backend | P1 |
| POST /api/webhooks/analysis-complete/ — Modal callback | Backend | P0 |
| GET /health/ — Fly.io health check | Backend | P0 |
| API rate limiting (analyze endpoint especially) | Backend | P0 |
| API security audit | Security | P0 |

---

## Epic 6 — Frontend (Next.js)
**Goal:** All Streamlit pages replaced with React equivalents.

| Story | Agent | Priority |
|---|---|---|
| Auth pages (login, register, forgot password) | Frontend | P0 |
| Dashboard (list past analyses) | Frontend | P0 |
| Upload/analyze flow + progress polling | Frontend | P0 |
| Report page — Track Overview section | Frontend | P0 |
| Report page — Authenticity Audit section | Frontend | P0 |
| Report page — Sync Readiness section | Frontend | P1 |
| Report page — Discovery & Licensing section | Frontend | P1 |
| Report page — Lyrics & Content Audit | Frontend | P1 |
| Audio player with timestamp seek | Frontend | P0 |
| Category label dropdown (replaces capture hack) | Frontend | P1 |
| Frontend security audit (token storage, CORS) | Security | P0 |

---

## Epic 7 — Testing & Parity
**Goal:** Full test suite, all existing verdicts validated, no regressions.

| Story | Agent | Priority |
|---|---|---|
| Port test_forensics.py to new stack | Test | P0 |
| Port test_compliance.py to new stack | Test | P0 |
| API endpoint integration tests | Test | P0 |
| Auth flow tests (all paths including failure) | Test | P0 |
| 10-fixture forensics regression suite passes | Test | P0 |
| Frontend E2E parity checklist | Test | P1 |
| Security audit final pass | Security | P0 |

---

# PART 4: SECURITY REQUIREMENTS (NON-NEGOTIABLE)

These must be satisfied before any production deployment.

## Authentication
- UUID primary keys on User model — integer IDs are enumerable
- JWT access tokens expire in 15 minutes
- Refresh tokens expire in 7 days, rotated on each use
- Email verification required before first analysis allowed
- Maximum 5 failed login attempts per IP per minute before lockout

## Data Isolation
- Every queryset that returns Analysis data must filter by user=request.user
- Webhook endpoint authenticated by MODAL_WEBHOOK_SECRET header
- Users cannot access other users' audio, results, or labels under any path

## File Upload
- Server-side MIME type validation (not extension — extensions are spoofable)
- Maximum file size enforced server-side (not just frontend)
- Audio processed only in isolated tempfile.TemporaryDirectory
  with try/finally cleanup — never written to a permanent path
- File content never stored in DB — only extracted metadata and result JSON

## Secrets
- All secrets via environment variables — none in code or git history
- Minimum-scope API keys (Last.fm read-only, etc.)
- Database password must be rotatable without code change
- MODAL_WEBHOOK_SECRET must be a random 32-byte hex string

## Production Hardening
- DEBUG = False in production (Fly.io env)
- ALLOWED_HOSTS locked to your domain(s) only
- HTTPS enforced (SECURE_SSL_REDIRECT = True)
- HSTS header enabled
- Error pages must not reveal stack traces
- Logging must redact audio bytes, API keys, and full URLs

---

# PART 5: MIGRATION EXECUTION ORDER

Run agents in this sequence. Do not start the next phase until the
current phase passes security audit.

```
Phase 1 (Foundation)
  → Architect Agent      [ADR — design review]
  → Infrastructure Agent [skeleton + config]
  → Security Audit       [config review]

Phase 2 (Auth + Data)
  → Auth Agent           [user model + auth endpoints]
  → Backend Agent        [models + API endpoints]  ← run in parallel with Auth
  → Security Audit       [auth flows + data isolation]
  → Test Agent           [auth tests + model tests]

Phase 3 (ML Worker)
  → Modal Agent          [port services/ to Modal functions]
  → Backend Agent        [webhook receiver + Celery tasks]
  → Test Agent           [regression suite on new stack]

Phase 4 (Frontend)
  → Frontend Agent       [Next.js UI]
  → Security Audit       [token handling + CORS]
  → Test Agent           [parity validation]

Phase 5 (Launch)
  → Full security audit  [final pass across all epics]
  → Cutover: point domain to Fly.io, keep HF Space as fallback for 2 weeks
  → Decommission HF Space after 2 weeks with no incidents
```

---

# PART 6: THINGS THE AI MUST NOT DO

Include this in every agent prompt to prevent common mistakes.

```
HARD CONSTRAINTS FOR ALL AGENTS:

1.  Never auto-commit. Propose commits and wait for explicit approval.
2.  Never push to main directly. All changes via feature branches.
3.  Never store audio bytes in the database or log files.
4.  Never use sqlite in production — Neon Postgres only.
5.  Never use integer primary keys on the User model.
6.  Never put secrets in code, even as example values.
7.  Never use st.html() or any Streamlit APIs — this is a Django app.
8.  Never use global state or singletons — constructor injection only.
9.  Never write bare except clauses — use domain-specific exceptions.
10. Never skip the Security Auditor review — it has veto power.
11. Never remove the MPS → CPU fallback in analysis.py logic.
12. Never change _EXPECTED_VERDICTS in test_forensics.py without
    a written explanation of why the verdict changed.
13. Never write a queryset that returns other users' data.
14. Before implementing any service, provide a 3-sentence summary
    of the proposed design and wait for OK.
```

---

# PART 7: QUICK REFERENCE — CURRENT CODEBASE MAP

Provide this to every agent for context.

```
Current stack:
  UI:         Streamlit (app.py + ui/pages/loading.py + ui/pages/report.py)
  Services:   services/ — forensics, analysis, nlp, compliance, ingestion,
              discovery, legal
  Models:     core/models.py — Pydantic models (AudioBuffer, AnalysisResult,
              ForensicsResult, StructureResult, ComplianceReport, etc.)
  Config:     core/config.py — CONSTANTS + ModelParams (pydantic)
  Logging:    core/logging.py — PipelineLogger (NDJSON, thread-safe)
  Exceptions: core/exceptions.py — ModelInferenceError, AudioSourceError
  Tests:      tests/ — test_forensics.py (46 tests), test_compliance.py (44 tests),
              conftest.py, fixtures/forensics/ (10 JSON fixtures)
  Hosting:    HuggingFace Spaces ZeroGPU (free tier, 25 min/day GPU)

What is NOT changing:
  - All business logic in services/ (forensics algorithms, compliance
    checks, key detection, etc.) — these are pure Python and move over
    unchanged
  - core/exceptions.py — exception types are reused
  - tests/fixtures/forensics/ — regression baselines stay the same
  - _EXPECTED_VERDICTS in test_forensics.py — verdicts must not drift

What IS changing:
  - Streamlit UI → Next.js
  - HF Spaces → Fly.io (Django) + Modal (GPU worker)
  - Pydantic models → Django ORM models (with Pydantic kept for
    ML service layer)
  - Ephemeral state → Postgres persistence
  - No auth → django-allauth + JWT
  - CONSTANTS → pydantic-settings from environment
```
