# Spectra Deployment Guide (Phase 6)

This document describes how to deploy Spectra in hosted mode with Clerk authentication
and Supabase persistence. **Local development requires no configuration** and works
exactly as before (Phase 4 behavior preserved).

---

## Architecture

```
User Browser  →  Vercel (Next.js frontend)  →  Railway/Fly.io (FastAPI backend)
                                                       ↓
                                              Supabase Postgres (metadata)
                                              Supabase Storage  (artifacts)
                 Clerk (authentication)
```

---

## Environment Variables

### Frontend (Vercel / `.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_SPECTRA_API_BASE` | Yes (hosted) | Backend API URL, e.g. `https://spectra-api.railway.app` |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Yes (hosted) | Clerk publishable key from dashboard |
| `CLERK_SECRET_KEY` | Yes (hosted) | Clerk secret key (server-side only) |

### Backend (Railway / Fly.io / `.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `SPECTRA_STORAGE_BACKEND` | Yes (hosted) | Set to `supabase` for hosted mode. Default: `local` |
| `SUPABASE_URL` | Yes (hosted) | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes (hosted) | Supabase service role key (server-side only) |
| `CLERK_ISSUER_URL` | Yes (hosted) | Clerk issuer URL for JWT verification, e.g. `https://my-app.clerk.accounts.dev` |
| `CLERK_JWKS_URL` | Optional | Explicit JWKS URL override (if not using `CLERK_ISSUER_URL`) |
| `SPECTRA_CORS_ORIGINS` | Recommended | Comma-separated allowed origins, e.g. `https://spectra.vercel.app` |
| `ANTHROPIC_API_KEY` | Optional | For LLM analyst features (Phase 4B) |

---

## Setup Steps

### 1. Supabase Project

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the migration file:

   ```
   supabase/migrations/001_phase6_schema.sql
   ```

   This creates:
   - `experiments` table with RLS
   - `runs` table with RLS
   - `artifacts` storage bucket with access policies
- **Important:** backend currently uses `SUPABASE_SERVICE_ROLE_KEY`, which bypasses RLS at DB layer.
  App code enforces owner scoping on every query (`owner_id` filters) as defense-in-depth.

3. Copy these values from **Settings → API**:
   - `SUPABASE_URL` (Project URL)
   - `SUPABASE_SERVICE_ROLE_KEY` (service_role key — keep secret)

### 2. Clerk Authentication

1. Create an application at [clerk.com](https://clerk.com)
2. Configure sign-in methods (email/password recommended)
3. Copy these values from **API Keys**:
   - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
   - `CLERK_SECRET_KEY`
4. Note your **Issuer URL** from **JWT Templates → Clerk** (or from the Clerk dashboard):
   - Usually `https://<your-app>.clerk.accounts.dev`
   - This becomes `CLERK_ISSUER_URL` for the backend

### 3. Deploy Backend (Railway)

1. Create a new Railway project
2. Connect your GitHub repo or use Railway CLI
3. Set the root directory to the repo root
4. Set the start command:

   ```bash
   cd web/backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

5. Add environment variables:
   - `SPECTRA_STORAGE_BACKEND=supabase`
   - `SUPABASE_URL=<your-url>`
   - `SUPABASE_SERVICE_ROLE_KEY=<your-key>`
   - `CLERK_ISSUER_URL=<your-clerk-issuer>`
   - `SPECTRA_CORS_ORIGINS=https://spectra.vercel.app`

6. Install Python dependencies:
   ```bash
   pip install -e ".[web]" supabase "PyJWT[crypto]"
   ```

### 4. Deploy Frontend (Vercel)

1. Create a new Vercel project
2. Set the root directory to `web/frontend`
3. Add environment variables:
   - `NEXT_PUBLIC_SPECTRA_API_BASE=https://spectra-api.railway.app`
   - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=<your-key>`
   - `CLERK_SECRET_KEY=<your-secret>`

4. Deploy

### 5. Verify

1. Visit your Vercel URL
2. Sign up with a new account
3. Create an experiment using a dataset preset
4. Run the experiment and verify results appear
5. Sign out, sign in with a different account, verify the experiment list is empty

---

## Local Development (unchanged)

No configuration needed. The system defaults to:
- `SPECTRA_STORAGE_BACKEND=local` (file-based persistence)
- No auth (all requests return `anonymous` owner_id)

```bash
# Terminal 1: Backend
cd web/backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd web/frontend
npm run dev
```

---

## Storage Path Convention

When hosted, artifacts are stored in Supabase Storage following this deterministic path:

```
artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/results.json
artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/analysis.json
artifacts/{owner_id}/experiments/{experiment_id}/runs/{run_id}/plots/*
```

---

## Database Schema

See `supabase/migrations/001_phase6_schema.sql` for the complete schema including:
- `experiments` table (id, owner_id, name, created_at, config)
- `runs` table (id, experiment_id, owner_id, status, created_at, results_key, analysis_key)
- Row Level Security policies on both tables
- Storage bucket with access policies

---

## Troubleshooting

**Backend returns 401 Unauthorized:**
- Verify `CLERK_ISSUER_URL` is correct
- Verify the Clerk publishable key matches the issuer

**CORS errors in browser:**
- Add your Vercel domain to `SPECTRA_CORS_ORIGINS`

**"supabase package required" error:**
- Install: `pip install supabase`

**"PyJWT[crypto] required" error:**
- Install: `pip install "PyJWT[crypto]"`

**Local dev still works without any env vars:**
- Yes, this is intentional. Phase 4 behavior is preserved exactly.
