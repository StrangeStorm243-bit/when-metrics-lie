# Spectra (when-metrics-lie) — Project Context for Claude Code

## What This Is

Spectra (package name: `metrics_lie`) is a **scenario-first ML evaluation engine** that surfaces where metrics lie by stress-testing models across realistic failure modes. It produces transparent, decision-oriented comparisons rather than headline metrics. Core engine at **v0.2.0**, Web API at **v0.3.0**.

**Philosophy**: Evaluation failures are often more dangerous than model failures. Prioritize robustness, visibility, and decision safety.

## Architecture

```
CLI (argparse) / Next.js Frontend
              ↓
       FastAPI Backend (web/)
              ↓
         Engine Bridge
              ↓
   Core Engine (execution.py)
       ├─→ Dataset Loading
       ├─→ Model Adapter (sklearn/pickle)
       ├─→ Scenario Runner (Monte Carlo trials)
       │     ├─→ label_noise, score_noise
       │     ├─→ class_imbalance, threshold_gaming
       │     └─→ Metrics (AUC, F1, Brier, ECE, etc.)
       ├─→ Diagnostics (calibration, gaming, subgroups)
       ├─→ Analysis (dashboard, disagreement, sensitivity)
       ├─→ Decision Framework (scorecard, components)
       ├─→ Artifacts (matplotlib plots)
       └─→ Database (SQLAlchemy → SQLite / Supabase)
```

### Key Directories

- `src/metrics_lie/` — Core engine (Python package)
  - `execution.py` — Main pipeline orchestrator
  - `runner.py` — Scenario runner & metric computation
  - `spec.py` — ExperimentSpec validation (Pydantic)
  - `schema.py` — Output schemas (ResultBundle, MetricSummary)
  - `cli.py` — CLI entry point (argparse, `spectra` command)
  - `cli_format.py` — Table formatting & string truncation helpers for CLI output
  - `validation.py` — Shared data validation helpers (arrays, datasets, surfaces)
  - `surface_compat.py` — Surface type routing & scenario compatibility filtering
  - `contract_check.py` — Spec loading & ResultBundle contract validation
  - `worker.py` — Job queue processing (claim job, execute, mark complete/failed)
  - `metrics/` — Metric implementations & registry (AUC, F1, accuracy, logloss, Brier, ECE, matthews_corrcoef, PR-AUC)
    - `core.py` — Metric functions (sklearn-based) & threshold/ranking/calibration categories
    - `applicability.py` — MetricResolver: filter metrics by surface type & dataset properties
    - `registry.py` — Declarative MetricRequirement definitions
  - `scenarios/` — Stress-test scenarios & registry (label_noise, score_noise, class_imbalance, threshold_gaming)
    - `base.py` — Scenario Protocol & ScenarioContext
    - `registry.py` — ScenarioFactory registry pattern
    - `label_noise.py`, `score_noise.py`, `class_imbalance.py`, `threshold_gaming.py`
  - `diagnostics/` — Calibration, metric gaming, subgroup analysis
    - `calibration.py` — Brier score & ECE computation
    - `metric_gaming.py` — Threshold optimization & accuracy inflation detection
    - `subgroups.py` — Per-group metrics, calibration, gap analysis
  - `analysis/` — Dashboard, disagreement, failure modes, sensitivity, threshold sweep
    - `dashboard.py` — Multi-metric risk summary (Phase 8)
    - `disagreement.py` — Metric disagreement analysis (pairwise threshold metrics)
    - `failure_modes.py` — Worst-case scenario + metric identification
    - `sensitivity.py` — Perturbation sensitivity ranking
    - `threshold_sweep.py` — Threshold optimization curves & crossover points
  - `decision/` — Decision components, scoring, scorecard
    - `components.py` — DecisionComponents dataclass
    - `extract.py` — Extract components from comparison report (aggregation strategies)
    - `score.py` — Weighted linear scoring of components
    - `scorecard.py` — DecisionScorecard with top contributors
  - `compare/` — Run comparison logic & rules
    - `compare.py` — compare_runs() with regression detection & decision logic
    - `loader.py` — Load & parse ResultBundles for comparison
    - `rules.py` — Regression heuristic thresholds (calibration, subgroup, metric, gaming)
  - `model/` — Model adapter, PredictionSurface, sources
    - `adapter.py` — ModelAdapter: load sklearn/pickle, detect capabilities, get surfaces
    - `surface.py` — PredictionSurface (probability/score/label), SurfaceType, validation
    - `sources.py` — ModelSource protocol: pickle, import, callable loading + MD5 hash
    - `errors.py` — CapabilityError, ModelNotFittedError, SurfaceValidationError
  - `experiments/` — Experiment registry (JSONL), identity (canonical JSON), fingerprinting
    - `definition.py` — ExperimentDefinition from spec
    - `identity.py` — Deterministic canonical_json, sha256, short_id
    - `datasets.py` — Dataset fingerprinting (CSV hash)
    - `registry.py` — JSONL-based experiment/run logging
    - `runs.py` — RunRecord with status transitions
  - `artifacts/` — Plot generation (matplotlib)
  - `datasets/` — CSV loaders
  - `db/` — SQLAlchemy ORM models (Experiment, Run, Artifact, Job), CRUD, session
  - `profiles/` — Decision profiles & presets
    - `schema.py` — DecisionProfile Pydantic model (aggregation, objectives, thresholds, weights)
    - `presets.py` — Built-in profiles (balanced, risk_averse, performance_focused)
    - `load.py` — get_profile_or_load() (preset name or JSON file)
  - `utils/` — Path helpers
- `web/backend/` — FastAPI REST API
  - `app/main.py` — FastAPI app (v0.3.0), CORS, router registration
  - `app/config.py` — Settings from env vars (storage backend, Supabase, Clerk)
  - `app/contracts.py` — Pydantic request/response schemas
  - `app/engine_bridge.py` — Bridge to core engine (`run_from_spec_dict`)
  - `app/auth.py` — Clerk JWT verification & anonymous fallback for local dev
  - `app/bundle_transform.py` — Transform engine ResultBundle to web API ResultSummary
  - `app/llm_contracts.py` — Pydantic contracts for LLM API endpoints
  - `app/model_validation.py` — Validate sklearn pickle models for binary classification
  - `app/persistence.py` — Persistence dispatcher: local filesystem vs Supabase
  - `app/storage.py` — In-memory preset registries
  - `app/storage_backend.py` — Storage backend abstraction (LocalFS or Supabase)
  - `app/supabase_db.py` — Supabase Postgres client with owner_id filtering
  - `app/routers/` — Route handlers (experiments, compare, models, llm, presets, results, share)
  - `app/services/claude_client.py` — Claude API integration
- `web/frontend/` — Next.js 14 + React 18 + Tailwind
  - `app/` — App Router pages (home, new, compare, experiments/[id], experiments/[id]/not-found, assistant, sign-in, sign-up)
  - `components/ui/` — Radix UI + Shadcn-style components (button, card, input, label, select, separator, textarea)
  - `lib/api.ts` — API client (fetch wrapper, TypeScript interfaces)
  - `lib/analyst.ts` — Deterministic analyst assistant for comparison insights
  - `lib/auth.tsx` — Auth token provider (client-side Clerk integration)
  - `lib/compare_insights.ts` — Comparison-specific insight derivation
  - `lib/compare_model.ts` — Comparison data models
  - `lib/insights.ts` — Shared insight derivation from experiment results
  - `lib/utils.ts` — Utility functions
  - `middleware.ts` — Clerk auth middleware
- `tests/` — pytest test suite
  - `golden/` — Expected test output files
  - Test files named by phase: `test_phase7_*.py`, `test_phase8_*.py`, etc.
- `data/` — Demo datasets (CSV) and pickled sklearn model
- `examples/` — Example experiment specs (JSON) and data
- `alembic/` — SQLAlchemy migration scripts
- `supabase/migrations/` — Supabase DB migrations

## Tech Stack

### Core Engine (Python)
- **Python 3.11+**
- **Pydantic v2** for spec validation
- **NumPy** >=1.26 for numerical computing
- **Pandas** >=2.0 for data manipulation
- **scikit-learn** >=1.3 for metrics & model support
- **Matplotlib** >=3.7 for plot generation
- **SQLAlchemy** >=2.0 for ORM
- **Alembic** >=1.13 for database migrations

### Web Backend
- **FastAPI** >=0.104 for REST API
- **Uvicorn** for ASGI server
- Optional: **Supabase** (PostgreSQL), **Clerk** (auth)

### Web Frontend
- **Next.js 14** (App Router, Server Components)
- **React 18** + **TypeScript 5**
- **Tailwind CSS 3.4** + Radix UI + lucide-react
- Optional: **Clerk** (@clerk/nextjs) for auth

### Dev Tools
- **pytest** >=7.0 for testing
- **ruff** >=0.3 for linting/formatting
- **ESLint** + next config for frontend

## CLI Entry Point

```bash
spectra run <spec.json>              # Run experiment from spec
spectra compare <run_a> <run_b>      # Compare two runs
spectra score <run_a> <run_b>        # Score with decision profile (--profile balanced)
spectra rerun <run_id>               # Deterministic rerun
spectra enqueue-run <experiment_id>  # Queue run job
spectra enqueue-rerun <run_id>       # Queue rerun job
spectra worker-once                  # Process one job from queue
spectra experiments list|show        # Query experiments (--limit)
spectra runs list|show               # Query runs (--limit, --status, --experiment)
spectra jobs list|show               # Query jobs (--limit, --status)
```

## Development Setup

```bash
# Core engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # Core + dev tools
pip install -e ".[web]"      # + web dependencies

# Backend
cd web/backend && uvicorn app.main:app --reload  # :8000

# Frontend
cd web/frontend && npm install && npm run dev    # :3000
```

## CI Requirements (must pass before merge)

1. `ruff check src tests` — no lint errors
2. `pytest` — all tests pass
3. GitHub Actions: `.github/workflows/ci.yml` (Python 3.11, ubuntu-latest)

## Code Conventions

- All Python files use `from __future__ import annotations`
- Full type annotations on all functions (type hints, Optional, etc.)
- Absolute imports from package root: `from metrics_lie.execution import ...`
- Pydantic models for all specs and API contracts
- Protocol pattern for abstract interfaces (e.g., Scenario)
- Registry pattern for metrics and scenarios (register in module, import to trigger)
- `src/` layout with setuptools
- Test files named by milestone/phase: `test_phase<N>_<feature>.py`

## Environment Variables (Backend)

| Variable | Default | Purpose |
|---|---|---|
| `SPECTRA_STORAGE_BACKEND` | `local` | `local` or `supabase` |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Supabase service role key |
| `CLERK_ISSUER_URL` | — | Clerk issuer for JWT auth |
| `CLERK_JWKS_URL` | derived from issuer | JWKS endpoint for JWT verification |
| `SPECTRA_CORS_ORIGINS` | — | Extra CORS origins (comma-separated) |
| `ANTHROPIC_API_KEY` | — | Required for LLM analyst features |
| `SPECTRA_LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for LLM analyst |
| `NEXT_PUBLIC_SPECTRA_API_BASE` | — | Frontend API base URL |

## Database

- **Local**: SQLite at `.spectra_registry/spectra.db`
- **Cloud**: PostgreSQL via Supabase REST
- **Tables**: experiments, runs, artifacts, jobs
- **Migrations**: Alembic (`alembic/`) + Supabase (`supabase/migrations/`)

## Key Concepts

| Term | Definition |
|---|---|
| **Experiment** | Reusable config: dataset + metric + scenarios |
| **Run** | Single execution of an experiment with specific seed |
| **Scenario** | Perturbation simulating failure modes (noise, imbalance, gaming) |
| **ResultBundle** | JSON output: baseline + scenarios + metrics + diagnostics + artifacts |
| **PredictionSurface** | Model predictions: probability, score, or label |
| **MetricSummary** | Stats across trials: mean, std, quantiles |
| **Decision Profile** | Weighted scoring for comparing runs |
| **Comparison Flags** | Rule-based findings (metric_regression, gaming_detected, etc.) |

## Workflow Rules

### Plan First
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan
- Write detailed specs upfront

### Subagent Strategy
- Use subagents to keep the main context window clean
- Offload research, exploration, and parallel analysis
- One task per subagent for focused execution

### Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness

### Engineering Standards
- **Simplicity first** — minimal impact, minimal code
- **No laziness** — find root causes, no temporary fixes
- **Minimal blast radius** — touch only what's necessary
- **Demand elegance** — pause and consider a cleaner approach for non-trivial changes
