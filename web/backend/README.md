# Spectra Web Backend

This backend is a **thin UI adapter** over the Spectra core engine.

## Architecture

- **Core engine** (`src/metrics_lie/`) remains the source of truth for evaluation logic
- **CLI** (`spectra` command) remains the primary interface for execution
- **Web backend** provides HTTP API for UI consumption
- **Frontend** (Phase 3.2+) will consume this API

## Phase 3.1 Scope

Phase 3.1 establishes:
- API boundary definition (FastAPI application structure)
- Data contracts (`contracts.py`) - stable Pydantic schemas
- Preset registries (`storage.py`) - in-memory placeholders

**No implementation yet**: Endpoints, experiment execution, and core engine integration come in Phase 3.2+.

## Non-Goals (Explicitly Out of Scope)

- Authentication / authorization
- Billing / subscription management
- Multi-user / tenant isolation
- File uploads / storage providers
- Background job queues (use core engine's job system)
- SaaS infrastructure / deployment automation

## Contracts

`contracts.py` defines the stable interface between frontend and backend:
- Request/response models for experiments, runs, and results
- These contracts will be used when endpoints are implemented in Phase 3.2+

## Installation

Install web dependencies:

```bash
pip install -e ".[web]"
```

## Running the Server

```bash
cd web/backend
python -m uvicorn app.main:app --reload --port 8000
```

The server will be available at `http://localhost:8000`.

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

