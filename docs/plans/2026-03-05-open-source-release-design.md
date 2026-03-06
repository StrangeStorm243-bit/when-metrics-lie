# Spectra 1.0 Open-Source Release Design

**Goal:** Ship Spectra as a polished, pip-installable ML stress-testing tool. CLI + Python SDK is the core product. Web UI runs locally via `spectra serve`. No hosting, no paid services.

## Decisions

- **PyPI name:** `spectra-ml` (`pip install spectra-ml`)
- **Import name:** `metrics_lie` (unchanged — renaming every import is high risk, low value. Many packages differ: `pip install Pillow` -> `import PIL`)
- **Version:** 1.0.0
- **License:** Apache 2.0 (already in place)
- **No hosting:** No Vercel, Railway, Supabase, Clerk in the default flow
- **Docker:** Minimal Dockerfile included

## What Ships

1. **`spectra serve` CLI command** — launches FastAPI backend via uvicorn, opens browser
2. **PyPI metadata** — authors, classifiers, URLs, keywords, license field
3. **README overhaul** — badges, features, install, quickstart, comparison
4. **Version 1.0.0** — across pyproject.toml, __init__.py, frontend package.json
5. **Dockerfile** — single-stage, `pip install spectra-ml[web] && spectra serve`
6. **GitHub polish** — CODE_OF_CONDUCT, issue/PR templates

## What Stays

- All existing code, tests, docs, CLI commands
- Internal package name `metrics_lie`
- Web frontend/backend architecture
- Apache 2.0 license
- CI pipeline
