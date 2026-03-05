# Phase 6: Production Integrations & Documentation — Design Document

## Goal

Add MLflow integration (logging + model loading), KServe V2 protocol support, and MkDocs documentation site to make Spectra production-ready.

## Scope

| Sub-phase | Deliverable |
|-----------|-------------|
| 6.1 MLflow | `log_to_mlflow(result)` + MLflow model adapter |
| 6.2 KServe V2 | KServe V2 protocol auto-detection on HTTP adapter |
| 6.4 MkDocs | API reference (auto-generated) + getting-started + CLI reference |

## 6.1 MLflow Integration

### Logging — `src/metrics_lie/integrations/mlflow.py`

- `log_to_mlflow(result: ResultBundle, run_id=None)` logs metrics, scenario results (JSON artifact), experiment spec (params), and plots (image artifacts)
- Creates new MLflow run if `run_id` is None
- Works with local tracking and remote MLflow servers
- Exported from public SDK: `spectra.log_to_mlflow()`

### Loading — `src/metrics_lie/model/adapters/mlflow_adapter.py`

- Implements `ModelAdapterProtocol`
- Loads from MLflow URIs: `runs:/run_id/model`, `models:/name/version`
- Uses `mlflow.pyfunc.load_model()` (handles any MLflow flavor)
- Registered in default registry as `"mlflow"` format
- Spec: `{"model_source": {"kind": "mlflow", "uri": "runs:/abc123/model"}}`

### Dependencies

`mlflow>=2.10` in `[project.optional-dependencies]` as `mlflow` group.

## 6.2 KServe V2 Protocol

Enhancement to existing `http_adapter.py`:

- Add `protocol: "kserve_v2" | "custom"` config field (default: `"custom"`)
- KServe V2 mode: auto-construct `/v2/models/{name}/infer` URL, format as KServe tensor request, parse KServe tensor response
- Auto-detect: if endpoint URL contains `/v2/`, assume KServe V2
- Spec: `{"model_source": {"kind": "http", "endpoint": "...", "model_name": "mymodel", "protocol": "kserve_v2"}}`
- ~100-150 lines added to existing adapter

## 6.4 MkDocs Documentation

### Setup

- `mkdocs.yml` at project root, Material for MkDocs theme
- `mkdocstrings[python]` for auto-generated API reference
- GitHub Actions step for GitHub Pages deployment

### Pages (~10)

1. `index.md` — Home/overview
2. `getting-started.md` — Install, first evaluation in 5 minutes
3. `cli-reference.md` — All CLI commands with examples
4. `api-reference/sdk.md` — Auto-generated from public API
5. `api-reference/spec.md` — ExperimentSpec format reference
6. `api-reference/result-bundle.md` — ResultBundle schema reference
7. `concepts.md` — Scenarios, metrics, decision profiles
8. `integrations/mlflow.md` — MLflow logging & loading guide
9. `integrations/model-formats.md` — Supported formats with examples
10. `contributing.md` — How to add metrics, scenarios, adapters

### Dependencies

`mkdocs-material>=9.0`, `mkdocstrings[python]>=0.24` as `docs` optional group.

## Streams

- **Stream A (MLflow)**: integrations module, adapter, registry, pyproject, SDK exports
- **Stream B (KServe)**: http_adapter.py enhancement
- **Stream C (MkDocs)**: mkdocs.yml, docs pages, GitHub Actions
- **Stream D (Tests)**: MLflow + KServe tests after A+B merge

Streams A, B, C run in parallel. Stream D after A+B merge.
