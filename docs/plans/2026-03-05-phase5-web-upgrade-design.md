# Phase 5: Web Platform Upgrade — Design Document

## Goal

Extend the FastAPI backend and Next.js frontend to support all task types (binary, multiclass, regression, ranking) and model formats (sklearn, ONNX, boosting) added in Phases 1-3, including task-specific visualizations.

## Key Insight

The web API contracts (ComponentScore, ScenarioResult, FindingFlag) are already generic. Binary classification is hardcoded in only 5 chokepoints:

1. `engine_bridge.py:233` — forces `task: "binary_classification"`
2. `model_validation.py` — requires `predict_proba` + shape `(n,2)`
3. `bundle_transform.py` — only extracts Brier/ECE diagnostics
4. `storage.py` — metric/scenario presets not filtered by task type
5. `app/new/page.tsx` — `.pkl` only, threshold 0.5, `y_score` default

## Architecture Decision

The core engine (Phases 1-3) already handles all task types end-to-end. The web layer just needs to:
1. Stop hardcoding binary classification in the bridge
2. Let the user choose task type + model format
3. Display task-specific results (confusion matrix, per-class metrics, residual stats)
4. Compute confusion matrix and per-class metrics in execution.py (small addition)

## Streams

- **Stream A (Backend)**: Contracts, bridge, validation, transform, presets, routers, core additions
- **Stream B (Frontend)**: TypeScript types, experiment form, results visualizations, compare, assistant
- **Stream C (Tests)**: Backend API tests + integration tests

Streams A and B run in parallel (same contract defined in plans). Stream C after merge.
