# Spectra Codebase Refactor Plan

## STEP 0 — Deep Codebase Understanding

### Architecture Overview

Spectra is a metrics stress-testing engine with five major layers:

```
┌─────────────────────────────────────────────────────────┐
│  CLI (cli.py 335L)  │  Web Backend (FastAPI, 6 routers) │
├─────────────────────┴───────────────────────────────────┤
│  Execution Orchestrator (execution.py 640L)             │
├────────────┬────────────┬───────────┬───────────────────┤
│ Spec Layer │ Model Layer│ Metrics   │ Analysis Layer    │
│ (spec.py)  │ (adapter,  │ (core,    │ (threshold_sweep, │
│ (schema.py)│  surface)  │  resolve) │  sensitivity,     │
│            │            │           │  disagreement,    │
│            │            │           │  dashboard,       │
│            │            │           │  failure_modes)   │
├────────────┴────────────┴───────────┴───────────────────┤
│ Scenario Engine (runner.py 244L, 4 scenarios)           │
├─────────────────────────┬───────────────────────────────┤
│ Persistence (DB/JSONL)  │ Compare Engine (compare.py)   │
├─────────────────────────┴───────────────────────────────┤
│ Diagnostics (calibration, metric_gaming, subgroups)     │
└─────────────────────────────────────────────────────────┘
```

**Total engine code:** ~3,400 LOC across 16 key files.
**Total test code:** ~3,600 LOC across 30 test files.
**Web backend:** ~1,600 LOC (contracts, engine_bridge, persistence, routers).

---

### Core Data Flow

```
spec.json (user input)
  │
  ▼
load_experiment_spec() ─── Pydantic validation (spec.py:135L)
  │
  ▼
ExperimentDefinition.from_spec() ─── SHA256 → deterministic experiment_id
  │
  ▼
DB: upsert_experiment() + JSONL log
  │
  ▼
load_binary_csv() ─── score_validation mode depends on surface_type
  │
  ├── [model_source] → ModelAdapter.get_all_surfaces(X)
  │     → SurfaceType preference: PROBABILITY > SCORE > LABEL
  │
  ├── [surface_source] → validate_surface() + PredictionSurface
  │     → _SURFACE_TYPE_MAP: string → enum (execution.py:181)
  │     → SCENARIO_SURFACE_COMPAT filtering (execution.py:242)
  │
  └── [neither] → default SurfaceType.PROBABILITY
  │
  ▼
MetricResolver.resolve() ─── surface_type + DatasetProperties → ApplicableMetricSet
  OR manual: ApplicableMetricSet(metrics=[spec.metric])
  │
  ▼
For each metric:
  ├── Baseline: single metric_fn(y_true, y_score) call
  └── run_scenarios() ─── n_trials Monte Carlo per scenario
        │
        ├── rng = np.random.default_rng(seed) ← determinism anchor
        ├── scenario.apply(y_true, y_score, rng, ctx) per trial
        ├── metric_fn(y_p, s_p) per trial
        ├── calibration (brier, ece) if PROBABILITY surface
        ├── subgroup metrics if subgroup column present
        └── metric_gaming diagnostics if accuracy metric
  │
  ▼
Artifact generation (plots) ─── separate rng_artifacts = default_rng(seed)
  │
  ▼
Analysis artifacts (if model_source or surface_source):
  ├── threshold_sweep (PROBABILITY/SCORE only)
  ├── sensitivity_analysis (score_noise perturbation)
  ├── metric_disagreements (threshold-based metrics)
  ├── failure_modes (all surface types)
  └── dashboard_summary (if >1 applicable metric)
  │
  ▼
ResultBundle → results.json on disk
  │
  ▼
DB: insert_artifacts(), update_run(status="completed")
```

---

### Where surface_type Is Decided and Propagated

**Decision point** (execution.py):
1. Line 134: `surface_type = SurfaceType.PROBABILITY` (default)
2. Line 169: If model_source → derived from `prediction_surface.surface_type`
3. Line 186: If surface_source → `_SURFACE_TYPE_MAP[spec.surface_source.surface_type]`

**Propagation path:**
- execution.py:273 → `ScenarioContext(surface_type=surface_type.value)` (enum → string)
- runner.py:84 → `if ctx.surface_type == "probability"` (string comparison)
- execution.py:242 → `SCENARIO_SURFACE_COMPAT[surface_type]` (enum lookup)
- execution.py:498 → threshold_sweep/sensitivity gated on PROBABILITY/SCORE
- analysis modules → `surface.surface_type` (enum from PredictionSurface)

**Critical asymmetry:** ScenarioContext uses **string**, analysis modules use **enum**.

---

### Where Determinism Is Enforced

| Location | Mechanism |
|----------|-----------|
| spec.py | `seed: int = 42` default |
| execution.py:95 | `canonical_json(spec_dict)` → sorted keys, no whitespace |
| identity.py | `sha256_hex(canonical_json(semantics))` → experiment_id |
| runner.py:53 | `rng = np.random.default_rng(cfg.seed)` — single RNG per scenario run |
| execution.py:337 | `rng_artifacts = np.random.default_rng(spec.seed)` — separate artifact RNG |
| datasets.py | `sha256_hex(Path(path).read_bytes())` — dataset fingerprint |
| dashboard.py | `sorted(metric_results.keys())` — deterministic metric ordering |

**No global numpy state:** All randomness flows through explicit `default_rng()` instances.

---

### Where Probability-Only Assumptions Still Exist

| Location | Assumption | Guard |
|----------|-----------|-------|
| runner.py:84-86 | brier/ece only for `ctx.surface_type == "probability"` | String check |
| runner.py:125 | Per-group calibration only for probability | String check |
| runner.py:195-201 | Gaming downstream (brier/ece) only for probability | String check |
| execution.py:296-300 | Baseline calibration diagnostics only for probability | Enum check |
| threshold_sweep.py:91-97 | SCORE_METRICS only computed for PROBABILITY | Enum check |
| sensitivity.py:85 | Score noise clips to [0,1] only for PROBABILITY | Enum check |
| score_noise scenario:29 | `np.clip(s, 0.0, 1.0)` only for probability | String check |
| metric_logloss | `np.clip(y_score, eps, 1-eps)` — implicit [0,1] | No guard |
| metric_brier_score | MSE against probabilities | MetricRequirement guard |
| metric_ece | Bins scores into [0,1] | MetricRequirement guard |
| engine_bridge.py:343 | Hardcoded `surface_type: "probability"` for multi-metric | None |

---

### Where Behavior Depends on Subtle Ordering or Defaults

1. **Surface type default** (execution.py:134): Falls through to PROBABILITY if neither model_source nor surface_source is set. This means the "legacy" Phase 4 path always assumes probability.

2. **Scenario filtering only with surface_source** (execution.py:241): `SCENARIO_SURFACE_COMPAT` is only applied when `spec.surface_source is not None`. Model-inferred surfaces run ALL scenarios regardless of surface type.

3. **Manual vs. auto metric selection** (execution.py:209-231): When neither model_source nor surface_source is set, MetricResolver is NOT called — just `[spec.metric]`. When either is set, full resolution runs. This means the Phase 4 path never gets metric warnings or exclusions.

4. **RNG ordering in runner.py**: Scenarios consume RNG sequentially. If scenario list order changes, all downstream RNG states change. The RNG is seeded once per `run_scenarios()` call, not per scenario.

5. **Threshold default 0.5**: Appears in spec.py (SurfaceSourceSpec validator), execution.py (baseline accuracy), runner.py (trial accuracy), and subgroups.py (safe_metric_for_group). Not centralized.

---

### Coupling Analysis

**Tightly coupled modules:**
- `execution.py` imports from 20+ internal modules — it's the God Function
- `runner.py` directly computes calibration diagnostics inline (not delegated)
- `engine_bridge.py` builds spec dicts with hardcoded field names matching engine schema

**Business logic leaking into execution:**
- Scenario compatibility table (`SCENARIO_SURFACE_COMPAT`) lives in execution.py
- Surface type mapping (`_SURFACE_TYPE_MAP`) is a local variable in execution.py
- Threshold defaults (0.5) are scattered across 4+ files
- Metric bifurcation (threshold vs. score metrics) duplicated in runner.py, threshold_sweep.py, sensitivity.py

**Decisions duplicated across spec/loaders/execution:**
- Threshold validation: spec.py (Pydantic), surface.py (validate_surface), execution.py (implicit)
- Binary label validation: loaders.py (_validate_binary_labels), surface.py (validate_surface)
- NaN/Inf validation: loaders.py (3 functions), surface.py (validate_surface)

---

## STEP 1 — Behavioral Invariants (Must Not Change)

### Invariant List

| # | Invariant | Where Enforced | Risk if Broken |
|---|-----------|---------------|----------------|
| 1 | Phase 4 CLI: `python -m metrics_lie.cli run examples/experiment_minimal.json` produces deterministic results | cli.py → execution.py → runner.py | Core product broken |
| 2 | ResultBundle JSON schema (schema_version "0.1") | schema.py | All consumers broken |
| 3 | Determinism: same spec+seed → same bundle (excluding run_id, created_at, notes) | runner.py:53 (seeded RNG), identity.py | 6 determinism tests fail |
| 4 | Metric resolution: surface_type filters which metrics are applicable | applicability.py METRIC_REQUIREMENTS | Wrong metrics computed |
| 5 | Scenario compatibility: PROBABILITY→4, SCORE→3, LABEL→2 scenarios | execution.py:65-69 | Incompatible scenarios run |
| 6 | Compare semantics: regression thresholds (metric:-0.01, cal:0.02, subgroup:0.03) | rules.py | Compare decisions change |
| 7 | Share token: 32-byte urlsafe, validated against stored token+run_id | persistence.py | Auth bypass |
| 8 | Backend contracts: ResultSummary, CompareResponse, ExperimentSummary shapes | contracts.py | Frontend broken |
| 9 | experiment_id is content-addressable (SHA256 of canonical spec) | identity.py + definition.py | Experiment dedup broken |
| 10 | Baseline = single metric_fn call (no trials, std=0) | execution.py:254-263 | Bundle schema violated |
| 11 | Artifact paths: `artifacts/{type}_{scenario_id}.png` | execution.py:343-466 | Plot loading broken |
| 12 | DB schema: experiments, runs, artifacts, jobs tables | models.py | All persistence broken |

### Validation Commands (Run After Every Patch)

```bash
# 1. Lint
ruff check src/ tests/

# 2. Full test suite
pytest tests/ -x -q

# 3. Phase 4 CLI smoke test
python -m metrics_lie.cli run examples/experiment_minimal.json

# 4. Determinism tests specifically
pytest tests/test_phase7_determinism.py tests/test_phase8_surface_ingest.py tests/test_phase9_label_ingest.py tests/test_phase9_score_ingest.py -v

# 5. Frontend build (if touching contracts)
cd web/frontend && npm run build
```

---

## STEP 2 — Refactor Hotspots (Top 10)

### Hotspot 1: execution.py — God Function (640L)

**File:** `src/metrics_lie/execution.py`, `run_from_spec_dict()`
**Why risky:** Single 500+ line function orchestrating 9 phases. Any change risks side effects across phases. Impossible to unit test individual phases.
**Future risk:** Adding new surface types, metrics, or analysis modes requires modifying this monolith.
**Determinism impact:** HIGH — RNG state, ordering, and phase gating all live here.

### Hotspot 2: Surface type string/enum inconsistency

**Files:** `execution.py:273` (enum→string), `runner.py:84` (string compare), `scenarios/score_noise.py:29` (string compare)
**Why risky:** ScenarioContext.surface_type is a string, but analysis modules use SurfaceType enum. Easy to introduce typos or miss a branch.
**Future risk:** Adding a new surface type requires updating string comparisons in multiple files.
**Determinism impact:** LOW — but correctness impact HIGH.

### Hotspot 3: _SURFACE_TYPE_MAP and SCENARIO_SURFACE_COMPAT in execution.py

**File:** `execution.py:65-69, 181-186`
**Why risky:** These are core routing tables buried inside a 640L orchestrator. They're not importable by other modules or tests.
**Future risk:** Adding surface types requires finding these in execution.py rather than a dedicated module.
**Determinism impact:** MEDIUM — scenario filtering directly affects results.

### Hotspot 4: Metric bifurcation (threshold vs. score) duplicated 3x

**Files:** `runner.py:79-105`, `threshold_sweep.py:85-97`, `sensitivity.py:94-106`
**Why risky:** Three independent if/else blocks deciding how to call metric functions. Adding a metric requires updating all three.
**Future risk:** New metric added to one but not others → silent wrong behavior.
**Determinism impact:** LOW.

### Hotspot 5: Threshold default 0.5 scattered across 4+ files

**Files:** `spec.py` (SurfaceSourceSpec validator), `execution.py:260`, `runner.py:80`, `subgroups.py:52`
**Why risky:** No single source of truth. If threshold default changes, must find all locations.
**Future risk:** Partial update → inconsistent thresholds between baseline and trials.
**Determinism impact:** HIGH — threshold affects binary predictions.

### Hotspot 6: Validation logic duplicated across loaders.py and surface.py

**Files:** `loaders.py:14-52`, `surface.py:55-100`
**Why risky:** Binary label validation, NaN/Inf checks, and probability range checks exist in both. Different implementations (pandas vs. numpy).
**Future risk:** Validation rule updated in one place but not the other → data passes one check but fails later.
**Determinism impact:** NONE directly, but incorrect data can cause nondeterministic failures.

### Hotspot 7: engine_bridge.py hardcoded spec construction

**File:** `web/backend/app/engine_bridge.py:316-370`
**Why risky:** Hardcoded task="binary_classification", n_trials=200, surface_type="probability", positive_label=1, column names y_true/y_score/group. None of these are configurable or share constants with engine.
**Future risk:** Engine spec changes require finding all hardcoded values in bridge.
**Determinism impact:** NONE — bridge is a consumer, not a producer.

### Hotspot 8: runner.py inline calibration/gaming logic

**File:** `runner.py:84-105, 125-128, 195-201`
**Why risky:** runner.py mixes Monte Carlo loop logic with diagnostic computation. Hard to understand which diagnostics apply when.
**Future risk:** Adding new diagnostic requires modifying the inner trial loop.
**Determinism impact:** MEDIUM — diagnostics computed inside RNG-consuming loop.

### Hotspot 9: compare/compare.py nested dict navigation

**File:** `compare/compare.py:1-399`
**Why risky:** Deep nested dict access with `.get()` chains. Easy to miss a changed key path.
**Future risk:** If ResultBundle structure changes, compare silently returns None deltas.
**Determinism impact:** NONE — compare is purely deterministic on its inputs.

### Hotspot 10: claim_next_job() race condition

**File:** `db/crud.py:175-186`
**Why risky:** No SELECT FOR UPDATE or row-level lock. Two workers can claim the same job.
**Future risk:** Duplicate runs in multi-worker deployments.
**Determinism impact:** NONE — but operational correctness impact.

---

## STEP 3 — Refactor Opportunities (Prioritized)

### 3.1 Extract surface type constants to dedicated module

**Problem:** `_SURFACE_TYPE_MAP`, `SCENARIO_SURFACE_COMPAT`, and `DEFAULT_THRESHOLD = 0.5` are buried in execution.py (local vars or module-level). Not importable by tests or other modules.

**Proposed change:** Create `src/metrics_lie/surface_compat.py` (~30 LOC) exporting:
- `SURFACE_TYPE_MAP: dict[str, SurfaceType]`
- `SCENARIO_SURFACE_COMPAT: dict[SurfaceType, set[str]]`
- `DEFAULT_THRESHOLD: float = 0.5`

Update execution.py to import from it. Update tests to import from it.

**Risk:** LOW — pure extraction, no logic change.
**LOC touched:** ~40 (new file + import changes in execution.py).
**Tests affected:** None broken; optionally add test for the constants.
**Rollback:** Delete new file, revert import.
**Determinism impact:** NONE.

### 3.2 Centralize threshold default constant

**Problem:** `0.5` appears as magic number in spec.py (validator), execution.py (baseline), runner.py (trial), subgroups.py (safe_metric_for_group), metric_gaming.py (accuracy_at_threshold caller context).

**Proposed change:** Import `DEFAULT_THRESHOLD` from surface_compat.py (created in 3.1) into all files that use 0.5 for threshold decisions.

**Risk:** LOW — literal replacement, no logic change.
**LOC touched:** ~15 (import + replace in 4 files).
**Tests affected:** None — same value.
**Rollback:** Revert to literal 0.5.
**Determinism impact:** NONE.

### 3.3 Extract metric dispatch helper

**Problem:** Three files (runner.py, threshold_sweep.py, sensitivity.py) each have if/else blocks for threshold-based vs. score-based metric calling.

**Proposed change:** Add helper to `metrics/core.py` (~20 LOC):
```python
def compute_metric(metric_id: str, metric_fn, y_true, y_score, *, threshold=DEFAULT_THRESHOLD):
    if metric_id in THRESHOLD_METRICS:
        return metric_fn(y_true, y_score, threshold=threshold)
    elif metric_id == "ece":
        return metric_fn(y_true, y_score, n_bins=10)
    else:
        return metric_fn(y_true, y_score)
```

Also add `THRESHOLD_METRICS` and `SCORE_METRICS` sets to core.py.

**Risk:** LOW-MEDIUM — must verify exact equivalence in all three call sites.
**LOC touched:** ~60 (20 new + 40 replaced across 3 files).
**Tests affected:** Run all scenario/analysis tests to confirm equivalence.
**Rollback:** Revert to inline if/else.
**Determinism impact:** NONE if equivalence verified.

### 3.4 Make ScenarioContext.surface_type use SurfaceType enum

**Problem:** ScenarioContext stores surface_type as string. Scenarios compare with string literals. Analysis modules use enum. Inconsistent.

**Proposed change:**
- Change `ScenarioContext.surface_type` from `str` to `SurfaceType` (or keep str but add a comment)
- Wait — this would break scenario code that does `ctx.surface_type == "probability"`.
- **Safer alternative:** Add `@property` to ScenarioContext that returns enum, or update scenarios to use enum comparison.
- **Safest:** Update execution.py:273 to pass `surface_type.value` (already does), but update runner.py:84 to compare against `SurfaceType.PROBABILITY.value`. This makes the comparison explicit without changing the dataclass.

**Risk:** MEDIUM — touches scenario protocol and runner inner loop.
**LOC touched:** ~15 (runner.py:84,125,195 + score_noise.py:29).
**Tests affected:** All scenario tests, all determinism tests.
**Rollback:** Revert string comparisons.
**Determinism impact:** NONE if string values are identical.

### 3.5 Extract scenario compatibility check from execution.py

**Problem:** Scenario filtering logic (execution.py:241-252) is inline in the God Function. Cannot be tested independently.

**Proposed change:** Move to surface_compat.py as:
```python
def filter_compatible_scenarios(
    scenarios: list[ScenarioSpec],
    surface_type: SurfaceType,
) -> tuple[list[ScenarioSpec], list[str]]:
    """Returns (compatible, skipped_ids)."""
```

Call from execution.py.

**Risk:** LOW — pure extraction.
**LOC touched:** ~25 (new function + call site change).
**Tests affected:** Phase 9 tests (scenario filtering). Add unit test for the function.
**Rollback:** Inline the logic back.
**Determinism impact:** NONE.

### 3.6 Consolidate validation helpers

**Problem:** NaN/Inf checks and binary label validation duplicated between loaders.py and surface.py.

**Proposed change:** Create shared validators in a new `src/metrics_lie/validation.py` (~40 LOC):
- `validate_no_nan_inf(arr, name)` — numpy-based
- `validate_binary_labels(arr, name)` — numpy-based
- `validate_probability_range(arr, name)` — numpy-based

Have both loaders.py and surface.py call these.

**Risk:** LOW-MEDIUM — must handle pandas Series → numpy conversion in loaders.py.
**LOC touched:** ~80 (40 new + 40 replaced across 2 files).
**Tests affected:** test_contracts.py, test_phase9_*.
**Rollback:** Delete validation.py, restore inline validators.
**Determinism impact:** NONE.

### 3.7 Add METRIC_CATEGORIES to metrics/core.py

**Problem:** "Threshold metrics" and "score metrics" sets are defined independently in threshold_sweep.py, sensitivity.py, and implicitly in runner.py. No canonical source.

**Proposed change:** Define in metrics/core.py:
```python
THRESHOLD_METRICS: set[str] = {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}
CALIBRATION_METRICS: set[str] = {"brier_score", "ece"}
SCORE_METRICS: set[str] = {"auc", "pr_auc", "logloss"}
```

Import these in threshold_sweep.py and sensitivity.py instead of local definitions.

**Risk:** LOW — pure constant extraction.
**LOC touched:** ~30.
**Tests affected:** None broken; optionally test membership.
**Rollback:** Restore local definitions.
**Determinism impact:** NONE.

### 3.8 Extract _bundle_to_result_summary transformation to dedicated module

**Problem:** engine_bridge.py:213-293 transforms engine ResultBundle to web contract. 80L of field mapping mixed with route logic.

**Proposed change:** Move to `web/backend/app/bundle_transform.py`. Pure function, independently testable.

**Risk:** LOW — pure extraction, no logic change.
**LOC touched:** ~90 (new file + import change).
**Tests affected:** Add unit test for transformation.
**Rollback:** Move function back to engine_bridge.py.
**Determinism impact:** NONE.

### 3.9 Add type annotation for analysis_artifacts dict

**Problem:** `analysis_artifacts` is typed as `dict` everywhere but has a known structure: threshold_sweep, sensitivity, metric_disagreements, failure_modes, dashboard_summary. No TypedDict.

**Proposed change:** Add TypedDict to schema.py:
```python
class AnalysisArtifacts(TypedDict, total=False):
    threshold_sweep: dict
    sensitivity: dict
    metric_disagreements: list
    failure_modes: dict
    dashboard_summary: dict
```

Use in ResultBundle and contracts.

**Risk:** LOW — type annotation only, no runtime change.
**LOC touched:** ~20.
**Tests affected:** None.
**Rollback:** Revert to `dict`.
**Determinism impact:** NONE.

### 3.10 Normalize error messages in loaders.py

**Problem:** Validation errors in loaders.py use inconsistent formats: some include first 5 bad values, some include column names, some don't.

**Proposed change:** Standardize to: `"{source}: {column} {problem}. First violations: {values[:5]}"`.

**Risk:** LOW — error message strings only. No behavioral change.
**LOC touched:** ~20.
**Tests affected:** Any tests asserting exact error messages (check first).
**Rollback:** Revert message strings.
**Determinism impact:** NONE.

### 3.11 Extract Phase 8 analysis orchestration from execution.py

**Problem:** Lines 494-560 of execution.py handle all Phase 8 analysis (threshold_sweep, sensitivity, disagreements, failure_modes, dashboard). This block is 66 lines that could be an independent function.

**Proposed change:** Extract to:
```python
def _run_analysis_phase(
    *, y_true, prediction_surface, surface_type, applicable, subgroup, spec, metric_results, scenario_results_by_metric, primary_metric
) -> dict:
```

Keep in execution.py but as a named function (not a new file).

**Risk:** MEDIUM — many parameters, must verify no implicit closure over execution state.
**LOC touched:** ~75 (extract + call site).
**Tests affected:** All Phase 8 tests.
**Rollback:** Inline back.
**Determinism impact:** NONE if parameter passing is exact.

### 3.12 Add golden snapshot test for Phase 4 CLI output

**Problem:** No test captures the exact JSON output of `python -m metrics_lie.cli run examples/experiment_minimal.json`. Determinism tests exist but are Phase 7+. A Phase 4-specific golden test would catch regressions early.

**Proposed change:** Add `tests/test_phase4_golden.py` that:
1. Runs minimal spec
2. Loads results.json
3. Strips non-deterministic fields (run_id, created_at, notes)
4. Compares against a committed golden file `tests/golden/phase4_minimal.json`

**Risk:** LOW — purely additive test.
**LOC touched:** ~40 (new test + golden file).
**Tests affected:** New test only.
**Rollback:** Delete test file.
**Determinism impact:** NONE (tests determinism, doesn't change it).

---

## STEP 4 — Proposed Refactor Milestones

### Milestone 1: Extract Constants & Centralize Threshold

**Scope:** Items 3.1 + 3.2 + 3.7
**Goal:** All surface type routing tables, threshold defaults, and metric categories live in importable, testable locations.

**Changes:**
1. Create `src/metrics_lie/surface_compat.py` with SURFACE_TYPE_MAP, SCENARIO_SURFACE_COMPAT, DEFAULT_THRESHOLD
2. Create THRESHOLD_METRICS, CALIBRATION_METRICS, SCORE_METRICS in `metrics/core.py`
3. Update execution.py, runner.py, spec.py, subgroups.py to import constants
4. Update threshold_sweep.py and sensitivity.py to import metric sets

**LOC touched:** ~80
**Tests affected:** 0 broken (same values, different import path)

**STOP LINE:** After this milestone, run full validation. Do not proceed until green.

**Validation:**
```bash
ruff check src/ tests/
pytest tests/ -x -q
python -m metrics_lie.cli run examples/experiment_minimal.json
pytest tests/test_phase7_determinism.py tests/test_phase9_label_ingest.py -v
```

**Rollback:** `git revert HEAD` (single commit)

---

### Milestone 2: Extract Scenario Compatibility + Metric Dispatch

**Scope:** Items 3.3 + 3.5
**Goal:** Scenario filtering and metric calling are testable functions, not inline code.

**Changes:**
1. Add `filter_compatible_scenarios()` to surface_compat.py
2. Add `compute_metric()` helper to metrics/core.py
3. Update execution.py to use filter_compatible_scenarios()
4. Update runner.py, threshold_sweep.py, sensitivity.py to use compute_metric()
5. Add unit tests for both new functions

**LOC touched:** ~100
**Tests affected:** All scenario and analysis tests (verify equivalence)

**STOP LINE:** After this milestone, run full validation. Do not proceed until green.

**Validation:**
```bash
ruff check src/ tests/
pytest tests/ -x -q
python -m metrics_lie.cli run examples/experiment_minimal.json
pytest tests/test_phase7_determinism.py tests/test_phase8_*.py tests/test_phase9_*.py -v
```

**Rollback:** `git revert HEAD` (single commit)

---

### Milestone 3: Consolidate Validation + Add Golden Test

**Scope:** Items 3.6 + 3.12
**Goal:** Single source of truth for data validation. Phase 4 golden snapshot prevents regression.

**Changes:**
1. Create `src/metrics_lie/validation.py` with shared validators
2. Update loaders.py and surface.py to use shared validators
3. Create `tests/test_phase4_golden.py` and `tests/golden/phase4_minimal.json`

**LOC touched:** ~120
**Tests affected:** test_contracts.py, test_phase9_* (validation paths). New golden test.

**STOP LINE:** After this milestone, run full validation. Do not proceed until green.

**Validation:**
```bash
ruff check src/ tests/
pytest tests/ -x -q
python -m metrics_lie.cli run examples/experiment_minimal.json
pytest tests/test_phase4_golden.py -v
```

**Rollback:** `git revert HEAD` (single commit)

---

### Milestone 4: Web Backend Contract Clarity

**Scope:** Items 3.8 + 3.9
**Goal:** Bundle transformation is independently testable. Analysis artifacts have type structure.

**Changes:**
1. Extract `_bundle_to_result_summary()` to `web/backend/app/bundle_transform.py`
2. Add AnalysisArtifacts TypedDict to schema.py
3. Add unit test for bundle transformation
4. Update engine_bridge.py imports

**LOC touched:** ~110
**Tests affected:** New transformation test. Existing backend tests unchanged.

**STOP LINE:** After this milestone, run full validation including frontend build.

**Validation:**
```bash
ruff check src/ tests/ web/backend/
pytest tests/ -x -q
python -m metrics_lie.cli run examples/experiment_minimal.json
cd web/frontend && npm run build
```

**Rollback:** `git revert HEAD` (single commit)

---

## STEP 5 — "Do First" Minimal Patch List

### Patch A: Extract surface_compat.py (~35 LOC new, ~15 LOC changed)

Create `src/metrics_lie/surface_compat.py`:
```python
"""Canonical surface type routing tables and defaults."""
from __future__ import annotations
from metrics_lie.model.surface import SurfaceType

SURFACE_TYPE_MAP: dict[str, SurfaceType] = {
    "probability": SurfaceType.PROBABILITY,
    "score": SurfaceType.SCORE,
    "label": SurfaceType.LABEL,
}

SCENARIO_SURFACE_COMPAT: dict[SurfaceType, set[str]] = {
    SurfaceType.PROBABILITY: {"label_noise", "score_noise", "class_imbalance", "threshold_gaming"},
    SurfaceType.SCORE: {"label_noise", "score_noise", "class_imbalance"},
    SurfaceType.LABEL: {"label_noise", "class_imbalance"},
}

DEFAULT_THRESHOLD: float = 0.5
```

Update execution.py:
- Remove local `_SURFACE_TYPE_MAP` (line 181-185)
- Remove module-level `SCENARIO_SURFACE_COMPAT` (lines 65-69)
- Add `from metrics_lie.surface_compat import SURFACE_TYPE_MAP, SCENARIO_SURFACE_COMPAT, DEFAULT_THRESHOLD`

**Risk:** Near-zero. Pure extraction. Values identical.

---

### Patch B: Add metric category sets to core.py (~15 LOC new)

Add to `src/metrics_lie/metrics/core.py`:
```python
THRESHOLD_METRICS: set[str] = {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}
CALIBRATION_METRICS: set[str] = {"brier_score", "ece"}
RANKING_METRICS: set[str] = {"auc", "pr_auc", "logloss"}
```

Update threshold_sweep.py and sensitivity.py to import these instead of defining locally.

**Risk:** Near-zero. Constant extraction only.

---

### Patch C: Centralize DEFAULT_THRESHOLD usage (~10 LOC changed)

In runner.py, replace `threshold=0.5` with `threshold=DEFAULT_THRESHOLD`.
In subgroups.py, replace `threshold=0.5` with `threshold=DEFAULT_THRESHOLD`.
In execution.py baseline, replace `threshold=0.5` with `threshold=DEFAULT_THRESHOLD`.

**Risk:** Near-zero. Same value, explicit import.

---

## STEP 6 — Test & CI Hardening

### Flaky Test Risks

| Test File | Risk | Root Cause | Mitigation |
|-----------|------|------------|------------|
| test_phase7_determinism.py | HIGH | Fresh SQLite DB per test via autouse fixture deleting DB_PATH | Use `tmp_path`-based DB or in-memory SQLite |
| test35_mvs_integration.py | HIGH | Same DB deletion pattern + pickle I/O | Same |
| test25_jobs_worker.py | MEDIUM | DB operations + job claiming | Isolate DB path per test |
| test26_db_queries.py | MEDIUM | Shared DB state between tests | Ensure autouse fixture runs first |
| test24_rerun.py | MEDIUM | DB read-after-write dependency | Verify fixture ordering |

### Nondeterministic Ordering Risks

1. **Scenario list iteration in runner.py**: Scenarios are processed in list order. If `effective_scenarios` order changes (e.g., from set operations), RNG state diverges.
   - **Current status:** List comprehension preserves input order. SAFE.

2. **Dict iteration in dashboard.py**: Uses `sorted(metric_results.keys())`. SAFE.

3. **Dict iteration in compare.py**: Uses `scenario_map()` which returns dict from list. Iteration order matches input. SAFE if input is stable.

### Missing Golden Tests

- **Phase 4 CLI golden test:** Does not exist. Should capture exact bundle JSON (stripped of non-deterministic fields).
- **Compare output golden test:** No golden file for compare output structure.
- **Analysis artifacts golden test:** No golden file for threshold_sweep/sensitivity output.

### Proposed Minimal Improvements

1. **Add Phase 4 golden test** (Item 3.12 above) — highest value, catches regressions early.

2. **Isolate DB per test using tmp_path:**
   ```python
   @pytest.fixture(autouse=True)
   def fresh_db(tmp_path, monkeypatch):
       db_path = str(tmp_path / "test.db")
       monkeypatch.setattr("metrics_lie.db.session.DB_PATH", db_path)
       init_db()
   ```
   This avoids deleting the shared DB_PATH and eliminates cross-test interference.

3. **Add `--tb=short` to pytest defaults** in pyproject.toml for clearer CI output.

4. **Consider marking slow tests** with `@pytest.mark.slow` (Phase 7-9 tests that run full experiments).

---

## STEP 7 — Future Query Knowledge Map

### How to Reason About the Codebase Quickly

- **Entry point:** `cli.py:main()` → `run()` → `execution.run_from_spec_dict()`
- **The God Function:** `execution.py:run_from_spec_dict()` orchestrates everything. Read this to understand the full pipeline.
- **Surface type is the pivot:** Once determined (line 134/169/186 of execution.py), it controls metric resolution, scenario filtering, diagnostic computation, and analysis gating.
- **Determinism anchor:** `np.random.default_rng(seed)` in runner.py:53. One RNG per scenario run. Artifact generation uses separate RNG (execution.py:337).
- **Two execution modes:** "Manual" (neither model_source nor surface_source) uses spec.metric directly. "Auto" (either present) uses MetricResolver.
- **Compare is read-only:** compare.py never modifies bundles. It loads JSON, computes deltas, applies thresholds from rules.py.
- **Web backend is a consumer:** engine_bridge.py builds spec dicts and calls `run_from_spec_dict()`. It transforms ResultBundle to web contracts.

### Where New Surface Types Would Plug In

1. Add enum value to `SurfaceType` in `model/surface.py`
2. Add validation branch in `surface.py:validate_surface()`
3. Add entry to `SURFACE_TYPE_MAP` in `surface_compat.py`
4. Add entry to `SCENARIO_SURFACE_COMPAT` in `surface_compat.py`
5. Add `MetricRequirement` entries in `metrics/applicability.py` (which metrics apply)
6. Add branch in `runner.py` for any surface-specific diagnostics
7. Add branch in `analysis/failure_modes.py` for contribution scoring
8. Update `loaders.py` score_validation if new validation needed

### Where New Metric Families Would Plug In

1. Add metric function to `metrics/core.py` with standard signature
2. Add to `METRICS` dict in `metrics/core.py`
3. Add to appropriate category set (THRESHOLD_METRICS, RANKING_METRICS, etc.)
4. Add `MetricRequirement` to `metrics/applicability.py` (surface type compatibility)
5. If threshold-based: works automatically via `compute_metric()` dispatcher
6. If special handling: add branch in `compute_metric()` (like ece's n_bins)
7. Update `analysis/threshold_sweep.py` if it should appear in sweep

### Where LLM Advisory Features Would Integrate Safely

1. **After compare:** LLM endpoint already exists at `web/backend/app/routers/llm.py`
2. **After dashboard_summary:** Build prompt from DashboardSummary.to_jsonable()
3. **Safe integration point:** LLM features should be read-only consumers of ResultBundle and compare output. Never modify bundle or execution path.
4. **Risk mitigation:** LLM responses should not be stored in ResultBundle. Use separate persistence key.

### Layering Model (15 bullets)

1. **Spec layer** (spec.py) defines input schema via Pydantic. Validates all user input.
2. **Identity layer** (experiments/) produces deterministic experiment_id from canonical JSON + SHA256.
3. **Dataset layer** (datasets/loaders.py) loads CSV with surface-type-aware validation.
4. **Model layer** (model/) loads models, generates PredictionSurface objects, validates surfaces.
5. **Surface routing** (surface_compat.py) maps surface types to compatible scenarios and metrics.
6. **Metric layer** (metrics/) defines metric functions and applicability rules.
7. **Scenario layer** (scenarios/) implements data perturbations via Protocol + Registry pattern.
8. **Runner layer** (runner.py) executes Monte Carlo trials with seeded RNG.
9. **Diagnostics layer** (diagnostics/) provides calibration, gaming, and subgroup utilities.
10. **Analysis layer** (analysis/) runs post-hoc analysis: sweep, sensitivity, disagreement, failure modes, dashboard.
11. **Artifact layer** (artifacts/plots.py) generates matplotlib visualizations.
12. **Schema layer** (schema.py) defines ResultBundle output structure.
13. **Orchestration layer** (execution.py) wires everything together.
14. **Persistence layer** (db/ + experiments/registry.py) stores experiments, runs, artifacts, jobs.
15. **Web layer** (web/backend/) transforms engine outputs to API contracts for frontend consumption.

---

## EXECUTION PROMPT — Milestone 1 Only

Copy-paste this prompt to begin implementing Milestone 1:

---

**TASK: Implement Milestone 1 — Extract Constants & Centralize Threshold**

**Hard constraints:**
- Do NOT add features.
- Do NOT change any behavior.
- Do NOT touch tests (except adding imports if needed).
- Preserve exact values — this is a pure extraction refactor.

**Steps:**

1. Create `src/metrics_lie/surface_compat.py` with:
   - `SURFACE_TYPE_MAP: dict[str, SurfaceType]` (copy from execution.py:181-185)
   - `SCENARIO_SURFACE_COMPAT: dict[SurfaceType, set[str]]` (copy from execution.py:65-69)
   - `DEFAULT_THRESHOLD: float = 0.5`

2. Update `src/metrics_lie/execution.py`:
   - Remove the local `_SURFACE_TYPE_MAP` dict (around line 181)
   - Remove the module-level `SCENARIO_SURFACE_COMPAT` dict (around line 65)
   - Add import: `from metrics_lie.surface_compat import SURFACE_TYPE_MAP, SCENARIO_SURFACE_COMPAT, DEFAULT_THRESHOLD`
   - Replace `_SURFACE_TYPE_MAP` references with `SURFACE_TYPE_MAP`
   - Replace any `threshold=0.5` in baseline computation with `threshold=DEFAULT_THRESHOLD`

3. Add to `src/metrics_lie/metrics/core.py`:
   - `THRESHOLD_METRICS: set[str] = {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}`
   - `CALIBRATION_METRICS: set[str] = {"brier_score", "ece"}`
   - `RANKING_METRICS: set[str] = {"auc", "pr_auc", "logloss"}`

4. Update `src/metrics_lie/analysis/threshold_sweep.py`:
   - Import `THRESHOLD_METRICS` from `metrics_lie.metrics.core`
   - Replace local THRESHOLD_METRICS definition

5. Update `src/metrics_lie/analysis/sensitivity.py`:
   - Import metric category sets from `metrics_lie.metrics.core` if local sets exist
   - Replace local definitions

6. Update `src/metrics_lie/runner.py`:
   - Import `DEFAULT_THRESHOLD` from `metrics_lie.surface_compat`
   - Replace `threshold=0.5` with `threshold=DEFAULT_THRESHOLD`

7. Update `src/metrics_lie/diagnostics/subgroups.py`:
   - Import `DEFAULT_THRESHOLD` from `metrics_lie.surface_compat`
   - Replace `threshold=0.5` with `threshold=DEFAULT_THRESHOLD`

**Validation (run all, must be green):**
```bash
ruff check src/ tests/
pytest tests/ -x -q
python -m metrics_lie.cli run examples/experiment_minimal.json
pytest tests/test_phase7_determinism.py tests/test_phase8_surface_ingest.py tests/test_phase9_label_ingest.py tests/test_phase9_score_ingest.py -v
```

**STOP after validation passes. Do not proceed to Milestone 2.**
