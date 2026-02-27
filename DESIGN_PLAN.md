# Spectra v1.0 — Universal ML Stress-Testing Platform: Design Plan

## Vision

Transform Spectra from a binary-classification-only evaluation tool into **the open-source standard for stress-testing any ML model**. Users upload any model (sklearn, PyTorch, TensorFlow, XGBoost, ONNX, HuggingFace, LLMs, or containerized), define stress scenarios, and get transparent, decision-oriented reports on where their model breaks.

**Tagline**: *"Don't trust your metrics. Prove them."*

---

## Current State (v0.2.0 engine / v0.3.0 web)

| Capability | Status |
|---|---|
| Binary classification only | Hardcoded throughout |
| sklearn pickle models only | ModelAdapter limited to pickle/import |
| 4 stress scenarios | label_noise, score_noise, class_imbalance, threshold_gaming |
| 8 metrics | AUC, F1, accuracy, logloss, Brier, ECE, MCC, PR-AUC |
| Comparison & decision scoring | Working (profiles, scorecards) |
| Web UI (Next.js + FastAPI) | Working with Clerk auth, Supabase option |
| CLI | Working (argparse) |

---

## Target State (v1.0)

| Capability | Target |
|---|---|
| Task types | Binary, multiclass, multi-label, regression, ranking, NLP, LLM |
| Model formats | sklearn, ONNX, PyTorch, TensorFlow, XGBoost/LightGBM/CatBoost, HuggingFace, GGUF, MLflow, Docker/HTTP endpoints |
| Stress scenarios | 15+ across tabular, text, image, distribution shift, fairness, adversarial |
| Metrics | 40+ via registry, community-extensible |
| Integrations | MLflow logging, Fairlearn fairness, Evidently drift, HF Evaluate metrics |
| Security | Sandboxed model loading, safetensors/ONNX preferred, pickle opt-in |
| Documentation | MkDocs site with API reference |
| SDK | `import spectra; spectra.evaluate(...)` public Python API |

---

## Competitive Landscape Summary

### Direct Competitors
| Tool | Overlap | Spectra's Edge |
|---|---|---|
| **Giskard** | Metamorphic testing, robustness scanning | Metric disagreement, calibration gaming, decision framework, scenario Monte Carlo |
| **Deepchecks** | Data/model validation checks | Deeper scenario-based stress testing, multi-metric comparison |

### Complementary Tools to Integrate
| Tool | License | What to Use |
|---|---|---|
| **Fairlearn** | MIT | MetricFrame for subgroup fairness (replace custom subgroup code) |
| **Evidently AI** | Apache 2.0 | Drift detection as pre-flight checks |
| **HF Evaluate** | Apache 2.0 | Community metrics backend (300+ metrics) |
| **MLflow** | Apache 2.0 | Model registry loading + result logging |
| **MAPIE** | BSD 3 | Conformal prediction / calibration for regression |
| **TextAttack** | MIT | NLP adversarial scenarios |
| **ART** | MIT | Adversarial robustness across modalities |
| **Cleanlab** | Apache 2.0 | Label quality detection via `find_label_issues()` |

### Unique Differentiators to Preserve
1. **Scenario-first Monte Carlo evaluation** (no other tool does this)
2. **Metric disagreement analysis** (unique)
3. **Decision framework with weighted scoring** (unique)
4. **Calibration gaming detection** (unique)
5. **Threshold sweep with crossover points** (unique)

---

## Architecture Evolution

### Current Architecture
```
ExperimentSpec → Runner (binary only) → ResultBundle → Compare → Decision
```

### Target Architecture
```
                     ┌─────────────────────────────────────┐
                     │         Spectra Platform             │
                     ├─────────────────────────────────────┤
                     │                                     │
                     │  ┌───────────┐   ┌──────────────┐  │
    Spec (YAML/JSON) │  │   Task    │   │   Model      │  │  Model (any format)
    ─────────────────┼─>│  Router   │   │   Adapter    │<─┼──────────────────
                     │  │           │   │   Registry   │  │
                     │  └─────┬─────┘   └──────┬───────┘  │
                     │        │                │          │
                     │        v                v          │
                     │  ┌─────────────────────────────┐   │
                     │  │     Scenario Engine          │   │
                     │  │  ┌────────┐ ┌────────────┐  │   │
                     │  │  │Tabular │ │   Text     │  │   │
                     │  │  │Perturb │ │  Perturb   │  │   │
                     │  │  ├────────┤ ├────────────┤  │   │
                     │  │  │ Image  │ │Distribution│  │   │
                     │  │  │Perturb │ │   Shift    │  │   │
                     │  │  ├────────┤ ├────────────┤  │   │
                     │  │  │Fairness│ │ Adversarial│  │   │
                     │  │  │  Test  │ │   Attack   │  │   │
                     │  │  └────────┘ └────────────┘  │   │
                     │  └──────────────┬──────────────┘   │
                     │                 │                   │
                     │                 v                   │
                     │  ┌─────────────────────────────┐   │
                     │  │      Metric Engine           │   │
                     │  │  (registry + HF Evaluate)    │   │
                     │  └──────────────┬──────────────┘   │
                     │                 │                   │
                     │                 v                   │
                     │  ┌─────────────────────────────┐   │
                     │  │   Analysis + Diagnostics     │   │
                     │  │  ┌──────┐ ┌───────┐ ┌─────┐ │   │
                     │  │  │Calib │ │Drift  │ │Fair │ │   │
                     │  │  │      │ │(Evid.)│ │learn│ │   │
                     │  │  └──────┘ └───────┘ └─────┘ │   │
                     │  └──────────────┬──────────────┘   │
                     │                 │                   │
                     │                 v                   │
                     │  ┌─────────────────────────────┐   │
                     │  │   ResultBundle (extended)     │   │
                     │  └──────┬───────────┬──────────┘   │
                     │         │           │              │
                     │         v           v              │
                     │  ┌──────────┐ ┌──────────────┐    │
                     │  │ Compare  │ │  MLflow Log  │    │
                     │  │ + Score  │ │  (optional)  │    │
                     │  └──────────┘ └──────────────┘    │
                     └─────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Task Router pattern**: A `TaskType` enum routes to task-specific metric sets, scenario filters, and output interpreters. The router is the single point where task-type knowledge lives.

2. **Model Adapter Registry**: Plugin-based. Each adapter registers for file extensions/format names. Loading priority: ONNX > safetensors > native formats > pickle (opt-in).

3. **Scenario Protocol stays**: The existing `Scenario.apply()` protocol generalizes cleanly. New scenarios just implement the same interface with task-type guards.

4. **Metric Registry extends**: Keep the current registry pattern but add an `evaluate` backend for community metrics. Custom metrics register the same way.

5. **ResultBundle extends, not replaces**: Add new fields for new task types. Existing binary classification results remain backward-compatible.

---

## Phased Implementation Plan

### Phase 1: Foundation — Universal Model Adapter (4-6 weeks)

**Goal**: Any model format loads and produces predictions through a unified interface.

#### 1.1 Model Adapter Protocol
```python
class ModelAdapter(Protocol):
    """Universal model adapter interface."""

    @property
    def task_type(self) -> TaskType: ...

    @property
    def capabilities(self) -> set[Capability]: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]: ...

    @property
    def metadata(self) -> ModelMetadata: ...
```

#### 1.2 Adapter Implementations (priority order)
| Adapter | Format | Security | Effort |
|---|---|---|---|
| `SklearnAdapter` | pickle/joblib/skops | Requires `--trust-pickle` | Refactor existing |
| `ONNXAdapter` | .onnx | Safe (protobuf graph) | New |
| `BoostingAdapter` | XGBoost (.ubj/.json), LightGBM (.txt), CatBoost (.cbm) | Safe (declarative) | New |
| `PyTorchAdapter` | .pt/.pth (state_dict), TorchScript | `weights_only=True` | New |
| `TensorFlowAdapter` | .keras, SavedModel | Generally safe | New |
| `HuggingFaceAdapter` | Transformers pipeline | Uses safetensors | New |
| `MLflowAdapter` | MLflow pyfunc | Depends on flavor | New |
| `HTTPAdapter` | REST endpoint (KServe V2 / custom) | Fully isolated | New |

#### 1.3 Security Layer (powered by modelscan, picklescan, fickling, magika)
- **Format detection**: `magika` auto-detects uploaded file type (10.1K stars, Google)
- **Pre-load scanning**: `modelscan` scans for malicious code across formats; `picklescan` for pickle-specific fast scan
- **Safe loading**: Replace `pickle.load()` with `fickling.load()` in `sources.py` (drop-in, Trail of Bits)
- **Auto-conversion**: `skl2onnx` converts sklearn → ONNX after scanning; infer via `onnxruntime` (no code execution)
- **Secure format**: Add `ModelSourceSkops` for `skops.io` format (MIT, no arbitrary code)
- Default: refuse raw pickle, require `--trust-pickle` or `trust_pickle=True`
- ONNX validation via `onnx.checker` before loading

#### 1.4 TaskType System
```python
class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"  # LLM
```

**Files to create/modify**:
- `src/metrics_lie/model/protocol.py` — New universal protocol
- `src/metrics_lie/model/adapters/` — New adapter directory (one file per adapter)
- `src/metrics_lie/model/registry.py` — Adapter registry (auto-discover by format)
- `src/metrics_lie/model/security.py` — Security validation layer
- `src/metrics_lie/task_types.py` — TaskType enum + routing logic
- Modify `src/metrics_lie/model/adapter.py` — Refactor existing to implement protocol
- Modify `src/metrics_lie/spec.py` — Add task_type field, new model_source kinds
- Modify `src/metrics_lie/execution.py` — Use adapter registry instead of hardcoded sklearn

#### Parallel work streams (for multiple terminals):
- **Terminal 1**: Adapter protocol + sklearn refactor + ONNX adapter
- **Terminal 2**: Boosting adapters + PyTorch/TF adapters
- **Terminal 3**: Tests for all adapters (TDD: write tests first)

---

### Phase 2: Multi-Task Metrics & Scenarios (4-6 weeks)

**Goal**: Metrics and scenarios work across all supported task types.

#### 2.1 Metric Registry Expansion

**Multiclass metrics** (extend existing):
- accuracy, weighted/macro/micro F1, precision, recall (generalize from binary)
- Cohen's Kappa, top-k accuracy, per-class metrics
- Multiclass ECE (per-class calibration)
- log_loss (already supports multiclass in sklearn)

**Regression metrics** (new):
- MAE, MSE, RMSE, MAPE, R-squared, adjusted R-squared
- Max error, explained variance, Huber loss
- Quantile loss, CRPS (via MAPIE integration)

**Ranking metrics** (new):
- NDCG@k, MRR, MAP@k, Precision@k, Recall@k, Hit Rate@k

**NLP metrics** (new, via HF Evaluate):
- ROUGE-1/2/L, BLEU, BERTScore, METEOR
- Exact match, token-level F1

**LLM metrics** (new):
- Perplexity, coherence (via LLM judge)
- Hallucination rate, factuality
- Safety/toxicity score

**Integration approach**:
```python
# Built-in metrics: direct implementation (fast, no dependencies)
# Community metrics: lazy-load via HF evaluate
# LLM metrics: separate optional dependency group

@metric_registry.register("rouge_l", task_types={TaskType.TEXT_GENERATION})
def rouge_l(y_true: list[str], y_pred: list[str]) -> float:
    import evaluate
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=y_pred, references=y_true)["rougeL"]
```

#### 2.2 Metric Applicability Extension
- Extend `MetricResolver` to filter by `TaskType` (not just `SurfaceType`)
- Each `MetricRequirement` gains a `task_types: set[TaskType]` field
- Backward compatible: binary classification metrics keep `{TaskType.BINARY_CLASSIFICATION}`

#### 2.3 Scenario Generalization

**Existing scenarios generalized**:
| Scenario | Binary | Multiclass | Regression | Text | Image |
|---|---|---|---|---|---|
| label_noise | Yes (flip) | Yes (random reassign) | Yes (add noise to target) | Yes (flip label) | Yes (flip label) |
| score_noise | Yes (Gaussian) | Yes (per-class noise) | Yes (Gaussian on output) | N/A | N/A |
| class_imbalance | Yes (subsample) | Yes (subsample per class) | N/A | Yes (subsample) | Yes (subsample) |
| threshold_gaming | Yes (threshold opt) | Yes (per-class threshold) | N/A | N/A | N/A |

**New scenarios** (with specific repo APIs):
| Scenario | Task Types | What It Does | Library | API Call |
|---|---|---|---|---|
| `missing_features` | Tabular | Drop features randomly | Built-in | — |
| `feature_corruption` | Tabular | Replace values with noise/outliers | Built-in | — |
| `covariate_shift` | Tabular | Reweight features to shift distribution | Evidently | `DataDriftPreset()` for validation |
| `typo_injection` | Text | Character-level perturbations | nlpaug | `nac.KeyboardAug().augment(text)` |
| `synonym_replacement` | Text | Semantic-preserving word swaps | nlpaug | `naw.SynonymAug(aug_src='wordnet').augment(text)` |
| `spelling_errors` | Text | Realistic misspellings | nlpaug | `naw.SpellingAug().augment(text)` |
| `back_translation` | Text | Paraphrase via translation roundtrip | AugLy | `txtaugs.simulate_typos(text, aug_p=0.3)` |
| `adversarial_text` | Text | TextFooler / BERT-Attack | TextAttack | `WordNetAugmenter(pct_words_to_swap=0.4).augment(text)` |
| `demographic_swap` | Text/Tabular | Swap protected attributes | AugLy | `txtaugs.swap_gendered_words(text)` |
| `image_corruption` | Image | 19 corruption types, severity 1-5 | imagecorruptions | `corrupt(image, corruption_name='gaussian_noise', severity=3)` |
| `adversarial_tabular` | Tabular | Gradient-based adversarial examples | ART | `FastGradientMethod(SklearnClassifier(model)).generate(x)` |
| `label_quality` | Any | Detect/inject realistic label errors | cleanlab | `find_label_issues(labels, pred_probs)` |
| `temporal_shift` | Time Series | Shift/missing timestamps | Built-in | — |

#### Parallel work streams:
- **Terminal 1**: Metric registry expansion (multiclass + regression)
- **Terminal 2**: Scenario generalization (tabular scenarios)
- **Terminal 3**: NLP/text scenarios (TextAttack integration)
- **Terminal 4**: Tests for all new metrics and scenarios

---

### Phase 3: Diagnostics & Analysis Generalization (3-4 weeks)

**Goal**: All analysis modules work across task types.

#### 3.1 Calibration Generalization
- Binary: Keep existing Brier/ECE (unchanged)
- Multiclass: Per-class reliability diagrams, multiclass ECE, top-label calibration
- Regression: Prediction interval coverage via MAPIE, quantile calibration
- Integration: `MAPIE` as optional dependency for conformal prediction

#### 3.2 Fairness Integration (Fairlearn)
- Replace `diagnostics/subgroups.py` internals with Fairlearn `MetricFrame`
- Preserve Spectra's API surface (subgroup_gap, group_sizes, etc.)
- Add new fairness metrics: demographic parity, equalized odds, disparate impact
- Add intersectional analysis (multiple protected attributes)

#### 3.3 Drift Detection (Evidently)
- Add pre-flight drift check: compare evaluation dataset to training data distribution
- Integrated as optional analysis step in `execution.py`
- Output drift report as part of `analysis_artifacts`
- Drift metrics: KS test, PSI, chi-squared (via Evidently presets)

#### 3.4 Analysis Module Extensions
- **Threshold sweep**: Already works for multiclass (per-class thresholds)
- **Sensitivity**: Generalize to any perturbation type (not just score_noise)
- **Disagreement**: Extend to multiclass metric pairs
- **Failure modes**: Per-sample contribution for regression (residual-based)
- **Dashboard**: Multi-task summary (metric risk by task type)

#### Parallel work streams:
- **Terminal 1**: Calibration generalization + MAPIE
- **Terminal 2**: Fairlearn integration
- **Terminal 3**: Evidently drift + analysis extensions

---

### Phase 4: Public Python SDK & CLI Upgrade (2-3 weeks)

**Goal**: Clean programmatic API for notebooks, scripts, and CI/CD.

#### 4.1 Public SDK
```python
import spectra

# Simple: one-line evaluation
result = spectra.evaluate("experiment.json")

# Programmatic: build spec in code
spec = spectra.ExperimentSpec(
    name="My Model Stress Test",
    task="multiclass_classification",
    dataset=spectra.Dataset.from_csv("data.csv", y_true="label", y_score="prediction"),
    model=spectra.Model.from_onnx("model.onnx"),
    scenarios=spectra.presets.standard_stress_suite,
    metrics="auto",  # auto-resolve based on task type
)
result = spectra.run(spec)

# Compare
comparison = spectra.compare(result_a, result_b)
comparison.summary()

# Score
decision = spectra.score(result_a, result_b, profile="risk_averse")
```

#### 4.2 CLI Upgrade (Typer)
```bash
# Current (argparse) → Target (Typer)
spectra run experiment.json                        # unchanged
spectra run experiment.json --model model.onnx     # new: override model
spectra run experiment.json --task multiclass       # new: override task type
spectra evaluate model.onnx --dataset data.csv     # new: quick evaluation
spectra compare run_a run_b --format table|json    # enhanced output
spectra models list                                # new: list supported formats
spectra scenarios list --task regression            # new: list scenarios per task
spectra metrics list --task multiclass              # new: list metrics per task
```

#### 4.3 Notebook Integration
- `spectra.display(result)` — Rich HTML display in Jupyter
- `spectra.plot(result)` — Interactive matplotlib/plotly figures
- Streamlit widget: `spectra.dashboard(result)` — Embedded dashboard

#### Parallel work streams:
- **Terminal 1**: SDK API design + implementation
- **Terminal 2**: CLI migration (argparse → Typer)
- **Terminal 3**: Notebook integration + display functions

---

### Phase 5: Web Platform Upgrade (3-4 weeks)

**Goal**: Web UI supports all task types and model formats.

#### 5.1 Backend Changes
- Extend `ExperimentCreateRequest` with `task_type` field
- Model upload: accept ONNX, safetensors, boosting native formats (not just pickle)
- Model validation: per-format validators (ONNX checker, sklearn check_is_fitted, etc.)
- Result contracts: extend `ResultSummary` for multiclass/regression/NLP results
- New endpoint: `GET /models/formats` — list supported formats and requirements

#### 5.2 Frontend Changes
- Task type selector on experiment creation page
- Model format auto-detection on upload
- Dynamic metric/scenario lists based on selected task type
- Results visualization per task type:
  - Multiclass: confusion matrix, per-class metrics
  - Regression: residual plots, prediction intervals
  - NLP: example perturbations, similarity scores
- Compare page: task-type-aware comparison

#### 5.3 LLM Analyst Enhancement
- Context-aware explanations per task type
- Scenario-specific insights (e.g., "This model is sensitive to synonym replacement")
- Decision recommendations with task-type knowledge

---

### Phase 6: Advanced Features & Ecosystem (4-6 weeks)

**Goal**: Production-grade platform with ecosystem integrations.

#### 6.1 MLflow Integration
- `spectra.log_to_mlflow(result)` — Log metrics, artifacts, and params to MLflow
- `spectra.Model.from_mlflow("runs:/run_id/model")` — Load from MLflow registry
- MLflow custom evaluator: use Spectra as an MLflow evaluation plugin

#### 6.2 Docker Model Runner Support
- `spectra.Model.from_endpoint("http://localhost:8080/v2/models/mymodel/infer")`
- KServe V2 protocol support (auto-detect from endpoint)
- Custom REST endpoint support (user specifies request/response schema)

#### 6.3 LLM Evaluation
- LLM-as-judge metrics (hallucination, factuality, safety)
- Prompt perturbation scenarios (paraphrase, instruction injection)
- Generation consistency testing
- Integration with DeepEval / RAGAS for RAG evaluation

#### 6.4 Documentation (MkDocs)
- API reference (auto-generated from docstrings)
- Getting started guide
- Task-type specific tutorials
- Integration guides (MLflow, Fairlearn, HF)
- Contributing guide (how to add metrics, scenarios, adapters)

#### 6.5 Community Features
- Plugin system for custom metrics, scenarios, and adapters
- `spectra plugin install <package>` — Install community plugins
- Template experiment specs per task type
- Example notebooks per task type

---

## Dependency Strategy

### Core (always installed)
```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.3
pydantic>=2.0
matplotlib>=3.7
typer>=0.9
```

### Optional Dependency Groups
```toml
[project.optional-dependencies]
onnx = ["onnxruntime>=1.16"]
pytorch = ["torch>=2.0"]
tensorflow = ["tensorflow>=2.15"]
boosting = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2"]
huggingface = ["transformers>=4.35", "safetensors"]
nlp = ["textattack>=0.3", "evaluate>=0.4"]
fairness = ["fairlearn>=0.10"]
drift = ["evidently>=0.4"]
calibration = ["mapie>=1.0"]
mlflow = ["mlflow>=2.10"]
llm = ["llama-cpp-python>=0.2"]
adversarial = ["foolbox>=3.3", "adversarial-robustness-toolbox>=1.17"]
web = ["fastapi>=0.104", "uvicorn"]
dev = ["pytest>=7.0", "ruff>=0.3"]
all = ["spectra[onnx,pytorch,tensorflow,boosting,huggingface,nlp,fairness,drift,calibration,mlflow,adversarial,web,dev]"]
```

### Why Optional Groups Matter
- Core install: `pip install spectra` — works for sklearn/tabular evaluation out of the box
- Users install only what they need: `pip install spectra[onnx,fairness]`
- CI tests each group independently
- No heavy dependencies forced on users who only need tabular evaluation

---

## Multi-Terminal Execution Strategy

Each phase naturally decomposes into 2-4 parallel work streams. The pattern:

```
Terminal 1 (Lead)          Terminal 2              Terminal 3              Terminal 4
────────────────          ──────────              ──────────              ──────────
Architecture/Protocol     Adapter impls           Tests (TDD)             Documentation
Core engine changes       Integration code        Integration tests       API examples
Review & merge            Review & merge          CI validation           Release prep
```

**Worktree isolation**: Each terminal works in a git worktree:
```bash
# Terminal 1: core protocol work
claude --worktree phase1-protocol

# Terminal 2: adapter implementations
claude --worktree phase1-adapters

# Terminal 3: test suite
claude --worktree phase1-tests
```

**Merge strategy**: Feature branches merge to a phase integration branch, which merges to main after all tests pass.

---

## Success Metrics

| Metric | Target |
|---|---|
| Supported task types | 7+ (binary, multiclass, multilabel, regression, ranking, text, LLM) |
| Model formats | 10+ (sklearn, ONNX, PyTorch, TF, XGB, LGBM, CatBoost, HF, MLflow, HTTP) |
| Stress scenarios | 15+ (across tabular, text, image, distribution shift, fairness) |
| Metrics | 40+ (built-in + HF Evaluate bridge) |
| Test coverage | >85% on core engine |
| Documentation pages | 30+ (MkDocs) |
| PyPI install | `pip install spectra` works with zero config |
| Time to first result | <5 minutes for a new user with an sklearn model |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Scope creep across task types | Ship each task type independently. Binary stays stable. |
| Dependency bloat | Strict optional groups. Core has zero ML framework deps beyond sklearn. |
| Breaking backward compatibility | Existing binary classification API is preserved. New features are additive. |
| Security of model loading | Pickle requires opt-in. ONNX/safetensors preferred. Docker sandbox option. |
| Integration library instability | Pin versions. Wrap integrations behind Spectra's own protocol. |
| LLM evaluation complexity | Ship deterministic metrics first. LLM-as-judge is Phase 6 (later). |

---

## Open Source Preparation Checklist

### License & Legal
- [ ] Choose license (Apache 2.0 recommended — compatible with all integration targets)
- [ ] Evaluate LGPL-3.0 compatibility for fickling dependency (or use modelscan only)
- [ ] Add LICENSE file, license headers

### Documentation (mkdocs-material + mkdocstrings)
- [ ] Set up `docs/` directory with `mkdocs.yml`
- [ ] Auto-generate API reference via mkdocstrings
- [ ] Write getting-started guide per task type
- [ ] Write integration guides (MLflow, Fairlearn, HF Evaluate)
- [ ] Create CONTRIBUTING.md with plugin development guide (pluggy hooks)
- [ ] Deploy to GitHub Pages

### CI/CD (nox + pre-commit + GitHub Actions)
- [ ] Create `noxfile.py` with sessions: tests, lint, docs, benchmarks
- [ ] Create `.pre-commit-config.yaml` (ruff, trailing-whitespace, check-yaml)
- [ ] Set up GitHub Actions matrix: Python 3.11/3.12, each optional group
- [ ] Add pytest-cov + codecov-action for coverage tracking
- [ ] Add pytest-benchmark for performance regression detection
- [ ] Set up pypa/gh-action-pypi-publish with Trusted Publishing

### Versioning (commitizen + towncrier)
- [ ] Configure commitizen in pyproject.toml for conventional commits
- [ ] Set up towncrier for changelog fragments (`changes/` directory)
- [ ] Add commitizen pre-commit hook

### Community
- [ ] Create `.github/ISSUE_TEMPLATE/` (bug, feature, new adapter, new scenario)
- [ ] Create `PULL_REQUEST_TEMPLATE.md` with checklist
- [ ] Install all-contributors bot for contributor recognition
- [ ] Set up Stale bot and Welcome bot via probot configs
- [ ] Set up GitHub Discussions
- [ ] Run `sp-repo-review` from scientific-python/cookie for audit

### Distribution
- [ ] Create PyPI package (`spectra-ml` or `spectra-eval` to avoid name conflicts)
- [ ] Add badges: CI, coverage, PyPI version, license, Python versions
- [ ] Write README with quick-start examples per task type
- [ ] Consider turborepo for cross-language build orchestration (Python + Next.js)

---

## Open Source Integration Catalog

This section catalogs every repo evaluated for integration, organized by verdict: **INTEGRATE** (add as dependency), **ADOPT** (copy the pattern/config), **REFERENCE** (study for ideas), or **SKIP** (not useful).

### INTEGRATE — Direct Dependencies

These repos become `pip install` dependencies in Spectra (core or optional groups).

#### Model Loading & Security

| Repo | Stars | License | What It Does for Spectra | Dependency Group |
|---|---|---|---|---|
| [**onnxruntime**](https://github.com/microsoft/onnxruntime) | 19.4K | MIT | Primary safe inference backend. `InferenceSession("model.onnx").run()` — no code execution risk. | `onnx` |
| [**skl2onnx**](https://github.com/onnx/sklearn-onnx) | 619 | Apache-2.0 | Auto-convert sklearn pickles → ONNX after security scan. Covers 133/194 sklearn operators. | `onnx` |
| [**onnxmltools**](https://github.com/onnx/onnxmltools) | 1.1K | Apache-2.0 | Convert XGBoost, LightGBM, CatBoost → ONNX. Extends model format coverage. | `boosting` |
| [**modelscan**](https://github.com/protectai/modelscan) | 647 | Apache-2.0 | Pre-load security scanner for pickle, H5, SavedModel, PyTorch. Severity-ranked findings. | `core` (web backend) |
| [**picklescan**](https://github.com/mmaitre314/picklescan) | 393 | MIT | Fast first-pass pickle-specific scanner. `scan_file_path()` → infected_count. | `core` |
| [**fickling**](https://github.com/trailofbits/fickling) | 604 | LGPL-3.0 | Drop-in safe pickle loader. Replace `pickle.load(f)` with `fickling.load(f)` in `sources.py`. | `core` (evaluate LGPL compat) |
| [**skops**](https://github.com/skops-dev/skops) | 512 | MIT | Secure sklearn serialization format. `sio.load("model.skops", trusted=types)` — no arbitrary code. | `core` |
| [**magika**](https://github.com/google/magika) | 10.1K | Apache-2.0 | AI-powered file type detection. Auto-detect uploaded model format before routing to loader. | `core` (web backend) |

**Model security pipeline** (the key integration):
```
Upload → Magika (detect format) → picklescan/modelscan (scan threats)
  → fickling.load() (safe load) → skl2onnx (convert to ONNX)
  → onnxruntime (safe inference)
```

#### Data Perturbation & Stress Testing

| Repo | Stars | License | What It Does for Spectra | Dependency Group |
|---|---|---|---|---|
| [**nlpaug**](https://github.com/makcedward/nlpaug) | 4.6K | MIT | Text perturbation: `KeyboardAug()`, `SynonymAug()`, `SpellingAug()`, `ContextualWordEmbsAug()`, `BackTranslationAug()`. Composable via `Flow`. | `nlp` |
| [**AugLy**](https://github.com/facebookresearch/AugLy) | 5.1K | MIT | Multi-modal perturbation (text + image + audio + video). `simulate_typos()`, `change_case()`, image corruption. Intensity metadata. | `nlp` / `image` |
| [**imagecorruptions**](https://github.com/bethgelab/imagecorruptions) | 463 | Apache-2.0 | 19 image corruption types with severity 1-5. `corrupt(img, 'gaussian_noise', severity=3)` — maps directly to Spectra scenarios. | `image` |
| [**TextAttack**](https://github.com/QData/TextAttack) | 3.4K | MIT | `textattack.augmentation` module: `WordNetAugmenter`, `CharSwapAugmenter`, `EmbeddingAugmenter`. Use augmenters only (not full attack framework). | `nlp` |
| [**ART**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | 5.8K | MIT | Adversarial attacks on tabular data. `SklearnClassifier` wrapper + `FastGradientMethod`/`ProjectedGradientDescent`. LF AI Foundation backing. | `adversarial` |

#### Diagnostics & Analysis

| Repo | Stars | License | What It Does for Spectra | Dependency Group |
|---|---|---|---|---|
| [**Fairlearn**](https://github.com/fairlearn/fairlearn) | 2K+ | MIT | `MetricFrame` for disaggregated subgroup metrics. Replace `diagnostics/subgroups.py` internals. Add demographic parity, equalized odds. | `fairness` |
| [**Evidently AI**](https://github.com/evidentlyai/evidently) | 7.2K | Apache-2.0 | Drift detection (PSI, KL, Wasserstein, KS-test). Pre-flight dataset checks + post-perturbation validation. Rich HTML reports. | `drift` |
| [**MAPIE**](https://github.com/scikit-learn-contrib/MAPIE) | 1.5K+ | BSD-3 | Conformal prediction for regression calibration. Prediction intervals with guaranteed coverage. | `calibration` |
| [**cleanlab**](https://github.com/cleanlab/cleanlab) | 11.3K | Apache-2.0 | Label quality detection. `find_label_issues(labels, pred_probs)` validates label_noise scenarios. sklearn-native. | `core` (optional diagnostic) |

#### CLI, Plugins & Infrastructure

| Repo | Stars | License | What It Does for Spectra | Dependency Group |
|---|---|---|---|---|
| [**Typer**](https://github.com/fastapi/typer) | 18.9K | MIT | Replace argparse with type-hint-driven CLI. Same author as FastAPI. Auto-completion, rich help. | `core` |
| [**pluggy**](https://github.com/pytest-dev/pluggy) | 1.6K | MIT | Plugin system (from pytest). `@hookspec`/`@hookimpl` decorators. Enable community metrics, scenarios, adapters. | `core` |
| [**HiPlot**](https://github.com/facebookresearch/hiplot) | 2.8K | MIT | Interactive parallel coordinates for scenario sweep visualization. `Experiment.from_iterable(data).display()`. | `notebook` |

#### Documentation & CI

| Repo | Stars | License | What It Does for Spectra | Dependency Group |
|---|---|---|---|---|
| [**mkdocs-material**](https://github.com/squidfunk/mkdocs-material) | 26.1K | MIT | Documentation site. Same ecosystem as FastAPI. Search, dark mode, versioning. | `docs` |
| [**mkdocstrings**](https://github.com/mkdocstrings/mkdocstrings) | 2.1K | ISC | Auto-generate API reference from docstrings + type annotations. Zero manual authoring. | `docs` |
| [**hypothesis**](https://github.com/HypothesisWorks/hypothesis) | 8.5K | MPL-2.0 | Property-based testing. `hypothesis.extra.numpy` strategies for array inputs. Test "for any valid surface, metric never raises." | `dev` |
| [**pytest-cov**](https://github.com/pytest-dev/pytest-cov) | 2K | MIT | Coverage tracking. `--cov=src/metrics_lie --cov-report=xml`. | `dev` |
| [**pytest-benchmark**](https://github.com/ionelmc/pytest-benchmark) | 1.4K | BSD-2 | Performance regression detection. Benchmark `run_scenario`, metric computation. JSON history. | `dev` |
| [**nox**](https://github.com/wntrblm/nox) | 1.5K | Apache-2.0 | Test automation via `noxfile.py`. Sessions for tests, lint, docs, benchmarks. Replaces bare pytest in CI. | `dev` |
| [**commitizen**](https://github.com/commitizen-tools/commitizen) | 3.3K | MIT | Automated versioning from conventional commits. `cz bump` → version bump + CHANGELOG generation. | `dev` |
| [**pre-commit**](https://github.com/pre-commit/pre-commit) | 15.1K | MIT | Git hooks framework. Ruff lint + format, trailing whitespace, check-yaml, detect-secrets. | `dev` |
| [**codecov-action**](https://github.com/codecov/codecov-action) | 1.7K | MIT | Upload coverage to Codecov. PR annotations, coverage diffs. | CI workflow |
| [**gh-action-pypi-publish**](https://github.com/pypa/gh-action-pypi-publish) | 1.1K | BSD-3 | Trusted Publishing to PyPI. OIDC, no tokens. Tag → build → publish. | CI workflow |
| [**all-contributors**](https://github.com/all-contributors/all-contributors) | 8K | MIT | Contributor recognition bot. Auto-update contributors table in README. | Community |

---

### ADOPT — Patterns to Copy (not dependencies)

These repos provide design patterns, configurations, or templates worth copying into Spectra's codebase.

#### Architecture Patterns

| Pattern | Source Repo | Stars | Apply To |
|---|---|---|---|
| Decorator-based registry (`@scorer`, `@metric`, `@task`) | [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) | 1.8K | Spectra's metric + scenario registries. Replace module-level registration with `@metric_registry.register()` decorators. |
| Typed lifecycle hooks (`RunStart`, `TaskEnd` dataclasses) | [inspect-ai](https://github.com/UKGovernmentBEIS/inspect_ai) | 1.8K | Add `on_run_start`, `on_scenario_complete`, `on_run_end` hooks to `execution.py`. Enables plugin lifecycle. |
| YAML task definition schema | [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 11.5K | Extend ExperimentSpec to support YAML alongside JSON. Their `metric_list` pattern maps to Spectra's metric config. |
| `MetricCollection` grouping | [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) | 2.4K | Group metrics by task type with `MetricCollection(metrics).compute()` pattern. |
| `Metric` base class with `update()`/`compute()` | [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) | 2.4K | Stateful metric accumulation for streaming/batched evaluation. |
| Report/TestSuite duality | [Evidently AI](https://github.com/evidentlyai/evidently) | 7.2K | Same computation, two output modes: Report (display) vs TestSuite (pass/fail). Apply to ResultBundle. |
| `evaluate.load("metric_name")` factory pattern | [HF Evaluate](https://github.com/huggingface/evaluate) | 2.4K | Lazy metric loading. `spectra.metrics.load("rouge_l")` for optional/community metrics. |
| MFT/INV/DIR test taxonomy | [CheckList](https://github.com/marcotcr/checklist) | 2K | Label Spectra scenarios: label_noise = INV (invariance), threshold_gaming = DIR (directional). |
| `DriverManager` for adapter selection | [stevedore](https://github.com/openstack/stevedore) | 242 | `entry_points`-based adapter discovery: `spectra.model_adapters` namespace. |

#### Configuration & Build

| Pattern | Source Repo | Stars | Apply To |
|---|---|---|---|
| Config composition with overrides | [Hydra](https://github.com/facebookresearch/hydra) | 10.2K | `spectra run experiment.yaml scenario=label_noise scenario.noise_rate=0.2` — CLI override of spec fields. |
| Structured Configs (dataclass → YAML) | [OmegaConf](https://github.com/omry/omegaconf) | 2.3K | Bidirectional ExperimentSpec ↔ YAML with interpolation (`${data_dir}/train.csv`). |
| Immutable config with key locking | [ml_collections](https://github.com/google/ml_collections) | 1K | `FrozenConfigDict` for experiment identity — prevent accidental mutation after fingerprinting. |
| Turborepo orchestration | [turborepo](https://github.com/vercel/turborepo) | 29.9K | `turbo.json` at repo root. Single `turbo run test` for both Python + Next.js with caching. |
| Towncrier changelog fragments | [towncrier](https://github.com/twisted/towncrier) | 883 | `changes/123.feature.md` per PR → `towncrier build` assembles CHANGELOG. No merge conflicts. |
| `sp-repo-review` audit | [scientific-python/cookie](https://github.com/scientific-python/cookie) | 385 | Run audit against Spectra repo for pyproject.toml, CI, typing, docs gaps. |

#### Community & Open Source

| Pattern | Source Repo | Stars | Apply To |
|---|---|---|---|
| Issue/PR templates | [huggingface/transformers](https://github.com/huggingface/transformers) | 145K | `.github/ISSUE_TEMPLATE/` (bug, feature, new adapter). `PULL_REQUEST_TEMPLATE.md` with checklist. |
| Welcome/Stale bots | [probot](https://github.com/probot/probot) | 9.5K | `.github/stale.yml` (close inactive), `.github/welcome.yml` (greet first-time contributors). |
| Test taxonomy (unit/behavioral/perf) | [eugeneyan/testing-ml](https://github.com/eugeneyan/testing-ml) | 265 | Organize `tests/` into implementation tests, behavioral tests (scenario invariance), performance benchmarks. |

---

### REFERENCE — Study But Don't Depend On

These repos provide valuable design lessons but are not direct dependencies.

#### Competitors to Study (Not Integrate)

| Repo | Stars | License | What to Learn |
|---|---|---|---|
| [Giskard](https://github.com/Giskard-AI/giskard-oss) | 5.1K | Apache-2.0 | Scanner/detector registry, model wrapper protocol, metamorphic testing patterns |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | 4K | **AGPL-3.0** | Check + Condition + Suite composable validation pattern. **Do not integrate (AGPL).** |
| [Stanford HELM](https://github.com/stanford-crfm/helm) | 2.7K | Apache-2.0 | RunSpec → Run → Summarize → Serve pipeline. `schema_classic.yaml` for benchmark definitions. |
| [PyCaret](https://github.com/pycaret/pycaret) | 9.7K | MIT | `compare_models()` tabular scoring grid — output format reference for comparison page. |

#### Libraries With License Issues

| Repo | Stars | License | Issue |
|---|---|---|---|
| [alibi-detect](https://github.com/SeldonIO/alibi-detect) | 2.5K | **BSL 1.1** | Great drift detectors (KS, MMD, Chi2) but requires commercial license for production. Use Evidently instead. |
| [SDV](https://github.com/sdv-dev/SDV) | 3.4K | **BSL 1.1** | Synthetic data generation. License prohibits production bundling. Reference patterns only. |
| [Albumentations](https://github.com/albumentations-team/albumentations) | 15.3K | **AGPL-3.0** (AlbumentationsX) | Original archived (MIT). New version AGPL. Use `imagecorruptions` instead. |

#### Design References

| Repo | Stars | License | Pattern to Study |
|---|---|---|---|
| [MLflow](https://github.com/mlflow/mlflow) | 24.4K | Apache-2.0 | `pyfunc.load_model()` universal interface, `@scorer` decorator, artifact logging. Too heavy as dependency. |
| [BentoML](https://github.com/bentoml/BentoML) | 8.5K | Apache-2.0 | Model store + runner abstraction. Study, don't depend. |
| [Hummingbird](https://github.com/microsoft/hummingbird) | 3.5K | MIT | Tree models → tensor computation conversion. Covered by skl2onnx. |
| [Foolbox](https://github.com/bethgelab/foolbox) | 2.9K | MIT | Decision-based adversarial attacks. Relevant when adding neural network support. |
| [CleverHans](https://github.com/cleverhans-lab/cleverhans) | 6.4K | MIT | Clean adversarial attack reference implementations. No sklearn support. |
| [River](https://github.com/online-ml/river) | 5.7K | BSD-3 | `ConceptDriftStream` for synthetic drift scenarios. Streaming paradigm doesn't fit batch eval. |
| [ydata-profiling](https://github.com/ydataai/ydata-profiling) | 13.4K | MIT | Jinja2 HTML report architecture, `compare()` for dataset comparison reports. |
| [Sweetviz](https://github.com/fbdesignpro/sweetviz) | 3.1K | MIT | One-liner comparison report. Self-contained HTML pattern. |
| [Plotly Dash](https://github.com/plotly/dash) | 24.4K | MIT | Reactive callback dashboard. Alternative to Next.js for Python-native users. |
| [DVC](https://github.com/iterative/dvc) | 15.4K | Apache-2.0 | CLI subcommand architecture. `dvc exp` for experiment tracking patterns. |
| [Great Expectations](https://github.com/great-expectations/great_expectations) | 11.2K | Apache-2.0 | Declarative data expectations pattern. "Data Docs" HTML report auto-generation. |
| [doubtlab](https://github.com/koaning/doubtlab) | 517 | MIT | Composable "Reason" pipeline for label quality checks. |
| [Trail of Bits ML File Formats](https://github.com/trailofbits/ml-file-formats) | 67 | Apache-2.0 | Documents 50+ ML file format internals. Magic bytes for format detection. |
| [Treelite](https://github.com/dmlc/treelite) | 812 | Apache-2.0 | Universal tree-model serialization. Covered by ONNX path. |

---

### SKIP — Evaluated and Rejected

| Repo | Stars | Why Skip |
|---|---|---|
| modelstore | 401 | Overlaps with Spectra's existing registry. No inference value. |
| ml2rt | 27 | Abandoned (2022). Not production-grade. |
| LitServe | 3.8K | Spectra already has FastAPI. Wrong abstraction level. |
| Cog (Replicate) | 9.3K | Docker-based deployment, not evaluation. |
| niacin | 18 | 18 stars, abandoned since 2022. |
| textaugment | 432 | Everything it does, nlpaug does better. |
| DeltaPy | 556 | Stale since 2023. API design reference only. |
| frouros | 252 | Good API but Evidently covers same ground with 30x community. |
| ydata-synthetic | 1.6K | GAN-based synthesis is heavyweight for perturbation. Reference only. |
| Google Model Card Toolkit | 444 | Archived/stale since 2023. Schema pattern is useful but the code isn't. |

---

## Updated Dependency Strategy

### Core (always installed)
```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.3
pydantic>=2.0
matplotlib>=3.7
typer>=0.9
pluggy>=1.4
```

### Optional Dependency Groups
```toml
[project.optional-dependencies]
# Model formats
onnx = ["onnxruntime>=1.16", "skl2onnx>=1.16"]
boosting = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2", "onnxmltools>=1.12"]
pytorch = ["torch>=2.0"]
tensorflow = ["tensorflow>=2.15"]
huggingface = ["transformers>=4.35", "safetensors"]
llm = ["llama-cpp-python>=0.2"]

# Model security
security = ["modelscan>=0.8", "picklescan>=1.0", "fickling>=0.1", "magika>=1.0"]

# Perturbation & stress testing
nlp = ["nlpaug>=1.1", "textattack>=0.3", "augly[text]>=1.0"]
image = ["imagecorruptions>=1.1", "augly[image]>=1.0"]
adversarial = ["adversarial-robustness-toolbox>=1.17"]

# Diagnostics & analysis
fairness = ["fairlearn>=0.10"]
drift = ["evidently>=0.4"]
calibration = ["mapie>=1.0"]
labels = ["cleanlab>=2.6"]
metrics = ["evaluate>=0.4"]

# Visualization
notebook = ["hiplot>=0.1"]

# Infrastructure
mlflow = ["mlflow>=2.10"]
web = ["fastapi>=0.104", "uvicorn"]

# Development
dev = [
    "pytest>=7.0", "ruff>=0.3", "hypothesis>=6.0",
    "pytest-cov>=4.0", "pytest-benchmark>=4.0",
    "nox>=2024.0", "pre-commit>=3.0", "commitizen>=3.0",
]
docs = ["mkdocs-material>=9.0", "mkdocstrings[python]>=0.24"]

# Everything
all = [
    "spectra[onnx,boosting,pytorch,tensorflow,huggingface,llm]",
    "spectra[security,nlp,image,adversarial]",
    "spectra[fairness,drift,calibration,labels,metrics]",
    "spectra[notebook,mlflow,web,dev,docs]",
]
```

---

## Tools & Skills to Leverage During Development

### MCP Servers
- **Context7** (already configured) — Library documentation lookup
- **Supabase MCP** — Direct schema management for hosted backend
- **GitHub MCP** — PR/issue management during development
- **Remotion docs** (already configured) — If video documentation needed

### Claude Code Skills (already available)
- **superpowers:brainstorming** — Before each phase's design
- **superpowers:writing-plans** — Detailed phase specs
- **superpowers:test-driven-development** — TDD for all new code
- **superpowers:dispatching-parallel-agents** — Multi-terminal coordination
- **superpowers:subagent-driven-development** — Parallel task execution
- **superpowers:using-git-worktrees** — Isolated development branches
- **superpowers:systematic-debugging** — When things break
- **superpowers:verification-before-completion** — Before marking phases done
- **superpowers:requesting-code-review** — Phase completion reviews
- **superpowers:finishing-a-development-branch** — Merge/PR decisions

### Development Patterns
- TDD throughout: tests first, implementation second
- Each adapter gets its own test file with golden output
- Integration tests per task type
- Benchmark suite for performance regression detection
