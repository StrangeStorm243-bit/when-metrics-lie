When Metrics Lie — Phase 1 Checklist (MVP Foundation)

This document tracks the completion of Phase 1.1 through Phase 1.8, covering the full MVP foundation of the When Metrics Lie evaluation system.

Phase 1 focuses on core evaluation infrastructure, stress testing, and decision-relevant diagnostics, without UI, cloud, or persistence layers.

✅ Phase 1.1 — Repository & Project Foundation

 Project repository initialized

 Clean Python package structure (src/metrics_lie)

 Virtual environment setup

 pyproject.toml with dependencies and dev tooling

 Git hygiene (.gitignore, no artifacts committed)

 Deterministic environment setup

Outcome: Stable development baseline.

✅ Phase 1.2 — Experiment Specification & Schemas

 JSON-based experiment spec

 Explicit schema versioning

 Dataset configuration (paths, columns)

 Metric selection via spec

 Scenario definitions via spec

 Reproducibility via random seed

 Strong separation of config vs execution

Outcome: Experiments are declarative, reproducible, and auditable.

✅ Phase 1.3 — Baseline Evaluation Engine

 CSV dataset loader

 Binary classification support

 Metric registry (AUC, accuracy, etc.)

 Baseline metric computation

 Results written to structured results.json

 Deterministic outputs

Outcome: Reliable baseline evaluation pipeline.

✅ Phase 1.4 — Scenario Stress Testing (Monte Carlo)

 Scenario abstraction (Scenario.apply)

 Multiple scenario types (e.g. label noise, score noise)

 Monte Carlo trials per scenario

 Metric distributions (mean, std, quantiles)

 Scenario-specific parameters

 Scenario registration and extensibility

Outcome: Robust stress testing across simulated failure modes.

✅ Phase 1.5 — Sensitivity Diagnostics

 Sensitivity metric (sensitivity_abs)

 Quantifies metric fragility under perturbation

 Integrated into scenario diagnostics

 No schema changes required

Outcome: Direct measurement of metric stability.

✅ Phase 1.6 — Calibration Diagnostics

 Brier score implementation

 Expected Calibration Error (ECE)

 Baseline calibration diagnostics

 Scenario-level calibration distributions

 Calibration stored alongside metric diagnostics

Outcome: Separation of ranking performance vs probability quality.

✅ Phase 1.7A — Subgroup Diagnostics

 Optional subgroup column support

 Per-group metric distributions

 Per-group calibration diagnostics

 Worst vs best subgroup gap computation

 Safe handling of missing or misaligned subgroups

 No schema changes

Outcome: Detection of hidden fairness and group-level failures.

✅ Phase 1.7B — Artifacts & Interpretability

 Metric distribution plots

 Calibration curves (reliability diagrams)

 Subgroup comparison bar charts

 Artifacts saved under runs/<id>/artifacts/

 Artifact paths registered in results

 Non-interactive plotting backend

 Graceful degradation if plots unavailable

Outcome: Human-readable, shareable evaluation outputs.

✅ Phase 1.8 — Metric Gaming & Decision Fragility

 Threshold optimization for accuracy

 Demonstrates metric inflation without real improvement

 Baseline vs optimized metric comparison

 Metric inflation diagnostic (delta)

 Downstream impacts on calibration

 Downstream impacts on subgroup gaps

 Guarded activation (accuracy only)

 Threshold vs metric curve artifact

 Unit tests validating inflation behavior

Outcome: Explicit demonstration of how metrics can mislead decision-making.

🎯 Phase 1 Summary

By the end of Phase 1, the system supports:

Reproducible experiment specs

Scenario-based stress testing

Metric distributions and sensitivity analysis

Calibration and subgroup diagnostics

Interpretable artifacts

Demonstrations of metric manipulation

Phase 1 delivers a complete evaluation engine MVP — focused on surfacing risk, fragility, and misleading signals in model evaluation.

🚧 Out of Scope (Intentionally Deferred)

UI / dashboards

Cloud deployment

Persistence or experiment comparison

Model training or fitting

Production monitoring

These are candidates for Phase 2+.
