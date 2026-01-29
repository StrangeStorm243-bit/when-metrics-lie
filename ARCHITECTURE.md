# Architecture & Design Notes

This document describes the **architecture, design principles, and evaluation philosophy**
behind *When Metrics Lie*.

It is intended to explain **how the system works** and **why it is structured this way**,
not to document individual functions or APIs.

---

## System Overview

At a high level, *When Metrics Lie* is an **evaluation pipeline**, not a training pipeline.

It takes a trained model’s outputs and asks:

> How reliable, stable, and decision-safe are the metrics we report?

The system is designed to be:
- deterministic
- modular
- auditable
- scenario-driven

---

## High-Level Architecture
<img width="1024" height="1536" alt="ChatGPT Image Jan 27, 2026, 05_37_13 PM" src="https://github.com/user-attachments/assets/8df7e244-a13e-47fe-b2f7-036591169e49" />

---
## Design Principles

### 1. Declarative Experiments
Experiments are defined via JSON specs:
- dataset
- metric
- scenarios
- random seed

This separates **what is evaluated** from **how it is executed** and enables
reproducibility and auditability.

---

### 2. Metrics as Distributions
Metrics are treated as **random variables**, not point estimates.

Instead of reporting:
- “Accuracy = 0.91”

The system reports:
- mean
- variance
- quantiles
- sensitivity to perturbation

This makes fragility visible.

---

### 3. Scenario-Driven Evaluation
Failure modes are modeled explicitly via scenarios.

Scenarios simulate realistic sources of evaluation risk, such as:
- noisy labels
- unstable scores
- distribution shifts

Each scenario is evaluated via Monte Carlo trials to capture variability.

---

### 4. Calibration as a First-Class Signal
Ranking metrics alone are insufficient for decision-making.

The system explicitly measures:
- Brier score
- Expected Calibration Error (ECE)

This distinguishes:
- “The model ranks well”
from
- “The model’s probabilities are trustworthy”

---

### 5. Subgroup Visibility
Global metrics can hide local failures.

When subgroup data is available, the system computes:
- per-group metrics
- per-group calibration
- worst vs best group gaps

Subgroup diagnostics are optional and computed only when alignment is safe.

---

### 6. Metric Gaming Is Modeled, Not Assumed Away
Evaluation choices themselves can inflate metrics.

For threshold-based metrics (e.g. accuracy), the system:
- optimizes the threshold
- compares baseline vs optimized performance
- quantifies metric inflation
- measures downstream harm

This makes evaluation manipulation explicit and measurable.

---

## Artifacts & Interpretability

The system produces artifacts alongside structured results to support human review.

Artifacts include:
- metric distribution plots
- calibration curves
- subgroup comparison charts
- threshold vs metric curves

Artifacts are:
- optional
- non-blocking
- deterministic
- stored alongside results

They are designed to be **shared and discussed**, not just logged.

---

## What This Architecture Intentionally Avoids

- Model training or fitting
- Online monitoring
- Persistent databases
- Dashboards or UI frameworks
- Cloud infrastructure

These are intentionally deferred to keep evaluation semantics clear and correct.

---

## Extension Points

The architecture is designed to be extended safely via:
- new scenarios
- additional diagnostics
- new artifact types
- experiment comparison (future phase)

Core evaluation contracts remain stable.

---

## Summary

*When Metrics Lie* is built around a simple idea:

> Evaluation failures are often more dangerous than model failures.

The architecture reflects this by prioritizing:
- robustness over optimization
- visibility over convenience
- decision safety over headline metrics
