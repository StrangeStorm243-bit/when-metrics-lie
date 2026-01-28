# When Metrics Lie

**An evaluation framework for detecting misleading model performance.**

Most model failures aren’t caused by bad models —  
they’re caused by **bad evaluation**.

When Metrics Lie is an evaluation system that surfaces failure modes hidden by standard metrics like accuracy and AUC, helping teams detect risk **before** deployment.

---

## Why This Exists

In practice, models often look “better” on paper while becoming:

- less reliable (poor calibration)
- less fair (subgroup collapse)
- more brittle (high sensitivity to noise)
- artificially inflated (metric gaming)

Traditional evaluation pipelines rarely catch this.

**When Metrics Lie is built to answer one question clearly:**

> Can I trust this metric as a decision signal?

---

## What It Does (At a Glance)

- **Declarative experiment specs**  
  Reproducible evaluation via versioned JSON configs.

- **Scenario stress testing**  
  Monte Carlo evaluation under controlled perturbations (label noise, score noise, class imbalance).

- **Distributional metrics**  
  Metrics are treated as distributions, not single numbers.

- **Calibration diagnostics**  
  Brier score and Expected Calibration Error (ECE).

- **Subgroup diagnostics**  
  Per-group performance, calibration, and worst-group gaps.

- **Sensitivity analysis**  
  Quantifies metric fragility under perturbation.

- **Metric gaming detection**  
  Shows how optimizing for a metric (e.g. threshold tuning) can inflate performance while degrading real quality.

- **Interpretable artifacts**  
  Automatic plots for sharing and review.

---

## A Real Failure Mode (Example)

A binary classifier is evaluated with accuracy:

- Accuracy improves after threshold optimization
- The model appears strictly “better”

But evaluation reveals:

- calibration worsens
- subgroup gaps increase
- decision confidence becomes misleading

The model didn’t change — the **evaluation did**.

When Metrics Lie detects and quantifies this behavior explicitly.

---

## How It Works


Each step is deterministic, auditable, and designed to surface evaluation risk.

---

## What Makes This Different

Most tools optimize models.  
**This tool evaluates evaluation.**

Key differences:

- focuses on decision reliability, not leaderboard metrics
- treats calibration and subgroup behavior as first-class signals
- explicitly models how metrics can be manipulated
- produces outputs designed for review, not just logging

This positions When Metrics Lie between model training and deployment decisions.

---

## Who This Is For

- ML engineers validating models pre-release
- Data scientists auditing performance claims
- Product and risk teams reviewing model readiness
- Researchers studying evaluation robustness

Not a training library.  
Not a monitoring dashboard.  
Intentionally scoped.

---

## Outputs

Each experiment produces:

- **Structured results** (`results.json`)
- **Interpretable artifacts**, including:
  - metric distribution plots
  - calibration curves
  - subgroup comparisons
  - threshold vs metric curves (when applicable)

Designed to be **shared, reviewed, and discussed**.

---

## Project Status

- Phase 1 MVP complete
- Core evaluation semantics stabilized
- Focused on correctness and interpretability

Future work will prioritize:

- experiment comparison
- regression detection
- decision-level summaries

UI and cloud layers are intentionally deferred.

---

## Philosophy

Metrics don’t fail loudly —  
they fail **quietly**.

When Metrics Lie exists to make those failures visible **before** they become costly.

---

## Notes

- Full Phase 1 implementation checklist: `docs/PHASE1_CHECKLIST.md`
- Architecture and design notes: `docs/ARCHITECTURE.md`
