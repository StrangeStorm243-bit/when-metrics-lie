# Spectra

**Spectra** is a scenario-first evaluation engine for machine learning models.  
It helps surface where metrics lie by stress-testing models across realistic failure modes and producing transparent, decision-oriented comparisons.

This repository contains the core engine and CLI.

---

## What Spectra Does

Spectra evaluates models by:

- Running **scenario-based perturbations** (e.g. label noise, score noise, class imbalance)
- Comparing runs using **structured diagnostics** (calibration, subgroup gaps, sensitivity, gaming)
- Aggregating results into **decision components**
- Producing **transparent, weighted scorecards** via decision profiles

The system is designed to be:

- Deterministic  
- Reproducible  
- Explainable  
- Extensible  

---

## Quickstart

### Install (editable)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

