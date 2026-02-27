"""Spectra — Scenario-first ML evaluation engine."""
from __future__ import annotations

__version__ = "0.3.0"

from metrics_lie.sdk import compare, evaluate, evaluate_file, score
from metrics_lie.schema import ResultBundle
from metrics_lie.spec import ExperimentSpec

__all__ = [
    "__version__",
    "evaluate",
    "evaluate_file",
    "compare",
    "score",
    "ResultBundle",
    "ExperimentSpec",
]
