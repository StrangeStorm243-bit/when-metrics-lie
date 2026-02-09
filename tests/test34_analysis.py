from __future__ import annotations

import numpy as np

from metrics_lie.analysis import (
    analyze_metric_disagreements,
    locate_failure_modes,
    run_sensitivity_analysis,
    run_threshold_sweep,
)
from metrics_lie.model.surface import CalibrationState, PredictionSurface, SurfaceType


def _surface() -> PredictionSurface:
    values = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)
    return PredictionSurface(
        surface_type=SurfaceType.PROBABILITY,
        values=values,
        dtype=values.dtype,
        n_samples=len(values),
        class_names=("negative", "positive"),
        positive_label=1,
        threshold=0.5,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )


def test_threshold_sweep_outputs() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    surface = _surface()
    res = run_threshold_sweep(
        y_true=y_true,
        surface=surface,
        metrics=["accuracy", "auc", "pr_auc", "f1"],
        n_points=11,
    )
    assert res.thresholds.shape[0] == 11
    assert "accuracy" in res.metric_curves
    assert "auc" in res.metric_curves
    assert "f1" in res.metric_curves
    assert "accuracy" in res.optimal_thresholds


def test_sensitivity_analysis_outputs() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    surface = _surface()
    res = run_sensitivity_analysis(
        y_true=y_true,
        surface=surface,
        metrics=["accuracy", "auc"],
        perturbation_type="score_noise",
        magnitudes=[0.01, 0.02],
        n_trials=3,
        seed=1,
    )
    assert len(res.magnitudes) == 2
    assert len(res.metric_responses["accuracy"]) == 2


def test_metric_disagreement_and_failure_modes() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    surface = _surface()
    sweep = run_threshold_sweep(
        y_true=y_true,
        surface=surface,
        metrics=["accuracy", "f1"],
        n_points=5,
    )
    disagreements = analyze_metric_disagreements(
        y_true=y_true,
        surface=surface,
        thresholds=sweep.optimal_thresholds,
        metrics=["accuracy", "f1"],
    )
    assert len(disagreements) >= 0
    failures = locate_failure_modes(
        y_true=y_true,
        surface=surface,
        metrics=["accuracy", "f1"],
        top_k=2,
    )
    assert failures.total_samples == 4
    assert len(failures.failure_samples) == 2


def test_failure_modes_worst_subgroup_is_max() -> None:
    """worst_subgroup must be the group with the HIGHEST mean contribution (worst performance)."""
    # "bad" group: y_true=1 but surface predicts ~0 → high contribution
    # "good" group: y_true=1 and surface predicts ~1 → low contribution
    y_true = np.array([1, 1, 1, 1], dtype=int)
    values = np.array([0.1, 0.15, 0.95, 0.9], dtype=float)  # bad, bad, good, good
    surface = PredictionSurface(
        surface_type=SurfaceType.PROBABILITY,
        values=values,
        dtype=values.dtype,
        n_samples=4,
        class_names=("negative", "positive"),
        positive_label=1,
        threshold=0.5,
        calibration_state=CalibrationState.UNKNOWN,
        model_hash=None,
        is_deterministic=True,
    )
    subgroup = np.array(["bad", "bad", "good", "good"])
    report = locate_failure_modes(
        y_true=y_true,
        surface=surface,
        metrics=["accuracy"],
        subgroup=subgroup,
        top_k=4,
    )
    assert report.worst_subgroup == "bad"
