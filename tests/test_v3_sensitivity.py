"""Tests for sensitivity analysis with multiclass and regression surfaces."""
from __future__ import annotations
import numpy as np
from metrics_lie.analysis.sensitivity import run_sensitivity_analysis
from metrics_lie.model.surface import CalibrationState, PredictionSurface, SurfaceType

def _make_surface(surface_type, values, **kwargs):
    return PredictionSurface(
        surface_type=surface_type, values=np.array(values), dtype=np.array(values).dtype,
        n_samples=len(values), class_names=kwargs.get("class_names", ("neg", "pos")),
        positive_label=kwargs.get("positive_label", 1),
        threshold=kwargs.get("threshold", 0.5),
        calibration_state=CalibrationState.UNKNOWN, model_hash=None, is_deterministic=True,
    )

def test_sensitivity_regression_continuous():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    surface = _make_surface(SurfaceType.CONTINUOUS, preds, threshold=None)
    result = run_sensitivity_analysis(
        y_true=y_true, surface=surface, metrics=["mae", "rmse"],
        perturbation_type="score_noise", magnitudes=[0.01, 0.05, 0.1], n_trials=10, seed=42,
    )
    assert result.most_sensitive_metric in ("mae", "rmse")
    assert len(result.magnitudes) == 3

def test_sensitivity_multiclass_label():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    surface = _make_surface(SurfaceType.LABEL, y_pred, class_names=("a", "b", "c"), threshold=None)
    result = run_sensitivity_analysis(
        y_true=y_true, surface=surface, metrics=["macro_f1", "cohens_kappa"],
        perturbation_type="score_noise", magnitudes=[0.01, 0.05, 0.1], n_trials=10, seed=42,
    )
    assert "macro_f1" in result.metric_responses
    assert "cohens_kappa" in result.metric_responses
