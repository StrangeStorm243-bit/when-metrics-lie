from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics_lie.metrics.core import (
    metric_accuracy,
    metric_f1,
    metric_logloss,
    metric_matthews_corrcoef,
    metric_precision,
    metric_pr_auc,
    metric_recall,
    metric_auc,
)
from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.model.surface import PredictionSurface, SurfaceType


METRIC_FUNCS = {
    "accuracy": metric_accuracy,
    "f1": metric_f1,
    "precision": metric_precision,
    "recall": metric_recall,
    "matthews_corrcoef": metric_matthews_corrcoef,
    "auc": metric_auc,
    "pr_auc": metric_pr_auc,
    "logloss": metric_logloss,
    "brier_score": brier_score,
    "ece": expected_calibration_error,
}


@dataclass(frozen=True)
class SensitivityAnalysisResult:
    perturbation_type: str
    magnitudes: list[float]
    metric_responses: dict[str, list[float]]
    metric_slopes: dict[str, float]
    most_sensitive_metric: str
    least_sensitive_metric: str

    def to_jsonable(self) -> dict:
        return {
            "perturbation_type": self.perturbation_type,
            "magnitudes": self.magnitudes,
            "metric_responses": self.metric_responses,
            "metric_slopes": self.metric_slopes,
            "most_sensitive_metric": self.most_sensitive_metric,
            "least_sensitive_metric": self.least_sensitive_metric,
        }


def _apply_score_noise(
    scores: np.ndarray, *, sigma: float, rng: np.random.Generator, clip: bool
) -> np.ndarray:
    noisy = scores + rng.normal(loc=0.0, scale=sigma, size=scores.shape[0])
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def run_sensitivity_analysis(
    *,
    y_true: np.ndarray,
    surface: PredictionSurface,
    metrics: list[str],
    perturbation_type: str,
    magnitudes: list[float],
    n_trials: int = 50,
    seed: int = 42,
) -> SensitivityAnalysisResult:
    rng = np.random.default_rng(seed)
    metric_responses: dict[str, list[float]] = {m: [] for m in metrics}

    for mag in magnitudes:
        per_metric_vals: dict[str, list[float]] = {m: [] for m in metrics}
        for _ in range(n_trials):
            if perturbation_type == "score_noise":
                noisy = _apply_score_noise(
                    surface.values,
                    sigma=mag,
                    rng=rng,
                    clip=surface.surface_type == SurfaceType.PROBABILITY,
                )
            else:
                raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

            for metric_id in metrics:
                fn = METRIC_FUNCS.get(metric_id)
                if fn is None:
                    continue
                if metric_id in {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}:
                    val = fn(y_true, noisy, threshold=0.5)
                elif metric_id == "ece":
                    val = fn(y_true, noisy, n_bins=10)
                else:
                    val = fn(y_true, noisy)
                per_metric_vals[metric_id].append(float(val))

        for metric_id, vals in per_metric_vals.items():
            metric_responses[metric_id].append(float(np.mean(vals)))

    metric_slopes: dict[str, float] = {}
    for metric_id, vals in metric_responses.items():
        if len(vals) >= 2:
            slope = np.polyfit(magnitudes, vals, 1)[0]
            metric_slopes[metric_id] = float(slope)
        else:
            metric_slopes[metric_id] = 0.0

    # Sensitivity = absolute slope magnitude
    sorted_by_sensitivity = sorted(
        metric_slopes.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    most_sensitive_metric = sorted_by_sensitivity[0][0] if sorted_by_sensitivity else ""
    least_sensitive_metric = (
        sorted_by_sensitivity[-1][0] if sorted_by_sensitivity else ""
    )

    return SensitivityAnalysisResult(
        perturbation_type=perturbation_type,
        magnitudes=magnitudes,
        metric_responses=metric_responses,
        metric_slopes=metric_slopes,
        most_sensitive_metric=most_sensitive_metric,
        least_sensitive_metric=least_sensitive_metric,
    )
