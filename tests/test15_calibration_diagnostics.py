import numpy as np

from metrics_lie.diagnostics.calibration import brier_score


def test_brier_score_perfect_predictions_is_zero():
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    assert brier_score(y_true, y_score) == 0.0
