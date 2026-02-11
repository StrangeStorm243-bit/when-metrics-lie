from __future__ import annotations

# Transparent, easy-to-tune heuristics (Phase 2.3)
CALIBRATION_REGRESSION_THRESHOLD = 0.02
SUBGROUP_GAP_REGRESSION_THRESHOLD = 0.03
METRIC_REGRESSION_THRESHOLD = -0.01  # baseline_mean_delta < this is a regression
