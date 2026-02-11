import tempfile
from pathlib import Path

import numpy as np

from metrics_lie.artifacts.plots import (
    plot_calibration_curve,
    plot_metric_distribution,
    plot_subgroup_bars,
)


def test_plot_functions_smoke():
    """Smoke test that plot functions run without error on tiny dummy inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test metric distribution
        metric_summary = {
            "mean": 0.8,
            "std": 0.1,
            "q05": 0.6,
            "q50": 0.8,
            "q95": 1.0,
            "n": 100,
        }
        plot_metric_distribution(
            metric_summary=metric_summary,
            metric_name="auc",
            scenario_id="test",
            out_path=tmp_path / "test_metric.png",
        )
        assert (tmp_path / "test_metric.png").exists()

        # Test calibration curve
        y_true = np.array([0, 1, 0, 1, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        plot_calibration_curve(
            y_true=y_true,
            y_score=y_score,
            scenario_id="test",
            out_path=tmp_path / "test_cal.png",
        )
        assert (tmp_path / "test_cal.png").exists()

        # Test subgroup bars
        group_means = {"A": 0.8, "B": 0.6}
        plot_subgroup_bars(
            group_means=group_means,
            scenario_id="test",
            out_path=tmp_path / "test_subgroup.png",
        )
        assert (tmp_path / "test_subgroup.png").exists()
