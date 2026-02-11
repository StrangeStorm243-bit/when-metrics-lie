import pytest

from metrics_lie.compare.compare import compare_bundles


def _make_bundle(
    run_id: str, baseline_mean: float, ece_mean: float, brier_mean: float, subgap: float
):
    return {
        "run_id": run_id,
        "metric_name": "auc",
        "baseline": {
            "mean": baseline_mean,
            "std": 0.0,
            "q05": baseline_mean,
            "q50": baseline_mean,
            "q95": baseline_mean,
            "n": 1,
        },
        "scenarios": [
            {
                "scenario_id": "label_noise",
                "metric": {
                    "mean": 0.85,
                    "std": 0.01,
                    "q05": 0.84,
                    "q50": 0.85,
                    "q95": 0.86,
                    "n": 200,
                },
                "diagnostics": {
                    "brier": {"mean": brier_mean},
                    "ece": {"mean": ece_mean},
                    "subgroup_gap": {"gap": subgap},
                    "sensitivity_abs": 0.05,
                },
                "artifacts": [],
            }
        ],
        "notes": {},
    }


def test_compare_bundles_baseline_and_flags():
    bundle_a = _make_bundle(
        "AAAA", baseline_mean=0.90, ece_mean=0.10, brier_mean=0.10, subgap=0.05
    )
    bundle_b = _make_bundle(
        "BBBB", baseline_mean=0.88, ece_mean=0.15, brier_mean=0.13, subgap=0.10
    )

    report = compare_bundles(bundle_a, bundle_b)

    assert report["baseline_delta"]["mean"] == pytest.approx(0.88 - 0.90)
    assert "label_noise" in report["scenario_deltas"]
    assert report["scenario_deltas"]["label_noise"]["ece_mean_delta"] == pytest.approx(
        0.15 - 0.10
    )

    assert report["regressions"]["metric"] is True
    assert report["regressions"]["calibration"] is True
    assert report["regressions"]["subgroup"] is True
    assert isinstance(report["risk_flags"], list)
    assert len(report["risk_flags"]) >= 2
