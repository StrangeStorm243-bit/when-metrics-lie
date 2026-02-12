"""Phase 8 M4 tests: Compare semantics for mismatched metric sets."""

from __future__ import annotations

from metrics_lie.compare.compare import compare_bundles


def test_compare_multi_metric_shared() -> None:
    """Two bundles with identical metric sets have correct shared_metrics."""
    bundle_a = {
        "run_id": "A",
        "metric_name": "auc",
        "baseline": {"mean": 0.85},
        "scenarios": [],
        "metric_results": {
            "auc": {"mean": 0.85, "std": 0.01},
            "accuracy": {"mean": 0.80, "std": 0.02},
            "logloss": {"mean": 0.45, "std": 0.03},
        },
    }
    bundle_b = {
        "run_id": "B",
        "metric_name": "auc",
        "baseline": {"mean": 0.87},
        "scenarios": [],
        "metric_results": {
            "auc": {"mean": 0.87, "std": 0.01},
            "accuracy": {"mean": 0.82, "std": 0.02},
            "logloss": {"mean": 0.42, "std": 0.03},
        },
    }

    report = compare_bundles(bundle_a, bundle_b)

    assert "multi_metric_comparison" in report
    mmc = report["multi_metric_comparison"]

    # Same metrics in both bundles
    assert mmc["shared_metrics"] == ["accuracy", "auc", "logloss"]
    assert mmc["only_in_a"] == []
    assert mmc["only_in_b"] == []

    # Check per-metric deltas
    assert "auc" in mmc["per_metric_deltas"]
    assert mmc["per_metric_deltas"]["auc"]["baseline_delta"] == 0.87 - 0.85
    assert mmc["per_metric_deltas"]["auc"]["a"] == 0.85
    assert mmc["per_metric_deltas"]["auc"]["b"] == 0.87

    # All three improved (auc: 0.85->0.87, accuracy: 0.80->0.82, logloss: 0.45->0.42 which is -0.03 so regressed)
    # Actually logloss 0.45->0.42 is improvement (lower is better for logloss)
    # But our comparison is purely delta-based, so -0.03 counts as regressed
    assert mmc["improved_count"] == 2  # auc and accuracy improved
    assert mmc["regressed_count"] == 1  # logloss (delta negative)


def test_compare_multi_metric_mismatch() -> None:
    """Bundles with different metric sets show correct intersection and only-in lists."""
    bundle_a = {
        "run_id": "A",
        "metric_name": "auc",
        "baseline": {"mean": 0.85},
        "scenarios": [],
        "metric_results": {
            "auc": {"mean": 0.85, "std": 0.01},
            "ece": {"mean": 0.10, "std": 0.01},
        },
    }
    bundle_b = {
        "run_id": "B",
        "metric_name": "auc",
        "baseline": {"mean": 0.87},
        "scenarios": [],
        "metric_results": {
            "auc": {"mean": 0.87, "std": 0.01},
            "accuracy": {"mean": 0.82, "std": 0.02},
        },
    }

    report = compare_bundles(bundle_a, bundle_b)

    mmc = report["multi_metric_comparison"]

    # Intersection
    assert mmc["shared_metrics"] == ["auc"]
    # Only in A
    assert mmc["only_in_a"] == ["ece"]
    # Only in B
    assert mmc["only_in_b"] == ["accuracy"]


def test_compare_multi_metric_absent_graceful() -> None:
    """Bundles without metric_results have no multi_metric_comparison (backward compat)."""
    bundle_a = {
        "run_id": "A",
        "metric_name": "auc",
        "baseline": {"mean": 0.85},
        "scenarios": [],
        # NO metric_results
    }
    bundle_b = {
        "run_id": "B",
        "metric_name": "auc",
        "baseline": {"mean": 0.87},
        "scenarios": [],
        # NO metric_results
    }

    report = compare_bundles(bundle_a, bundle_b)

    # multi_metric_comparison should not be present
    assert "multi_metric_comparison" not in report


def test_compare_multi_metric_partial() -> None:
    """One bundle with metric_results, one without, still works."""
    bundle_a = {
        "run_id": "A",
        "metric_name": "auc",
        "baseline": {"mean": 0.85},
        "scenarios": [],
        "metric_results": {
            "auc": {"mean": 0.85},
        },
    }
    bundle_b = {
        "run_id": "B",
        "metric_name": "auc",
        "baseline": {"mean": 0.87},
        "scenarios": [],
        # NO metric_results
    }

    report = compare_bundles(bundle_a, bundle_b)

    # Should still have multi_metric_comparison since one has results
    mmc = report["multi_metric_comparison"]
    assert mmc["shared_metrics"] == []
    assert mmc["only_in_a"] == ["auc"]
    assert mmc["only_in_b"] == []
