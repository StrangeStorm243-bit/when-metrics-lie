"""Phase 7: Unit tests for compare_bundles regression flags and decision semantics."""

from __future__ import annotations

from metrics_lie.compare.compare import compare_bundles


def test_auc_up_subgroup_down_triggers_fairness_flag() -> None:
    """AUC improvement + subgroup gap regression must trigger subgroup risk flag."""
    bundle_a = {
        "run_id": "a",
        "metric_name": "auc",
        "baseline": {"mean": 0.80},
        "scenarios": [
            {
                "scenario_id": "class_imbalance",
                "metric": {"mean": 0.78},
                "diagnostics": {"subgroup_gap": {"gap": 0.05}},
            }
        ],
        "notes": {},
    }
    bundle_b = {
        "run_id": "b",
        "metric_name": "auc",
        "baseline": {"mean": 0.85},
        "scenarios": [
            {
                "scenario_id": "class_imbalance",
                "metric": {"mean": 0.83},
                "diagnostics": {"subgroup_gap": {"gap": 0.12}},
            }
        ],
        "notes": {},
    }
    result = compare_bundles(bundle_a, bundle_b)
    assert result["regressions"]["subgroup"] is True
    assert result["decision"]["winner"] == "no_clear_winner"
    assert any("subgroup_regression" in f for f in result["risk_flags"])


def test_missing_baseline_yields_low_confidence() -> None:
    """No baseline data yields winner=no_clear_winner, confidence=low."""
    bundle_a = {
        "run_id": "a",
        "metric_name": "auc",
        "baseline": {},
        "scenarios": [],
        "notes": {},
    }
    bundle_b = {
        "run_id": "b",
        "metric_name": "auc",
        "baseline": {},
        "scenarios": [],
        "notes": {},
    }
    result = compare_bundles(bundle_a, bundle_b)
    assert result["decision"]["winner"] == "no_clear_winner"
    assert result["decision"]["confidence"] == "low"
    assert result["baseline_delta"].get("mean") is None
