from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from metrics_lie.db.session import get_session
from metrics_lie.db.crud import get_results_path_for_run

from .loader import load_resultbundle_from_path, scenario_map
from .rules import (
    CALIBRATION_REGRESSION_THRESHOLD,
    SUBGROUP_GAP_REGRESSION_THRESHOLD,
    METRIC_REGRESSION_THRESHOLD,
)


def _get_mean(obj: Dict[str, Any]) -> Optional[float]:
    v = obj.get("mean")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _get_diag_mean(diag: Dict[str, Any], key: str) -> Optional[float]:
    # Diagnostics often stored as {"brier": {"mean": ...}} etc.
    d = diag.get(key)
    if isinstance(d, dict):
        return _get_mean(d)
    return None


def _get_nested(diag: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = diag
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def compare_bundles(
    bundle_a: Dict[str, Any], bundle_b: Dict[str, Any]
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "run_a": bundle_a.get("run_id"),
        "run_b": bundle_b.get("run_id"),
        "metric_name": bundle_b.get("metric_name") or bundle_a.get("metric_name"),
        "baseline_delta": {},
        "scenario_deltas": {},
        "metric_gaming_delta": None,
        "regressions": {
            "calibration": False,
            "subgroup": False,
            "metric": False,
            "gaming": False,
        },
        "risk_flags": [],
        "decision": {"winner": "no_clear_winner", "confidence": "low", "reasoning": []},
    }

    # A) Baseline delta (mean + a few extras if present)
    base_a = bundle_a.get("baseline") or {}
    base_b = bundle_b.get("baseline") or {}
    a_mean = _get_mean(base_a) if isinstance(base_a, dict) else None
    b_mean = _get_mean(base_b) if isinstance(base_b, dict) else None
    if a_mean is not None and b_mean is not None:
        report["baseline_delta"] = {"mean": b_mean - a_mean, "a": a_mean, "b": b_mean}
    else:
        report["baseline_delta"] = {"mean": None, "a": a_mean, "b": b_mean}

    # B) Scenario deltas (intersection)
    sm_a = scenario_map(bundle_a)
    sm_b = scenario_map(bundle_b)
    shared = sorted(set(sm_a.keys()) & set(sm_b.keys()))

    worst_ece_delta: Optional[float] = None
    worst_brier_delta: Optional[float] = None
    worst_subgap_delta: Optional[float] = None

    for sid in shared:
        sa = sm_a[sid]
        sb = sm_b[sid]
        da = sa.get("diagnostics") if isinstance(sa.get("diagnostics"), dict) else {}
        db = sb.get("diagnostics") if isinstance(sb.get("diagnostics"), dict) else {}

        ma = sa.get("metric") if isinstance(sa.get("metric"), dict) else {}
        mb = sb.get("metric") if isinstance(sb.get("metric"), dict) else {}
        ma_mean = _get_mean(ma) if isinstance(ma, dict) else None
        mb_mean = _get_mean(mb) if isinstance(mb, dict) else None
        metric_delta = (
            (mb_mean - ma_mean)
            if (ma_mean is not None and mb_mean is not None)
            else None
        )

        brier_a = _get_diag_mean(da, "brier")
        brier_b = _get_diag_mean(db, "brier")
        brier_delta = (
            (brier_b - brier_a)
            if (brier_a is not None and brier_b is not None)
            else None
        )

        ece_a = _get_diag_mean(da, "ece")
        ece_b = _get_diag_mean(db, "ece")
        ece_delta = (
            (ece_b - ece_a) if (ece_a is not None and ece_b is not None) else None
        )

        subgap_a = _get_nested(da, ("subgroup_gap", "gap"))
        subgap_b = _get_nested(db, ("subgroup_gap", "gap"))
        if isinstance(subgap_a, (int, float)) and isinstance(subgap_b, (int, float)):
            subgap_delta = float(subgap_b) - float(subgap_a)
        else:
            subgap_delta = None

        sens_a = da.get("sensitivity_abs")
        sens_b = db.get("sensitivity_abs")
        if isinstance(sens_a, (int, float)) and isinstance(sens_b, (int, float)):
            sens_delta = float(sens_b) - float(sens_a)
        else:
            sens_delta = None

        report["scenario_deltas"][sid] = {
            "metric_mean_delta": metric_delta,
            "metric_mean": {"a": ma_mean, "b": mb_mean},
            "brier_mean_delta": brier_delta,
            "ece_mean_delta": ece_delta,
            "subgroup_gap_delta": subgap_delta,
            "sensitivity_abs_delta": sens_delta,
        }

        if isinstance(ece_delta, float):
            worst_ece_delta = (
                ece_delta
                if worst_ece_delta is None
                else max(worst_ece_delta, ece_delta)
            )
        if isinstance(brier_delta, float):
            worst_brier_delta = (
                brier_delta
                if worst_brier_delta is None
                else max(worst_brier_delta, brier_delta)
            )
        if isinstance(subgap_delta, float):
            worst_subgap_delta = (
                subgap_delta
                if worst_subgap_delta is None
                else max(worst_subgap_delta, subgap_delta)
            )

    # C) metric_inflation deltas (if present in both)
    # We check the first scenario that has it; this is intentionally simple.
    gaming_a = None
    gaming_b = None
    for sid in shared:
        da = (
            sm_a[sid].get("diagnostics")
            if isinstance(sm_a[sid].get("diagnostics"), dict)
            else {}
        )
        db = (
            sm_b[sid].get("diagnostics")
            if isinstance(sm_b[sid].get("diagnostics"), dict)
            else {}
        )
        if isinstance(da.get("metric_inflation"), dict) and isinstance(
            db.get("metric_inflation"), dict
        ):
            gaming_a = da["metric_inflation"]
            gaming_b = db["metric_inflation"]
            break

    if isinstance(gaming_a, dict) and isinstance(gaming_b, dict):

        def _f(d: Dict[str, Any], k: str) -> Optional[float]:
            v = d.get(k)
            return float(v) if isinstance(v, (int, float)) else None

        delta_baseline = None
        delta_optimized = None
        delta_inflation = None
        if (
            _f(gaming_a, "baseline") is not None
            and _f(gaming_b, "baseline") is not None
        ):
            delta_baseline = _f(gaming_b, "baseline") - _f(gaming_a, "baseline")
        if (
            _f(gaming_a, "optimized") is not None
            and _f(gaming_b, "optimized") is not None
        ):
            delta_optimized = _f(gaming_b, "optimized") - _f(gaming_a, "optimized")
        if _f(gaming_a, "delta") is not None and _f(gaming_b, "delta") is not None:
            delta_inflation = _f(gaming_b, "delta") - _f(gaming_a, "delta")

        # downstream deltas (optional)
        down_a = (
            gaming_a.get("downstream")
            if isinstance(gaming_a.get("downstream"), dict)
            else {}
        )
        down_b = (
            gaming_b.get("downstream")
            if isinstance(gaming_b.get("downstream"), dict)
            else {}
        )

        def _down_delta(k: str) -> Optional[float]:
            va = down_a.get(k)
            vb = down_b.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                return float(vb) - float(va)
            return None

        report["metric_gaming_delta"] = {
            "delta_baseline": delta_baseline,
            "delta_optimized": delta_optimized,
            "delta_inflation": delta_inflation,
            "downstream": {
                "brier_delta": _down_delta("brier"),
                "ece_delta": _down_delta("ece"),
                "subgroup_gap_delta": _down_delta("subgroup_gap"),
            },
        }

    # 4) Regression flags (transparent)
    baseline_delta = report["baseline_delta"].get("mean")
    if (
        isinstance(baseline_delta, (int, float))
        and float(baseline_delta) < METRIC_REGRESSION_THRESHOLD
    ):
        report["regressions"]["metric"] = True
        report["risk_flags"].append(
            f"metric_regression: baseline_mean_delta={baseline_delta:.6f} < {METRIC_REGRESSION_THRESHOLD}"
        )

    if (
        isinstance(worst_ece_delta, float)
        and worst_ece_delta > CALIBRATION_REGRESSION_THRESHOLD
    ):
        report["regressions"]["calibration"] = True
        report["risk_flags"].append(
            f"calibration_regression: worst_ece_mean_delta={worst_ece_delta:.6f} > {CALIBRATION_REGRESSION_THRESHOLD}"
        )
    if (
        isinstance(worst_brier_delta, float)
        and worst_brier_delta > CALIBRATION_REGRESSION_THRESHOLD
    ):
        report["regressions"]["calibration"] = True
        report["risk_flags"].append(
            f"calibration_regression: worst_brier_mean_delta={worst_brier_delta:.6f} > {CALIBRATION_REGRESSION_THRESHOLD}"
        )

    if (
        isinstance(worst_subgap_delta, float)
        and worst_subgap_delta > SUBGROUP_GAP_REGRESSION_THRESHOLD
    ):
        report["regressions"]["subgroup"] = True
        report["risk_flags"].append(
            f"subgroup_regression: worst_subgroup_gap_delta={worst_subgap_delta:.6f} > {SUBGROUP_GAP_REGRESSION_THRESHOLD}"
        )

    # gaming regression: inflation delta increases materially (simple)
    mgd = report.get("metric_gaming_delta")
    if (
        isinstance(mgd, dict)
        and isinstance(mgd.get("delta_inflation"), (int, float))
        and mgd["delta_inflation"] > 0.02
    ):
        report["regressions"]["gaming"] = True
        report["risk_flags"].append(
            f"gaming_regression: delta_inflation_delta={mgd['delta_inflation']:.6f} > 0.02"
        )

    # 5) Decision summary
    reasoning = []
    winner = "no_clear_winner"
    confidence = "low"

    regressions = report["regressions"]
    any_regression = any(bool(v) for v in regressions.values())

    if isinstance(baseline_delta, (int, float)):
        if baseline_delta > 0 and not (
            regressions["calibration"] or regressions["subgroup"]
        ):
            winner = "run_b"
            confidence = "high" if not any_regression else "medium"
            reasoning.append(
                "Baseline metric improved and no calibration/subgroup regressions detected."
            )
        elif baseline_delta > 0 and any_regression:
            winner = "no_clear_winner"
            confidence = "medium"
            reasoning.append(
                "Baseline metric improved but regressions detected; review tradeoffs."
            )
        elif baseline_delta <= 0:
            winner = "run_a"
            confidence = "medium" if not any_regression else "low"
            reasoning.append(
                "Baseline metric did not improve; prefer run_a unless risk flags justify tradeoff."
            )
    else:
        reasoning.append(
            "Baseline metric mean not available; insufficient data for clear decision."
        )

    if report["risk_flags"]:
        reasoning.append(f"Risk flags: {len(report['risk_flags'])} triggered.")

    report["decision"] = {
        "winner": winner,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    # Phase 8: Multi-metric comparison
    multi_metric_comparison = _compare_multi_metric(bundle_a, bundle_b)
    if multi_metric_comparison:
        report["multi_metric_comparison"] = multi_metric_comparison

    return report


def _compare_multi_metric(
    bundle_a: Dict[str, Any], bundle_b: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Compare bundles across multiple metrics from metric_results.

    Returns None if neither bundle has metric_results (backward compat).
    """
    mr_a = bundle_a.get("metric_results") or {}
    mr_b = bundle_b.get("metric_results") or {}

    if not mr_a and not mr_b:
        return None

    metrics_a = set(mr_a.keys())
    metrics_b = set(mr_b.keys())

    shared = sorted(metrics_a & metrics_b)
    only_in_a = sorted(metrics_a - metrics_b)
    only_in_b = sorted(metrics_b - metrics_a)

    per_metric_deltas: Dict[str, Dict[str, Any]] = {}
    improved_count = 0
    regressed_count = 0

    for metric_id in shared:
        a_mean = _get_mean(mr_a.get(metric_id, {}))
        b_mean = _get_mean(mr_b.get(metric_id, {}))

        if a_mean is not None and b_mean is not None:
            delta = b_mean - a_mean
            per_metric_deltas[metric_id] = {
                "baseline_delta": delta,
                "a": a_mean,
                "b": b_mean,
            }
            if delta > 0.001:  # Small epsilon for "improvement"
                improved_count += 1
            elif delta < -0.001:
                regressed_count += 1
        else:
            per_metric_deltas[metric_id] = {
                "baseline_delta": None,
                "a": a_mean,
                "b": b_mean,
            }

    total = len(shared)
    if total > 0:
        summary = f"{improved_count}/{total} shared metrics improved; {regressed_count} regressed"
    else:
        summary = "No shared metrics to compare"

    return {
        "shared_metrics": shared,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "per_metric_deltas": per_metric_deltas,
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "summary": summary,
    }


def compare_runs(run_id_a: str, run_id_b: str) -> Dict[str, Any]:
    with get_session() as session:
        path_a = get_results_path_for_run(session, run_id_a)
        path_b = get_results_path_for_run(session, run_id_b)

    bundle_a = load_resultbundle_from_path(path_a)
    bundle_b = load_resultbundle_from_path(path_b)

    report = compare_bundles(bundle_a, bundle_b)
    report["results_path_a"] = path_a
    report["results_path_b"] = path_b
    return report
