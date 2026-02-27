"""Public SDK entry points for Spectra."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir

logger = logging.getLogger(__name__)


def evaluate(
    *,
    name: str,
    dataset: str,
    model: str | None = None,
    metric: str = "auc",
    task: str = "binary_classification",
    scenarios: list[dict[str, Any]] | None = None,
    n_trials: int = 200,
    seed: int = 42,
    y_true_col: str = "y_true",
    y_score_col: str = "y_score",
    feature_cols: list[str] | None = None,
    subgroup_col: str | None = None,
    sensitive_feature: str | None = None,
    reference_dataset: str | None = None,
) -> ResultBundle:
    """Run a quick evaluation: model + dataset -> ResultBundle."""
    from metrics_lie.execution import run_from_spec_dict

    spec_dict: dict[str, Any] = {
        "name": name,
        "task": task,
        "dataset": {
            "source": "csv",
            "path": str(dataset),
            "y_true_col": y_true_col,
            "y_score_col": y_score_col,
        },
        "metric": metric,
        "scenarios": _normalize_scenarios(scenarios or []),
        "n_trials": n_trials,
        "seed": seed,
    }

    if subgroup_col:
        spec_dict["dataset"]["subgroup_col"] = subgroup_col

    if feature_cols:
        spec_dict["dataset"]["feature_cols"] = feature_cols
    elif model:
        import pandas as pd

        df = pd.read_csv(dataset, nrows=0)
        non_feature = {y_true_col, y_score_col, subgroup_col} - {None}
        auto_features = [
            c
            for c in df.columns
            if c not in non_feature and not c.startswith("y_score_")
        ]
        if auto_features:
            spec_dict["dataset"]["feature_cols"] = auto_features

    if model:
        model_path = Path(model)
        kind = _detect_model_kind(model_path)
        spec_dict["model_source"] = {"kind": kind, "path": str(model_path)}

    if sensitive_feature:
        spec_dict["sensitive_feature"] = sensitive_feature
    if reference_dataset:
        spec_dict["reference_dataset"] = reference_dataset

    run_id = run_from_spec_dict(spec_dict)
    return _load_bundle(run_id)


def evaluate_file(path: str | Path) -> ResultBundle:
    """Run an evaluation from a spec file and return the ResultBundle."""
    from metrics_lie.execution import run_from_spec_dict

    spec_dict = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=str(path))
    return _load_bundle(run_id)


def compare(
    result_a: ResultBundle,
    result_b: ResultBundle,
) -> dict[str, Any]:
    """Compare two ResultBundles and return a comparison report."""
    from metrics_lie.compare.compare import compare_bundles

    bundle_a = json.loads(result_a.to_pretty_json())
    bundle_b = json.loads(result_b.to_pretty_json())
    return compare_bundles(bundle_a, bundle_b)


def score(
    result_a: ResultBundle,
    result_b: ResultBundle,
    profile: str = "balanced",
) -> Any:
    """Score a comparison using a decision profile."""
    from metrics_lie.decision.extract import extract_components
    from metrics_lie.decision.scorecard import build_scorecard
    from metrics_lie.profiles.load import get_profile_or_load

    report = compare(result_a, result_b)
    prof = get_profile_or_load(profile)
    components = extract_components(report, prof)
    return build_scorecard(components, prof)


def _load_bundle(run_id: str) -> ResultBundle:
    """Load a ResultBundle from disk by run ID."""
    paths = get_run_dir(run_id)
    return ResultBundle.model_validate_json(
        paths.results_json.read_text(encoding="utf-8")
    )


def _detect_model_kind(path: Path) -> str:
    """Detect model format from file extension."""
    suffix = path.suffix.lower()
    kind_map = {
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".joblib": "pickle",
        ".onnx": "onnx",
        ".ubj": "xgboost",
        ".xgb": "xgboost",
        ".txt": "lightgbm",
        ".cbm": "catboost",
    }
    return kind_map.get(suffix, "pickle")


def _normalize_scenarios(
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize scenario dicts to {id, params} form."""
    normalized = []
    for s in scenarios:
        if "id" in s and "params" in s:
            normalized.append(s)
        else:
            s_copy = dict(s)
            scenario_type = s_copy.pop("type", s_copy.pop("id", None))
            if scenario_type is None:
                raise ValueError(f"Scenario must have 'type' or 'id' key: {s}")
            normalized.append({"id": scenario_type, "params": dict(s_copy)})
    return normalized
