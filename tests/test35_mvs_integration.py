from __future__ import annotations

import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.utils.paths import get_run_dir


def test_mvs_model_inference_flow(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3, 4, 5],
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_score": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    X = df[["feature1"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    model = LogisticRegression(random_state=0).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    spec = {
        "name": "mvs_test",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
            "feature_cols": ["feature1"],
        },
        "metric": "accuracy",
        "model_source": {"kind": "pickle", "path": str(model_path), "threshold": 0.5},
        "scenarios": [],
        "n_trials": 5,
        "seed": 123,
        "tags": {"test": "mvs"},
    }

    run_id = run_from_spec_dict(spec)
    run_paths = get_run_dir(run_id)
    bundle_json = run_paths.results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    assert bundle.prediction_surface is not None
    assert "accuracy" in bundle.applicable_metrics


def _strip_nondeterministic(d: dict) -> dict:
    """Remove fields that vary between runs so we can compare deterministic content.

    Stripped fields:
      - run_id: unique per run by design
      - created_at: timestamp of run creation
      - Any nested key named 'run_id' or 'created_at'
    """
    out = {}
    for k, v in d.items():
        if k in ("run_id", "created_at"):
            continue
        if isinstance(v, dict):
            out[k] = _strip_nondeterministic(v)
        elif isinstance(v, list):
            out[k] = [
                _strip_nondeterministic(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def test_mvs_determinism_golden(tmp_path: Path) -> None:
    """Two runs with identical spec + seed must produce identical deterministic content."""
    df = pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3, 4, 5],
            "y_true": [0, 0, 0, 1, 1, 1],
            "y_score": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    X = df[["feature1"]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    model = LogisticRegression(random_state=0).fit(X, y)
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    spec = {
        "name": "determinism_test",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
            "feature_cols": ["feature1"],
        },
        "metric": "accuracy",
        "model_source": {"kind": "pickle", "path": str(model_path), "threshold": 0.5},
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
        ],
        "n_trials": 10,
        "seed": 777,
        "tags": {"test": "determinism"},
    }

    # Run 1
    run_id_a = run_from_spec_dict(spec)
    bundle_a = ResultBundle.model_validate_json(
        get_run_dir(run_id_a).results_json.read_text(encoding="utf-8")
    )

    # Run 2 (identical spec + seed)
    run_id_b = run_from_spec_dict(spec)
    bundle_b = ResultBundle.model_validate_json(
        get_run_dir(run_id_b).results_json.read_text(encoding="utf-8")
    )

    # Strip nondeterministic fields and compare
    dict_a = _strip_nondeterministic(bundle_a.model_dump())
    dict_b = _strip_nondeterministic(bundle_b.model_dump())

    json_a = json.dumps(dict_a, sort_keys=True, default=str)
    json_b = json.dumps(dict_b, sort_keys=True, default=str)

    assert json_a == json_b, (
        "Two runs with identical spec+seed produced different deterministic content"
    )
