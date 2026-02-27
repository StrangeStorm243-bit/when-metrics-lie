from __future__ import annotations

import pandas as pd


def _create_binary_csv(tmp_path):
    df = pd.DataFrame({
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "p": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75, 0.3, 0.7],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_execution_binary_still_works(tmp_path):
    from metrics_lie.execution import run_from_spec_dict
    csv_path = _create_binary_csv(tmp_path)
    spec = {
        "name": "binary test",
        "task": "binary_classification",
        "dataset": {"path": csv_path, "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "n_trials": 5,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
    assert len(run_id) == 10


def test_execution_default_task_binary(tmp_path):
    from metrics_lie.execution import run_from_spec_dict
    csv_path = _create_binary_csv(tmp_path)
    spec = {
        "name": "default task test",
        "dataset": {"path": csv_path, "y_true_col": "y", "y_score_col": "p"},
        "metric": "auc",
        "n_trials": 5,
        "seed": 42,
    }
    run_id = run_from_spec_dict(spec)
    assert run_id is not None
