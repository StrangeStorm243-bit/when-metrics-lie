from __future__ import annotations

import pandas as pd


def _write_csv(tmp_path, filename, df):
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def test_load_binary_csv_unchanged(tmp_path):
    from metrics_lie.datasets.loaders import load_binary_csv
    df = pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.2, 0.8, 0.3, 0.9]})
    path = _write_csv(tmp_path, "binary.csv", df)
    ds = load_binary_csv(path, "y", "p")
    assert len(ds.y_true) == 4


def test_load_dataset_multiclass(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset
    df = pd.DataFrame({"label": [0, 1, 2, 3], "pred": [0, 2, 1, 3]})
    path = _write_csv(tmp_path, "multi.csv", df)
    ds = load_dataset(
        path=path, y_true_col="label", y_score_col="pred",
        task_type="multiclass_classification",
    )
    assert len(ds.y_true) == 4
    assert set(ds.y_true.unique()) == {0, 1, 2, 3}


def test_load_dataset_regression(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset
    df = pd.DataFrame({"target": [1.5, 2.3, -0.7, 100.0], "pred": [1.4, 2.5, -0.5, 99.0]})
    path = _write_csv(tmp_path, "reg.csv", df)
    ds = load_dataset(
        path=path, y_true_col="target", y_score_col="pred",
        task_type="regression",
    )
    assert len(ds.y_true) == 4


def test_load_dataset_binary_backward_compat(tmp_path):
    from metrics_lie.datasets.loaders import load_dataset
    df = pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.2, 0.8, 0.3, 0.9]})
    path = _write_csv(tmp_path, "binary2.csv", df)
    ds = load_dataset(
        path=path, y_true_col="y", y_score_col="p",
        task_type="binary_classification",
    )
    assert len(ds.y_true) == 4
