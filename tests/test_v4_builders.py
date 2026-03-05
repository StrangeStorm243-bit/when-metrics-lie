from __future__ import annotations


def test_dataset_from_csv():
    from metrics_lie.builders import Dataset

    ds = Dataset.from_csv("data.csv", y_true="label", y_score="pred")
    assert ds.path == "data.csv"
    assert ds.y_true_col == "label"
    assert ds.y_score_col == "pred"


def test_dataset_from_csv_with_features():
    from metrics_lie.builders import Dataset

    ds = Dataset.from_csv("data.csv", y_true="y", y_score="p", features=["a", "b"])
    assert ds.feature_cols == ["a", "b"]


def test_dataset_to_spec_dict():
    from metrics_lie.builders import Dataset

    ds = Dataset.from_csv("data.csv", y_true="label", y_score="pred", subgroup="group")
    d = ds.to_spec_dict()
    assert d["source"] == "csv"
    assert d["path"] == "data.csv"
    assert d["y_true_col"] == "label"
    assert d["subgroup_col"] == "group"


def test_model_from_pickle():
    from metrics_lie.builders import Model

    m = Model.from_pickle("model.pkl")
    assert m.kind == "pickle"
    assert m.path == "model.pkl"
    assert m.trust_pickle is True


def test_model_from_onnx():
    from metrics_lie.builders import Model

    m = Model.from_onnx("model.onnx")
    assert m.kind == "onnx"
    assert m.path == "model.onnx"


def test_model_from_pytorch():
    from metrics_lie.builders import Model

    m = Model.from_pytorch("model.pt")
    assert m.kind == "pytorch"


def test_model_from_tensorflow():
    from metrics_lie.builders import Model

    m = Model.from_tensorflow("model.keras")
    assert m.kind == "tensorflow"


def test_model_from_endpoint():
    from metrics_lie.builders import Model

    m = Model.from_endpoint("http://localhost:8080/v2/models/m/infer")
    assert m.kind == "http"
    assert m.endpoint == "http://localhost:8080/v2/models/m/infer"


def test_model_from_mlflow():
    from metrics_lie.builders import Model

    m = Model.from_mlflow("runs:/abc123/model")
    assert m.kind == "mlflow"
    assert m.uri == "runs:/abc123/model"


def test_model_to_spec_dict():
    from metrics_lie.builders import Model

    m = Model.from_pickle("model.pkl")
    d = m.to_spec_dict()
    assert d["kind"] == "pickle"
    assert d["path"] == "model.pkl"
    assert d["trust_pickle"] is True


def test_model_from_onnx_no_trust():
    from metrics_lie.builders import Model

    m = Model.from_onnx("model.onnx")
    d = m.to_spec_dict()
    assert "trust_pickle" not in d


def test_builders_exported():
    from metrics_lie import Dataset, Model

    assert Dataset is not None
    assert Model is not None
