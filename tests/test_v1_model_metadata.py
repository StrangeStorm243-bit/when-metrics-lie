from __future__ import annotations

from metrics_lie.model.metadata import ModelMetadata


def test_model_metadata_creation():
    meta = ModelMetadata(
        model_class="LogisticRegression",
        model_module="sklearn.linear_model",
        model_format="pickle",
        model_hash="sha256:abc123",
        capabilities={"predict", "predict_proba"},
    )
    assert meta.model_class == "LogisticRegression"
    assert meta.model_format == "pickle"
    assert "predict_proba" in meta.capabilities


def test_model_metadata_optional_fields():
    meta = ModelMetadata(
        model_class="CustomModel",
        model_module="custom",
        model_format="onnx",
    )
    assert meta.model_hash is None
    assert meta.capabilities == set()


def test_model_metadata_frozen():
    meta = ModelMetadata(
        model_class="X", model_module="x", model_format="pickle"
    )
    try:
        meta.model_class = "Y"
        assert False, "Should be frozen"
    except (AttributeError, TypeError):
        pass
