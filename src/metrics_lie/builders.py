"""Convenience builder classes for programmatic SDK usage.

Usage:
    from metrics_lie import Dataset, Model

    ds = Dataset.from_csv("data.csv", y_true="label", y_score="pred")
    model = Model.from_onnx("model.onnx")
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    """Builder for dataset configuration."""

    path: str
    y_true_col: str = "y_true"
    y_score_col: str = "y_score"
    subgroup_col: str | None = None
    feature_cols: list[str] | None = field(default=None)

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        y_true: str = "y_true",
        y_score: str = "y_score",
        subgroup: str | None = None,
        features: list[str] | None = None,
    ) -> Dataset:
        """Create a Dataset from a CSV file path."""
        return cls(
            path=path,
            y_true_col=y_true,
            y_score_col=y_score,
            subgroup_col=subgroup,
            feature_cols=features,
        )

    def to_spec_dict(self) -> dict:
        """Convert to ExperimentSpec dataset dict."""
        d: dict = {
            "source": "csv",
            "path": self.path,
            "y_true_col": self.y_true_col,
            "y_score_col": self.y_score_col,
        }
        if self.subgroup_col:
            d["subgroup_col"] = self.subgroup_col
        if self.feature_cols:
            d["feature_cols"] = self.feature_cols
        return d


@dataclass(frozen=True)
class Model:
    """Builder for model source configuration."""

    kind: str
    path: str | None = None
    endpoint: str | None = None
    uri: str | None = None
    trust_pickle: bool = False

    @classmethod
    def from_pickle(cls, path: str, *, trust: bool = True) -> Model:
        """Load a scikit-learn pickle model."""
        return cls(kind="pickle", path=path, trust_pickle=trust)

    @classmethod
    def from_onnx(cls, path: str) -> Model:
        """Load an ONNX model."""
        return cls(kind="onnx", path=path)

    @classmethod
    def from_pytorch(cls, path: str) -> Model:
        """Load a PyTorch TorchScript model."""
        return cls(kind="pytorch", path=path)

    @classmethod
    def from_tensorflow(cls, path: str) -> Model:
        """Load a TensorFlow/Keras model."""
        return cls(kind="tensorflow", path=path)

    @classmethod
    def from_xgboost(cls, path: str) -> Model:
        """Load an XGBoost model."""
        return cls(kind="xgboost", path=path)

    @classmethod
    def from_lightgbm(cls, path: str) -> Model:
        """Load a LightGBM model."""
        return cls(kind="lightgbm", path=path)

    @classmethod
    def from_catboost(cls, path: str) -> Model:
        """Load a CatBoost model."""
        return cls(kind="catboost", path=path)

    @classmethod
    def from_endpoint(cls, url: str) -> Model:
        """Connect to an HTTP model endpoint."""
        return cls(kind="http", endpoint=url)

    @classmethod
    def from_mlflow(cls, uri: str) -> Model:
        """Load from MLflow model registry."""
        return cls(kind="mlflow", uri=uri)

    def to_spec_dict(self) -> dict:
        """Convert to ExperimentSpec model_source dict."""
        d: dict = {"kind": self.kind}
        if self.path:
            d["path"] = self.path
        if self.endpoint:
            d["endpoint"] = self.endpoint
        if self.trust_pickle:
            d["trust_pickle"] = True
        return d
