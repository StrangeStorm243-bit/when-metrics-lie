from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

from metrics_lie.model import (
    ModelAdapter,
    ModelSourceCallable,
    ModelSourceImport,
    ModelSourcePickle,
)


def _train_model() -> LogisticRegression:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)
    model = LogisticRegression(random_state=0)
    model.fit(X, y)
    return model


def test_model_adapter_pickle_predict_proba(tmp_path: Path) -> None:
    model = _train_model()
    pkl = tmp_path / "model.pkl"
    with pkl.open("wb") as f:
        pickle.dump(model, f)

    adapter = ModelAdapter(ModelSourcePickle(path=str(pkl)))
    surface = adapter.predict_proba(np.array([[0.5], [2.5]], dtype=float))

    assert surface.surface_type.value == "probability"
    assert surface.n_samples == 2
    assert surface.model_hash is not None
    assert np.all(surface.values >= 0.0) and np.all(surface.values <= 1.0)


def test_model_adapter_import_path(tmp_path: Path) -> None:
    module_path = tmp_path / "temp_model_module.py"
    module_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from sklearn.linear_model import LogisticRegression",
                "X = np.array([[0.0],[1.0],[2.0],[3.0]], dtype=float)",
                "y = np.array([0,0,1,1], dtype=int)",
                "MODEL = LogisticRegression(random_state=0).fit(X, y)",
            ]
        ),
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        adapter = ModelAdapter(ModelSourceImport(import_path="temp_model_module:MODEL"))
        surface = adapter.predict_proba(np.array([[0.5], [2.5]], dtype=float))
        assert surface.surface_type.value == "probability"
        assert surface.n_samples == 2
    finally:
        sys.path.remove(str(tmp_path))
        spec = importlib.util.find_spec("temp_model_module")
        if spec and "temp_model_module" in sys.modules:
            del sys.modules["temp_model_module"]


def test_model_adapter_callable() -> None:
    def infer(X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0],), 0.7, dtype=float)

    adapter = ModelAdapter(ModelSourceCallable(fn=infer, name="test_fn"))
    surface = adapter.predict_proba(np.array([[1.0], [2.0], [3.0]], dtype=float))
    assert surface.surface_type.value == "probability"
    assert surface.n_samples == 3
    assert np.allclose(surface.values, 0.7)
