from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from metrics_lie.model.metadata import ModelMetadata
from metrics_lie.model.surface import PredictionSurface, SurfaceType
from metrics_lie.task_types import TaskType


class ModelAdapterProtocol(Protocol):
    """Universal model adapter interface.

    All model adapters (sklearn, ONNX, PyTorch, etc.) implement this protocol.
    The adapter is responsible for loading the model, detecting capabilities,
    and producing PredictionSurfaces.
    """

    @property
    def task_type(self) -> TaskType: ...

    @property
    def metadata(self) -> ModelMetadata: ...

    def predict(self, X: np.ndarray) -> PredictionSurface: ...

    def predict_proba(self, X: np.ndarray) -> PredictionSurface | None: ...

    def predict_raw(self, X: np.ndarray) -> dict[str, Any]: ...

    def get_all_surfaces(self, X: np.ndarray) -> dict[SurfaceType, PredictionSurface]: ...
