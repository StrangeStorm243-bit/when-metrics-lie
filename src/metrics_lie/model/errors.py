from __future__ import annotations


class ModelLoadError(RuntimeError):
    pass


class ModelNotFittedError(RuntimeError):
    pass


class SurfaceValidationError(ValueError):
    pass


class CapabilityError(RuntimeError):
    pass
