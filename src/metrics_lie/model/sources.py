from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Union
import hashlib
import pickle

from .errors import ModelLoadError


@dataclass(frozen=True)
class ModelSourcePickle:
    path: str


@dataclass(frozen=True)
class ModelSourceCallable:
    fn: Callable[[Any], Any]
    name: str = "anonymous"


@dataclass(frozen=True)
class ModelSourceImport:
    import_path: str


ModelSource = Union[ModelSourcePickle, ModelSourceCallable, ModelSourceImport]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _hash_import(import_path: str, module_file: str | None, mtime: float | None) -> str:
    h = hashlib.sha256()
    h.update(import_path.encode("utf-8"))
    if module_file:
        h.update(module_file.encode("utf-8"))
    if mtime is not None:
        h.update(str(mtime).encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def load_model(source: ModelSource) -> tuple[Any, str | None]:
    if isinstance(source, ModelSourcePickle):
        p = Path(source.path)
        if not p.exists():
            raise ModelLoadError(f"Model artifact not found: {p}")
        try:
            with p.open("rb") as f:
                model = pickle.load(f)
        except Exception as exc:  # pragma: no cover - passthrough
            raise ModelLoadError(f"Failed to load pickle model: {exc}") from exc
        return model, _hash_file(p)

    if isinstance(source, ModelSourceCallable):
        if not callable(source.fn):
            raise ModelLoadError("ModelSourceCallable.fn must be callable")
        return source.fn, None

    if isinstance(source, ModelSourceImport):
        if ":" not in source.import_path:
            raise ModelLoadError(
                "ModelSourceImport.import_path must be 'module:attr' format"
            )
        module_name, attr_name = source.import_path.split(":", 1)
        try:
            module = import_module(module_name)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to import module '{module_name}': {exc}"
            ) from exc
        if not hasattr(module, attr_name):
            raise ModelLoadError(
                f"Imported module '{module_name}' has no attribute '{attr_name}'"
            )
        model = getattr(module, attr_name)
        module_file = getattr(module, "__file__", None)
        mtime = None
        if module_file:
            try:
                mtime = Path(module_file).stat().st_mtime
            except OSError:
                mtime = None
        return model, _hash_import(source.import_path, module_file, mtime)

    raise ModelLoadError(f"Unsupported model source: {source}")
