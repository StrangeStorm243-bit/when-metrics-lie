from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelMetadata:
    model_class: str
    model_module: str
    model_format: str
    model_hash: str | None = None
    capabilities: set[str] = field(default_factory=set)
