from __future__ import annotations

from typing import Any, Callable


AdapterFactory = Callable[..., Any]


class AdapterRegistry:
    """Registry mapping model format names and file extensions to adapter factories."""

    def __init__(self) -> None:
        self._by_format: dict[str, AdapterFactory] = {}
        self._by_extension: dict[str, AdapterFactory] = {}

    def register(
        self,
        format_name: str,
        *,
        factory: AdapterFactory,
        extensions: set[str],
    ) -> None:
        if format_name in self._by_format:
            raise ValueError(f"Format '{format_name}' already registered")
        self._by_format[format_name] = factory
        for ext in extensions:
            self._by_extension[ext.lower()] = factory

    def resolve_format(self, format_name: str) -> AdapterFactory:
        if format_name not in self._by_format:
            raise KeyError(
                f"Unknown model format: '{format_name}'. "
                f"Available: {sorted(self._by_format.keys())}"
            )
        return self._by_format[format_name]

    def resolve_extension(self, ext: str) -> AdapterFactory:
        ext = ext.lower()
        if ext not in self._by_extension:
            raise KeyError(
                f"No adapter registered for extension '{ext}'. "
                f"Available: {sorted(self._by_extension.keys())}"
            )
        return self._by_extension[ext]

    def list_formats(self) -> list[str]:
        return sorted(self._by_format.keys())
