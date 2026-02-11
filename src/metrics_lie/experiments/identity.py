from __future__ import annotations

from collections.abc import Mapping
import json
import hashlib
from typing import Any


def _normalize_for_json(obj: Any) -> Any:
    """
    Normalize objects into JSON-serializable primitives with stable ordering.

    - Mappings: sort keys recursively
    - Sequences: normalize elements; tuples/lists treated the same
    - numpy scalars: converted to Python scalars via item() if available
    - Other types: returned as-is and rely on json to handle or fail
    """
    # Convert numpy scalars / similar objects with item()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    if isinstance(obj, Mapping):
        return {
            str(k): _normalize_for_json(v)
            for k, v in sorted(obj.items(), key=lambda kv: kv[0])
        }

    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(v) for v in obj]

    return obj


def canonical_json(obj: dict) -> str:
    """
    Produce a deterministic JSON string for a Python mapping.

    - Keys are sorted recursively
    - No insignificant whitespace
    """
    normalized = _normalize_for_json(obj)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def sha256_hex(text_or_bytes: Any) -> str:
    """
    Compute a SHA-256 hex digest from text or bytes.
    """
    if isinstance(text_or_bytes, str):
        data = text_or_bytes.encode("utf-8")
    else:
        data = bytes(text_or_bytes)
    return hashlib.sha256(data).hexdigest()


def short_id(prefix: str, hex_digest: str, n: int = 10) -> str:
    """
    Return a short identifier like 'exp_ABCDEF1234' from a hex digest.
    """
    return f"{prefix}_{hex_digest[:n].upper()}"
