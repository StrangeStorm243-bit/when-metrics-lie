from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_resultbundle_from_path(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Minimal validation (Phase 2.3): keep it light and auditable.
    for key in ("baseline", "scenarios", "notes"):
        if key not in data:
            raise ValueError(f"Invalid results.json (missing '{key}') at: {path}")
    if not isinstance(data.get("scenarios"), list):
        raise ValueError(f"Invalid results.json (scenarios not a list) at: {path}")

    return data


def scenario_map(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in bundle.get("scenarios", []) or []:
        sid = s.get("scenario_id")
        if isinstance(sid, str):
            out[sid] = s
    return out
