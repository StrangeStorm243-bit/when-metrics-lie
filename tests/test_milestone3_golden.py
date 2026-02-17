"""Golden test: locks Phase 4 CLI output format.

Runs a minimal experiment end-to-end through run_from_spec_dict,
strips non-deterministic fields (run_id, created_at, notes, artifact paths),
and compares byte-for-byte against a checked-in golden JSON file.

If the golden file is absent the test auto-generates it on the first run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from metrics_lie.db.session import DB_PATH, engine, init_db
from metrics_lie.execution import run_from_spec_dict
from metrics_lie.utils.paths import get_run_dir

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "phase4_minimal_golden.json"


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure a clean local SQLite for every test."""
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


def _create_csv(tmp_path: Path) -> Path:
    """Write a small deterministic CSV dataset."""
    df = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "y_score": [0.10, 0.20, 0.15, 0.35, 0.45, 0.55, 0.65, 0.80, 0.90, 0.95],
            "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "B"],
        }
    )
    csv_path = tmp_path / "golden_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _build_spec(csv_path: Path) -> dict:
    return {
        "name": "golden_test",
        "task": "binary_classification",
        "dataset": {
            "source": "csv",
            "path": str(csv_path),
            "y_true_col": "y_true",
            "y_score_col": "y_score",
            "subgroup_col": "group",
        },
        "metric": "auc",
        "scenarios": [
            {"id": "label_noise", "params": {"p": 0.1}},
            {"id": "score_noise", "params": {"sigma": 0.05}},
            {
                "id": "class_imbalance",
                "params": {"target_pos_rate": 0.2, "max_remove_frac": 0.8},
            },
        ],
        "n_trials": 50,
        "seed": 42,
    }


def _strip_nondeterministic(d: dict) -> dict:
    """Recursively strip fields that change between runs."""
    skip_keys = {"run_id", "created_at", "notes"}
    out = {}
    for k, v in d.items():
        if k in skip_keys:
            continue
        if isinstance(v, dict):
            out[k] = _strip_nondeterministic(v)
        elif isinstance(v, list):
            out[k] = [
                _strip_nondeterministic(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def _strip_artifact_paths(d: dict) -> dict:
    """Replace artifact path values (which contain run_id) with a stable placeholder."""
    out = {}
    for k, v in d.items():
        if k == "path" and isinstance(v, str) and "artifacts/" in v:
            out[k] = "<artifact>"
        elif isinstance(v, dict):
            out[k] = _strip_artifact_paths(v)
        elif isinstance(v, list):
            out[k] = [
                _strip_artifact_paths(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def _run_and_strip(tmp_path: Path) -> dict:
    """Run a minimal experiment and return the stripped result dict."""
    csv_path = _create_csv(tmp_path)
    spec = _build_spec(csv_path)
    run_id = run_from_spec_dict(spec)
    paths = get_run_dir(run_id)
    raw = json.loads(paths.results_json.read_text(encoding="utf-8"))
    stripped = _strip_nondeterministic(raw)
    stripped = _strip_artifact_paths(stripped)
    return stripped


def test_phase4_golden(tmp_path: Path) -> None:
    """Output of the Phase 4 CLI path matches the golden reference."""
    result = _run_and_strip(tmp_path)

    if not GOLDEN_FILE.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        pytest.skip("Golden file generated; re-run to validate.")

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    result_json = json.dumps(result, indent=2, sort_keys=True)
    golden_json = json.dumps(golden, indent=2, sort_keys=True)
    assert result_json == golden_json, (
        "Phase 4 output diverged from golden reference.\n"
        "If the change is intentional, delete tests/golden/phase4_minimal_golden.json "
        "and re-run to regenerate."
    )


def test_phase4_determinism(tmp_path: Path) -> None:
    """Two back-to-back runs with the same spec+seed produce identical output."""
    result_a = _run_and_strip(tmp_path)

    # Need a fresh DB for the second run to avoid experiment_id collision
    engine.dispose()
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()

    result_b = _run_and_strip(tmp_path)
    json_a = json.dumps(result_a, sort_keys=True)
    json_b = json.dumps(result_b, sort_keys=True)
    assert json_a == json_b, "Two identical runs produced different results"
