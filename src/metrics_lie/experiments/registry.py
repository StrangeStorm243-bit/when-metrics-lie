from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .definition import ExperimentDefinition
from .runs import RunRecord


REGISTRY_DIR = Path(".spectra_registry")
EXPERIMENTS_FILE = REGISTRY_DIR / "experiments.jsonl"
RUNS_FILE = REGISTRY_DIR / "runs.jsonl"


def ensure_registry_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj_dict: Dict) -> None:
    ensure_registry_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj_dict, separators=(",", ":")))
        f.write("\n")


def experiment_exists(experiment_id: str) -> bool:
    if not EXPERIMENTS_FILE.exists():
        return False
    with EXPERIMENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("experiment_id") == experiment_id:
                return True
    return False


def upsert_experiment(defn: ExperimentDefinition) -> None:
    if experiment_exists(defn.experiment_id):
        return
    append_jsonl(EXPERIMENTS_FILE, defn.model_dump())


def log_run(run_record: RunRecord) -> None:
    append_jsonl(RUNS_FILE, run_record.__dict__)
