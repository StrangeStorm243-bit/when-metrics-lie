from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def results_json(self) -> Path:
        return self.root / "results.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def get_runs_dir() -> Path:
    return Path("runs")


def get_run_dir(run_id: str) -> RunPaths:
    return RunPaths(root=get_runs_dir() / run_id)
