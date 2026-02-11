from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


RunStatus = Literal["queued", "running", "completed", "failed"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    run_id: str
    experiment_id: str
    status: RunStatus = "queued"
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    results_path: str = ""
    artifacts_dir: str = ""
    seed_used: int = 0
    error: Optional[str] = None
    # Optional linkage to an original run when this record represents a rerun.
    rerun_of: Optional[str] = None

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = _now_iso()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished_at = _now_iso()

    def mark_failed(self, error_msg: str) -> None:
        self.status = "failed"
        self.error = error_msg
        self.finished_at = _now_iso()
