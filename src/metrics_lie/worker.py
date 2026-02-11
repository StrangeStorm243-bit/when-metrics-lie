from __future__ import annotations

import json
import time

from metrics_lie.db.session import get_session
from metrics_lie.db.crud import (
    claim_next_job,
    mark_job_completed,
    mark_job_failed,
    get_experiment_spec_json,
)
from metrics_lie.execution import run_from_spec_dict, rerun


def process_one_job(poll: bool = False) -> int:
    """
    Process one job from the queue.

    Args:
        poll: If True, wait briefly if no job is available (not used in single-worker mode).

    Returns:
        1 if a job was processed, 0 if no job was available.
    """
    with get_session() as session:
        job = claim_next_job(session)
        if job is None:
            return 0

        try:
            if job.kind == "run_experiment":
                if job.experiment_id is None:
                    raise ValueError(
                        f"Job {job.job_id} has kind='run_experiment' but experiment_id is None"
                    )

                spec_json_str = get_experiment_spec_json(session, job.experiment_id)
                if not spec_json_str:
                    raise ValueError(
                        f"No spec_json found for experiment {job.experiment_id}"
                    )

                spec_dict = json.loads(spec_json_str)
                spec_path_for_notes = f"<job:{job.job_id}>"
                result_run_id = run_from_spec_dict(
                    spec_dict, spec_path_for_notes=spec_path_for_notes, rerun_of=None
                )

                with get_session() as update_session:
                    mark_job_completed(update_session, job.job_id, result_run_id)

            elif job.kind == "rerun_run":
                if job.run_id is None:
                    raise ValueError(
                        f"Job {job.job_id} has kind='rerun_run' but run_id is None"
                    )

                result_run_id = rerun(job.run_id)

                with get_session() as update_session:
                    mark_job_completed(update_session, job.job_id, result_run_id)
            else:
                raise ValueError(f"Unknown job kind: {job.kind}")

            return 1

        except Exception as exc:
            error_msg = str(exc)
            with get_session() as update_session:
                mark_job_failed(update_session, job.job_id, error_msg)
            raise


def main() -> None:
    """Main worker loop: process jobs continuously."""
    print("[WORKER] Starting worker loop...")
    while True:
        try:
            processed = process_one_job()
            if not processed:
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n[WORKER] Shutting down...")
            break
        except Exception as exc:
            print(f"[WORKER] Error processing job: {exc}")
            time.sleep(2)


if __name__ == "__main__":
    main()
