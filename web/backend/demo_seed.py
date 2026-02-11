#!/usr/bin/env python3
"""Demo seed script to create and run a sample experiment.

This script creates an experiment using the first available metric and stress suite preset,
runs it, and prints the experiment ID and frontend URL.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from app
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timezone
import uuid

from app.contracts import ExperimentCreateRequest, ExperimentSummary
from app.engine_bridge import run_experiment
from app.persistence import save_experiment, save_result
from app.storage import METRIC_PRESETS, STRESS_SUITE_PRESETS


def main():
    """Create and run a demo experiment."""
    # Get first metric and stress suite
    if not METRIC_PRESETS:
        print("ERROR: No metric presets available")
        sys.exit(1)
    if not STRESS_SUITE_PRESETS:
        print("ERROR: No stress suite presets available")
        sys.exit(1)

    metric = METRIC_PRESETS[0]
    stress_suite = STRESS_SUITE_PRESETS[0]

    print("Creating demo experiment with:")
    print(f"  Metric: {metric['name']} ({metric['id']})")
    print(f"  Stress Suite: {stress_suite['name']} ({stress_suite['id']})")
    print()

    # Create experiment
    experiment_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    create_req = ExperimentCreateRequest(
        name=f"Demo Experiment - {metric['name']} + {stress_suite['name']}",
        metric_id=metric["id"],
        stress_suite_id=stress_suite["id"],
        notes="Created by demo_seed.py",
    )

    summary = ExperimentSummary(
        id=experiment_id,
        name=create_req.name,
        metric_id=create_req.metric_id,
        stress_suite_id=create_req.stress_suite_id,
        status="created",
        created_at=now,
        last_run_at=None,
        error_message=None,
    )

    save_experiment(experiment_id, create_req, summary)
    print(f"✓ Experiment created: {experiment_id}")

    # Run experiment
    print("Running experiment...")
    run_id = str(uuid.uuid4())

    try:
        # Update status to running
        summary.status = "running"
        summary.last_run_at = now
        save_experiment(experiment_id, create_req, summary)

        # Run
        result = run_experiment(create_req, experiment_id, run_id, seed=42)

        # Save result
        save_result(experiment_id, run_id, result)

        # Update status to completed
        summary.status = "completed"
        save_experiment(experiment_id, create_req, summary)

        print("✓ Experiment completed successfully")
        print()
        print("=" * 60)
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID: {run_id}")
        print(f"Headline Score: {result.headline_score:.4f}")
        print()
        print("Frontend URL:")
        print(f"  http://localhost:3000/experiments/{experiment_id}")
        print("=" * 60)

    except Exception as e:
        # Update status to failed
        summary.status = "failed"
        summary.error_message = str(e)
        save_experiment(experiment_id, create_req, summary)
        print(f"✗ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
