"""Verification script for Phase 3.2 endpoints."""
import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def main():
    """Verify Phase 3.2 endpoints."""
    print("Phase 3.2 Verification")
    print("=" * 50)

    # 1. Create experiment
    print("\n1. Creating experiment...")
    create_req = {
        "name": "Phase 3.2 Test",
        "metric_id": "auc",
        "stress_suite_id": "balanced",
        "notes": "Verification test",
    }
    response = requests.post(f"{BASE_URL}/experiments", json=create_req)
    response.raise_for_status()
    experiment = response.json()
    experiment_id = experiment["id"]
    print(f"   Created experiment: {experiment_id}")
    print(f"   Status: {experiment['status']}")

    # 2. List experiments
    print("\n2. Listing experiments...")
    response = requests.get(f"{BASE_URL}/experiments")
    response.raise_for_status()
    experiments = response.json()
    print(f"   Found {len(experiments)} experiment(s)")

    # 3. Run experiment
    print("\n3. Running experiment...")
    run_req = {"seed": 42}
    response = requests.post(f"{BASE_URL}/experiments/{experiment_id}/run", json=run_req)
    response.raise_for_status()
    run_result = response.json()
    run_id = run_result["run_id"]
    print(f"   Run ID: {run_id}")
    print(f"   Status: {run_result['status']}")

    # 4. Get results
    print("\n4. Fetching results...")
    response = requests.get(f"{BASE_URL}/experiments/{experiment_id}/results")
    response.raise_for_status()
    results = response.json()
    print(f"   Headline score: {results['headline_score']:.6f}")
    print(f"   Component scores: {len(results['component_scores'])}")
    print(f"   Scenario results: {len(results['scenario_results'])}")
    print(f"   Flags: {len(results['flags'])}")

    # Print component scores
    if results["component_scores"]:
        print("\n   Component Scores:")
        for comp in results["component_scores"]:
            print(f"     - {comp['name']}: {comp['score']:.6f}")

    # Print scenario results
    if results["scenario_results"]:
        print("\n   Scenario Results:")
        for scenario in results["scenario_results"]:
            print(f"     - {scenario['scenario_name']}: delta={scenario['delta']:.6f}, score={scenario['score']:.6f}")

    # Print flags
    if results["flags"]:
        print("\n   Flags:")
        for flag in results["flags"]:
            print(f"     - [{flag['severity']}] {flag['title']}: {flag['detail']}")

    print("\n" + "=" * 50)
    print("✓ Phase 3.2 verification complete!")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Make sure it's running:")
        print("  cd web/backend")
        print("  python -m uvicorn app.main:app --reload --port 8000")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

