"""Bridge module to call Spectra core engine functions directly."""
from datetime import datetime, timezone
from pathlib import Path

from metrics_lie.execution import run_from_spec_dict
from metrics_lie.schema import ResultBundle
from metrics_lie.spec import ExperimentSpec, DatasetSpec, ScenarioSpec
from metrics_lie.utils.paths import get_run_dir

from .contracts import (
    ComponentScore,
    ExperimentCreateRequest,
    FindingFlag,
    ResultSummary,
    ScenarioResult as ContractScenarioResult,
)


def _find_repo_root() -> Path:
    """Find repository root by walking up until pyproject.toml is found."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current.resolve()
        current = current.parent
    raise RuntimeError("Could not find repository root (pyproject.toml not found)")


def ensure_core_db_initialized() -> None:
    """
    Ensure core engine database tables are created.

    This must be called before run_from_spec_dict() to ensure tables exist.
    The core engine uses .spectra_registry/spectra.db relative to CWD at import time.
    """
    repo_root = _find_repo_root()
    original_cwd = Path.cwd()

    try:
        import os
        os.chdir(repo_root)

        # IMPORTANT: import AFTER changing CWD so engine binds correctly
        from metrics_lie.db.session import init_db

        init_db()
    finally:
        os.chdir(original_cwd)



def _get_default_scenarios(stress_suite_id: str) -> list[dict]:
    """Map stress_suite_id to default scenario configurations."""
    # For Phase 3.2, use a standard set of scenarios
    # This will be enhanced in later phases to map to actual suite configs
    return [
        {"id": "label_noise", "params": {"p": 0.1}},
        {"id": "score_noise", "params": {"sigma": 0.05}},
        {"id": "class_imbalance", "params": {"target_pos_rate": 0.2, "max_remove_frac": 0.8}},
    ]


def _get_dataset_path(create_req: ExperimentCreateRequest) -> Path:
    """
    Determine dataset path using config or candidate fallbacks.
    
    Returns:
        Path to dataset CSV file (absolute, resolved)
    
    Raises:
        ValueError: If no dataset file can be found
    """
    repo_root = _find_repo_root()
    searched_locations = []
    
    # Check config first
    if "dataset_path" in create_req.config:
        config_path = Path(create_req.config["dataset_path"])
        if config_path.is_absolute():
            dataset_path = config_path.resolve()
        else:
            dataset_path = (repo_root / config_path).resolve()
        
        searched_locations.append(f"config['dataset_path']: {dataset_path}")
        if not dataset_path.exists():
            raise ValueError(
                f"Dataset file does not exist: {create_req.config['dataset_path']} "
                f"(resolved to: {dataset_path})"
            )
        if not dataset_path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {create_req.config['dataset_path']} "
                f"(resolved to: {dataset_path})"
            )
        print(f"[DEBUG] Using dataset path: {dataset_path}")
        return dataset_path
    
    # Try candidate paths in order (ONLY if they exist)
    candidates = [
        "data/demo_binary.csv",
        "data/demo.csv",
        "data/sample.csv",
        "data/example.csv",
    ]
    
    for candidate in candidates:
        candidate_path = (repo_root / candidate).resolve()
        searched_locations.append(str(candidate_path))
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    
    # Try any CSV in data/ directory (deterministically sorted)
    data_dir = repo_root / "data"
    if data_dir.exists() and data_dir.is_dir():
        csv_files = sorted(data_dir.glob("*.csv"))
        searched_locations.append(f"{data_dir}/*.csv (found {len(csv_files)} files)")
        if csv_files:
            return csv_files[0].resolve()
    else:
        searched_locations.append(f"{data_dir} (directory does not exist)")
    
    # Broaden search: any CSV under repo_root, excluding venv/node_modules/.git/runs/.spectra_ui
    exclude_dirs = {".venv", "venv", "node_modules", ".git", "runs", ".spectra_ui"}
    all_csvs = []
    for csv_file in repo_root.rglob("*.csv"):
        # Check if any parent directory is in exclude list
        if not any(excluded in csv_file.parts for excluded in exclude_dirs):
            all_csvs.append(csv_file)
    
    if all_csvs:
        # Deterministically sort and pick first
        sorted_csvs = sorted(all_csvs, key=lambda p: str(p))
        searched_locations.append(f"repo_root/**/*.csv (found {len(sorted_csvs)} files, excluding venv/node_modules/.git/runs/.spectra_ui)")
        return sorted_csvs[0].resolve()
    else:
        searched_locations.append(f"repo_root/**/*.csv (none found, excluding venv/node_modules/.git/runs/.spectra_ui)")
    
    # None found - raise clear error
    raise ValueError(
        f"No dataset file found. Searched locations:\n  " + "\n  ".join(searched_locations) +
        f"\n\nTo fix: Set create_req.config['dataset_path'] to a valid CSV file path "
        f"(relative to repo root or absolute)."
    )


def _get_default_dataset(create_req: ExperimentCreateRequest) -> dict:
    """Get default dataset configuration."""
    dataset_path = _get_dataset_path(create_req)
    
    # ALWAYS use absolute path in spec dict
    path_str = str(dataset_path.resolve())
    
    return {
        "source": "csv",
        "path": path_str,
        "y_true_col": "y_true",
        "y_score_col": "y_score",
        "subgroup_col": "group",
    }


def _bundle_to_result_summary(
    bundle: ResultBundle, experiment_id: str, run_id: str
) -> ResultSummary:
    """Convert ResultBundle to ResultSummary."""
    # Extract headline score from baseline
    headline_score = bundle.baseline.mean if bundle.baseline else 0.0

    # Convert scenario results
    scenario_results = []
    for sr in bundle.scenarios:
        # Calculate delta: scenario mean - baseline mean
        delta = sr.metric.mean - headline_score if bundle.baseline else 0.0
        scenario_results.append(
            ContractScenarioResult(
                scenario_id=sr.scenario_id,
                scenario_name=sr.scenario_id.replace("_", " ").title(),
                delta=delta,
                score=sr.metric.mean,
                severity=None,  # Will be calculated in later phases
                notes=None,
            )
        )

    # Extract component scores from diagnostics
    component_scores = []
    # For Phase 3.2, extract basic diagnostics
    if bundle.baseline:
        baseline_diag = bundle.notes.get("baseline_diagnostics", {})
        if "brier" in baseline_diag:
            component_scores.append(
                ComponentScore(
                    name="brier_score",
                    score=baseline_diag["brier"],
                    weight=None,
                    notes="Baseline Brier score",
                )
            )
        if "ece" in baseline_diag:
            component_scores.append(
                ComponentScore(
                    name="ece_score",
                    score=baseline_diag["ece"],
                    weight=None,
                    notes="Baseline ECE",
                )
            )

    # Extract flags from diagnostics
    flags = []
    # For Phase 3.2, add basic flags if diagnostics indicate issues
    if bundle.baseline:
        baseline_diag = bundle.notes.get("baseline_diagnostics", {})
        if baseline_diag.get("ece", 0) > 0.1:
            flags.append(
                FindingFlag(
                    code="high_ece",
                    title="High Expected Calibration Error",
                    detail=f"ECE is {baseline_diag.get('ece', 0):.4f}, indicating poor calibration",
                    severity="warn",
                )
            )

    return ResultSummary(
        experiment_id=experiment_id,
        run_id=run_id,
        headline_score=headline_score,
        weighted_score=None,  # Will be calculated with decision profiles in later phases
        component_scores=component_scores,
        scenario_results=scenario_results,
        flags=flags,
        generated_at=datetime.fromisoformat(bundle.created_at.replace("Z", "+00:00")),
    )


def run_experiment(
    create_req: ExperimentCreateRequest, experiment_id: str, run_id: str, seed: int | None = None
) -> ResultSummary:
    """
    Run an experiment using Spectra core engine.

    Args:
        create_req: Experiment creation request
        experiment_id: Experiment ID
        run_id: Run ID
        seed: Optional random seed

    Returns:
        ResultSummary with experiment results
    """
    # Build ExperimentSpec from create_req
    dataset_dict = _get_default_dataset(create_req)
    scenarios = _get_default_scenarios(create_req.stress_suite_id)

    spec_dict = {
        "name": create_req.name,
        "task": "binary_classification",
        "dataset": dataset_dict,
        "metric": create_req.metric_id,
        "scenarios": scenarios,
        "n_trials": 200,  # Default
        "seed": seed if seed is not None else 42,
        "tags": {
            "experiment_id": experiment_id,
            "run_id": run_id,
        },
    }

    # Ensure core DB is initialized before calling engine
    ensure_core_db_initialized()
    
    # Call core engine
    spec_path_for_notes = f"<ui_experiment:{experiment_id}:{run_id}>"
    returned_run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=spec_path_for_notes)

    # Read ResultBundle from disk
    run_paths = get_run_dir(returned_run_id)
    bundle_json = run_paths.results_json.read_text(encoding="utf-8")
    bundle = ResultBundle.model_validate_json(bundle_json)

    # Convert to ResultSummary
    return _bundle_to_result_summary(bundle, experiment_id, run_id)

