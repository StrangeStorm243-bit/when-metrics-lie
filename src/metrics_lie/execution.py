from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from metrics_lie.artifacts.plots import (
    plot_calibration_curve,
    plot_metric_distribution,
    plot_subgroup_bars,
    plot_threshold_curve,
)
from metrics_lie.datasets.loaders import load_binary_csv
from metrics_lie.metrics.applicability import ApplicableMetricSet, DatasetProperties, MetricResolver
from metrics_lie.model import (
    CalibrationState,
    ModelAdapter,
    ModelSourceImport,
    ModelSourcePickle,
    SurfaceType,
)
from metrics_lie.analysis import (
    analyze_metric_disagreements,
    locate_failure_modes,
    run_sensitivity_analysis,
    run_threshold_sweep,
)
from metrics_lie.diagnostics.calibration import brier_score, expected_calibration_error
from metrics_lie.metrics.core import METRICS
from metrics_lie.schema import Artifact, MetricSummary, ResultBundle, ScenarioResult
from metrics_lie.spec import load_experiment_spec
from metrics_lie.utils.paths import get_run_dir
from metrics_lie.experiments.datasets import dataset_fingerprint_csv
from metrics_lie.experiments.definition import ExperimentDefinition
from metrics_lie.experiments.registry import upsert_experiment as upsert_experiment_jsonl, log_run as log_run_jsonl
from metrics_lie.experiments.runs import RunRecord
from metrics_lie.db.session import get_session
from metrics_lie.db.crud import (
    upsert_experiment,
    insert_run,
    update_run,
    insert_artifacts,
    get_experiment_spec_json,
    get_experiment_id_for_run,
)
from metrics_lie.compare.compare import compare_runs
from metrics_lie.experiments.identity import canonical_json

# Ensure scenario registration occurs (import-time registration)
from metrics_lie.scenarios import class_imbalance, label_noise, score_noise, threshold_gaming  # noqa: F401
from metrics_lie.runner import RunConfig, run_scenarios
from metrics_lie.scenarios.base import ScenarioContext
from metrics_lie.scenarios.registry import create_scenario


def _summary_from_single_value(v: float) -> MetricSummary:
    # Baseline is a single deterministic value (Phase 1.3/1.4)
    return MetricSummary(mean=v, std=0.0, q05=v, q50=v, q95=v, n=1)


def run_from_spec_dict(spec_dict: dict, *, spec_path_for_notes: str | None = None, rerun_of: str | None = None) -> str:
    """
    Execute an experiment run given a parsed spec dictionary.

    This is the canonical execution path used both for CLI `run` (from file)
    and for `rerun` (from a stored spec_json snapshot in the DB).
    """
    spec = load_experiment_spec(spec_dict)

    dataset_fp = dataset_fingerprint_csv(spec.dataset.path)
    exp_def = ExperimentDefinition.from_spec(spec, dataset_fingerprint=dataset_fp)

    # Canonical JSON snapshot of the original spec for deterministic reruns.
    spec_json_str = canonical_json(spec_dict)

    # Phase 2.2: Write to DB
    with get_session() as session:
        upsert_experiment(session, exp_def, spec_json_str)

    # Phase 2.1: Keep JSONL logging optional
    upsert_experiment_jsonl(exp_def)

    if spec.metric not in METRICS:
        raise ValueError(f"Unknown metric '{spec.metric}'. Supported: {sorted(METRICS.keys())}")

    # Load dataset. If model_source is provided, require features.
    require_features = spec.model_source is not None
    allow_missing_score = spec.model_source is not None
    ds = load_binary_csv(
        path=spec.dataset.path,
        y_true_col=spec.dataset.y_true_col,
        y_score_col=spec.dataset.y_score_col,
        subgroup_col=spec.dataset.subgroup_col,
        feature_cols=spec.dataset.feature_cols,
        require_features=require_features,
        allow_missing_score=allow_missing_score,
    )

    y_true = ds.y_true.to_numpy(dtype=int)
    y_score = ds.y_score.to_numpy(dtype=float)
    subgroup = None
    if ds.subgroup is not None:
        subgroup = ds.subgroup.to_numpy()

    # If model source is provided, run inference to produce surface.
    prediction_surface = None
    surface_type = SurfaceType.PROBABILITY
    if spec.model_source is not None:
        if ds.X is None:
            raise ValueError("Model inference requires feature columns in dataset.")
        if spec.model_source.kind == "pickle":
            if not spec.model_source.path:
                raise ValueError("model_source.path is required for kind=pickle")
            source = ModelSourcePickle(path=spec.model_source.path)
        elif spec.model_source.kind == "import":
            if not spec.model_source.import_path:
                raise ValueError("model_source.import_path is required for kind=import")
            source = ModelSourceImport(import_path=spec.model_source.import_path)
        else:
            raise ValueError(f"Unsupported model_source kind: {spec.model_source.kind}")

        adapter = ModelAdapter(
            source,
            threshold=spec.model_source.threshold or 0.5,
            positive_label=spec.model_source.positive_label or 1,
            calibration_state=CalibrationState.UNKNOWN,
        )
        surfaces = adapter.get_all_surfaces(ds.X.to_numpy())
        # Prefer probability surface when available.
        if SurfaceType.PROBABILITY in surfaces:
            prediction_surface = surfaces[SurfaceType.PROBABILITY]
        elif SurfaceType.SCORE in surfaces:
            prediction_surface = surfaces[SurfaceType.SCORE]
        elif SurfaceType.LABEL in surfaces:
            prediction_surface = surfaces[SurfaceType.LABEL]
        else:
            raise ValueError("Model adapter produced no usable surfaces.")
        surface_type = prediction_surface.surface_type
        y_score = prediction_surface.values.astype(float)

    # Resolve applicable metrics based on surface type and dataset properties.
    if spec.model_source is not None:
        dataset_props = DatasetProperties(
            n_samples=int(len(y_true)),
            n_positive=int(np.sum(y_true == 1)),
            n_negative=int(np.sum(y_true == 0)),
            has_subgroups=subgroup is not None,
            positive_rate=float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
        )
        resolver = MetricResolver()
        applicable = resolver.resolve(
            task_type=spec.task,
            surface_type=surface_type,
            dataset_props=dataset_props,
        )
    else:
        applicable = ApplicableMetricSet(
            task_type=spec.task,
            surface_type=surface_type,
            metrics=[spec.metric],
            excluded=[],
            reasoning_trace=["manual_metric_selection"],
            warnings=[],
        )

    # Primary metric selection: respect spec.metric if applicable.
    primary_metric = spec.metric if spec.metric in applicable.metrics else None
    if primary_metric is None and applicable.metrics:
        primary_metric = applicable.metrics[0]
    if primary_metric is None:
        raise ValueError("No applicable metrics available for this surface.")

    metric_results: dict[str, MetricSummary] = {}
    scenario_results_by_metric: dict[str, list[ScenarioResult]] = {}

    for metric_id in applicable.metrics:
        metric_fn = METRICS[metric_id]
        if metric_id in {"accuracy", "f1", "precision", "recall", "matthews_corrcoef"}:
            baseline_value = metric_fn(y_true, y_score, threshold=0.5)
        else:
            baseline_value = metric_fn(y_true, y_score)
        metric_results[metric_id] = _summary_from_single_value(baseline_value)

        # --- Phase 1.4: run scenario stress tests (Monte Carlo) ---
        scenario_results = run_scenarios(
            y_true=y_true,
            y_score=y_score,
            metric_name=metric_id,
            metric_fn=metric_fn,
            scenario_specs=[s.model_dump() for s in spec.scenarios],
            cfg=RunConfig(n_trials=spec.n_trials, seed=spec.seed),
            ctx=ScenarioContext(task=spec.task, surface_type=surface_type.value),
            subgroup=subgroup,
        )

        # --- Phase 1.5: add sensitivity_abs diagnostic ---
        baseline_mean = baseline_value
        scenario_results_with_diag = []
        for sr in scenario_results:
            sensitivity_abs = abs(sr.metric.mean - baseline_mean)
            diag = sr.diagnostics.copy()
            diag["sensitivity_abs"] = sensitivity_abs
            scenario_results_with_diag.append(
                ScenarioResult(
                    scenario_id=sr.scenario_id,
                    params=sr.params,
                    metric=sr.metric,
                    diagnostics=diag,
                    artifacts=sr.artifacts,
                )
            )
        scenario_results_by_metric[metric_id] = scenario_results_with_diag

    baseline_cal = {}
    if surface_type == SurfaceType.PROBABILITY:
        baseline_cal = {
            "brier": brier_score(y_true, y_score),
            "ece": expected_calibration_error(y_true, y_score, n_bins=10),
        }

    # Use primary metric for headline baseline/scenarios.
    baseline_value = metric_results[primary_metric].mean
    scenario_results_with_diag = scenario_results_by_metric.get(primary_metric, [])

    run_id = uuid.uuid4().hex[:10].upper()
    paths = get_run_dir(run_id)
    paths.ensure()

    run_record = RunRecord(
        run_id=run_id,
        experiment_id=exp_def.experiment_id,
        results_path=str(paths.results_json),
        artifacts_dir=str(paths.artifacts_dir),
        seed_used=spec.seed,
        rerun_of=rerun_of,
    )
    
    # Phase 2.2: Write to DB (queued)
    with get_session() as session:
        insert_run(session, run_record)
    
    # Phase 2.1: Keep JSONL logging optional
    log_run_jsonl(run_record)

    try:
        run_record.mark_running()
        
        # Phase 2.2: Update DB (running)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)

        # --- Phase 1.7B: generate artifacts (plots) ---
        rng_artifacts = np.random.default_rng(spec.seed)
        scenario_results_with_artifacts: list[ScenarioResult] = []
        for sr in scenario_results_with_diag:
            artifacts_list: list[Artifact] = []
            scenario_id = sr.scenario_id

            # 1. Metric distribution plot
            try:
                metric_dist_path = paths.artifacts_dir / f"metric_dist_{scenario_id}.png"
                plot_metric_distribution(
                    metric_summary=sr.metric.model_dump(),
                    metric_name=spec.metric,
                    scenario_id=scenario_id,
                    out_path=metric_dist_path,
                )
                artifacts_list.append(
                    Artifact(
                        kind="plot",
                        path=f"artifacts/metric_dist_{scenario_id}.png",
                        meta={"type": "metric_distribution"},
                    )
                )
            except Exception:
                pass  # Skip if plot generation fails

            # 2. Calibration curve (run one representative trial)
            try:
                scenario = create_scenario(scenario_id, sr.params)
                y_p_rep, s_p_rep = scenario.apply(
                    y_true, y_score, rng_artifacts, ScenarioContext(task=spec.task, surface_type=surface_type.value)
                )
                if len(y_p_rep) > 0 and len(s_p_rep) > 0:
                    cal_path = paths.artifacts_dir / f"calibration_{scenario_id}.png"
                    plot_calibration_curve(
                        y_true=y_p_rep,
                        y_score=s_p_rep,
                        scenario_id=scenario_id,
                        out_path=cal_path,
                    )
                    artifacts_list.append(
                        Artifact(
                            kind="plot",
                            path=f"artifacts/calibration_{scenario_id}.png",
                            meta={"type": "calibration_curve"},
                        )
                    )
            except Exception:
                pass  # Skip if plot generation fails

            # 3. Subgroup metric bars (if subgroup diagnostics exist)
            try:
                subgroup_metric = sr.diagnostics.get("subgroup_metric")
                if subgroup_metric:
                    group_means = {k: v["mean"] for k, v in subgroup_metric.items()}
                    if group_means:
                        subgroup_path = paths.artifacts_dir / f"subgroup_metric_{scenario_id}.png"
                        plot_subgroup_bars(
                            group_means=group_means,
                            scenario_id=scenario_id,
                            out_path=subgroup_path,
                        )
                        artifacts_list.append(
                            Artifact(
                                kind="plot",
                                path=f"artifacts/subgroup_metric_{scenario_id}.png",
                                meta={"type": "subgroup_comparison"},
                            )
                        )
            except Exception:
                pass  # Skip if plot generation fails

            # 4. Threshold curve (only for accuracy with metric_inflation)
            if spec.metric == "accuracy":
                try:
                    metric_inflation = sr.diagnostics.get("metric_inflation")
                    if metric_inflation:
                        scenario = create_scenario(scenario_id, sr.params)
                        y_p_rep, s_p_rep = scenario.apply(
                            y_true, y_score, rng_artifacts, ScenarioContext(task=spec.task, surface_type=surface_type.value)
                        )
                        if len(y_p_rep) > 0 and len(s_p_rep) > 0:
                            # Get mean optimal threshold from diagnostics (approximate)
                            # Use a representative threshold from the inflation data
                            baseline_thresh = 0.5
                            # Estimate optimized threshold from delta (use 0.5 + small adjustment as proxy)
                            # Actually, we need to recompute or store it - let's use a simple heuristic
                            # For now, use 0.5 as baseline and compute optimal from representative trial
                            from metrics_lie.diagnostics.metric_gaming import find_optimal_threshold
                            thresholds = np.linspace(0.05, 0.95, 19)
                            opt_thresh, _ = find_optimal_threshold(y_p_rep, s_p_rep, thresholds)
                            
                            threshold_path = paths.artifacts_dir / f"threshold_curve_{scenario_id}.png"
                            plot_threshold_curve(
                                y_true=y_p_rep,
                                y_score=s_p_rep,
                                baseline_threshold=baseline_thresh,
                                optimized_threshold=opt_thresh,
                                scenario_id=scenario_id,
                                out_path=threshold_path,
                            )
                            artifacts_list.append(
                                Artifact(
                                    kind="plot",
                                    path=f"artifacts/threshold_curve_{scenario_id}.png",
                                    meta={"type": "threshold_optimization"},
                                )
                            )
                except Exception:
                    pass  # Skip if plot generation fails

            scenario_results_with_artifacts.append(
                ScenarioResult(
                    scenario_id=sr.scenario_id,
                    params=sr.params,
                    metric=sr.metric,
                    diagnostics=sr.diagnostics,
                    artifacts=artifacts_list,
                )
            )

            if artifacts_list:
                print(f"[PLOT] Saved {len(artifacts_list)} artifacts for scenario {scenario_id}")

        notes = {
            "phase": "1.7B",
            "spec_path": spec_path_for_notes,
            "baseline_diagnostics": baseline_cal,
            "applicable_metrics": applicable.metrics,
            "metric_resolution": {
                "excluded": applicable.excluded,
                "warnings": applicable.warnings,
                "reasoning_trace": applicable.reasoning_trace,
            },
        }

        analysis_artifacts: dict[str, Any] = {}
        if prediction_surface is not None and prediction_surface.surface_type == SurfaceType.PROBABILITY:
            sweep = run_threshold_sweep(
                y_true=y_true, surface=prediction_surface, metrics=applicable.metrics, n_points=101
            )
            analysis_artifacts["threshold_sweep"] = sweep.to_jsonable()
            sensitivity = run_sensitivity_analysis(
                y_true=y_true,
                surface=prediction_surface,
                metrics=applicable.metrics,
                perturbation_type="score_noise",
                magnitudes=[0.01, 0.02, 0.05, 0.1, 0.2],
                n_trials=50,
                seed=spec.seed,
            )
            analysis_artifacts["sensitivity"] = sensitivity.to_jsonable()
            disagreements = analyze_metric_disagreements(
                y_true=y_true,
                surface=prediction_surface,
                thresholds=sweep.optimal_thresholds,
                metrics=applicable.metrics,
            )
            analysis_artifacts["metric_disagreements"] = [d.to_jsonable() for d in disagreements]
            failures = locate_failure_modes(
                y_true=y_true,
                surface=prediction_surface,
                metrics=applicable.metrics,
                subgroup=subgroup,
                top_k=20,
            )
            analysis_artifacts["failure_modes"] = failures.to_jsonable()

        bundle = ResultBundle(
            run_id=run_id,
            experiment_name=spec.name,
            metric_name=primary_metric,
            baseline=_summary_from_single_value(baseline_value),
            scenarios=scenario_results_with_artifacts,
            prediction_surface=prediction_surface.to_jsonable() if prediction_surface else None,
            applicable_metrics=applicable.metrics,
            metric_results=metric_results,
            scenario_results_by_metric=scenario_results_by_metric,
            analysis_artifacts=analysis_artifacts,
            notes=notes,
        )

        paths.results_json.write_text(bundle.to_pretty_json(), encoding="utf-8")
        print(f"[OK] Wrote results: {paths.results_json}")
        print(f"Baseline {spec.metric} = {baseline_value:.6f}")
        if spec.scenarios:
            print(f"[OK] Ran {len(spec.scenarios)} scenario(s) with n_trials={spec.n_trials}")

        # Phase 2.2: Insert artifacts into DB
        all_artifacts: list[Artifact] = []
        for sr in scenario_results_with_artifacts:
            all_artifacts.extend(sr.artifacts)
        if all_artifacts:
            with get_session() as session:
                insert_artifacts(session, run_id, all_artifacts)

        run_record.mark_completed()
        
        # Phase 2.2: Update DB (completed)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)
    except Exception as exc:  # pragma: no cover - simple logging wrapper
        run_record.mark_failed(str(exc))
        
        # Phase 2.2: Update DB (failed)
        with get_session() as session:
            update_run(session, run_record)
        
        # Phase 2.1: Keep JSONL logging optional
        log_run_jsonl(run_record)
        raise

    return run_id


def rerun(run_id: str) -> str:
    """
    Deterministically rerun a completed experiment using the stored spec_json snapshot.

    This creates a new run (with a new run_id) linked to the same experiment_id,
    and optionally records the original run_id in the DB.
    """
    with get_session() as session:
        experiment_id = get_experiment_id_for_run(session, run_id)
        spec_json_str = get_experiment_spec_json(session, experiment_id)

    if not spec_json_str:
        raise ValueError(
            "No stored spec_json found for experiment. "
            "Please run database migrations with 'alembic upgrade head' before using 'rerun'."
        )

    spec_dict = json.loads(spec_json_str)
    # Use a descriptive marker in notes so results.json keeps the same schema.
    spec_path_for_notes = f"<rerun_of:{run_id}>"
    new_run_id = run_from_spec_dict(spec_dict, spec_path_for_notes=spec_path_for_notes, rerun_of=run_id)
    print(new_run_id)
    return new_run_id

