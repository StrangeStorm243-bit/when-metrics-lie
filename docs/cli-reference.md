# CLI Reference

Spectra's CLI is available via the `spectra` command after installation.

## Core Commands

### `spectra run <spec.json>`

Run an experiment from a JSON spec file.

```bash
spectra run experiments/binary_eval.json
```

### `spectra evaluate`

Quick evaluation without a spec file.

```bash
spectra evaluate --name quick_test \
  --dataset data.csv \
  --metric auc \
  --task binary_classification
```

### `spectra compare <run_a> <run_b>`

Compare two runs side-by-side.

```bash
spectra compare abc123 def456
```

Output includes baseline delta, per-scenario deltas, regression flags, and a winner recommendation.

### `spectra score <run_a> <run_b>`

Compare with decision profile scoring.

```bash
spectra score abc123 def456 --profile balanced
spectra score abc123 def456 --profile risk_averse
spectra score abc123 def456 --profile performance_focused
```

Profiles weight different components (calibration, robustness, fairness, etc.) differently.

### `spectra rerun <run_id>`

Deterministically rerun an experiment.

```bash
spectra rerun abc123
```

## Job Queue Commands

### `spectra enqueue-run <experiment_id>`

Queue a run for background processing.

### `spectra worker-once`

Process one job from the queue.

## Query Commands

### `spectra experiments list`

List experiments.

```bash
spectra experiments list --limit 10
```

### `spectra experiments show <id>`

Show experiment details.

### `spectra runs list`

List runs.

```bash
spectra runs list --limit 10 --status completed
spectra runs list --experiment <experiment_id>
```

### `spectra runs show <id>`

Show run details.

### `spectra jobs list`

List queued jobs.

```bash
spectra jobs list --limit 10 --status pending
```

## Decision Profiles

Three built-in profiles for `spectra score`:

| Profile | Focus |
|---------|-------|
| `balanced` | Equal weight across all components |
| `risk_averse` | Prioritizes calibration and robustness |
| `performance_focused` | Prioritizes headline metric performance |

Custom profiles: pass a JSON file path instead of a profile name.
