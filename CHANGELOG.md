# Changelog

All notable changes to Spectra will be documented in this file.

## [0.2.0] - 2026-01-29

### Added
- Console entrypoint `spectra` for CLI access
- Version flag: `spectra --version` displays version
- Decision profiles with configurable aggregation modes (worst_case, mean, percentile)
- Decision component extraction from compare reports
- Transparent weighted scorecard generation
- CLI commands: `score`, `experiments`, `runs`, `jobs` with list/show subcommands
- Async job queue system for experiment execution
- Database persistence layer with SQLAlchemy and Alembic migrations
- Deterministic rerun capability using stored experiment specs
- Compare reports with scenario deltas and risk flags
- Public-facing README with Quickstart guide

### Changed
- Internal package structure remains `metrics_lie`; external CLI branded as `spectra`

### Fixed
- UnboundLocalError in CLI when `run` variable shadowed `run()` function

---

## [0.1.0] - Initial Release

### Added
- Core evaluation engine with scenario-based stress testing
- Experiment specification schema (JSON-based)
- Baseline metric computation
- Scenario perturbations: label noise, score noise, class imbalance
- Diagnostic computations: calibration (ECE, Brier), subgroup gaps, sensitivity analysis
- Metric gaming detection (metric inflation)
- Artifact generation: plots and visualizations
- Result bundle schema with versioning
- Deterministic Monte Carlo trials with configurable seed

