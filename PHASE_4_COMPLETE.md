# Phase 4 Complete: Demo-Ready Frontend

## What Phase 4 Delivers (Demo-Ready)

- **`/new` flow**: Create and run experiments with dataset selection
  - Metric and stress suite presets
  - Dataset selection from available CSV files
  - Automatic experiment execution after creation

- **`/experiments/[id]` results UI**: Comprehensive experiment results view
  - Headline score and component scores visualization
  - Scenario stress results with severity indicators
  - Flags organized by severity (critical, warn, info)
  - Key findings summary
  - Export JSON functionality
  - Print-friendly styling
  - Share functionality (copy summary/JSON)

- **`/compare` run comparison**: Side-by-side comparison of two experiment runs
  - Visual diff of headline scores, components, scenarios, and flags
  - Deterministic analyst with prompt chips
  - Optional LLM-powered analyst (Phase 4B) with fallback to deterministic

- **`/assistant` deterministic chat**: Interactive Q&A over loaded run context
  - Experiment and run selection
  - Context loading for specific runs
  - Deterministic responses based on loaded result data
  - Evidence-based answers mentioning headline score, worst scenario, and flags
  - Optional LLM integration toggle (Phase 4B) but not required for demo

## What Phase 4 Does Not Do Yet (Intentional)

- **No model upload / no model training**: Spectra evaluates existing model outputs, not training models
- **No multi-dataset metric comparability guarantees**: Each experiment uses a single dataset; cross-dataset comparisons require manual interpretation
- **No full "ask anything" global search**: Assistant requires context to be loaded for a specific run before answering
- **No production auth, multi-user, or hosted infra**: Designed for local development and evaluation workflows

## Next (Phase 5 Ideas)

- **True conversational query layer**: Tool-like interface for asking questions across loaded context with follow-up questions
- **Model artifact upload + evaluation harness**: Direct integration with model checkpoints and evaluation pipelines
- **Scenario library expansion + metric scoring policies**: User-defined scenarios and custom metric weighting
- **Hosted deployment & auth**: Multi-user support with authentication and shared experiment libraries
