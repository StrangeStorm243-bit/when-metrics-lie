/**
 * API client for Spectra backend.
 *
 * Auth token injection:
 * - Client-side: AuthTokenProvider (lib/auth.ts) registers a global token provider.
 * - Server-side: callers pass token explicitly via the optional token parameter.
 * - Local dev: no token → backend returns 'anonymous' owner_id.
 */

const API_BASE = process.env.NEXT_PUBLIC_SPECTRA_API_BASE || "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Global token provider (set by AuthTokenProvider on the client)
// ---------------------------------------------------------------------------

let _tokenProvider: (() => Promise<string | null>) | null = null;

export function setTokenProvider(provider: (() => Promise<string | null>) | null): void {
  _tokenProvider = provider;
}

// TypeScript types matching backend contracts
export interface ExperimentSummary {
  id: string;
  name: string;
  metric_id: string;
  stress_suite_id: string;
  status: "created" | "running" | "completed" | "failed";
  created_at: string;
  last_run_at: string | null;
  error_message?: string | null;
}

export interface MetricPreset {
  id: string;
  name: string;
  description: string;
}

export interface StressSuitePreset {
  id: string;
  name: string;
  description: string;
}

export interface DatasetPreset {
  id: string;
  name: string;
  path: string;
  description?: string;
}

export interface RunRequest {
  seed?: number | null;
}

export interface RunResponse {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed";
}

export interface ComponentScore {
  name: string;
  score: number;
  weight: number | null;
  notes: string | null;
}

export interface ScenarioResult {
  scenario_id: string;
  scenario_name: string;
  delta: number;
  score: number;
  severity: "low" | "med" | "high" | null;
  notes: string | null;
}

export interface FindingFlag {
  code: string;
  title: string;
  detail: string;
  severity: "info" | "warn" | "critical";
}

export interface ResultSummary {
  experiment_id: string;
  run_id: string;
  headline_score: number;
  weighted_score: number | null;
  component_scores: ComponentScore[];
  scenario_results: ScenarioResult[];
  flags: FindingFlag[];
  prediction_surface?: Record<string, unknown> | null;
  applicable_metrics?: string[];
  analysis_artifacts?: Record<string, unknown> | null;
  generated_at: string;
}

export interface ExperimentCreateRequest {
  name: string;
  metric_id: string;
  stress_suite_id: string;
  notes?: string | null;
  config?: Record<string, unknown>;
}

/**
 * Custom error class with status code.
 */
export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Fetch wrapper that throws on non-2xx with parsed JSON error if present.
 *
 * Auth token is injected automatically from the global token provider (client)
 * or from an explicit token parameter (server components).
 */
async function apiFetch<T>(
  path: string,
  init?: RequestInit & { token?: string },
): Promise<T> {
  // Separate our custom 'token' field from standard RequestInit
  const { token: explicitToken, ...fetchInit } = init || {};

  // Resolve auth token: explicit > global provider > none
  let authToken: string | null = explicitToken ?? null;
  if (!authToken && _tokenProvider) {
    try {
      authToken = await _tokenProvider();
    } catch {
      // Token provider failed; proceed without auth
    }
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(fetchInit?.headers as Record<string, string> | undefined),
  };

  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...fetchInit,
    headers,
  });

  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Ignore JSON parse errors, use status text
    }
    throw new ApiError(errorMessage, response.status);
  }

  return response.json();
}

/**
 * List all experiments.
 * @param token Optional auth token (for server-side calls).
 */
export async function listExperiments(token?: string): Promise<ExperimentSummary[]> {
  return apiFetch<ExperimentSummary[]>("/experiments", { token });
}

/**
 * Create a new experiment.
 */
export async function createExperiment(
  payload: ExperimentCreateRequest
): Promise<ExperimentSummary> {
  return apiFetch<ExperimentSummary>("/experiments", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Run an experiment.
 */
export async function runExperiment(
  experimentId: string,
  payload?: RunRequest
): Promise<RunResponse> {
  return apiFetch<RunResponse>(`/experiments/${experimentId}/run`, {
    method: "POST",
    body: JSON.stringify(payload || {}),
  });
}

/**
 * Get results for an experiment.
 */
export async function getResults(experimentId: string): Promise<ResultSummary> {
  return apiFetch<ResultSummary>(`/experiments/${experimentId}/results`);
}

/**
 * Get a specific experiment by ID.
 */
export async function getExperiment(experimentId: string): Promise<ExperimentSummary> {
  return apiFetch<ExperimentSummary>(`/experiments/${experimentId}`);
}

/**
 * Get available metric presets.
 */
export async function getMetricPresets(): Promise<MetricPreset[]> {
  return apiFetch<MetricPreset[]>("/presets/metrics");
}

/**
 * Get available stress suite presets.
 */
export async function getStressSuitePresets(): Promise<StressSuitePreset[]> {
  return apiFetch<StressSuitePreset[]>("/presets/stress-suites");
}

/**
 * Get available dataset presets.
 */
export async function getDatasetPresets(): Promise<DatasetPreset[]> {
  return apiFetch<DatasetPreset[]>("/presets/datasets");
}

/**
 * Run summary for listing runs.
 */
export interface RunSummary {
  run_id: string;
  generated_at: string | null;
}

/**
 * List all runs for an experiment.
 */
export async function listRuns(experimentId: string): Promise<RunSummary[]> {
  return apiFetch<RunSummary[]>(`/experiments/${experimentId}/runs`);
}

/**
 * Get results for a specific run.
 */
export async function getRunResult(experimentId: string, runId: string): Promise<ResultSummary> {
  return apiFetch<ResultSummary>(`/experiments/${experimentId}/runs/${runId}/results`);
}

export interface RunAnalysisResponse {
  run_id: string;
  analysis_artifacts: Record<string, unknown>;
}

export async function getRunAnalysis(
  experimentId: string,
  runId: string
): Promise<RunAnalysisResponse> {
  return apiFetch<RunAnalysisResponse>(`/experiments/${experimentId}/runs/${runId}/analysis`);
}

/**
 * LLM explanation request.
 */
export interface CompareExplainRequest {
  intent: string;
  focus?: { type: "scenario" | "component" | "flag"; key: string } | null;
  context: Record<string, unknown>;
  user_question?: string | null;
}

/**
 * LLM explanation response.
 */
export interface CompareExplainResponse {
  title: string;
  body_markdown: string;
  evidence_keys: string[];
}

/**
 * Get LLM explanation for a comparison.
 */
export async function compareExplain(request: CompareExplainRequest): Promise<CompareExplainResponse> {
  return apiFetch<CompareExplainResponse>("/llm/compare-explain", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

