/**
 * API client for Spectra backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_SPECTRA_API_BASE || "http://127.0.0.1:8000";

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
 */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
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
 */
export async function listExperiments(): Promise<ExperimentSummary[]> {
  return apiFetch<ExperimentSummary[]>("/experiments");
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

