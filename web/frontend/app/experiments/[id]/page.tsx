"use client";

import { useRouter, useParams } from "next/navigation";
import { useState, useEffect, useRef, useCallback } from "react";
import {
  getExperiment,
  getResults,
  runExperiment,
  ApiError,
  type ExperimentSummary,
  type ResultSummary,
} from "@/lib/api";

export default function ExperimentPage() {
  const router = useRouter();
  const params = useParams();
  const experimentId = params.id as string;

  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null);
  const [result, setResult] = useState<ResultSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [retrying, setRetrying] = useState(false);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const loadData = useCallback(async (silent = false) => {
    if (!silent) {
      setLoading(true);
    }
    setError(null);
    setResultsError(null);

    try {
      // Load experiment status
      const exp = await getExperiment(experimentId);
      setExperiment(exp);
      setLastUpdated(new Date());

      // Try to load results
      try {
        const res = await getResults(experimentId);
        setResult(res);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          setResult(null);
        } else {
          setResultsError(e instanceof Error ? e.message : "Failed to load results");
        }
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        router.push("/experiments/not-found");
        return;
      }
      setError(e instanceof Error ? e.message : "Failed to load experiment");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [experimentId, router]);

  function stopPolling() {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }

  function startPolling() {
    stopPolling();
    pollingIntervalRef.current = setInterval(() => {
      loadData(true);
    }, 2000);
  }

  useEffect(() => {
    if (experimentId) {
      loadData();
    }

    return () => {
      stopPolling();
    };
  }, [experimentId, loadData]);

  useEffect(() => {
    if (!experiment) return;

    const shouldPoll =
      experiment.status === "running" || (experiment.status === "completed" && !result);

    if (shouldPoll) {
      startPolling();
    } else {
      stopPolling();
    }

    return () => {
      stopPolling();
    };
  }, [experiment?.status, result]);

  async function handleRetry() {
    if (!experiment) return;
    setRetrying(true);
    setError(null);

    try {
      await runExperiment(experiment.id);
      // Reload data to get updated status
      await loadData();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to retry experiment");
    } finally {
      setRetrying(false);
    }
  }

  if (loading && !experiment) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (error && !experiment) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fee",
            color: "#c00",
            borderRadius: "4px",
            marginTop: "1rem",
          }}
        >
          Error: {error}
        </div>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <p>Experiment not found.</p>
      </div>
    );
  }

  // Handle failed status
  if (experiment.status === "failed") {
    return (
      <div>
        <h1>Experiment Results</h1>
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fee",
            color: "#c00",
            borderRadius: "4px",
            marginTop: "1rem",
          }}
        >
          <h2 style={{ marginTop: 0 }}>Experiment Failed</h2>
          {experiment.error_message ? (
            <p>{experiment.error_message}</p>
          ) : (
            <p>The experiment run failed. Please check the backend logs for details.</p>
          )}
        </div>
        {error && (
          <div
            style={{
              padding: "0.75rem",
              backgroundColor: "#fff3cd",
              color: "#856404",
              borderRadius: "4px",
              marginTop: "1rem",
            }}
          >
            {error}
          </div>
        )}
        <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
          <button
            onClick={handleRetry}
            disabled={retrying}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: retrying ? "#ccc" : "#28a745",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: retrying ? "not-allowed" : "pointer",
              fontWeight: "bold",
            }}
          >
            {retrying ? "Retrying..." : "Retry Run"}
          </button>
          <button
            onClick={() => loadData()}
            disabled={loading}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: loading ? "#ccc" : "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>
    );
  }

  // Handle running status or no results yet
  if (experiment.status === "running" || !result) {
    const isPolling = pollingIntervalRef.current !== null;
    return (
      <div>
        <h1>Experiment Results</h1>
        <div
          style={{
            padding: "2rem",
            textAlign: "center",
            color: "#666",
          }}
        >
          <div style={{ marginBottom: "1rem", display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}>
            <div
              style={{
                width: "20px",
                height: "20px",
                border: "3px solid #f3f3f3",
                borderTop: "3px solid #0070f3",
                borderRadius: "50%",
                animation: "spin 1s linear infinite",
              }}
            />
            <p style={{ fontSize: "1.25rem", margin: 0 }}>
              {experiment.status === "running" ? "Experiment is running..." : "No results available yet."}
            </p>
          </div>
          {lastUpdated && (
            <p style={{ fontSize: "0.875rem", color: "#999", marginTop: "0.5rem" }}>
              Last updated: {lastUpdated.toLocaleTimeString()}
              {isPolling && " (auto-updating)"}
            </p>
          )}
          <button
            onClick={() => loadData()}
            disabled={loading}
            style={{
              marginTop: "1rem",
              padding: "0.75rem 1.5rem",
              backgroundColor: loading ? "#ccc" : "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              fontSize: "1rem",
              fontWeight: "bold",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Refreshing..." : "Refresh Now"}
          </button>
        </div>
      </div>
    );
  }

  // Show results
  return (
    <div>
      <h1>Experiment Results</h1>
      {resultsError && (
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fff3cd",
            color: "#856404",
            borderRadius: "4px",
            marginTop: "1rem",
          }}
        >
          Warning: {resultsError}
        </div>
      )}
      <div style={{ marginTop: "2rem" }}>
        <div style={{ marginBottom: "2rem" }}>
          <h2>Headline Score</h2>
          <div style={{ fontSize: "2rem", fontWeight: "bold", marginTop: "0.5rem" }}>
            {result.headline_score.toFixed(6)}
          </div>
          {result.weighted_score !== null && (
            <div style={{ fontSize: "1rem", color: "#666", marginTop: "0.25rem" }}>
              Weighted: {result.weighted_score.toFixed(6)}
            </div>
          )}
        </div>

        {result.component_scores.length > 0 && (
          <div style={{ marginBottom: "2rem" }}>
            <h2>Component Scores ({result.component_scores.length})</h2>
            <div style={{ marginTop: "1rem", display: "grid", gap: "0.5rem" }}>
              {result.component_scores.map((comp, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "0.75rem",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    display: "flex",
                    justifyContent: "space-between",
                  }}
                >
                  <span>{comp.name}</span>
                  <span style={{ fontWeight: "bold" }}>{comp.score.toFixed(6)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.scenario_results.length > 0 && (
          <div style={{ marginBottom: "2rem" }}>
            <h2>Scenario Results ({result.scenario_results.length})</h2>
            <div style={{ marginTop: "1rem", display: "grid", gap: "0.5rem" }}>
              {result.scenario_results.map((scenario, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "0.75rem",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                    <span style={{ fontWeight: "bold" }}>{scenario.scenario_name}</span>
                    <span>Score: {scenario.score.toFixed(6)}</span>
                  </div>
                  <div style={{ fontSize: "0.875rem", color: "#666" }}>
                    Delta: {scenario.delta > 0 ? "+" : ""}{scenario.delta.toFixed(6)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.flags.length > 0 && (
          <div style={{ marginBottom: "2rem" }}>
            <h2>Flags ({result.flags.length})</h2>
            <div style={{ marginTop: "1rem", display: "grid", gap: "0.5rem" }}>
              {result.flags.map((flag, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "0.75rem",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    backgroundColor:
                      flag.severity === "critical"
                        ? "#f8d7da"
                        : flag.severity === "warn"
                        ? "#fff3cd"
                        : "#d1ecf1",
                  }}
                >
                  <div style={{ fontWeight: "bold", marginBottom: "0.25rem" }}>{flag.title}</div>
                  <div style={{ fontSize: "0.875rem" }}>{flag.detail}</div>
                  <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                    [{flag.severity}] {flag.code}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ fontSize: "0.875rem", color: "#666", marginTop: "2rem" }}>
          Generated: {new Date(result.generated_at).toLocaleString()}
        </div>
      </div>
    </div>
  );
}
