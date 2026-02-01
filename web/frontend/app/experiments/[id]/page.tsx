"use client";

import { useRouter, useParams } from "next/navigation";
import { useState, useEffect } from "react";
import { getExperiment, getResults, type ExperimentSummary, type ResultSummary } from "@/lib/api";

export default function ExperimentPage() {
  const router = useRouter();
  const params = useParams();
  const experimentId = params.id as string;

  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null);
  const [result, setResult] = useState<ResultSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resultsError, setResultsError] = useState<string | null>(null);

  async function loadData() {
    setLoading(true);
    setError(null);
    setResultsError(null);

    try {
      // Load experiment status
      const exp = await getExperiment(experimentId);
      setExperiment(exp);

      // Try to load results
      try {
        const res = await getResults(experimentId);
        setResult(res);
      } catch (e) {
        if (e instanceof Error && e.message.includes("404")) {
          setResult(null);
        } else {
          setResultsError(e instanceof Error ? e.message : "Failed to load results");
        }
      }
    } catch (e) {
      if (e instanceof Error && e.message.includes("404")) {
        router.push("/experiments/not-found");
        return;
      }
      setError(e instanceof Error ? e.message : "Failed to load experiment");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (experimentId) {
      loadData();
    }
  }, [experimentId]);

  if (loading && !experiment) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (error) {
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
        <button
          onClick={loadData}
          style={{
            marginTop: "1rem",
            padding: "0.5rem 1rem",
            backgroundColor: "#0070f3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </div>
    );
  }

  // Handle running status or no results yet
  if (experiment.status === "running" || !result) {
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
          <p style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>
            {experiment.status === "running" ? "Experiment is running..." : "No results available yet."}
          </p>
          <button
            onClick={loadData}
            disabled={loading}
            style={{
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
            {loading ? "Refreshing..." : "Refresh"}
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
