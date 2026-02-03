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
import { deriveFindings, getSeverityFromDelta } from "@/lib/insights";

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
  const [copiedMessage, setCopiedMessage] = useState<string | null>(null);
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

  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    stopPolling();
    pollingIntervalRef.current = setInterval(() => {
      loadData(true);
    }, 2000);
  }, [loadData, stopPolling]);

  useEffect(() => {
    if (experimentId) {
      loadData();
    }

    return () => {
      stopPolling();
    };
  }, [experimentId, loadData, stopPolling]);

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
  }, [experiment, result, startPolling, stopPolling]);

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

  function handleExportJSON() {
    if (!result) return;
    const jsonStr = JSON.stringify(result, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spectra_${result.experiment_id}_${result.run_id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function handleCopySummary() {
    if (!experiment || !result) return;
    
    const worstScenario = result.scenario_results.length > 0
      ? result.scenario_results.reduce((min, s) => (s.delta < min.delta ? s : min))
      : null;
    const topFlags = result.flags
      .sort((a, b) => {
        const severityOrder = { critical: 0, warn: 1, info: 2 };
        return severityOrder[a.severity] - severityOrder[b.severity];
      })
      .slice(0, 3)
      .map(f => f.code);
    
    const summary = [
      `Experiment: ${experiment.name} (${experiment.id})`,
      `Status: ${experiment.status}`,
      `Metric: ${experiment.metric_id}`,
      `Stress Suite: ${experiment.stress_suite_id}`,
      `Headline Score: ${result.headline_score.toFixed(4)}`,
      worstScenario ? `Worst Scenario: ${worstScenario.scenario_name} (Δ=${worstScenario.delta.toFixed(4)})` : "Worst Scenario: N/A",
      `Flags: ${result.flags.length} total (top: ${topFlags.join(", ") || "none"})`,
      `Run ID: ${result.run_id}`,
    ].join("\n");

    try {
      await navigator.clipboard.writeText(summary);
      setCopiedMessage("Summary copied!");
      setTimeout(() => setCopiedMessage(null), 2000);
    } catch (err) {
      console.error("Failed to copy summary:", err);
    }
  }

  async function handleCopyJSON() {
    if (!result) return;
    const jsonStr = JSON.stringify(result, null, 2);
    try {
      await navigator.clipboard.writeText(jsonStr);
      setCopiedMessage("JSON copied!");
      setTimeout(() => setCopiedMessage(null), 2000);
    } catch (err) {
      console.error("Failed to copy JSON:", err);
    }
  }

  function handlePrint() {
    window.print();
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case "completed":
        return "#28a745";
      case "running":
        return "#ffc107";
      case "failed":
        return "#dc3545";
      default:
        return "#6c757d";
    }
  }

  function getStatusBgColor(status: string): string {
    switch (status) {
      case "completed":
        return "#d4edda";
      case "running":
        return "#fff3cd";
      case "failed":
        return "#f8d7da";
      default:
        return "#e2e3e5";
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
  const findings = deriveFindings(result);
  const sortedScenarios = [...result.scenario_results].sort((a, b) => a.delta - b.delta);
  const maxComponentScore = Math.max(
    ...result.component_scores.map((c) => c.score),
    1
  );
  const flagsBySeverity = {
    critical: result.flags.filter((f) => f.severity === "critical"),
    warn: result.flags.filter((f) => f.severity === "warn"),
    info: result.flags.filter((f) => f.severity === "info"),
  };

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "2rem",
          flexWrap: "wrap",
          gap: "1rem",
        }}
      >
        <div>
          <h1 style={{ marginBottom: "0.5rem" }}>{experiment.name}</h1>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
            <span
              style={{
                padding: "0.25rem 0.75rem",
                borderRadius: "12px",
                fontSize: "0.875rem",
                fontWeight: "bold",
                backgroundColor: getStatusBgColor(experiment.status),
                color: getStatusColor(experiment.status),
                textTransform: "uppercase",
              }}
            >
              {experiment.status}
            </span>
            {lastUpdated && (
              <span style={{ fontSize: "0.875rem", color: "#666" }}>
                Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button
            onClick={() => loadData()}
            disabled={loading}
            className="no-print"
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: loading ? "#ccc" : "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "0.875rem",
            }}
          >
            {loading ? "Refreshing..." : "Refresh Now"}
          </button>
          <button
            onClick={handleExportJSON}
            className="no-print"
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.875rem",
            }}
          >
            Export JSON
          </button>
          <button
            onClick={handlePrint}
            className="no-print"
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#28a745",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.875rem",
            }}
          >
            Print
          </button>
        </div>
      </div>

      {/* Share Block */}
      <div
        className="no-print"
        style={{
          padding: "1rem",
          border: "1px solid #e0e0e0",
          borderRadius: "8px",
          backgroundColor: "#f8f9fa",
          marginBottom: "1.5rem",
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: "0.75rem", fontSize: "1rem" }}>Share</h2>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <button
            onClick={handleCopySummary}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.875rem",
            }}
          >
            Copy Summary
          </button>
          <button
            onClick={handleCopyJSON}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.875rem",
            }}
          >
            Copy JSON
          </button>
          {copiedMessage && (
            <span
              style={{
                padding: "0.25rem 0.75rem",
                backgroundColor: "#28a745",
                color: "white",
                borderRadius: "4px",
                fontSize: "0.875rem",
                fontWeight: "500",
              }}
            >
              {copiedMessage}
            </span>
          )}
        </div>
      </div>

      {resultsError && (
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fff3cd",
            color: "#856404",
            borderRadius: "4px",
            marginBottom: "1rem",
          }}
        >
          Warning: {resultsError}
        </div>
      )}

      <div style={{ display: "grid", gap: "1.5rem" }}>
        {/* Key Findings */}
        {findings.length > 0 && (
          <div
            style={{
              padding: "1.5rem",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              backgroundColor: "#f8f9fa",
            }}
          >
            <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Key Findings</h2>
            <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
              {findings.map((finding, idx) => (
                <li key={idx} style={{ marginBottom: "0.5rem", lineHeight: "1.5" }}>
                  {finding.text}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Headline Score */}
        <div
          style={{
            padding: "1.5rem",
            border: "1px solid #e0e0e0",
            borderRadius: "8px",
            backgroundColor: "white",
          }}
        >
          <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Headline Score</h2>
          <div style={{ fontSize: "3rem", fontWeight: "bold", color: "#0070f3" }}>
            {result.headline_score.toFixed(4)}
          </div>
          {result.weighted_score !== null && (
            <div style={{ fontSize: "1rem", color: "#666", marginTop: "0.5rem" }}>
              Weighted: {result.weighted_score.toFixed(4)}
            </div>
          )}
        </div>

        {/* Component Scores */}
        <div
          style={{
            padding: "1.5rem",
            border: "1px solid #e0e0e0",
            borderRadius: "8px",
            backgroundColor: "white",
          }}
        >
          <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>
            Component Scores ({result.component_scores.length})
          </h2>
          {result.component_scores.length === 0 ? (
            <p style={{ color: "#666", margin: 0 }}>No component-level diagnostics available for this run.</p>
          ) : (
            <div style={{ display: "grid", gap: "1rem" }}>
              {result.component_scores.map((comp, idx) => {
                const barWidth = Math.max(0, Math.min(100, (comp.score / maxComponentScore) * 100));
                return (
                  <div key={idx}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                      <span style={{ fontWeight: "500" }}>{comp.name}</span>
                      <span style={{ fontWeight: "bold" }}>{comp.score.toFixed(4)}</span>
                    </div>
                    <div
                      style={{
                        width: "100%",
                        height: "24px",
                        backgroundColor: "#e9ecef",
                        borderRadius: "4px",
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${barWidth}%`,
                          height: "100%",
                          backgroundColor: comp.score >= 0.7 ? "#28a745" : comp.score >= 0.5 ? "#ffc107" : "#dc3545",
                          transition: "width 0.3s ease",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Scenario Results */}
        {sortedScenarios.length > 0 && (
          <div
            style={{
              padding: "1.5rem",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              backgroundColor: "white",
            }}
          >
            <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>
              Scenario Stress Results ({sortedScenarios.length})
            </h2>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {sortedScenarios.map((scenario, idx) => {
                const severity = getSeverityFromDelta(scenario.delta);
                const severityColors = {
                  high: { bg: "#f8d7da", text: "#721c24", border: "#f5c6cb" },
                  med: { bg: "#fff3cd", text: "#856404", border: "#ffeaa7" },
                  low: { bg: "#d1ecf1", text: "#0c5460", border: "#bee5eb" },
                };
                const colors = severityColors[severity];
                return (
                  <div
                    key={idx}
                    style={{
                      padding: "1rem",
                      border: `1px solid ${colors.border}`,
                      borderRadius: "6px",
                      backgroundColor: colors.bg,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                      <div>
                        <div style={{ fontWeight: "bold", marginBottom: "0.25rem" }}>{scenario.scenario_name}</div>
                        <div style={{ fontSize: "0.875rem", color: colors.text }}>
                          Score: {scenario.score.toFixed(4)}
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <span
                          style={{
                            padding: "0.25rem 0.5rem",
                            borderRadius: "4px",
                            fontSize: "0.75rem",
                            fontWeight: "bold",
                            backgroundColor: colors.text,
                            color: colors.bg,
                            textTransform: "uppercase",
                            marginBottom: "0.25rem",
                            display: "inline-block",
                          }}
                        >
                          {severity}
                        </span>
                        <div
                          style={{
                            fontSize: "1.25rem",
                            fontWeight: "bold",
                            color: colors.text,
                          }}
                        >
                          {scenario.delta > 0 ? "+" : ""}
                          {scenario.delta.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Flags by Severity */}
        {(flagsBySeverity.critical.length > 0 ||
          flagsBySeverity.warn.length > 0 ||
          flagsBySeverity.info.length > 0) && (
          <div
            style={{
              padding: "1.5rem",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              backgroundColor: "white",
            }}
          >
            <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Flags ({result.flags.length})</h2>
            <div style={{ display: "grid", gap: "1rem" }}>
              {/* Critical Flags */}
              {flagsBySeverity.critical.length > 0 && (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#721c24" }}>
                    Critical ({flagsBySeverity.critical.length})
                  </h3>
                  <div style={{ display: "grid", gap: "0.5rem" }}>
                    {flagsBySeverity.critical.map((flag, idx) => (
                      <div
                        key={idx}
                        style={{
                          padding: "0.75rem",
                          border: "1px solid #f5c6cb",
                          borderRadius: "4px",
                          backgroundColor: "#f8d7da",
                        }}
                      >
                        <div style={{ fontWeight: "bold", marginBottom: "0.25rem", color: "#721c24" }}>
                          {flag.title}
                        </div>
                        <div style={{ fontSize: "0.875rem", color: "#721c24" }}>{flag.detail}</div>
                        <div style={{ fontSize: "0.75rem", color: "#856404", marginTop: "0.25rem" }}>
                          [{flag.code}]
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Warn Flags */}
              {flagsBySeverity.warn.length > 0 && (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#856404" }}>
                    Warnings ({flagsBySeverity.warn.length})
                  </h3>
                  <div style={{ display: "grid", gap: "0.5rem" }}>
                    {flagsBySeverity.warn.map((flag, idx) => (
                      <div
                        key={idx}
                        style={{
                          padding: "0.75rem",
                          border: "1px solid #ffeaa7",
                          borderRadius: "4px",
                          backgroundColor: "#fff3cd",
                        }}
                      >
                        <div style={{ fontWeight: "bold", marginBottom: "0.25rem", color: "#856404" }}>
                          {flag.title}
                        </div>
                        <div style={{ fontSize: "0.875rem", color: "#856404" }}>{flag.detail}</div>
                        <div style={{ fontSize: "0.75rem", color: "#856404", marginTop: "0.25rem" }}>
                          [{flag.code}]
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Info Flags */}
              {flagsBySeverity.info.length > 0 && (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#0c5460" }}>
                    Info ({flagsBySeverity.info.length})
                  </h3>
                  <div style={{ display: "grid", gap: "0.5rem" }}>
                    {flagsBySeverity.info.map((flag, idx) => (
                      <div
                        key={idx}
                        style={{
                          padding: "0.75rem",
                          border: "1px solid #bee5eb",
                          borderRadius: "4px",
                          backgroundColor: "#d1ecf1",
                        }}
                      >
                        <div style={{ fontWeight: "bold", marginBottom: "0.25rem", color: "#0c5460" }}>
                          {flag.title}
                        </div>
                        <div style={{ fontSize: "0.875rem", color: "#0c5460" }}>{flag.detail}</div>
                        <div style={{ fontSize: "0.75rem", color: "#0c5460", marginTop: "0.25rem" }}>
                          [{flag.code}]
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Metadata */}
        <div style={{ fontSize: "0.875rem", color: "#666", textAlign: "center", paddingTop: "1rem" }}>
          Generated: {new Date(result.generated_at).toLocaleString()} • Run ID: {result.run_id}
        </div>
      </div>
    </div>
  );
}
