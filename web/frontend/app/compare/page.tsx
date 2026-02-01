"use client";

import { useState, useEffect } from "react";
import {
  listExperiments,
  listRuns,
  getRunResult,
  ApiError,
  type ExperimentSummary,
  type RunSummary,
  type ResultSummary,
} from "@/lib/api";
import { deriveComparisonFindings } from "@/lib/compare_insights";

export default function ComparePage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loadingExperiments, setLoadingExperiments] = useState(true);

  // Selector A
  const [experimentIdA, setExperimentIdA] = useState<string>("");
  const [runsA, setRunsA] = useState<RunSummary[]>([]);
  const [loadingRunsA, setLoadingRunsA] = useState(false);
  const [runIdA, setRunIdA] = useState<string>("");

  // Selector B
  const [experimentIdB, setExperimentIdB] = useState<string>("");
  const [runsB, setRunsB] = useState<RunSummary[]>([]);
  const [loadingRunsB, setLoadingRunsB] = useState(false);
  const [runIdB, setRunIdB] = useState<string>("");

  // Comparison results
  const [resultA, setResultA] = useState<ResultSummary | null>(null);
  const [resultB, setResultB] = useState<ResultSummary | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load experiments on mount
  useEffect(() => {
    async function loadExperiments() {
      try {
        const exps = await listExperiments();
        setExperiments(exps);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load experiments");
      } finally {
        setLoadingExperiments(false);
      }
    }
    loadExperiments();
  }, []);

  // Load runs when experiment A changes
  useEffect(() => {
    if (!experimentIdA) {
      setRunsA([]);
      setRunIdA("");
      return;
    }

    setLoadingRunsA(true);
    setRunIdA("");
    listRuns(experimentIdA)
      .then((runs) => {
        setRunsA(runs);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Failed to load runs for experiment A");
        setRunsA([]);
      })
      .finally(() => {
        setLoadingRunsA(false);
      });
  }, [experimentIdA]);

  // Load runs when experiment B changes
  useEffect(() => {
    if (!experimentIdB) {
      setRunsB([]);
      setRunIdB("");
      return;
    }

    setLoadingRunsB(true);
    setRunIdB("");
    listRuns(experimentIdB)
      .then((runs) => {
        setRunsB(runs);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Failed to load runs for experiment B");
        setRunsB([]);
      })
      .finally(() => {
        setLoadingRunsB(false);
      });
  }, [experimentIdB]);

  async function handleCompare() {
    if (!experimentIdA || !runIdA || !experimentIdB || !runIdB) {
      setError("Please select both experiments and runs");
      return;
    }

    setError(null);
    setLoadingComparison(true);
    setResultA(null);
    setResultB(null);

    try {
      const [resA, resB] = await Promise.all([
        getRunResult(experimentIdA, runIdA),
        getRunResult(experimentIdB, runIdB),
      ]);
      setResultA(resA);
      setResultB(resB);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load comparison results");
    } finally {
      setLoadingComparison(false);
    }
  }

  // Comparison data
  const findings = resultA && resultB ? deriveComparisonFindings(resultA, resultB) : [];

  // Scenario diff
  const scenarioMapA = new Map(resultA?.scenario_results.map((s) => [s.scenario_id, s]) || []);
  const scenarioMapB = new Map(resultB?.scenario_results.map((s) => [s.scenario_id, s]) || []);
  const scenarioDiffs: Array<{
    scenario_id: string;
    scenario_name: string;
    deltaA: number;
    deltaB: number;
    change: number;
  }> = [];

  for (const [scenarioId, scenarioB] of scenarioMapB.entries()) {
    const scenarioA = scenarioMapA.get(scenarioId);
    if (scenarioA) {
      scenarioDiffs.push({
        scenario_id: scenarioId,
        scenario_name: scenarioB.scenario_name,
        deltaA: scenarioA.delta,
        deltaB: scenarioB.delta,
        change: scenarioB.delta - scenarioA.delta,
      });
    }
  }
  scenarioDiffs.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

  // Component diff
  const componentMapA = new Map(resultA?.component_scores.map((c) => [c.name, c]) || []);
  const componentMapB = new Map(resultB?.component_scores.map((c) => [c.name, c]) || []);
  const componentDiffs: Array<{
    name: string;
    scoreA: number;
    scoreB: number;
    delta: number;
  }> = [];

  for (const [name, componentB] of componentMapB.entries()) {
    const componentA = componentMapA.get(name);
    if (componentA) {
      componentDiffs.push({
        name,
        scoreA: componentA.score,
        scoreB: componentB.score,
        delta: componentB.score - componentA.score,
      });
    }
  }
  componentDiffs.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  // Flags diff
  const flagsA = resultA?.flags || [];
  const flagsB = resultB?.flags || [];
  const flagsA_set = new Set(flagsA.map((f) => f.code));
  const flagsB_set = new Set(flagsB.map((f) => f.code));

  const addedFlags = flagsB.filter((f) => !flagsA_set.has(f.code));
  const removedFlags = flagsA.filter((f) => !flagsB_set.has(f.code));
  const persistingFlags = flagsA.filter((f) => flagsB_set.has(f.code));

  const groupFlagsBySeverity = (flags: typeof flagsA) => {
    return {
      critical: flags.filter((f) => f.severity === "critical"),
      warn: flags.filter((f) => f.severity === "warn"),
      info: flags.filter((f) => f.severity === "info"),
    };
  };

  if (loadingExperiments) {
    return (
      <div>
        <h1>Compare Runs</h1>
        <p>Loading experiments...</p>
      </div>
    );
  }

  return (
    <div>
      <h1>Compare Runs</h1>

      {error && (
        <div
          style={{
            padding: "1rem",
            backgroundColor: "#fee",
            color: "#c00",
            borderRadius: "4px",
            marginBottom: "1rem",
          }}
        >
          Error: {error}
        </div>
      )}

      {/* Selectors */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "2rem",
          marginBottom: "2rem",
          padding: "1.5rem",
          border: "1px solid #e0e0e0",
          borderRadius: "8px",
          backgroundColor: "#f8f9fa",
        }}
      >
        {/* Selector A */}
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Run A</h2>
          <div style={{ marginBottom: "1rem" }}>
            <label
              htmlFor="experiment-a"
              style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
            >
              Experiment
            </label>
            <select
              id="experiment-a"
              value={experimentIdA}
              onChange={(e) => setExperimentIdA(e.target.value)}
              disabled={loadingExperiments}
              style={{
                width: "100%",
                padding: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: "4px",
                fontSize: "1rem",
              }}
            >
              <option value="">Select experiment...</option>
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label
              htmlFor="run-a"
              style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
            >
              Run
            </label>
            <select
              id="run-a"
              value={runIdA}
              onChange={(e) => setRunIdA(e.target.value)}
              disabled={!experimentIdA || loadingRunsA || runsA.length === 0}
              style={{
                width: "100%",
                padding: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: "4px",
                fontSize: "1rem",
              }}
            >
              <option value="">Select run...</option>
              {loadingRunsA ? (
                <option disabled>Loading runs...</option>
              ) : (
                runsA.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.generated_at
                      ? new Date(run.generated_at).toLocaleString()
                      : run.run_id.slice(0, 8)}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        {/* Selector B */}
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Run B</h2>
          <div style={{ marginBottom: "1rem" }}>
            <label
              htmlFor="experiment-b"
              style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
            >
              Experiment
            </label>
            <select
              id="experiment-b"
              value={experimentIdB}
              onChange={(e) => setExperimentIdB(e.target.value)}
              disabled={loadingExperiments}
              style={{
                width: "100%",
                padding: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: "4px",
                fontSize: "1rem",
              }}
            >
              <option value="">Select experiment...</option>
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label
              htmlFor="run-b"
              style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
            >
              Run
            </label>
            <select
              id="run-b"
              value={runIdB}
              onChange={(e) => setRunIdB(e.target.value)}
              disabled={!experimentIdB || loadingRunsB || runsB.length === 0}
              style={{
                width: "100%",
                padding: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: "4px",
                fontSize: "1rem",
              }}
            >
              <option value="">Select run...</option>
              {loadingRunsB ? (
                <option disabled>Loading runs...</option>
              ) : (
                runsB.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.generated_at
                      ? new Date(run.generated_at).toLocaleString()
                      : run.run_id.slice(0, 8)}
                  </option>
                ))
              )}
            </select>
          </div>
        </div>
      </div>

      {/* Compare Button */}
      <div style={{ marginBottom: "2rem", textAlign: "center" }}>
        <button
          onClick={handleCompare}
          disabled={!experimentIdA || !runIdA || !experimentIdB || !runIdB || loadingComparison}
          style={{
            padding: "0.75rem 2rem",
            backgroundColor:
              !experimentIdA || !runIdA || !experimentIdB || !runIdB || loadingComparison
                ? "#ccc"
                : "#0070f3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            fontSize: "1rem",
            fontWeight: "bold",
            cursor:
              !experimentIdA || !runIdA || !experimentIdB || !runIdB || loadingComparison
                ? "not-allowed"
                : "pointer",
          }}
        >
          {loadingComparison ? "Comparing..." : "Compare"}
        </button>
      </div>

      {/* Comparison Dashboard */}
      {resultA && resultB && (
        <div style={{ display: "grid", gap: "1.5rem" }}>
          {/* Analyst Panel */}
          {findings.length > 0 && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "#f8f9fa",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Analyst Panel v1</h2>
              <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
                {findings.map((finding, idx) => (
                  <li key={idx} style={{ marginBottom: "0.5rem", lineHeight: "1.5" }}>
                    {finding.text}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Headline Score Comparison */}
          <div
            style={{
              padding: "1.5rem",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              backgroundColor: "white",
            }}
          >
            <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Headline Score</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
              <div>
                <div style={{ fontSize: "0.875rem", color: "#666", marginBottom: "0.25rem" }}>
                  Run A
                </div>
                <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                  {resultA.headline_score.toFixed(4)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: "0.875rem", color: "#666", marginBottom: "0.25rem" }}>
                  Run B
                </div>
                <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                  {resultB.headline_score.toFixed(4)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: "0.875rem", color: "#666", marginBottom: "0.25rem" }}>
                  Delta (B-A)
                </div>
                <div
                  style={{
                    fontSize: "2rem",
                    fontWeight: "bold",
                    color: resultB.headline_score - resultA.headline_score >= 0 ? "#28a745" : "#dc3545",
                  }}
                >
                  {(resultB.headline_score - resultA.headline_score >= 0 ? "+" : "") +
                    (resultB.headline_score - resultA.headline_score).toFixed(4)}
                </div>
              </div>
            </div>
          </div>

          {/* Scenario Diff Table */}
          {scenarioDiffs.length > 0 && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "white",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Scenario Comparison</h2>
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "0.875rem",
                  }}
                >
                  <thead>
                    <tr style={{ borderBottom: "2px solid #e0e0e0" }}>
                      <th style={{ textAlign: "left", padding: "0.5rem" }}>Scenario</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Δ A</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Δ B</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scenarioDiffs.map((diff) => (
                      <tr key={diff.scenario_id} style={{ borderBottom: "1px solid #e0e0e0" }}>
                        <td style={{ padding: "0.5rem" }}>{diff.scenario_name}</td>
                        <td style={{ textAlign: "right", padding: "0.5rem" }}>
                          {diff.deltaA.toFixed(4)}
                        </td>
                        <td style={{ textAlign: "right", padding: "0.5rem" }}>
                          {diff.deltaB.toFixed(4)}
                        </td>
                        <td
                          style={{
                            textAlign: "right",
                            padding: "0.5rem",
                            color: diff.change >= 0 ? "#28a745" : "#dc3545",
                            fontWeight: "bold",
                          }}
                        >
                          {(diff.change >= 0 ? "+" : "") + diff.change.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Component Diff Table */}
          {componentDiffs.length > 0 && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "white",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Component Comparison</h2>
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "0.875rem",
                  }}
                >
                  <thead>
                    <tr style={{ borderBottom: "2px solid #e0e0e0" }}>
                      <th style={{ textAlign: "left", padding: "0.5rem" }}>Component</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Score A</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Score B</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {componentDiffs.map((diff) => (
                      <tr key={diff.name} style={{ borderBottom: "1px solid #e0e0e0" }}>
                        <td style={{ padding: "0.5rem" }}>{diff.name}</td>
                        <td style={{ textAlign: "right", padding: "0.5rem" }}>
                          {diff.scoreA.toFixed(4)}
                        </td>
                        <td style={{ textAlign: "right", padding: "0.5rem" }}>
                          {diff.scoreB.toFixed(4)}
                        </td>
                        <td
                          style={{
                            textAlign: "right",
                            padding: "0.5rem",
                            color: diff.delta >= 0 ? "#28a745" : "#dc3545",
                            fontWeight: "bold",
                          }}
                        >
                          {(diff.delta >= 0 ? "+" : "") + diff.delta.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Flags Diff */}
          {(addedFlags.length > 0 || removedFlags.length > 0 || persistingFlags.length > 0) && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "white",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Flags Comparison</h2>
              <div style={{ display: "grid", gap: "1rem" }}>
                {/* Added Flags */}
                {addedFlags.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#856404" }}>
                      Added ({addedFlags.length})
                    </h3>
                    {Object.entries(groupFlagsBySeverity(addedFlags)).map(([severity, flags]) => {
                      if (flags.length === 0) return null;
                      const colors = {
                        critical: { bg: "#f8d7da", text: "#721c24", border: "#f5c6cb" },
                        warn: { bg: "#fff3cd", text: "#856404", border: "#ffeaa7" },
                        info: { bg: "#d1ecf1", text: "#0c5460", border: "#bee5eb" },
                      };
                      const color = colors[severity as keyof typeof colors];
                      return (
                        <div key={severity} style={{ marginBottom: "0.5rem" }}>
                          <div
                            style={{
                              padding: "0.5rem",
                              fontSize: "0.75rem",
                              fontWeight: "bold",
                              color: color.text,
                            }}
                          >
                            {severity.toUpperCase()}
                          </div>
                          {flags.map((flag, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: "0.75rem",
                                border: `1px solid ${color.border}`,
                                borderRadius: "4px",
                                backgroundColor: color.bg,
                                marginBottom: "0.5rem",
                              }}
                            >
                              <div style={{ fontWeight: "bold", color: color.text }}>
                                {flag.title}
                              </div>
                              <div style={{ fontSize: "0.875rem", color: color.text }}>
                                {flag.detail}
                              </div>
                              <div style={{ fontSize: "0.75rem", color: color.text, marginTop: "0.25rem" }}>
                                [{flag.code}]
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Removed Flags */}
                {removedFlags.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#28a745" }}>
                      Removed ({removedFlags.length})
                    </h3>
                    {Object.entries(groupFlagsBySeverity(removedFlags)).map(([severity, flags]) => {
                      if (flags.length === 0) return null;
                      const colors = {
                        critical: { bg: "#f8d7da", text: "#721c24", border: "#f5c6cb" },
                        warn: { bg: "#fff3cd", text: "#856404", border: "#ffeaa7" },
                        info: { bg: "#d1ecf1", text: "#0c5460", border: "#bee5eb" },
                      };
                      const color = colors[severity as keyof typeof colors];
                      return (
                        <div key={severity} style={{ marginBottom: "0.5rem" }}>
                          <div
                            style={{
                              padding: "0.5rem",
                              fontSize: "0.75rem",
                              fontWeight: "bold",
                              color: color.text,
                            }}
                          >
                            {severity.toUpperCase()}
                          </div>
                          {flags.map((flag, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: "0.75rem",
                                border: `1px solid ${color.border}`,
                                borderRadius: "4px",
                                backgroundColor: color.bg,
                                marginBottom: "0.5rem",
                                opacity: 0.7,
                              }}
                            >
                              <div style={{ fontWeight: "bold", color: color.text }}>
                                {flag.title}
                              </div>
                              <div style={{ fontSize: "0.875rem", color: color.text }}>
                                {flag.detail}
                              </div>
                              <div style={{ fontSize: "0.75rem", color: color.text, marginTop: "0.25rem" }}>
                                [{flag.code}]
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Persisting Flags */}
                {persistingFlags.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#666" }}>
                      Persisting ({persistingFlags.length})
                    </h3>
                    {Object.entries(groupFlagsBySeverity(persistingFlags)).map(([severity, flags]) => {
                      if (flags.length === 0) return null;
                      const colors = {
                        critical: { bg: "#f8d7da", text: "#721c24", border: "#f5c6cb" },
                        warn: { bg: "#fff3cd", text: "#856404", border: "#ffeaa7" },
                        info: { bg: "#d1ecf1", text: "#0c5460", border: "#bee5eb" },
                      };
                      const color = colors[severity as keyof typeof colors];
                      return (
                        <div key={severity} style={{ marginBottom: "0.5rem" }}>
                          <div
                            style={{
                              padding: "0.5rem",
                              fontSize: "0.75rem",
                              fontWeight: "bold",
                              color: color.text,
                            }}
                          >
                            {severity.toUpperCase()}
                          </div>
                          {flags.map((flag, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: "0.75rem",
                                border: `1px solid ${color.border}`,
                                borderRadius: "4px",
                                backgroundColor: color.bg,
                                marginBottom: "0.5rem",
                              }}
                            >
                              <div style={{ fontWeight: "bold", color: color.text }}>
                                {flag.title}
                              </div>
                              <div style={{ fontSize: "0.875rem", color: color.text }}>
                                {flag.detail}
                              </div>
                              <div style={{ fontSize: "0.75rem", color: color.text, marginTop: "0.25rem" }}>
                                [{flag.code}]
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

