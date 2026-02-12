"use client";

import { useState, useEffect, useRef } from "react";
import {
  listExperiments,
  listRuns,
  getRunResult,
  compareRuns,
  compareExplain,
  ApiError,
  type ExperimentSummary,
  type RunSummary,
  type ResultSummary,
  type CompareExplainRequest,
  type CompareResponse,
} from "@/lib/api";
import {
  buildScenarioDiff,
  buildComponentDiff,
  buildFlagDiff,
  getWorstCaseScenario,
  getBiggestRegression,
  getBiggestImprovement,
  getBiggestComponentChange,
} from "@/lib/compare_model";
import { respond, type AnalystIntent, type AnalystMessage } from "@/lib/analyst";

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
  const [backendCompare, setBackendCompare] = useState<CompareResponse | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Analyst panel
  const [analystMessages, setAnalystMessages] = useState<AnalystMessage[]>([]);
  const [pinnedMessages, setPinnedMessages] = useState<Set<string>>(new Set());
  const [focus, setFocus] = useState<{ type: "scenario" | "component" | "flag"; key: string } | null>(null);
  const analystPanelRef = useRef<HTMLDivElement>(null);
  const [useAI, setUseAI] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

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
    setBackendCompare(null);
    setAnalystMessages([]);
    setPinnedMessages(new Set());
    setFocus(null);

    try {
      const [resA, resB] = await Promise.all([
        getRunResult(experimentIdA, runIdA),
        getRunResult(experimentIdB, runIdB),
      ]);
      setResultA(resA);
      setResultB(resB);

      // Phase 7: backend compare (regressions, risk flags, decision) when bundles exist
      try {
        const compareData = await compareRuns(
          { experiment_id: experimentIdA, run_id: runIdA },
          { experiment_id: experimentIdB, run_id: runIdB },
        );
        setBackendCompare(compareData);
      } catch {
        // Bundles may not exist for pre-Phase-7 runs; keep existing client-side comparison
        setBackendCompare(null);
      }

      // Auto-trigger overview
      if (resA && resB) {
        const scenarioDiffs = buildScenarioDiff(resA, resB);
        const componentDiffs = buildComponentDiff(resA, resB);
        const flagDiff = buildFlagDiff(resA.flags, resB.flags);
        const overview = respond("overview", {
          resultA: resA,
          resultB: resB,
          scenarioDiffs,
          componentDiffs,
          flagDiff,
        });
        setAnalystMessages(overview);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load comparison results");
    } finally {
      setLoadingComparison(false);
    }
  }

  // Build compact context bundle for LLM
  function buildContextBundle() {
    if (!resultA || !resultB) return null;

    const scenarioDiffs = buildScenarioDiff(resultA, resultB);
    const componentDiffs = buildComponentDiff(resultA, resultB);
    const flagDiff = buildFlagDiff(resultA.flags, resultB.flags);

    const expA = experiments.find((e) => e.id === experimentIdA);
    const expB = experiments.find((e) => e.id === experimentIdB);

    return {
      experiments: {
        a: {
          id: experimentIdA,
          name: expA?.name || experimentIdA,
          run_id: runIdA,
        },
        b: {
          id: experimentIdB,
          name: expB?.name || experimentIdB,
          run_id: runIdB,
        },
      },
      headline: {
        score_a: resultA.headline_score,
        score_b: resultB.headline_score,
        delta: resultB.headline_score - resultA.headline_score,
      },
      worst_case: getWorstCaseScenario(scenarioDiffs)
        ? {
            scenario_id: getWorstCaseScenario(scenarioDiffs)!.scenario_id,
            scenario_name: getWorstCaseScenario(scenarioDiffs)!.scenario_name,
            delta_a: getWorstCaseScenario(scenarioDiffs)!.deltaA,
            delta_b: getWorstCaseScenario(scenarioDiffs)!.deltaB,
          }
        : null,
      top_scenario_changes: scenarioDiffs.slice(0, 10).map((d) => ({
        scenario_id: d.scenario_id,
        scenario_name: d.scenario_name,
        delta_a: d.deltaA,
        delta_b: d.deltaB,
        change: d.change,
      })),
      top_component_changes: componentDiffs.slice(0, 10).map((d) => ({
        name: d.name,
        score_a: d.scoreA,
        score_b: d.scoreB,
        delta: d.delta,
      })),
      flags_summary: {
        added: flagDiff.added.length,
        removed: flagDiff.removed.length,
        persisting: flagDiff.persisting.length,
        by_severity: {
          critical: {
            added: flagDiff.bySeverity.critical.added.length,
            removed: flagDiff.bySeverity.critical.removed.length,
          },
          warn: {
            added: flagDiff.bySeverity.warn.added.length,
            removed: flagDiff.bySeverity.warn.removed.length,
          },
        },
        top_added: flagDiff.added.slice(0, 5).map((f) => ({
          code: f.code,
          title: f.title,
          severity: f.severity,
        })),
      },
    };
  }

  async function handlePrompt(intent: AnalystIntent) {
    if (!resultA || !resultB) return;

    const scenarioDiffs = buildScenarioDiff(resultA, resultB);
    const componentDiffs = buildComponentDiff(resultA, resultB);
    const flagDiff = buildFlagDiff(resultA.flags, resultB.flags);

    const userMessage: AnalystMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      body: intent === "overview" ? "Show overview" : intent === "worse" ? "What got worse?" : intent === "improved" ? "What improved?" : intent === "worst_case" ? "Explain worst-case change" : "Explain new flags",
    };

    setAnalystMessages((prev) => [...prev, userMessage]);
    setAiError(null);

    // Try AI if enabled
    if (useAI) {
      const context = buildContextBundle();
      if (context) {
        try {
          const request: CompareExplainRequest = {
            intent,
            focus: focus || null,
            context,
            user_question: userMessage.body,
          };

          const aiResponse = await compareExplain(request);

          const aiMessage: AnalystMessage = {
            id: `ai-${Date.now()}`,
            role: "assistant",
            title: `AI Analyst: ${aiResponse.title}`,
            body: aiResponse.body_markdown,
            evidence: aiResponse.evidence_keys.length > 0 ? `Evidence: ${aiResponse.evidence_keys.join(", ")}` : undefined,
          };

          setAnalystMessages((prev) => [...prev, aiMessage]);
          setTimeout(() => {
            analystPanelRef.current?.scrollTo({
              top: analystPanelRef.current.scrollHeight,
              behavior: "smooth",
            });
          }, 100);
          return;
        } catch (e) {
          // Fallback to deterministic on error (including 501)
          if (e instanceof ApiError && e.status === 501) {
            setAiError("AI features require ANTHROPIC_API_KEY. Falling back to deterministic analyst.");
          } else {
            setAiError(`AI request failed: ${e instanceof Error ? e.message : "Unknown error"}. Falling back to deterministic analyst.`);
          }
        }
      }
    }

    // Deterministic fallback (default or on error)
    const responses = respond(intent, {
      resultA,
      resultB,
      scenarioDiffs,
      componentDiffs,
      flagDiff,
      focus: focus || undefined,
    });

    setAnalystMessages((prev) => [...prev, ...responses]);

    // Scroll to bottom
    setTimeout(() => {
      analystPanelRef.current?.scrollTo({
        top: analystPanelRef.current.scrollHeight,
        behavior: "smooth",
      });
    }, 100);
  }

  function handlePin(messageId: string) {
    setPinnedMessages((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }

  async function handleFocus(type: "scenario" | "component" | "flag", key: string) {
    setFocus({ type, key });

    if (!resultA || !resultB) return;

    const scenarioDiffs = buildScenarioDiff(resultA, resultB);
    const componentDiffs = buildComponentDiff(resultA, resultB);
    const flagDiff = buildFlagDiff(resultA.flags, resultB.flags);

    const intent: AnalystIntent = type === "scenario" ? "scenario_focus" : type === "component" ? "component_focus" : "flag_focus";
    setAiError(null);

    // Try AI if enabled
    if (useAI) {
      const context = buildContextBundle();
      if (context) {
        try {
          const request: CompareExplainRequest = {
            intent,
            focus: { type, key },
            context,
          };

          const aiResponse = await compareExplain(request);

          const aiMessage: AnalystMessage = {
            id: `ai-${Date.now()}`,
            role: "assistant",
            title: `AI Analyst: ${aiResponse.title}`,
            body: aiResponse.body_markdown,
            evidence: aiResponse.evidence_keys.length > 0 ? `Evidence: ${aiResponse.evidence_keys.join(", ")}` : undefined,
          };

          setAnalystMessages((prev) => [...prev, aiMessage]);
          setTimeout(() => {
            analystPanelRef.current?.scrollTo({
              top: analystPanelRef.current.scrollHeight,
              behavior: "smooth",
            });
          }, 100);
          return;
        } catch (e) {
          // Fallback to deterministic on error (including 501)
          if (e instanceof ApiError && e.status === 501) {
            setAiError("AI features require ANTHROPIC_API_KEY. Falling back to deterministic analyst.");
          } else {
            setAiError(`AI request failed: ${e instanceof Error ? e.message : "Unknown error"}. Falling back to deterministic analyst.`);
          }
        }
      }
    }

    // Deterministic fallback (default or on error)
    const responses = respond(intent, {
      resultA,
      resultB,
      scenarioDiffs,
      componentDiffs,
      flagDiff,
      focus: { type, key },
    });

    setAnalystMessages((prev) => [...prev, ...responses]);

    setTimeout(() => {
      analystPanelRef.current?.scrollTo({
        top: analystPanelRef.current.scrollHeight,
        behavior: "smooth",
      });
    }, 100);
  }

  function handleScrollTo(type: "scenario" | "component" | "flag", key: string) {
    const elementId = `${type}-${key}`;
    const element = document.getElementById(elementId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
      element.classList.add("focusGlow");
      setTimeout(() => {
        element.classList.remove("focusGlow");
      }, 2000);
      setFocus({ type, key });
    }
  }

  // Build diffs using model builders
  const scenarioDiffs = resultA && resultB ? buildScenarioDiff(resultA, resultB) : [];
  const componentDiffs = resultA && resultB ? buildComponentDiff(resultA, resultB) : [];
  const flagDiff = resultA && resultB ? buildFlagDiff(resultA.flags, resultB.flags) : null;

  const pinnedMessagesList = analystMessages.filter((m) => pinnedMessages.has(m.id));

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
          {/* Phase 7: Backend compare (regressions, risk flags, decision) */}
          {backendCompare && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "#f8f9fa",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Comparison Summary</h2>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
                {backendCompare.regressions && typeof backendCompare.regressions === "object" && (
                  <>
                    {Object.entries(backendCompare.regressions).map(([key, val]) => (
                      <span
                        key={key}
                        style={{
                          padding: "0.25rem 0.5rem",
                          borderRadius: "4px",
                          fontSize: "0.875rem",
                          backgroundColor: val ? "#f8d7da" : "#d4edda",
                          color: val ? "#721c24" : "#155724",
                        }}
                      >
                        {key}: {val ? "regression" : "ok"}
                      </span>
                    ))}
                  </>
                )}
              </div>
              {backendCompare.risk_flags && backendCompare.risk_flags.length > 0 && (
                <div style={{ marginBottom: "1rem" }}>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Risk flags</h3>
                  <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
                    {backendCompare.risk_flags.map((flag, i) => (
                      <li key={i} style={{ marginBottom: "0.25rem" }}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}
              {backendCompare.decision && typeof backendCompare.decision === "object" && (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Decision</h3>
                  <p style={{ margin: "0.25rem 0" }}>
                    <strong>Winner:</strong> {(backendCompare.decision as Record<string, unknown>).winner as string}
                    {" · "}
                    <strong>Confidence:</strong> {(backendCompare.decision as Record<string, unknown>).confidence as string}
                  </p>
                  {Array.isArray((backendCompare.decision as Record<string, unknown>).reasoning) &&
                    ((backendCompare.decision as Record<string, unknown>).reasoning as string[]).length > 0 && (
                      <ul style={{ margin: "0.5rem 0 0 1.25rem", padding: 0 }}>
                        {((backendCompare.decision as Record<string, unknown>).reasoning as string[]).map((r, i) => (
                          <li key={i} style={{ marginBottom: "0.25rem" }}>{r}</li>
                        ))}
                      </ul>
                    )}
                </div>
              )}
            </div>
          )}

          {/* Phase 8: Multi-Metric Comparison */}
          {backendCompare?.multi_metric_comparison && (
            <div
              style={{
                padding: "1.5rem",
                border: "1px solid #e0e0e0",
                borderRadius: "8px",
                backgroundColor: "white",
              }}
            >
              <h2 style={{ marginTop: 0, marginBottom: "1rem" }}>Multi-Metric Comparison</h2>
              <div style={{ overflowX: "auto", marginBottom: "1rem" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
                  <thead>
                    <tr style={{ borderBottom: "2px solid #e0e0e0" }}>
                      <th style={{ textAlign: "left", padding: "0.5rem" }}>Metric</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>A Baseline</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>B Baseline</th>
                      <th style={{ textAlign: "right", padding: "0.5rem" }}>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backendCompare.multi_metric_comparison.shared_metrics.map((metricId) => {
                      const d = backendCompare.multi_metric_comparison!.per_metric_deltas[metricId];
                      const delta = d.baseline_delta;
                      return (
                        <tr key={metricId} style={{ borderBottom: "1px solid #e0e0e0" }}>
                          <td style={{ padding: "0.5rem" }}>{metricId}</td>
                          <td style={{ textAlign: "right", padding: "0.5rem" }}>
                            {typeof d.a === "number" ? d.a.toFixed(4) : "—"}
                          </td>
                          <td style={{ textAlign: "right", padding: "0.5rem" }}>
                            {typeof d.b === "number" ? d.b.toFixed(4) : "—"}
                          </td>
                          <td style={{
                            textAlign: "right",
                            padding: "0.5rem",
                            color: delta !== null ? (delta > 0 ? "#28a745" : delta < 0 ? "#dc3545" : "#666") : "#666",
                            fontWeight: "bold",
                          }}>
                            {delta !== null ? (delta > 0 ? "+" : "") + delta.toFixed(4) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize: "0.875rem", color: "#666" }}>
                <p style={{ margin: "0 0 0.25rem 0" }}>
                  <strong>Summary:</strong> {backendCompare.multi_metric_comparison.summary}
                </p>
                {backendCompare.multi_metric_comparison.only_in_a.length > 0 && (
                  <p style={{ margin: "0 0 0.25rem 0" }}>
                    <strong>Only in A:</strong> {backendCompare.multi_metric_comparison.only_in_a.join(", ")}
                  </p>
                )}
                {backendCompare.multi_metric_comparison.only_in_b.length > 0 && (
                  <p style={{ margin: 0 }}>
                    <strong>Only in B:</strong> {backendCompare.multi_metric_comparison.only_in_b.join(", ")}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Pinned Messages Strip */}
          {pinnedMessagesList.length > 0 && (
            <div className="pinnedStrip">
              <h3 style={{ marginTop: 0, marginBottom: "0.5rem", fontSize: "0.875rem" }}>Pinned</h3>
              {pinnedMessagesList.map((msg) => (
                <div key={msg.id} className="pinnedMessage">
                  <strong>{msg.title || "Message"}:</strong> {msg.body}
                </div>
              ))}
            </div>
          )}

          {/* Analyst Panel */}
          <div
            style={{
              padding: "1.5rem",
              border: "1px solid #e0e0e0",
              borderRadius: "8px",
              backgroundColor: "white",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h2 style={{ marginTop: 0, marginBottom: 0 }}>Analyst Assistant</h2>
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={useAI}
                  onChange={(e) => setUseAI(e.target.checked)}
                  style={{ cursor: "pointer" }}
                />
                <span style={{ fontSize: "0.875rem" }}>Enhanced Analyst (AI)</span>
              </label>
            </div>

            {aiError && (
              <div
                style={{
                  padding: "0.75rem",
                  backgroundColor: "#fff3cd",
                  border: "1px solid #ffc107",
                  borderRadius: "4px",
                  marginBottom: "1rem",
                  fontSize: "0.875rem",
                  color: "#856404",
                }}
              >
                {aiError}
              </div>
            )}

            {/* Prompt Chips */}
            <div style={{ marginBottom: "1rem", display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              <button className="chip" onClick={() => handlePrompt("overview")}>
                Overview
              </button>
              <button className="chip" onClick={() => handlePrompt("worse")}>
                What got worse?
              </button>
              <button className="chip" onClick={() => handlePrompt("improved")}>
                What improved?
              </button>
              <button className="chip" onClick={() => handlePrompt("worst_case")}>
                Explain worst-case
              </button>
              <button className="chip" onClick={() => handlePrompt("new_flags")}>
                Explain new flags
              </button>
            </div>

            {/* Chat Feed */}
            <div ref={analystPanelRef} className="analystPanel">
              {analystMessages.length === 0 ? (
                <p style={{ color: "#666", fontStyle: "italic" }}>Click a prompt above to start...</p>
              ) : (
                analystMessages.map((msg) => (
                  <div key={msg.id} className={`analystMessage ${msg.role}`}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                      <div>
                        {msg.title && <strong>{msg.title}</strong>}
                        {msg.title && <br />}
                        <span>{msg.body}</span>
                      </div>
                      {msg.role === "assistant" && (
                        <button
                          onClick={() => handlePin(msg.id)}
                          style={{
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.75rem",
                            backgroundColor: pinnedMessages.has(msg.id) ? "#ffc107" : "#f0f0f0",
                            border: "1px solid #ccc",
                            borderRadius: "4px",
                            cursor: "pointer",
                          }}
                        >
                          {pinnedMessages.has(msg.id) ? "Unpin" : "Pin"}
                        </button>
                      )}
                    </div>
                    {msg.evidence && (
                      <div style={{ fontSize: "0.875rem", color: "#666", marginTop: "0.5rem", fontStyle: "italic" }}>
                        {msg.evidence}
                      </div>
                    )}
                    {msg.actions && msg.actions.length > 0 && (
                      <div className="analystActions">
                        {msg.actions.map((action, idx) => (
                          <button
                            key={idx}
                            className="analystAction"
                            onClick={() => {
                              if (action.type === "scroll_to") {
                                handleScrollTo(action.payload.type, action.payload.key);
                              } else if (action.type === "focus") {
                                handleFocus(action.payload.type, action.payload.key);
                              }
                            }}
                          >
                            {action.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

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
                    {scenarioDiffs.map((diff) => {
                      const isFocused = focus?.type === "scenario" && focus.key === diff.scenario_id;
                      return (
                        <tr
                          key={diff.scenario_id}
                          id={`scenario-${diff.scenario_id}`}
                          className={isFocused ? "focusGlow clickableRow" : "clickableRow"}
                          onClick={() => handleFocus("scenario", diff.scenario_id)}
                          style={{ borderBottom: "1px solid #e0e0e0" }}
                        >
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
                      );
                    })}
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
                    {componentDiffs.map((diff) => {
                      const isFocused = focus?.type === "component" && focus.key === diff.name;
                      return (
                        <tr
                          key={diff.name}
                          id={`component-${diff.name}`}
                          className={isFocused ? "focusGlow clickableRow" : "clickableRow"}
                          onClick={() => handleFocus("component", diff.name)}
                          style={{ borderBottom: "1px solid #e0e0e0" }}
                        >
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
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Flags Diff */}
          {flagDiff && (flagDiff.added.length > 0 || flagDiff.removed.length > 0 || flagDiff.persisting.length > 0) && (
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
                {flagDiff.added.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#856404" }}>
                      Added ({flagDiff.added.length})
                    </h3>
                    {Object.entries(flagDiff.bySeverity).map(([severity, flags]) => {
                      const addedFlags = flags.added;
                      if (addedFlags.length === 0) return null;
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
                          {addedFlags.map((flag, idx) => {
                            const isFocused = focus?.type === "flag" && focus.key === flag.code;
                            return (
                              <div
                                key={idx}
                                id={`flag-${flag.code}`}
                                className={isFocused ? "focusGlow clickableRow" : "clickableRow"}
                                onClick={() => handleFocus("flag", flag.code)}
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
                            );
                          })}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Removed Flags */}
                {flagDiff.removed.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#28a745" }}>
                      Removed ({flagDiff.removed.length})
                    </h3>
                    {Object.entries(flagDiff.bySeverity).map(([severity, flags]) => {
                      const removedFlags = flags.removed;
                      if (removedFlags.length === 0) return null;
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
                          {removedFlags.map((flag, idx) => {
                            const isFocused = focus?.type === "flag" && focus.key === flag.code;
                            return (
                              <div
                                key={idx}
                                id={`flag-${flag.code}`}
                                className={isFocused ? "focusGlow clickableRow" : "clickableRow"}
                                onClick={() => handleFocus("flag", flag.code)}
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
                            );
                          })}
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Persisting Flags */}
                {flagDiff.persisting.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem", color: "#666" }}>
                      Persisting ({flagDiff.persisting.length})
                    </h3>
                    {Object.entries(flagDiff.bySeverity).map(([severity, flags]) => {
                      const persistingFlags = flags.persisting;
                      if (persistingFlags.length === 0) return null;
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
                          {persistingFlags.map((flag, idx) => {
                            const isFocused = focus?.type === "flag" && focus.key === flag.code;
                            return (
                              <div
                                key={idx}
                                id={`flag-${flag.code}`}
                                className={isFocused ? "focusGlow clickableRow" : "clickableRow"}
                                onClick={() => handleFocus("flag", flag.code)}
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
                            );
                          })}
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
