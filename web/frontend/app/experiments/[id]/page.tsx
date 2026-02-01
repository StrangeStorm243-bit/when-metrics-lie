import { notFound } from "next/navigation";
import { getResults } from "@/lib/api";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function ExperimentPage({ params }: PageProps) {
  const { id } = await params;
  let result;
  let error: string | null = null;

  try {
    result = await getResults(id);
  } catch (e) {
    if (e instanceof Error && e.message.includes("404")) {
      notFound();
    }
    error = e instanceof Error ? e.message : "Failed to load results";
  }

  if (error) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <div style={{ padding: "1rem", backgroundColor: "#fee", color: "#c00", borderRadius: "4px", marginTop: "1rem" }}>
          Error: {error}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div>
        <h1>Experiment Results</h1>
        <div style={{ padding: "2rem", textAlign: "center", color: "#666" }}>
          No results available for this experiment yet.
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>Experiment Results</h1>
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

