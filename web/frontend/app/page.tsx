import Link from "next/link";
import { listExperiments } from "@/lib/api";
import type { ExperimentSummary } from "@/lib/api";

export default async function HomePage() {
  let experiments: ExperimentSummary[] = [];
  let error: string | null = null;

  try {
    experiments = await listExperiments();
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load experiments";
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "2rem",
        }}
      >
        <h1>Experiments</h1>
        <Link
          href="/new"
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: "#0070f3",
            color: "white",
            borderRadius: "4px",
            textDecoration: "none",
          }}
        >
          New Experiment
        </Link>
      </div>

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

      {experiments.length === 0 ? (
        <div
          style={{
            padding: "2rem",
            textAlign: "center",
            color: "#666",
          }}
        >
          No experiments yet.{" "}
          <Link href="/new" style={{ color: "#0070f3" }}>
            Create one
          </Link>{" "}
          to get started.
        </div>
      ) : (
        <div style={{ display: "grid", gap: "1rem" }}>
          {experiments.map((exp) => (
            <Link
              key={exp.id}
              href={`/experiments/${exp.id}`}
              style={{
                display: "block",
                padding: "1rem",
                border: "1px solid #e0e0e0",
                borderRadius: "4px",
                textDecoration: "none",
                color: "inherit",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "start",
                }}
              >
                <div>
                  <h2 style={{ marginBottom: "0.5rem" }}>{exp.name}</h2>
                  <div style={{ fontSize: "0.875rem", color: "#666" }}>
                    Metric: {exp.metric_id} • Suite: {exp.stress_suite_id}
                  </div>
                </div>
                <div style={{ fontSize: "0.875rem" }}>
                  <span
                    style={{
                      padding: "0.25rem 0.5rem",
                      borderRadius: "4px",
                      backgroundColor:
                        exp.status === "completed"
                          ? "#d4edda"
                          : exp.status === "failed"
                          ? "#f8d7da"
                          : exp.status === "running"
                          ? "#fff3cd"
                          : "#e2e3e5",
                      color:
                        exp.status === "completed"
                          ? "#155724"
                          : exp.status === "failed"
                          ? "#721c24"
                          : exp.status === "running"
                          ? "#856404"
                          : "#383d41",
                    }}
                  >
                    {exp.status}
                  </span>
                </div>
              </div>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "#999",
                  marginTop: "0.5rem",
                }}
              >
                Created: {new Date(exp.created_at).toLocaleString()}
                {exp.last_run_at &&
                  ` • Last run: ${new Date(
                    exp.last_run_at
                  ).toLocaleString()}`}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

