"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import {
  createExperiment,
  runExperiment,
  getMetricPresets,
  getStressSuitePresets,
  getDatasetPresets,
  type MetricPreset,
  type StressSuitePreset,
  type DatasetPreset,
} from "@/lib/api";

export default function NewExperimentPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [metricId, setMetricId] = useState("");
  const [stressSuiteId, setStressSuiteId] = useState("");
  const [notes, setNotes] = useState("");
  const [metrics, setMetrics] = useState<MetricPreset[]>([]);
  const [stressSuites, setStressSuites] = useState<StressSuitePreset[]>([]);
  const [datasets, setDatasets] = useState<DatasetPreset[]>([]);
  const [datasetPath, setDatasetPath] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingPresets, setLoadingPresets] = useState(true);

  useEffect(() => {
    async function loadPresets() {
      try {
        const [metricsData, suitesData, datasetsData] = await Promise.all([
          getMetricPresets(),
          getStressSuitePresets(),
          getDatasetPresets(),
        ]);
        setMetrics(metricsData);
        setStressSuites(suitesData);
        setDatasets(datasetsData);
        if (metricsData.length > 0) {
          setMetricId(metricsData[0].id);
        }
        if (suitesData.length > 0) {
          setStressSuiteId(suitesData[0].id);
        }
        if (datasetsData.length > 0) {
          setDatasetPath(datasetsData[0].path);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load presets");
      } finally {
        setLoadingPresets(false);
      }
    }
    loadPresets();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      // Create experiment
      const experiment = await createExperiment({
        name,
        metric_id: metricId,
        stress_suite_id: stressSuiteId,
        notes: notes || undefined,
        config: datasetPath ? { dataset_path: datasetPath } : undefined,
      });

      // Run experiment
      await runExperiment(experiment.id);

      // Redirect to results page
      router.push(`/experiments/${experiment.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create or run experiment");
      setLoading(false);
    }
  }

  if (loadingPresets) {
    return (
      <div>
        <h1>New Experiment</h1>
        <p>Loading presets...</p>
      </div>
    );
  }

  return (
    <div>
      <h1>New Experiment</h1>
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
      <form onSubmit={handleSubmit} style={{ maxWidth: "600px", marginTop: "2rem" }}>
        <div style={{ marginBottom: "1.5rem" }}>
          <label
            htmlFor="name"
            style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
          >
            Experiment Name *
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "1rem",
            }}
          />
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label
            htmlFor="metric"
            style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
          >
            Metric *
          </label>
          <select
            id="metric"
            value={metricId}
            onChange={(e) => setMetricId(e.target.value)}
            required
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "1rem",
            }}
          >
            {metrics.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} - {m.description}
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label
            htmlFor="stressSuite"
            style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
          >
            Stress Suite *
          </label>
          <select
            id="stressSuite"
            value={stressSuiteId}
            onChange={(e) => setStressSuiteId(e.target.value)}
            required
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "1rem",
            }}
          >
            {stressSuites.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} - {s.description}
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label
            htmlFor="dataset"
            style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
          >
            Dataset {datasets.length === 0 ? "(none found)" : ""}
          </label>
          {datasets.length === 0 ? (
            <div
              style={{
                padding: "0.75rem",
                backgroundColor: "#fff3cd",
                border: "1px solid #ffc107",
                borderRadius: "4px",
                fontSize: "0.875rem",
                color: "#856404",
              }}
            >
              No datasets found. Place CSV files under the repo's data/ directory (e.g., data/demo_binary_label_noise.csv).
            </div>
          ) : (
            <select
              id="dataset"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              disabled={loading}
              style={{
                width: "100%",
                padding: "0.5rem",
                border: "1px solid #ccc",
                borderRadius: "4px",
                fontSize: "1rem",
              }}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.path}>
                  {d.name}
                </option>
              ))}
            </select>
          )}
        </div>

        <div style={{ marginBottom: "1.5rem" }}>
          <label
            htmlFor="notes"
            style={{ display: "block", marginBottom: "0.5rem", fontWeight: "bold" }}
          >
            Notes (optional)
          </label>
          <textarea
            id="notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={loading}
            rows={4}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              fontSize: "1rem",
              fontFamily: "inherit",
            }}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !name || !metricId || !stressSuiteId}
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
          {loading ? "Creating & Running..." : "Create & Run"}
        </button>
      </form>
    </div>
  );
}
