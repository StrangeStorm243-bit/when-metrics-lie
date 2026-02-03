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
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
      <div className="container max-w-2xl py-8">
        <h1 className="text-3xl font-bold mb-4">New Experiment</h1>
        <p className="text-muted-foreground">Loading presets...</p>
      </div>
    );
  }

  return (
    <div className="container max-w-2xl py-8">
      <h1 className="text-3xl font-bold mb-6">New Experiment</h1>
      {error && (
        <div className="mb-6 p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md">
          Error: {error}
        </div>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Create New Experiment</CardTitle>
          <CardDescription>Configure your experiment settings and start a new run.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="name">
                Experiment Name <span className="text-destructive">*</span>
              </Label>
              <Input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                disabled={loading}
                placeholder="Enter experiment name"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="metric">
                Metric <span className="text-destructive">*</span>
              </Label>
              <Select value={metricId} onValueChange={setMetricId} disabled={loading} required>
                <SelectTrigger id="metric">
                  <SelectValue placeholder="Select a metric" />
                </SelectTrigger>
                <SelectContent>
                  {metrics.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.name} - {m.description}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="stressSuite">
                Stress Suite <span className="text-destructive">*</span>
              </Label>
              <Select
                value={stressSuiteId}
                onValueChange={setStressSuiteId}
                disabled={loading}
                required
              >
                <SelectTrigger id="stressSuite">
                  <SelectValue placeholder="Select a stress suite" />
                </SelectTrigger>
                <SelectContent>
                  {stressSuites.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.name} - {s.description}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="dataset">
                Dataset {datasets.length === 0 ? <span className="text-muted-foreground">(none found)</span> : ""}
              </Label>
              {datasets.length === 0 ? (
                <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md text-sm text-yellow-800">
                  No datasets found. Place CSV files under the repo&apos;s data/ directory (e.g., data/demo_binary_label_noise.csv).
                </div>
              ) : (
                <Select value={datasetPath} onValueChange={setDatasetPath} disabled={loading}>
                  <SelectTrigger id="dataset">
                    <SelectValue placeholder="Select a dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((d) => (
                      <SelectItem key={d.id} value={d.path}>
                        {d.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes">Notes (optional)</Label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                disabled={loading}
                rows={4}
                placeholder="Add any notes about this experiment..."
              />
            </div>

            <Button
              type="submit"
              disabled={loading || !name || !metricId || !stressSuiteId}
              className="w-full"
            >
              {loading ? "Creating & Running..." : "Create & Run"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
