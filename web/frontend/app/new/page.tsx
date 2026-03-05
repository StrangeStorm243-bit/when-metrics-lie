"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import {
  createExperiment,
  runExperiment,
  uploadModel,
  getMetricPresets,
  getStressSuitePresets,
  getDatasetPresets,
  ApiError,
  type MetricPreset,
  type StressSuitePreset,
  type DatasetPreset,
  type ModelUploadResponse,
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

const TASK_TYPES = [
  { id: "binary_classification", name: "Binary Classification" },
  { id: "multiclass_classification", name: "Multiclass Classification" },
  { id: "regression", name: "Regression" },
  { id: "ranking", name: "Ranking" },
];

export default function NewExperimentPage() {
  const router = useRouter();
  const [taskType, setTaskType] = useState("binary_classification");
  const [name, setName] = useState("");
  const [metricId, setMetricId] = useState("");
  const [stressSuiteId, setStressSuiteId] = useState("");
  const [notes, setNotes] = useState("");
  const [metrics, setMetrics] = useState<MetricPreset[]>([]);
  const [stressSuites, setStressSuites] = useState<StressSuitePreset[]>([]);
  const [datasets, setDatasets] = useState<DatasetPreset[]>([]);
  const [datasetPath, setDatasetPath] = useState<string>("");
  const [modelId, setModelId] = useState<string>("");
  const [modelMeta, setModelMeta] = useState<ModelUploadResponse | null>(null);
  const [modelUploadError, setModelUploadError] = useState<string | null>(null);
  const [modelUploading, setModelUploading] = useState(false);
  const [featureCols, setFeatureCols] = useState("y_score");
  const [yTrueCol, setYTrueCol] = useState("y_true");
  const [yScoreCol, setYScoreCol] = useState("y_score");
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingPresets, setLoadingPresets] = useState(true);

  useEffect(() => {
    async function loadPresets() {
      try {
        const [metricsData, suitesData, datasetsData] = await Promise.all([
          getMetricPresets(taskType),
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
  }, [taskType]);

  async function handleModelUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setModelUploadError(null);
    setModelMeta(null);
    setModelId("");
    setModelUploading(true);
    try {
      const res = await uploadModel(file);
      setModelId(res.model_id);
      setModelMeta(res);
    } catch (err) {
      setModelUploadError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setModelUploading(false);
      e.target.value = "";
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const config: Record<string, unknown> = {
      task_type: taskType,
      y_true_col: yTrueCol,
      y_score_col: yScoreCol,
    };
    if (datasetPath) config.dataset_path = datasetPath;
    if (modelId) {
      config.model_id = modelId;
      config.feature_cols = featureCols
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      if (config.feature_cols instanceof Array && (config.feature_cols as string[]).length === 0) {
        (config.feature_cols as string[]) = ["y_score"];
      }
    }
    if (taskType !== "regression" && taskType !== "ranking") {
      config.threshold = threshold;
    }

    try {
      // Create experiment
      const experiment = await createExperiment({
        name,
        metric_id: metricId,
        stress_suite_id: stressSuiteId,
        notes: notes || undefined,
        config: Object.keys(config).length > 0 ? config : undefined,
      });

      // Run experiment
      await runExperiment(experiment.id);

      // Redirect to results page
      router.push(`/experiments/${experiment.id}`);
    } catch (e) {
      if (e instanceof ApiError) {
        // Backend error - show the detail message directly
        setError(e.message);
      } else if (e instanceof Error) {
        setError(e.message);
      } else {
        setError("Failed to create or run experiment");
      }
      setLoading(false);
    }
  }

  if (loadingPresets) {
    return (
      <div className="container max-w-2xl py-8">
        <h1 className="text-3xl font-bold mb-2">New Experiment</h1>
        <p className="text-sm text-muted-foreground mb-4">
          Spectra evaluates metrics against the same model outputs under stress scenarios — it doesn&apos;t train models.
        </p>
        <p className="text-muted-foreground">Loading presets...</p>
      </div>
    );
  }

  return (
    <div className="container max-w-2xl py-8">
      <h1 className="text-3xl font-bold mb-2">New Experiment</h1>
      <p className="text-sm text-muted-foreground mb-6">
        Spectra evaluates metrics against the same model outputs under stress scenarios — it doesn&apos;t train models.
      </p>
      {error && (
        <div className="mb-6 p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md">
          {error}
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
              <Label htmlFor="task-type">Task Type</Label>
              <Select value={taskType} onValueChange={(v) => { setTaskType(v); setMetricId(""); }}>
                <SelectTrigger id="task-type">
                  <SelectValue placeholder="Select task type" />
                </SelectTrigger>
                <SelectContent>
                  {TASK_TYPES.map((t) => (
                    <SelectItem key={t.id} value={t.id}>{t.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

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
                <>
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
                  <p className="text-xs text-muted-foreground">
                    Tip: use two different datasets to produce different runs for Compare.
                  </p>
                </>
              )}
            </div>

            <div className="space-y-2">
              <Label>Model (optional)</Label>
              <p className="text-xs text-muted-foreground">
                Supported: sklearn (.pkl, .joblib), ONNX (.onnx), XGBoost (.ubj, .xgb), LightGBM (.lgb), CatBoost (.cbm)
              </p>
              <div className="flex gap-2 items-center">
                <Input
                  type="file"
                  accept=".pkl,.joblib,.onnx,.ubj,.xgb,.lgb,.cbm"
                  onChange={handleModelUpload}
                  disabled={loading || modelUploading}
                  className="max-w-xs"
                />
              </div>
              {modelUploading && <p className="text-sm text-muted-foreground">Uploading...</p>}
              {modelUploadError && (
                <div className="p-3 bg-destructive/10 text-destructive border border-destructive/20 rounded-md text-sm">
                  {modelUploadError}
                </div>
              )}
              {modelMeta && (
                <div className="p-3 bg-muted rounded-md text-sm">
                  <span className="font-medium">Uploaded:</span> {modelMeta.original_filename} — {modelMeta.model_class}
                  {modelMeta.capabilities?.predict_proba && " (predict_proba ✓)"}
                </div>
              )}
              {modelId && (
                <div className="space-y-1">
                  <Label htmlFor="featureCols">Feature columns (comma-separated)</Label>
                  <Input
                    id="featureCols"
                    type="text"
                    value={featureCols}
                    onChange={(e) => setFeatureCols(e.target.value)}
                    disabled={loading}
                    placeholder="y_score"
                  />
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="y-true-col">Ground Truth Column</Label>
                <Input id="y-true-col" value={yTrueCol} onChange={(e) => setYTrueCol(e.target.value)}
                  placeholder="y_true" />
              </div>
              <div>
                <Label htmlFor="y-score-col">
                  {taskType === "regression" ? "Prediction Column" : "Score/Probability Column"}
                </Label>
                <Input id="y-score-col" value={yScoreCol} onChange={(e) => setYScoreCol(e.target.value)}
                  placeholder={taskType === "regression" ? "y_pred" : "y_score"} />
              </div>
            </div>

            {(taskType === "binary_classification" || taskType === "multiclass_classification") && (
              <div className="space-y-2">
                <Label htmlFor="threshold">Decision Threshold</Label>
                <Input id="threshold" type="number" step="0.01" min="0" max="1"
                  value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value) || 0.5)} />
              </div>
            )}

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
