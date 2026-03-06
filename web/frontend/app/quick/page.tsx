"use client";

import { useRouter } from "next/navigation";
import { useState, useCallback } from "react";
import {
  uploadModel,
  uploadDataset,
  autoDetect,
  createExperiment,
  runExperiment,
  ApiError,
  type ModelUploadResponse,
  type DatasetUploadResponse,
  type AutoDetectResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

type Step = "upload" | "review" | "running";

export default function QuickTestPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("upload");

  // Upload state
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelUploadResponse | null>(null);
  const [datasetMeta, setDatasetMeta] = useState<DatasetUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Detection state
  const [detection, setDetection] = useState<AutoDetectResponse | null>(null);

  // Run state
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const handleModelSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setModelFile(file);
  }, []);

  const handleDatasetSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setDatasetFile(file);
  }, []);

  async function handleUploadAndDetect() {
    if (!datasetFile) {
      setUploadError("Please select a dataset CSV file");
      return;
    }
    setUploading(true);
    setUploadError(null);

    try {
      // Upload files in parallel
      const [dsResult, modelResult] = await Promise.all([
        uploadDataset(datasetFile),
        modelFile ? uploadModel(modelFile) : Promise.resolve(null),
      ]);

      setDatasetMeta(dsResult);
      if (modelResult) setModelMeta(modelResult);

      // Auto-detect configuration
      const detectResult = await autoDetect({
        model_id: modelResult?.model_id ?? undefined,
        dataset_id: dsResult.dataset_id,
      });

      setDetection(detectResult);
      setStep("review");
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleRunStressTest() {
    if (!detection || !datasetMeta) return;
    setRunning(true);
    setRunError(null);

    try {
      const config: Record<string, unknown> = {
        task_type: detection.task_type,
        y_true_col: detection.y_true_col || "y_true",
        y_score_col: detection.y_score_col || "y_score",
        dataset_id: datasetMeta.dataset_id,
      };

      if (modelMeta) {
        config.model_id = modelMeta.model_id;
        config.feature_cols = detection.feature_cols.length > 0
          ? detection.feature_cols
          : [detection.y_score_col || "y_score"];
      }

      const experiment = await createExperiment({
        name: `Quick Test - ${datasetMeta.original_filename}`,
        metric_id: detection.recommended_metric,
        stress_suite_id: detection.recommended_stress_suite,
        config,
      });

      await runExperiment(experiment.id);
      router.push(`/experiments/${experiment.id}`);
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : "Run failed");
      setRunning(false);
    }
  }

  return (
    <div className="container max-w-2xl py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-1">Quick Stress Test</h1>
          <p className="text-sm text-muted-foreground">
            Upload your model and dataset — Spectra handles the rest.
          </p>
        </div>
        <Link href="/new" className="text-sm text-muted-foreground hover:text-foreground underline">
          Advanced Mode
        </Link>
      </div>

      {/* Step 1: Upload */}
      {step === "upload" && (
        <Card>
          <CardHeader>
            <CardTitle>Upload Files</CardTitle>
            <CardDescription>
              Drop your model file and dataset CSV. Spectra will auto-detect everything.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {uploadError && (
              <div className="p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md text-sm">
                {uploadError}
              </div>
            )}

            <div className="space-y-2">
              <Label>Dataset CSV <span className="text-destructive">*</span></Label>
              <Input
                type="file"
                accept=".csv"
                onChange={handleDatasetSelect}
                disabled={uploading}
              />
              {datasetFile && (
                <p className="text-sm text-muted-foreground">Selected: {datasetFile.name}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label>Model File (optional)</Label>
              <p className="text-xs text-muted-foreground">
                Supports: sklearn (.pkl, .joblib), ONNX (.onnx), XGBoost (.ubj, .xgb), LightGBM (.lgb), CatBoost (.cbm)
              </p>
              <Input
                type="file"
                accept=".pkl,.joblib,.onnx,.ubj,.xgb,.lgb,.cbm"
                onChange={handleModelSelect}
                disabled={uploading}
              />
              {modelFile && (
                <p className="text-sm text-muted-foreground">Selected: {modelFile.name}</p>
              )}
            </div>

            <Button
              onClick={handleUploadAndDetect}
              disabled={uploading || !datasetFile}
              className="w-full"
            >
              {uploading ? "Uploading & Analyzing..." : "Upload & Analyze"}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Review detected config */}
      {step === "review" && detection && (
        <Card>
          <CardHeader>
            <CardTitle>Detected Configuration</CardTitle>
            <CardDescription>
              Review what Spectra detected. Click Run to start the stress test.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {runError && (
              <div className="p-4 bg-destructive/10 text-destructive border border-destructive/20 rounded-md text-sm">
                {runError}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-muted-foreground">Task Type</span>
                <p>{detection.task_type.replace(/_/g, " ")}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Confidence</span>
                <p className={detection.confidence === "high" ? "text-green-600" : "text-yellow-600"}>
                  {detection.confidence}
                </p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Ground Truth Column</span>
                <p className="font-mono">{detection.y_true_col || "\u2014"}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Score Column</span>
                <p className="font-mono">{detection.y_score_col || "\u2014"}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Metric</span>
                <p>{detection.recommended_metric}</p>
              </div>
              <div>
                <span className="font-medium text-muted-foreground">Stress Suite</span>
                <p>{detection.recommended_stress_suite}</p>
              </div>
              {detection.model_class && (
                <div>
                  <span className="font-medium text-muted-foreground">Model</span>
                  <p>{detection.model_class}</p>
                </div>
              )}
              <div>
                <span className="font-medium text-muted-foreground">Dataset Rows</span>
                <p>{detection.n_rows.toLocaleString()}</p>
              </div>
            </div>

            {detection.feature_cols.length > 0 && (
              <div className="text-sm">
                <span className="font-medium text-muted-foreground">Features</span>
                <p className="font-mono text-xs mt-1">
                  {detection.feature_cols.join(", ")}
                </p>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                onClick={() => setStep("upload")}
                disabled={running}
              >
                Back
              </Button>
              <Button
                onClick={handleRunStressTest}
                disabled={running}
                className="flex-1"
              >
                {running ? "Running Stress Test..." : "Run Stress Test"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
