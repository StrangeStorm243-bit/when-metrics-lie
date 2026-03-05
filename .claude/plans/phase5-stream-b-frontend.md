# Plan: Phase 5 Stream B — Frontend (Multi-Task UI & Visualizations)

> **Execution model:** This plan was written by Opus for execution by Sonnet.
> Run: `claude --model claude-sonnet-4-6` then say "Execute plan .claude/plans/phase5-stream-b-frontend.md"

## Goal

Update the Next.js frontend to support all task types with a task type selector, dynamic form fields, and task-specific visualizations (confusion matrix, per-class metrics table, regression stats, ranking metrics).

## Context

- Stream A (backend) adds these fields to the API contracts:
  - `task_type` on ExperimentSummary, ResultSummary, ModelUploadResponse
  - `confusion_matrix`, `class_names`, `per_class_metrics` (classification)
  - `residual_stats` (regression), `ranking_metrics` (ranking)
  - `GET /presets/metrics?task_type=X` for filtered presets
  - `GET /models/formats` for supported model formats
- The frontend changes can be made independently — TypeScript types are updated first

## Prerequisites

- [ ] Create branch: `git checkout -b phase5-stream-b-frontend`
- [ ] Read each file before modifying it

## Tasks

### Task B1: Update TypeScript types

**Files:**
- Modify: `web/frontend/lib/api.ts`

**Steps:**

1. Add `task_type` to `ExperimentSummary` interface (after `stress_suite_id`):
```typescript
  task_type: string;
```

2. Add task-specific fields to `ResultSummary` interface (after `dashboard_summary`):
```typescript
  task_type: string;
  confusion_matrix?: number[][] | null;
  class_names?: string[] | null;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1: number; support: number }> | null;
  residual_stats?: {
    mean: number; std: number; min: number; max: number; median: number;
    mae: number; rmse: number;
  } | null;
  ranking_metrics?: Record<string, number> | null;
```

3. Add `task_type` and `n_classes` to `ModelUploadResponse` interface:
```typescript
  task_type: string;
  n_classes: number | null;
```

4. Update `uploadModel` function to accept any supported extension (update the comment on line 390):
```typescript
/**
 * Upload a model file (supports pickle, ONNX, boosting formats).
 */
```

5. Add `getMetricPresets` task_type filter support:
```typescript
export async function getMetricPresets(taskType?: string): Promise<MetricPreset[]> {
  const params = taskType ? `?task_type=${encodeURIComponent(taskType)}` : "";
  return apiFetch<MetricPreset[]>(`/presets/metrics${params}`);
}
```

6. Add model formats API:
```typescript
export interface SupportedFormat {
  format_id: string;
  name: string;
  extensions: string[];
  task_types: string[];
}

export async function getModelFormats(): Promise<SupportedFormat[]> {
  return apiFetch<SupportedFormat[]>("/models/formats");
}
```

**Acceptance:** TypeScript compiles with no errors. `npm run build` or `npx tsc --noEmit` passes.

### Task B2: Update experiment creation form

**Files:**
- Modify: `web/frontend/app/new/page.tsx`

**Steps:**

1. Add task type state and options at the top of the component (with the other state):

```typescript
const TASK_TYPES = [
  { id: "binary_classification", name: "Binary Classification" },
  { id: "multiclass_classification", name: "Multiclass Classification" },
  { id: "regression", name: "Regression" },
  { id: "ranking", name: "Ranking" },
];

// Inside the component:
const [taskType, setTaskType] = useState("binary_classification");
```

2. Add a Task Type selector as the FIRST form field (before Experiment Name):

```tsx
<div>
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
```

3. Update the metrics fetch to filter by task type. Replace the `getMetricPresets()` call in useEffect:

```typescript
getMetricPresets(taskType).then((data) => {
  setMetrics(data);
  if (data.length > 0) setMetricId(data[0].id);
});
```

Add `taskType` to the useEffect dependency array so metrics refresh when task type changes.

4. Update the model upload to accept all supported extensions. Replace the file input:

```tsx
<Input
  id="model-upload"
  type="file"
  accept=".pkl,.joblib,.onnx,.ubj,.xgb,.lgb,.cbm"
  onChange={handleModelUpload}
/>
<p className="text-xs text-gray-500 mt-1">
  Supported: sklearn (.pkl, .joblib), ONNX (.onnx), XGBoost (.ubj, .xgb), LightGBM (.lgb), CatBoost (.cbm)
</p>
```

5. Make the threshold field conditional (only for classification):

```tsx
{(taskType === "binary_classification" || taskType === "multiclass_classification") && (
  <div>
    <Label htmlFor="threshold">Decision Threshold</Label>
    <Input id="threshold" type="number" step="0.01" min="0" max="1"
      value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value) || 0.5)} />
  </div>
)}
```

6. Add `y_true_col` and `y_score_col` fields that adapt based on task type:

```tsx
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
```

Add state for these:
```typescript
const [yTrueCol, setYTrueCol] = useState("y_true");
const [yScoreCol, setYScoreCol] = useState("y_score");
const [threshold, setThreshold] = useState(0.5);
```

7. Update the config object in the submit handler to include task_type and column names:

```typescript
const config: Record<string, unknown> = {
  task_type: taskType,
  y_true_col: yTrueCol,
  y_score_col: yScoreCol,
};
if (datasetPath) config.dataset_path = datasetPath;
if (modelId) {
  config.model_id = modelId;
  config.feature_cols = featureCols.split(",").map((c) => c.trim()).filter(Boolean);
}
if (taskType !== "regression" && taskType !== "ranking") {
  config.threshold = threshold;
}
```

8. Run: `cd web/frontend && npx tsc --noEmit 2>&1 | tail -5`
   Expected: No type errors

**Acceptance:** Form shows task type selector. Metrics dropdown updates when task type changes. Model upload accepts all formats.

### Task B3: Add confusion matrix component

**Files:**
- Create: `web/frontend/components/viz/confusion-matrix.tsx`

**Steps:**

1. Create the file:

```tsx
"use client";

interface ConfusionMatrixProps {
  matrix: number[][];
  classNames?: string[];
}

export function ConfusionMatrix({ matrix, classNames }: ConfusionMatrixProps) {
  if (!matrix || matrix.length === 0) return null;

  const labels = classNames || matrix.map((_, i) => `Class ${i}`);
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2 border border-gray-300 bg-gray-50"></th>
            <th className="p-2 border border-gray-300 bg-gray-50 text-center" colSpan={labels.length}>
              Predicted
            </th>
          </tr>
          <tr>
            <th className="p-2 border border-gray-300 bg-gray-50"></th>
            {labels.map((label) => (
              <th key={label} className="p-2 border border-gray-300 bg-gray-50 text-center min-w-[60px]">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-2 border border-gray-300 bg-gray-50 text-right">
                {i === 0 && <span className="text-xs text-gray-500 block">Actual</span>}
                {labels[i]}
              </th>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isDiagonal = i === j;
                const bg = isDiagonal
                  ? `rgba(34, 197, 94, ${0.1 + intensity * 0.6})`
                  : `rgba(239, 68, 68, ${intensity * 0.5})`;
                return (
                  <td
                    key={j}
                    className="p-2 border border-gray-300 text-center font-mono"
                    style={{ backgroundColor: bg }}
                  >
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Acceptance:** Component renders a color-coded confusion matrix table. Diagonal cells green, off-diagonal cells red.

### Task B4: Add per-class metrics component

**Files:**
- Create: `web/frontend/components/viz/per-class-metrics.tsx`

**Steps:**

1. Create the file:

```tsx
"use client";

interface PerClassMetricsProps {
  metrics: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  classNames?: string[];
}

export function PerClassMetrics({ metrics, classNames }: PerClassMetricsProps) {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const classes = classNames || Object.keys(metrics).sort();

  return (
    <div className="overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Per-Class Metrics</h3>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-50">
            <th className="p-2 border border-gray-300 text-left">Class</th>
            <th className="p-2 border border-gray-300 text-right">Precision</th>
            <th className="p-2 border border-gray-300 text-right">Recall</th>
            <th className="p-2 border border-gray-300 text-right">F1</th>
            <th className="p-2 border border-gray-300 text-right">Support</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((cls) => {
            const m = metrics[cls];
            if (!m) return null;
            return (
              <tr key={cls} className="hover:bg-gray-50">
                <td className="p-2 border border-gray-300 font-medium">{cls}</td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  {m.precision.toFixed(3)}
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  {m.recall.toFixed(3)}
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  <span style={{ color: m.f1 >= 0.7 ? "#16a34a" : m.f1 >= 0.5 ? "#d97706" : "#dc2626" }}>
                    {m.f1.toFixed(3)}
                  </span>
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">{m.support}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

**Acceptance:** Table renders with precision/recall/F1 per class, F1 color-coded.

### Task B5: Add regression stats component

**Files:**
- Create: `web/frontend/components/viz/residual-stats.tsx`

**Steps:**

1. Create the file:

```tsx
"use client";

interface ResidualStatsProps {
  stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
    mae: number;
    rmse: number;
  };
}

export function ResidualStats({ stats }: ResidualStatsProps) {
  if (!stats) return null;

  const items = [
    { label: "MAE", value: stats.mae, description: "Mean Absolute Error" },
    { label: "RMSE", value: stats.rmse, description: "Root Mean Squared Error" },
    { label: "Mean Residual", value: stats.mean, description: "Avg prediction error (bias)" },
    { label: "Std Residual", value: stats.std, description: "Spread of errors" },
    { label: "Median Residual", value: stats.median, description: "Middle prediction error" },
    { label: "Min Residual", value: stats.min, description: "Largest underprediction" },
    { label: "Max Residual", value: stats.max, description: "Largest overprediction" },
  ];

  return (
    <div>
      <h3 className="text-lg font-semibold mb-2">Regression Diagnostics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {items.map((item) => (
          <div key={item.label} className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide">{item.label}</div>
            <div className="text-xl font-mono font-bold mt-1">
              {item.value >= 0 ? "+" : ""}{item.value.toFixed(4)}
            </div>
            <div className="text-xs text-gray-400 mt-1">{item.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Acceptance:** Card grid renders with all 7 residual statistics.

### Task B6: Update experiment detail page with task-specific visualizations

**Files:**
- Modify: `web/frontend/app/experiments/[id]/page.tsx`

**Steps:**

1. Add imports at the top of the file:

```typescript
import { ConfusionMatrix } from "@/components/viz/confusion-matrix";
import { PerClassMetrics } from "@/components/viz/per-class-metrics";
import { ResidualStats } from "@/components/viz/residual-stats";
```

2. Find the section where results are displayed (after flags, before metadata footer). Add task-specific visualization sections:

After the Flags section and before the metadata footer, add:

```tsx
{/* Task-Specific Visualizations */}
{result.task_type && result.task_type !== "binary_classification" && (
  <div className="text-xs text-gray-500 mb-2 uppercase tracking-wide">
    Task: {result.task_type.replace(/_/g, " ")}
  </div>
)}

{/* Confusion Matrix (classification only) */}
{result.confusion_matrix && (
  <div className="bg-white rounded-lg border p-4 mb-4">
    <ConfusionMatrix
      matrix={result.confusion_matrix}
      classNames={result.class_names || undefined}
    />
  </div>
)}

{/* Per-Class Metrics (multiclass only) */}
{result.per_class_metrics && (
  <div className="bg-white rounded-lg border p-4 mb-4">
    <PerClassMetrics
      metrics={result.per_class_metrics}
      classNames={result.class_names || undefined}
    />
  </div>
)}

{/* Regression Stats */}
{result.residual_stats && (
  <div className="bg-white rounded-lg border p-4 mb-4">
    <ResidualStats stats={result.residual_stats} />
  </div>
)}

{/* Ranking Metrics */}
{result.ranking_metrics && (
  <div className="bg-white rounded-lg border p-4 mb-4">
    <h3 className="text-lg font-semibold mb-2">Ranking Metrics</h3>
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {Object.entries(result.ranking_metrics).map(([name, value]) => (
        <div key={name} className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 uppercase">{name.replace(/_/g, " ")}</div>
          <div className="text-xl font-mono font-bold">{(value as number).toFixed(4)}</div>
        </div>
      ))}
    </div>
  </div>
)}
```

3. Update the headline score section to show task-appropriate label:

Find where `headline_score` is displayed. Add task context:

```tsx
<div className="text-xs text-gray-500">
  {result.task_type === "regression" ? "Primary Metric" : "Headline Score"}
</div>
```

4. Run: `cd web/frontend && npx tsc --noEmit 2>&1 | tail -5`
   Expected: No type errors

**Acceptance:** Experiment detail page shows confusion matrix for classification, per-class metrics for multiclass, residual stats for regression. Binary classification display unchanged.

### Task B7: Update compare page for multi-task

**Files:**
- Modify: `web/frontend/app/compare/page.tsx`

**Steps:**

1. In the comparison results display, add task_type indicator if results have different task types or non-binary type:

Find where `result_a.headline_score` and `result_b.headline_score` are displayed. Add:

```tsx
{result_a?.task_type && result_a.task_type !== "binary_classification" && (
  <span className="text-xs bg-gray-100 px-2 py-0.5 rounded ml-2">
    {result_a.task_type.replace(/_/g, " ")}
  </span>
)}
```

Do the same for result_b.

2. The rest of the compare page (scenario diffs, component diffs, flag diffs) is already generic and works across task types. No other changes needed.

**Acceptance:** Compare page shows task type labels for non-binary experiments.

### Task B8: Update insights derivation

**Files:**
- Modify: `web/frontend/lib/insights.ts`

**Steps:**

1. Read the file first. Find the `deriveFindings` function.

2. Add task-type awareness to the findings. After the existing findings logic, add:

```typescript
// Task-specific findings
if (result.task_type === "regression" && result.residual_stats) {
  const stats = result.residual_stats;
  findings.push(`Regression MAE: ${stats.mae.toFixed(4)}, RMSE: ${stats.rmse.toFixed(4)}`);
  if (Math.abs(stats.mean) > stats.std * 0.5) {
    findings.push(`Model shows prediction bias (mean residual: ${stats.mean.toFixed(4)})`);
  }
}

if (result.per_class_metrics) {
  const classes = Object.entries(result.per_class_metrics);
  const worstClass = classes.reduce((worst, [cls, m]) =>
    m.f1 < (worst[1]?.f1 ?? 1) ? [cls, m] : worst, classes[0]);
  if (worstClass) {
    findings.push(`Weakest class: ${worstClass[0]} (F1: ${worstClass[1].f1.toFixed(3)})`);
  }
}
```

**Acceptance:** Findings include regression-specific and multiclass-specific insights.

### Task B9: Commit all frontend changes

**Steps:**

1. Run type check: `cd web/frontend && npx tsc --noEmit`
2. Run lint: `cd web/frontend && npx next lint`
3. Fix any errors.
4. Commit:

```bash
git add web/frontend/
git commit -m "feat: Phase 5 Stream B — multi-task frontend with visualizations

- Add task_type to TypeScript interfaces (ExperimentSummary, ResultSummary, ModelUploadResponse)
- Add task type selector to experiment creation form
- Dynamic metric presets filtered by task type
- Multi-format model upload (.pkl, .onnx, .joblib, .ubj, .xgb, .lgb, .cbm)
- Confusion matrix component (CSS heatmap, green diagonal, red off-diagonal)
- Per-class metrics table (precision, recall, F1 with color coding)
- Regression diagnostics card grid (MAE, RMSE, residual statistics)
- Ranking metrics display
- Task-aware insights derivation
- Compare page task type labels"
```

**Acceptance:** TypeScript compiles, lint passes, commit created.

## Boundaries

**DO:**
- Follow steps exactly as written
- Commit after all tasks pass
- Run the specified verification commands

**DO NOT:**
- Modify backend Python files (that's Stream A)
- Add new npm dependencies (all visualizations use plain HTML/CSS)
- Change the auth or middleware logic
- Modify the assistant page question routing (works generically)

## Escalation Triggers

Stop and flag for Opus review if:
- TypeScript compilation errors related to backend contract mismatches
- Import errors for new components
- The existing experiment detail page layout breaks
- You need to modify files not listed in any task

When escalating, write to `.claude/plans/phase5-stream-b-blockers.md`.

## Verification

After all tasks complete:
- [ ] `cd web/frontend && npx tsc --noEmit` passes
- [ ] `cd web/frontend && npx next lint` passes
- [ ] No files modified outside the plan's file list
- [ ] Single commit with all changes
