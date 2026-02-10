"use client";

import { useState, useEffect, useRef } from "react";
import {
  listExperiments,
  listRuns,
  getRunResult,
  type ExperimentSummary,
  type RunSummary,
  type ResultSummary,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const PINNED_MESSAGE: Message = {
  role: "assistant",
  content: "I analyze experiment runs by stress-testing metrics against the same model outputs. I don't train models — I reveal where metrics disagree, drift, or fail under stress.",
};

export default function AssistantPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>("");
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [loadedResult, setLoadedResult] = useState<ResultSummary | null>(null);
  const [contextLoaded, setContextLoaded] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [enhancedAnalyst, setEnhancedAnalyst] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingExperiments, setLoadingExperiments] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingContext, setLoadingContext] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const hasInitializedRef = useRef(false);

  // Add pinned initial assistant message on mount
  useEffect(() => {
    if (!hasInitializedRef.current && messages.length === 0) {
      setMessages([PINNED_MESSAGE]);
      hasInitializedRef.current = true;
    }
  }, [messages.length]);

  // Load experiments on mount
  useEffect(() => {
    async function loadExperiments() {
      try {
        setLoadingExperiments(true);
        const data = await listExperiments();
        setExperiments(data);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load experiments");
      } finally {
        setLoadingExperiments(false);
      }
    }
    loadExperiments();
  }, []);

  // Load runs when experiment changes
  useEffect(() => {
    if (!selectedExperimentId) {
      setRuns([]);
      setSelectedRunId("");
      setLoadedResult(null);
      setContextLoaded(false);
      return;
    }

    async function loadRuns() {
      try {
        setLoadingRuns(true);
        const data = await listRuns(selectedExperimentId);
        setRuns(data);
        setSelectedRunId("");
        setLoadedResult(null);
        setContextLoaded(false);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load runs");
        setRuns([]);
      } finally {
        setLoadingRuns(false);
      }
    }

    loadRuns();
  }, [selectedExperimentId]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleLoadContext() {
    if (!selectedExperimentId || !selectedRunId) {
      setError("Please select both experiment and run");
      return;
    }

    try {
      setLoadingContext(true);
      setError(null);
      const result = await getRunResult(selectedExperimentId, selectedRunId);
      setLoadedResult(result);
      setContextLoaded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load context");
      setLoadedResult(null);
      setContextLoaded(false);
    } finally {
      setLoadingContext(false);
    }
  }

  function routeQuestion(question: string, result: ResultSummary): string {
    const q = question.toLowerCase();

    // Help
    if (/\b(help|what can you|how do i|what do you)\b/.test(q)) {
      return "I can answer questions about this run. Try asking about:\n\u2022 Headline score\n\u2022 Scenarios and worst-case stress results\n\u2022 Flags and risks\n\u2022 Component scores\n\u2022 Applicable metrics\n\u2022 Threshold sweep\n\u2022 Sensitivity analysis\n\u2022 Failure modes\n\u2022 Metric disagreements\n\u2022 Calibration (Brier / ECE)\n\u2022 Prediction surface";
    }

    // Headline / score
    if (/\b(score|headline|overall|total|summary)\b/.test(q) && !/\b(scenario|component|threshold|sensitiv)\b/.test(q)) {
      const parts = [`Headline score: ${result.headline_score.toFixed(4)}`];
      if (result.weighted_score != null) {
        parts.push(`Weighted score: ${result.weighted_score.toFixed(4)}`);
      }
      if (result.component_scores.length > 0) {
        parts.push(`${result.component_scores.length} component score(s) computed`);
      }
      parts.push(`${result.flags.length} flag(s) found`);
      return parts.join(". ") + ".";
    }

    // Scenarios / worst-case
    if (/\b(scenario|worst|stress|perturb)\b/.test(q)) {
      if (result.scenario_results.length === 0) return "No scenarios ran in this experiment.";
      const sorted = [...result.scenario_results].sort((a, b) => a.delta - b.delta);
      const worst = sorted[0];
      const lines = sorted.map(
        (s) => `\u2022 ${s.scenario_name}: score=${s.score.toFixed(4)}, \u0394=${s.delta.toFixed(4)}${s.severity ? ` [${s.severity}]` : ""}`
      );
      return `Worst scenario: ${worst.scenario_name} (\u0394=${worst.delta.toFixed(4)}).\n\n${lines.join("\n")}`;
    }

    // Flags / risks
    if (/\b(flag|risk|finding|issue|warning|critical)\b/.test(q)) {
      if (result.flags.length === 0) return "No flags or findings for this run.";
      const sorted = [...result.flags].sort((a, b) => {
        const order: Record<string, number> = { critical: 0, warn: 1, info: 2 };
        return (order[a.severity] ?? 3) - (order[b.severity] ?? 3);
      });
      const lines = sorted.map((f) => `\u2022 [${f.severity.toUpperCase()}] ${f.title}: ${f.detail}`);
      return `${result.flags.length} flag(s) found:\n\n${lines.join("\n")}`;
    }

    // Component scores
    if (/\b(component|decision|weight)\b/.test(q)) {
      if (result.component_scores.length === 0) return "No component scores available.";
      const lines = result.component_scores.map(
        (c) => `\u2022 ${c.name}: ${c.score.toFixed(4)}${c.weight != null ? ` (weight: ${c.weight})` : ""}`
      );
      return `Component scores:\n\n${lines.join("\n")}`;
    }

    // Applicable metrics
    if (/\b(metric|applicable|available)\b/.test(q) && !/\b(disagree|gaming|inflat)\b/.test(q)) {
      if (!result.applicable_metrics || result.applicable_metrics.length === 0) {
        return "No applicable metrics information available.";
      }
      return `Applicable metrics: ${result.applicable_metrics.join(", ")}.`;
    }

    // Threshold sweep
    if (/\b(threshold|sweep|optimal)\b/.test(q)) {
      const sweep = result.analysis_artifacts?.threshold_sweep as Record<string, unknown> | undefined;
      if (!sweep) return "No threshold sweep data available for this run.";
      const optimal = sweep.optimal_thresholds as Record<string, number> | undefined;
      if (optimal && typeof optimal === "object") {
        const lines = Object.entries(optimal).map(
          ([metric, thresh]) => `\u2022 ${metric}: optimal threshold = ${thresh.toFixed(4)}`
        );
        const crossovers = sweep.crossover_points as Array<Record<string, unknown>> | undefined;
        if (crossovers && crossovers.length > 0) {
          lines.push(`\n${crossovers.length} crossover point(s) detected between metrics.`);
        }
        return `Threshold sweep results:\n\n${lines.join("\n")}`;
      }
      return "Threshold sweep ran but no optimal thresholds found.";
    }

    // Sensitivity
    if (/\b(sensitiv|robust|fragil)\b/.test(q)) {
      const sens = result.analysis_artifacts?.sensitivity as Record<string, unknown> | undefined;
      if (!sens) return "No sensitivity analysis data available for this run.";
      const parts: string[] = [];
      if (sens.most_sensitive_metric) parts.push(`Most sensitive metric: ${sens.most_sensitive_metric}`);
      if (sens.least_sensitive_metric) parts.push(`Least sensitive metric: ${sens.least_sensitive_metric}`);
      const slopes = sens.metric_slopes as Record<string, number> | undefined;
      if (slopes) {
        const lines = Object.entries(slopes).map(
          ([m, s]) => `\u2022 ${m}: slope = ${s.toFixed(6)}`
        );
        parts.push(`\nMetric slopes (higher magnitude = more fragile):\n${lines.join("\n")}`);
      }
      return parts.length > 0 ? `Sensitivity analysis:\n\n${parts.join("\n")}` : "Sensitivity analysis ran but data is unavailable.";
    }

    // Failure modes
    if (/\b(fail|failure|mode)\b/.test(q)) {
      const fm = result.analysis_artifacts?.failure_modes as Record<string, unknown> | undefined;
      if (!fm) return "No failure mode data available for this run.";
      const parts: string[] = [`Total samples: ${fm.total_samples}`];
      if (fm.worst_subgroup) parts.push(`Worst subgroup: ${fm.worst_subgroup}`);
      const summary = fm.summary as Record<string, unknown> | undefined;
      if (summary) {
        if (summary.mean_contribution != null) parts.push(`Mean contribution: ${(summary.mean_contribution as number).toFixed(4)}`);
        if (summary.max_contribution != null) parts.push(`Max contribution: ${(summary.max_contribution as number).toFixed(4)}`);
      }
      const samples = fm.failure_samples as number[] | undefined;
      if (samples && samples.length > 0) {
        parts.push(`Top failure sample indices: ${samples.slice(0, 5).join(", ")}`);
      }
      return `Failure mode report:\n\n${parts.join("\n")}`;
    }

    // Metric disagreements
    if (/\b(disagree|conflict|contradict)\b/.test(q)) {
      const dis = result.analysis_artifacts?.metric_disagreements as Array<Record<string, unknown>> | undefined;
      if (!dis || dis.length === 0) return "No metric disagreement data available for this run.";
      const lines = dis.map(
        (d) => `\u2022 ${d.metric_a} vs ${d.metric_b}: disagreement rate = ${((d.disagreement_rate as number) * 100).toFixed(1)}%`
      );
      return `Metric disagreements:\n\n${lines.join("\n")}`;
    }

    // Prediction surface
    if (/\b(surface|prediction|inference)\b/.test(q)) {
      if (!result.prediction_surface) return "No prediction surface data available for this run.";
      const ps = result.prediction_surface;
      return `Prediction surface: type=${ps.surface_type || "unknown"}, samples=${ps.n_samples || "unknown"}.`;
    }

    // Calibration
    if (/\b(calibrat|brier|ece)\b/.test(q)) {
      const calComponents = result.component_scores.filter(
        (c) => c.name.includes("brier") || c.name.includes("ece") || c.name.includes("calibrat")
      );
      if (calComponents.length > 0) {
        const lines = calComponents.map((c) => `\u2022 ${c.name}: ${c.score.toFixed(4)}`);
        return `Calibration components:\n\n${lines.join("\n")}`;
      }
      return "No calibration data available in component scores.";
    }

    // Fallback: general overview
    const worstScenario = result.scenario_results.length > 0
      ? result.scenario_results.reduce((min, s) => (s.delta < min.delta ? s : min))
      : null;
    const topFlags = [...result.flags]
      .sort((a, b) => {
        const order: Record<string, number> = { critical: 0, warn: 1, info: 2 };
        return (order[a.severity] ?? 3) - (order[b.severity] ?? 3);
      })
      .slice(0, 2)
      .map((f) => f.code);

    const parts: string[] = [];
    parts.push(`Headline score: ${result.headline_score.toFixed(4)}`);
    if (worstScenario) {
      parts.push(`Worst scenario: ${worstScenario.scenario_name} (\u0394=${worstScenario.delta.toFixed(4)})`);
    }
    parts.push(`${result.flags.length} flag(s) found`);
    if (topFlags.length > 0) {
      parts.push(`Top flags: ${topFlags.join(", ")}`);
    }
    if (result.prediction_surface) {
      parts.push(`Prediction surface: ${String(result.prediction_surface.surface_type)}`);
    }
    if (result.applicable_metrics && result.applicable_metrics.length > 0) {
      parts.push(`Applicable metrics: ${result.applicable_metrics.join(", ")}`);
    }
    return parts.join(". ") + ".";
  }

  function handleSend() {
    if (!inputValue.trim()) return;

    const userMessage: Message = { role: "user", content: inputValue.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    let assistantContent: string;

    if (!contextLoaded || !loadedResult) {
      if (!selectedExperimentId || !selectedRunId) {
        assistantContent = "Select an experiment run and click \"Load Context\" so I can answer with evidence from metrics, scenarios, and flags.";
      } else {
        assistantContent = "I have context metadata but no result loaded yet. Click 'Load Context'.";
      }
    } else {
      assistantContent = routeQuestion(inputValue.trim(), loadedResult);
    }

    const assistantMessage: Message = { role: "assistant", content: assistantContent };
    setTimeout(() => {
      setMessages((prev) => [...prev, assistantMessage]);
    }, 100);
  }

  function handleReset() {
    setError(null);
    // Reset to pinned message only (no duplication)
    setMessages([PINNED_MESSAGE]);
  }

  const selectedExperiment = experiments.find((e) => e.id === selectedExperimentId);
  const selectedRun = runs.find((r) => r.run_id === selectedRunId);

  return (
    <div className="flex h-[calc(100vh-12rem)] gap-4">
      {/* Left Sidebar */}
      <div className="w-full max-w-[320px] flex-shrink-0 border-r border-border p-4 space-y-4 overflow-y-auto">
        <h2 className="text-lg font-semibold">Context</h2>

        {/* Experiment Selector */}
        <div className="space-y-2">
          <Label htmlFor="experiment">Experiment</Label>
          {loadingExperiments ? (
            <div className="text-sm text-muted-foreground">Loading...</div>
          ) : experiments.length === 0 ? (
            <div className="text-sm text-muted-foreground">No experiments found</div>
          ) : (
            <Select
              value={selectedExperimentId}
              onValueChange={setSelectedExperimentId}
            >
              <SelectTrigger id="experiment">
                <SelectValue placeholder="Select experiment" />
              </SelectTrigger>
              <SelectContent>
                {experiments.map((exp) => (
                  <SelectItem key={exp.id} value={exp.id}>
                    {exp.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {/* Run Selector */}
        <div className="space-y-2">
          <Label htmlFor="run">Run</Label>
          {loadingRuns ? (
            <div className="text-sm text-muted-foreground">Loading...</div>
          ) : !selectedExperimentId ? (
            <div className="text-sm text-muted-foreground">Select experiment first</div>
          ) : runs.length === 0 ? (
            <div className="text-sm text-muted-foreground">No runs found</div>
          ) : (
            <Select
              value={selectedRunId}
              onValueChange={setSelectedRunId}
              disabled={!selectedExperimentId}
            >
              <SelectTrigger id="run">
                <SelectValue placeholder="Select run" />
              </SelectTrigger>
              <SelectContent>
                {runs.map((run) => (
                  <SelectItem key={run.run_id} value={run.run_id}>
                    {run.run_id}
                    {run.generated_at && ` (${new Date(run.generated_at).toLocaleDateString()})`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {/* Load Context Button */}
        <Button
          onClick={handleLoadContext}
          disabled={!selectedExperimentId || !selectedRunId || loadingContext}
          className="w-full"
        >
          {loadingContext ? "Loading..." : "Load Context"}
        </Button>

        {/* Context Status */}
        {contextLoaded && loadedResult && selectedExperiment && selectedRun && (
          <div className="text-sm text-muted-foreground p-2 bg-muted rounded-md">
            Context loaded: {selectedExperiment.name} • {selectedRun.run_id}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="text-sm text-destructive p-2 bg-destructive/10 border border-destructive/20 rounded-md">
            Error: {error}
          </div>
        )}
      </div>

      {/* Right Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="p-4 border-b border-border">
          <h1 className="text-2xl font-bold mb-1">Assistant</h1>
          <p className="text-sm text-muted-foreground mb-2">
            Ask questions about a specific experiment run and its metric stress results.
          </p>
          <p className="text-xs text-muted-foreground">
            Spectra evaluates metrics against the same model outputs under stress scenarios — it doesn&apos;t train models.
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4 border-b border-border">
          {messages.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              Start a conversation by sending a message.
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 space-y-2">
          {/* Enhanced Analyst Toggle */}
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="enhanced-analyst"
                checked={enhancedAnalyst}
                onChange={(e) => setEnhancedAnalyst(e.target.checked)}
                className="h-4 w-4 rounded border-input"
              />
              <Label htmlFor="enhanced-analyst" className="text-sm font-normal cursor-pointer">
                Enhanced Analyst (LLM · coming soon)
              </Label>
            </div>
            <p className="text-xs text-muted-foreground ml-6">
              Uses an external LLM if configured. Falls back to deterministic analysis.
            </p>
          </div>

          {/* Input and Buttons */}
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              className="flex-1"
            />
            <Button onClick={handleSend} disabled={!inputValue.trim()}>
              Send
            </Button>
            <Button onClick={handleReset} variant="outline">
              Reset
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
