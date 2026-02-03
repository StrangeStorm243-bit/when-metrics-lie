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

  function handleSend() {
    if (!inputValue.trim()) return;

    const userMessage: Message = { role: "user", content: inputValue.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    // Generate deterministic response
    let assistantContent: string;

    if (!contextLoaded || !loadedResult) {
      if (!selectedExperimentId || !selectedRunId) {
        assistantContent = "Select an experiment run and click \"Load Context\" so I can answer with evidence from metrics, scenarios, and flags.";
      } else {
        assistantContent = "I have context metadata but no result loaded yet. Click 'Load Context'.";
      }
    } else {
      // Generate evidence-based response
      const worstScenario = loadedResult.scenario_results.length > 0
        ? loadedResult.scenario_results.reduce((min, s) => (s.delta < min.delta ? s : min))
        : null;

      const topFlags = loadedResult.flags
        .sort((a, b) => {
          const severityOrder = { critical: 0, warn: 1, info: 2 };
          return severityOrder[a.severity] - severityOrder[b.severity];
        })
        .slice(0, 2)
        .map((f) => f.code);

      const parts: string[] = [];
      parts.push(`Headline score: ${loadedResult.headline_score.toFixed(4)}`);

      if (worstScenario) {
        parts.push(`Worst scenario: ${worstScenario.scenario_name} (Δ=${worstScenario.delta.toFixed(4)})`);
      }

      parts.push(`${loadedResult.flags.length} flag(s) found`);
      if (topFlags.length > 0) {
        parts.push(`Top flags: ${topFlags.join(", ")}`);
      }

      assistantContent = parts.join(". ") + ".";
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
