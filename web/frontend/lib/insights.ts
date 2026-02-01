/**
 * Deterministic helper functions for deriving insights from experiment results.
 */

import type { ResultSummary } from "./api";

export interface Finding {
  text: string;
}

/**
 * Derive key findings from result summary.
 * Returns 3-6 bullet points based on headline score, scenario deltas, and flags.
 */
export function deriveFindings(result: ResultSummary): Finding[] {
  const findings: Finding[] = [];

  // 1. Headline score assessment
  if (result.headline_score >= 0.9) {
    findings.push({ text: `Strong overall performance with headline score of ${result.headline_score.toFixed(3)}` });
  } else if (result.headline_score >= 0.7) {
    findings.push({ text: `Moderate performance with headline score of ${result.headline_score.toFixed(3)}` });
  } else if (result.headline_score >= 0.5) {
    findings.push({ text: `Below-average performance with headline score of ${result.headline_score.toFixed(3)}` });
  } else {
    findings.push({ text: `Poor performance with headline score of ${result.headline_score.toFixed(3)}` });
  }

  // 2. Worst scenario delta
  if (result.scenario_results.length > 0) {
    const sortedByDelta = [...result.scenario_results].sort((a, b) => a.delta - b.delta);
    const worst = sortedByDelta[0];
    if (worst.delta < -0.1) {
      findings.push({
        text: `Significant degradation in "${worst.scenario_name}" with delta of ${worst.delta.toFixed(3)}`,
      });
    } else if (worst.delta < -0.05) {
      findings.push({
        text: `Moderate degradation in "${worst.scenario_name}" with delta of ${worst.delta.toFixed(3)}`,
      });
    } else if (worst.delta < 0) {
      findings.push({
        text: `Minor degradation in "${worst.scenario_name}" with delta of ${worst.delta.toFixed(3)}`,
      });
    }
  }

  // 3. Scenario spread
  if (result.scenario_results.length > 1) {
    const deltas = result.scenario_results.map((s) => s.delta);
    const minDelta = Math.min(...deltas);
    const maxDelta = Math.max(...deltas);
    const spread = maxDelta - minDelta;
    if (spread > 0.2) {
      findings.push({
        text: `High variability across scenarios (spread: ${spread.toFixed(3)}) indicating inconsistent performance`,
      });
    } else if (spread > 0.1) {
      findings.push({
        text: `Moderate variability across scenarios (spread: ${spread.toFixed(3)})`,
      });
    }
  }

  // 4. Top 2 flags (critical first, then warn)
  const sortedFlags = [...result.flags].sort((a, b) => {
    const severityOrder = { critical: 0, warn: 1, info: 2 };
    return severityOrder[a.severity] - severityOrder[b.severity];
  });
  const topFlags = sortedFlags.slice(0, 2);
  topFlags.forEach((flag) => {
    findings.push({
      text: `[${flag.severity.toUpperCase()}] ${flag.title}: ${flag.detail}`,
    });
  });

  // 5. Component score insights (if available)
  if (result.component_scores.length > 0) {
    const sortedComponents = [...result.component_scores].sort((a, b) => a.score - b.score);
    const worstComponent = sortedComponents[0];
    const bestComponent = sortedComponents[sortedComponents.length - 1];
    if (worstComponent.score < 0.5 && bestComponent.score > 0.8) {
      findings.push({
        text: `Wide range in component performance: "${worstComponent.name}" (${worstComponent.score.toFixed(3)}) vs "${bestComponent.name}" (${bestComponent.score.toFixed(3)})`,
      });
    }
  }

  // 6. Weighted score comparison (if available)
  if (result.weighted_score !== null && result.weighted_score !== result.headline_score) {
    const diff = result.weighted_score - result.headline_score;
    if (Math.abs(diff) > 0.05) {
      findings.push({
        text: `Weighted score (${result.weighted_score.toFixed(3)}) ${diff > 0 ? "exceeds" : "falls below"} headline score by ${Math.abs(diff).toFixed(3)}`,
      });
    }
  }

  // Return 3-6 findings (prioritize most important)
  return findings.slice(0, 6);
}

/**
 * Derive severity from delta value.
 */
export function getSeverityFromDelta(delta: number): "high" | "med" | "low" {
  if (delta <= -0.1) return "high";
  if (delta <= -0.05) return "med";
  return "low";
}

