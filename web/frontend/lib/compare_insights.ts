/**
 * Deterministic helper functions for deriving comparison insights.
 */

import type { ResultSummary } from "./api";

export interface ComparisonFinding {
  text: string;
}

/**
 * Derive key findings from comparing two result summaries.
 * Returns 3-8 bullet points summarizing the biggest changes.
 */
export function deriveComparisonFindings(
  resultA: ResultSummary,
  resultB: ResultSummary
): ComparisonFinding[] {
  const findings: ComparisonFinding[] = [];

  // 1. Headline score change
  const headlineDelta = resultB.headline_score - resultA.headline_score;
  if (Math.abs(headlineDelta) > 0.05) {
    const direction = headlineDelta > 0 ? "improved" : "degraded";
    findings.push({
      text: `Headline score ${direction} from ${resultA.headline_score.toFixed(4)} to ${resultB.headline_score.toFixed(4)} (Δ=${headlineDelta > 0 ? "+" : ""}${headlineDelta.toFixed(4)})`,
    });
  } else {
    findings.push({
      text: `Headline score remained stable: ${resultA.headline_score.toFixed(4)} → ${resultB.headline_score.toFixed(4)} (Δ=${headlineDelta > 0 ? "+" : ""}${headlineDelta.toFixed(4)})`,
    });
  }

  // 2. Scenario changes - find biggest improvements and degradations
  const scenarioMapA = new Map(resultA.scenario_results.map((s) => [s.scenario_id, s]));
  const scenarioMapB = new Map(resultB.scenario_results.map((s) => [s.scenario_id, s]));

  const scenarioChanges: Array<{
    scenario_id: string;
    scenario_name: string;
    deltaA: number;
    deltaB: number;
    change: number;
  }> = [];

  // Find common scenarios
  for (const [scenarioId, scenarioB] of scenarioMapB.entries()) {
    const scenarioA = scenarioMapA.get(scenarioId);
    if (scenarioA) {
      scenarioChanges.push({
        scenario_id: scenarioId,
        scenario_name: scenarioB.scenario_name,
        deltaA: scenarioA.delta,
        deltaB: scenarioB.delta,
        change: scenarioB.delta - scenarioA.delta,
      });
    }
  }

  // Sort by absolute change, descending
  scenarioChanges.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

  // Top improvement
  const topImprovement = scenarioChanges.find((s) => s.change > 0);
  if (topImprovement && topImprovement.change > 0.05) {
    findings.push({
      text: `Best improvement: "${topImprovement.scenario_name}" improved from Δ=${topImprovement.deltaA.toFixed(4)} to Δ=${topImprovement.deltaB.toFixed(4)} (change: +${topImprovement.change.toFixed(4)})`,
    });
  }

  // Top degradation
  const topDegradation = scenarioChanges.find((s) => s.change < 0);
  if (topDegradation && topDegradation.change < -0.05) {
    findings.push({
      text: `Worst degradation: "${topDegradation.scenario_name}" worsened from Δ=${topDegradation.deltaA.toFixed(4)} to Δ=${topDegradation.deltaB.toFixed(4)} (change: ${topDegradation.change.toFixed(4)})`,
    });
  }

  // 3. Component score changes
  const componentMapA = new Map(resultA.component_scores.map((c) => [c.name, c]));
  const componentMapB = new Map(resultB.component_scores.map((c) => [c.name, c]));

  const componentChanges: Array<{
    name: string;
    scoreA: number;
    scoreB: number;
    delta: number;
  }> = [];

  for (const [name, componentB] of componentMapB.entries()) {
    const componentA = componentMapA.get(name);
    if (componentA) {
      componentChanges.push({
        name,
        scoreA: componentA.score,
        scoreB: componentB.score,
        delta: componentB.score - componentA.score,
      });
    }
  }

  componentChanges.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const topComponentChange = componentChanges[0];
  if (topComponentChange && Math.abs(topComponentChange.delta) > 0.05) {
    const direction = topComponentChange.delta > 0 ? "improved" : "degraded";
    findings.push({
      text: `Component "${topComponentChange.name}" ${direction}: ${topComponentChange.scoreA.toFixed(4)} → ${topComponentChange.scoreB.toFixed(4)} (Δ=${topComponentChange.delta > 0 ? "+" : ""}${topComponentChange.delta.toFixed(4)})`,
    });
  }

  // 4. Flags comparison
  const flagsA = new Set(resultA.flags.map((f) => f.code));
  const flagsB = new Set(resultB.flags.map((f) => f.code));

  const addedFlags = resultB.flags.filter((f) => !flagsA.has(f.code));
  const removedFlags = resultA.flags.filter((f) => !flagsB.has(f.code));

  if (addedFlags.length > 0) {
    const criticalAdded = addedFlags.filter((f) => f.severity === "critical").length;
    const warnAdded = addedFlags.filter((f) => f.severity === "warn").length;
    if (criticalAdded > 0) {
      findings.push({
        text: `${criticalAdded} critical flag(s) added, ${warnAdded} warning(s) added`,
      });
    } else if (warnAdded > 0) {
      findings.push({
        text: `${warnAdded} warning flag(s) added`,
      });
    }
  }

  if (removedFlags.length > 0) {
    const criticalRemoved = removedFlags.filter((f) => f.severity === "critical").length;
    const warnRemoved = removedFlags.filter((f) => f.severity === "warn").length;
    if (criticalRemoved > 0) {
      findings.push({
        text: `${criticalRemoved} critical flag(s) resolved, ${warnRemoved} warning(s) resolved`,
      });
    } else if (warnRemoved > 0) {
      findings.push({
        text: `${warnRemoved} warning flag(s) resolved`,
      });
    }
  }

  // 5. Overall trend assessment
  const scenarioImprovements = scenarioChanges.filter((s) => s.change > 0.01).length;
  const scenarioDegradations = scenarioChanges.filter((s) => s.change < -0.01).length;

  if (scenarioImprovements > scenarioDegradations && scenarioImprovements > 0) {
    findings.push({
      text: `Overall positive trend: ${scenarioImprovements} scenario(s) improved vs ${scenarioDegradations} degraded`,
    });
  } else if (scenarioDegradations > scenarioImprovements && scenarioDegradations > 0) {
    findings.push({
      text: `Overall negative trend: ${scenarioDegradations} scenario(s) degraded vs ${scenarioImprovements} improved`,
    });
  }

  // Return 3-8 findings
  return findings.slice(0, 8);
}

