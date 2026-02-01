/**
 * Comparison model builders for deterministic diff computation.
 */

import type { ResultSummary, ScenarioResult, ComponentScore, FindingFlag } from "./api";

export interface ScenarioDiffRow {
  scenario_id: string;
  scenario_name: string;
  deltaA: number;
  deltaB: number;
  change: number;
  scoreA?: number;
  scoreB?: number;
}

export interface ComponentDiffRow {
  name: string;
  scoreA: number;
  scoreB: number;
  delta: number;
}

export interface FlagDiff {
  added: FindingFlag[];
  removed: FindingFlag[];
  persisting: FindingFlag[];
  bySeverity: {
    critical: { added: FindingFlag[]; removed: FindingFlag[]; persisting: FindingFlag[] };
    warn: { added: FindingFlag[]; removed: FindingFlag[]; persisting: FindingFlag[] };
    info: { added: FindingFlag[]; removed: FindingFlag[]; persisting: FindingFlag[] };
  };
}

/**
 * Build scenario diff between two results.
 * Joins by scenario_id, sorted by abs(change) descending.
 */
export function buildScenarioDiff(
  resultA: ResultSummary,
  resultB: ResultSummary
): ScenarioDiffRow[] {
  const scenarioMapA = new Map(resultA.scenario_results.map((s) => [s.scenario_id, s]));
  const scenarioMapB = new Map(resultB.scenario_results.map((s) => [s.scenario_id, s]));

  const diffs: ScenarioDiffRow[] = [];

  // Find common scenarios
  for (const [scenarioId, scenarioB] of scenarioMapB.entries()) {
    const scenarioA = scenarioMapA.get(scenarioId);
    if (scenarioA) {
      diffs.push({
        scenario_id: scenarioId,
        scenario_name: scenarioB.scenario_name,
        deltaA: scenarioA.delta,
        deltaB: scenarioB.delta,
        change: scenarioB.delta - scenarioA.delta,
        scoreA: scenarioA.score,
        scoreB: scenarioB.score,
      });
    }
  }

  // Sort by absolute change descending
  diffs.sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

  return diffs;
}

/**
 * Build component diff between two results.
 * Joins by component name, sorted by abs(delta) descending.
 */
export function buildComponentDiff(
  resultA: ResultSummary,
  resultB: ResultSummary
): ComponentDiffRow[] {
  const componentMapA = new Map(resultA.component_scores.map((c) => [c.name, c]));
  const componentMapB = new Map(resultB.component_scores.map((c) => [c.name, c]));

  const diffs: ComponentDiffRow[] = [];

  // Find common components
  for (const [name, componentB] of componentMapB.entries()) {
    const componentA = componentMapA.get(name);
    if (componentA) {
      diffs.push({
        name,
        scoreA: componentA.score,
        scoreB: componentB.score,
        delta: componentB.score - componentA.score,
      });
    }
  }

  // Sort by absolute delta descending
  diffs.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return diffs;
}

/**
 * Build flag diff between two flag arrays.
 * Groups by added/removed/persisting and by severity.
 */
export function buildFlagDiff(flagsA: FindingFlag[], flagsB: FindingFlag[]): FlagDiff {
  const flagsA_set = new Set(flagsA.map((f) => f.code));
  const flagsB_set = new Set(flagsB.map((f) => f.code));

  const added = flagsB.filter((f) => !flagsA_set.has(f.code));
  const removed = flagsA.filter((f) => !flagsB_set.has(f.code));
  const persisting = flagsA.filter((f) => flagsB_set.has(f.code));

  const groupBySeverity = (flags: FindingFlag[]) => ({
    critical: flags.filter((f) => f.severity === "critical"),
    warn: flags.filter((f) => f.severity === "warn"),
    info: flags.filter((f) => f.severity === "info"),
  });

  return {
    added,
    removed,
    persisting,
    bySeverity: {
      critical: {
        added: groupBySeverity(added).critical,
        removed: groupBySeverity(removed).critical,
        persisting: groupBySeverity(persisting).critical,
      },
      warn: {
        added: groupBySeverity(added).warn,
        removed: groupBySeverity(removed).warn,
        persisting: groupBySeverity(persisting).warn,
      },
      info: {
        added: groupBySeverity(added).info,
        removed: groupBySeverity(removed).info,
        persisting: groupBySeverity(persisting).info,
      },
    },
  };
}

/**
 * Get worst-case scenario (lowest deltaB).
 */
export function getWorstCaseScenario(diffs: ScenarioDiffRow[]): ScenarioDiffRow | null {
  if (diffs.length === 0) return null;
  return diffs.reduce((worst, current) => (current.deltaB < worst.deltaB ? current : worst));
}

/**
 * Get biggest regression (most negative change).
 */
export function getBiggestRegression(diffs: ScenarioDiffRow[]): ScenarioDiffRow | null {
  const regressions = diffs.filter((d) => d.change < 0);
  if (regressions.length === 0) return null;
  return regressions.reduce((worst, current) => (current.change < worst.change ? current : worst));
}

/**
 * Get biggest improvement (most positive change).
 */
export function getBiggestImprovement(diffs: ScenarioDiffRow[]): ScenarioDiffRow | null {
  const improvements = diffs.filter((d) => d.change > 0);
  if (improvements.length === 0) return null;
  return improvements.reduce((best, current) => (current.change > best.change ? current : best));
}

/**
 * Get biggest component change (by absolute delta).
 */
export function getBiggestComponentChange(diffs: ComponentDiffRow[]): ComponentDiffRow | null {
  if (diffs.length === 0) return null;
  return diffs[0]; // Already sorted by abs(delta) desc
}

