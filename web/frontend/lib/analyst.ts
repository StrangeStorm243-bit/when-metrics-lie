/**
 * Deterministic analyst assistant for comparison insights.
 */

import type { ResultSummary } from "./api";
import type { ScenarioDiffRow, ComponentDiffRow, FlagDiff } from "./compare_model";
import {
  getWorstCaseScenario,
  getBiggestRegression,
  getBiggestImprovement,
} from "./compare_model";

export type AnalystIntent =
  | "overview"
  | "worse"
  | "improved"
  | "worst_case"
  | "new_flags"
  | "scenario_focus"
  | "component_focus"
  | "flag_focus";

export interface AnalystAction {
  label: string;
  type: "scroll_to" | "focus";
  payload: { type: "scenario" | "component" | "flag"; key: string };
}

export interface AnalystMessage {
  id: string;
  role: "user" | "assistant";
  title?: string;
  body: string;
  evidence?: string;
  actions?: AnalystAction[];
}

export interface AnalystContext {
  resultA: ResultSummary;
  resultB: ResultSummary;
  scenarioDiffs: ScenarioDiffRow[];
  componentDiffs: ComponentDiffRow[];
  flagDiff: FlagDiff;
  focus?: { type: "scenario" | "component" | "flag"; key: string };
}

/**
 * Generate analyst response for a given intent and context.
 */
export function respond(intent: AnalystIntent, context: AnalystContext): AnalystMessage[] {
  const { resultA, resultB, scenarioDiffs, componentDiffs, flagDiff, focus } = context;

  const headlineDelta = resultB.headline_score - resultA.headline_score;
  const headlineDeltaFormatted = headlineDelta >= 0 ? `+${headlineDelta.toFixed(4)}` : headlineDelta.toFixed(4);

  switch (intent) {
    case "overview": {
      const worstCase = getWorstCaseScenario(scenarioDiffs);
      const biggestReg = getBiggestRegression(scenarioDiffs);
      const criticalAdded = flagDiff.bySeverity.critical.added.length;
      const criticalRemoved = flagDiff.bySeverity.critical.removed.length;

      const body = `Headline score changed by ${headlineDeltaFormatted} (${resultA.headline_score.toFixed(4)} → ${resultB.headline_score.toFixed(4)}).`;
      let evidence = "";
      const actions: AnalystAction[] = [];

      if (worstCase) {
        evidence += `Worst-case scenario: "${worstCase.scenario_name}" with Δ=${worstCase.deltaB.toFixed(4)}. `;
        actions.push({
          label: "Show worst-case scenario",
          type: "scroll_to",
          payload: { type: "scenario", key: worstCase.scenario_id },
        });
      }

      if (biggestReg) {
        evidence += `Biggest regression: "${biggestReg.scenario_name}" worsened by ${Math.abs(biggestReg.change).toFixed(4)}. `;
      }

      if (criticalAdded > 0 || criticalRemoved > 0) {
        evidence += `${criticalAdded} critical flag(s) added, ${criticalRemoved} resolved.`;
      }

      return [
        {
          id: `overview-${Date.now()}`,
          role: "assistant",
          title: "Comparison Overview",
          body,
          evidence: evidence.trim() || undefined,
          actions: actions.length > 0 ? actions : undefined,
        },
      ];
    }

    case "worse": {
      const regressions = scenarioDiffs.filter((d) => d.change < 0);
      if (regressions.length === 0) {
        return [
          {
            id: `worse-${Date.now()}`,
            role: "assistant",
            body: "No scenarios worsened. All changes are neutral or improvements.",
          },
        ];
      }

      const top3 = regressions.slice(0, 3);
      const body = `${regressions.length} scenario(s) worsened. Top regressions:`;
      const evidence = top3
        .map((r) => `"${r.scenario_name}": Δ ${r.deltaA.toFixed(4)} → ${r.deltaB.toFixed(4)} (change: ${r.change.toFixed(4)})`)
        .join("; ");

      return [
        {
          id: `worse-${Date.now()}`,
          role: "assistant",
          title: "What Got Worse?",
          body,
          evidence,
          actions: top3.map((r) => ({
            label: `Show "${r.scenario_name}"`,
            type: "scroll_to",
            payload: { type: "scenario", key: r.scenario_id },
          })),
        },
      ];
    }

    case "improved": {
      const improvements = scenarioDiffs.filter((d) => d.change > 0);
      if (improvements.length === 0) {
        return [
          {
            id: `improved-${Date.now()}`,
            role: "assistant",
            body: "No scenarios improved. All changes are neutral or regressions.",
          },
        ];
      }

      const top3 = improvements.slice(0, 3);
      const body = `${improvements.length} scenario(s) improved. Top improvements:`;
      const evidence = top3
        .map((i) => `"${i.scenario_name}": Δ ${i.deltaA.toFixed(4)} → ${i.deltaB.toFixed(4)} (change: +${i.change.toFixed(4)})`)
        .join("; ");

      return [
        {
          id: `improved-${Date.now()}`,
          role: "assistant",
          title: "What Improved?",
          body,
          evidence,
          actions: top3.map((i) => ({
            label: `Show "${i.scenario_name}"`,
            type: "scroll_to",
            payload: { type: "scenario", key: i.scenario_id },
          })),
        },
      ];
    }

    case "worst_case": {
      const worstCase = getWorstCaseScenario(scenarioDiffs);
      if (!worstCase) {
        return [
          {
            id: `worst-case-${Date.now()}`,
            role: "assistant",
            body: "No scenario data available.",
          },
        ];
      }

      return [
        {
          id: `worst-case-${Date.now()}`,
          role: "assistant",
          title: "Worst-Case Scenario",
          body: `"${worstCase.scenario_name}" has the worst performance in Run B with Δ=${worstCase.deltaB.toFixed(4)}.`,
          evidence: `Changed from Δ=${worstCase.deltaA.toFixed(4)} in Run A (change: ${worstCase.change >= 0 ? "+" : ""}${worstCase.change.toFixed(4)}).`,
          actions: [
            {
              label: "Show scenario",
              type: "scroll_to",
              payload: { type: "scenario", key: worstCase.scenario_id },
            },
          ],
        },
      ];
    }

    case "new_flags": {
      const added = flagDiff.added;
      if (added.length === 0) {
        return [
          {
            id: `new-flags-${Date.now()}`,
            role: "assistant",
            body: "No new flags in Run B.",
          },
        ];
      }

      const critical = flagDiff.bySeverity.critical.added;
      const warn = flagDiff.bySeverity.warn.added;
      const info = flagDiff.bySeverity.info.added;

      const body = `${added.length} new flag(s) in Run B: ${critical.length} critical, ${warn.length} warning(s), ${info.length} info.`;
      const evidence = critical
        .slice(0, 3)
        .map((f) => `[${f.code}] ${f.title}`)
        .join("; ");

      return [
        {
          id: `new-flags-${Date.now()}`,
          role: "assistant",
          title: "New Flags",
          body,
          evidence: evidence || undefined,
          actions: added.slice(0, 3).map((f) => ({
            label: `Show "${f.title}"`,
            type: "scroll_to",
            payload: { type: "flag", key: f.code },
          })),
        },
      ];
    }

    case "scenario_focus": {
      if (!focus || focus.type !== "scenario") {
        return [];
      }

      const scenario = scenarioDiffs.find((d) => d.scenario_id === focus.key);
      if (!scenario) {
        return [
          {
            id: `scenario-focus-${Date.now()}`,
            role: "assistant",
            body: "Scenario not found in comparison.",
          },
        ];
      }

      const direction = scenario.change > 0 ? "improved" : scenario.change < 0 ? "worsened" : "unchanged";
      const changeFormatted = scenario.change >= 0 ? `+${scenario.change.toFixed(4)}` : scenario.change.toFixed(4);

      return [
        {
          id: `scenario-focus-${Date.now()}`,
          role: "assistant",
          title: `Scenario: ${scenario.scenario_name}`,
          body: `This scenario ${direction} by ${changeFormatted}.`,
          evidence: `Delta changed from ${scenario.deltaA.toFixed(4)} (Run A) to ${scenario.deltaB.toFixed(4)} (Run B). Score: ${scenario.scoreA?.toFixed(4) ?? "N/A"} → ${scenario.scoreB?.toFixed(4) ?? "N/A"}.`,
        },
      ];
    }

    case "component_focus": {
      if (!focus || focus.type !== "component") {
        return [];
      }

      const component = componentDiffs.find((d) => d.name === focus.key);
      if (!component) {
        return [
          {
            id: `component-focus-${Date.now()}`,
            role: "assistant",
            body: "Component not found in comparison.",
          },
        ];
      }

      const direction = component.delta > 0 ? "improved" : component.delta < 0 ? "degraded" : "unchanged";
      const deltaFormatted = component.delta >= 0 ? `+${component.delta.toFixed(4)}` : component.delta.toFixed(4);

      return [
        {
          id: `component-focus-${Date.now()}`,
          role: "assistant",
          title: `Component: ${component.name}`,
          body: `This component ${direction} by ${deltaFormatted}.`,
          evidence: `Score changed from ${component.scoreA.toFixed(4)} (Run A) to ${component.scoreB.toFixed(4)} (Run B).`,
        },
      ];
    }

    case "flag_focus": {
      if (!focus || focus.type !== "flag") {
        return [];
      }

      const flagA = resultA.flags.find((f) => f.code === focus.key);
      const flagB = resultB.flags.find((f) => f.code === focus.key);

      if (!flagA && !flagB) {
        return [
          {
            id: `flag-focus-${Date.now()}`,
            role: "assistant",
            body: "Flag not found in either run.",
          },
        ];
      }

      if (flagA && !flagB) {
        return [
          {
            id: `flag-focus-${Date.now()}`,
            role: "assistant",
            title: `Flag: ${flagA.title}`,
            body: "This flag was present in Run A but resolved in Run B.",
            evidence: flagA.detail,
          },
        ];
      }

      if (!flagA && flagB) {
        return [
          {
            id: `flag-focus-${Date.now()}`,
            role: "assistant",
            title: `Flag: ${flagB.title}`,
            body: "This is a new flag in Run B.",
            evidence: flagB.detail,
          },
        ];
      }

      // Persisting flag
      return [
        {
          id: `flag-focus-${Date.now()}`,
          role: "assistant",
          title: `Flag: ${flagB!.title}`,
          body: "This flag persists in both runs.",
          evidence: flagB!.detail,
        },
      ];
    }

    default:
      return [];
  }
}

