"""Display functions for notebooks and terminals."""
from __future__ import annotations

from typing import Any

from metrics_lie.schema import ResultBundle


def format_summary(bundle: ResultBundle) -> str:
    """Format a ResultBundle as a readable text summary."""
    lines = [
        f"Spectra Run: {bundle.run_id}",
        f"Experiment: {bundle.experiment_name}",
        f"Task: {bundle.task_type}",
        f"Primary metric: {bundle.metric_name} = {bundle.baseline.mean:.4f}"
        if bundle.baseline
        else f"Primary metric: {bundle.metric_name} (no baseline)",
        f"Applicable metrics: {', '.join(bundle.applicable_metrics)}",
        f"Scenarios: {len(bundle.scenarios)}",
    ]

    if bundle.metric_results:
        lines.append("\nMetric Results:")
        for mid, ms in bundle.metric_results.items():
            summary = ms if isinstance(ms, dict) else ms.model_dump()
            lines.append(
                f"  {mid}: {summary.get('mean', 0):.4f}"
                f" (std={summary.get('std', 0):.4f})"
            )

    aa = bundle.analysis_artifacts or {}
    if "dashboard_summary" in aa:
        ds = aa["dashboard_summary"]
        risk = ds.get("risk_summary", {})
        drops = risk.get("metrics_with_large_drops", [])
        if drops:
            lines.append(
                f"\nRisk: {len(drops)} metric(s) with large drops:"
                f" {', '.join(drops)}"
            )
        else:
            lines.append("\nRisk: No large metric drops detected.")

    return "\n".join(lines)


def format_comparison(report: dict[str, Any]) -> str:
    """Format a comparison report as readable text."""
    lines = [
        f"Comparison: {report.get('metric_name', '?')}",
    ]
    bd = report.get("baseline_delta", {})
    lines.append(
        f"Baseline delta: {bd.get('mean', 0):+.4f}"
        f" (A={bd.get('a', 0):.4f}, B={bd.get('b', 0):.4f})"
    )

    decision = report.get("decision", {})
    lines.append(
        f"Winner: {decision.get('winner', '?')}"
        f" ({decision.get('confidence', '?')})"
    )
    lines.append(f"Reasoning: {decision.get('reasoning', '?')}")

    flags = report.get("risk_flags", [])
    if flags:
        lines.append(f"Risk flags: {', '.join(flags)}")

    return "\n".join(lines)


def display(bundle: ResultBundle) -> None:
    """Display a ResultBundle. Uses HTML in Jupyter, text elsewhere."""
    try:
        from IPython.display import display as ipy_display, HTML

        ipy_display(HTML(_to_html(bundle)))
    except ImportError:
        print(format_summary(bundle))


def _to_html(bundle: ResultBundle) -> str:
    """Convert ResultBundle to HTML for Jupyter display."""
    summary = format_summary(bundle)
    escaped = (
        summary.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f"<pre style='background:#f5f5f5;padding:12px;"
        f"border-radius:4px;'>{escaped}</pre>"
    )
