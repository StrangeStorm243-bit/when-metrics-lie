from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is required for artifact generation. "
        "Install with: pip install matplotlib"
    )


def plot_metric_distribution(
    metric_summary: dict,
    metric_name: str,
    scenario_id: str,
    out_path: Path,
) -> None:
    """
    Plot histogram of metric values from summary statistics.
    
    Note: The distribution is an approximation reconstructed from summary
    statistics (mean, std, quantiles). It is intended for visualization purposes
    only and should not be used for exact statistical inference.
    """
    mean = metric_summary["mean"]
    std = metric_summary["std"]
    q05 = metric_summary["q05"]
    q50 = metric_summary["q50"]
    q95 = metric_summary["q95"]
    n = metric_summary["n"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate representative distribution from quantiles
    # Use normal approximation with bounds
    if std > 0:
        values = np.random.RandomState(seed=42).normal(mean, std, size=min(n, 1000))
        values = np.clip(values, q05, q95)
    else:
        # Single value case
        values = np.array([mean])

    ax.hist(values, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.4f}")
    ax.axvline(q50, color="orange", linestyle="--", linewidth=1, label=f"Median: {q50:.4f}")
    ax.set_xlabel(metric_name.upper())
    ax.set_ylabel("Frequency")
    ax.set_title(f"{metric_name.upper()} distribution – {scenario_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    scenario_id: str,
    out_path: Path,
) -> None:
    """
    Plot calibration curve (reliability diagram).
    Shows predicted confidence vs empirical accuracy.
    """
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_score, bins[1:-1], right=False)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        conf = float(y_score[mask].mean())
        acc = float(y_true[mask].mean())
        count = int(mask.sum())
        bin_confs.append(conf)
        bin_accs.append(acc)
        bin_counts.append(count)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)

    # Calibration curve
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "o-", label="Calibration", linewidth=2, markersize=8)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration curve – {scenario_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_subgroup_bars(
    group_means: dict[str, float],
    scenario_id: str,
    out_path: Path,
) -> None:
    """
    Plot bar chart of subgroup metric means.
    Highlights worst and best groups.
    """
    if not group_means:
        return

    groups = list(group_means.keys())
    means = [group_means[g] for g in groups]

    worst_idx = np.argmin(means)
    best_idx = np.argmax(means)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["lightblue"] * len(groups)
    colors[worst_idx] = "lightcoral"
    colors[best_idx] = "lightgreen"

    bars = ax.bar(groups, means, color=colors, edgecolor="black", alpha=0.7)

    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Metric mean")
    ax.set_xlabel("Subgroup")
    ax.set_title(f"Subgroup metric comparison – {scenario_id}")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgreen", label="Best group"),
        Patch(facecolor="lightcoral", label="Worst group"),
        Patch(facecolor="lightblue", label="Other groups"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

