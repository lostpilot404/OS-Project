"""Figures for the self-healing study, drawn only from the machine-readable tables.

Every figure is a rendering of a frame that the experiment also writes to CSV, so a
reviewer can reproduce any plotted value from the artifacts in ``results/healing/``.
Importing Matplotlib is deferred to the plotting functions, so the study itself runs
without a plotting backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from visualization.style import DPI, FIGSIZE_WIDE, PALETTE, apply_style, color_for

__all__ = ["plot_healing_figures"]

_FIGURE_ARMS: Sequence[str] = (
    "fixed_fcfs",
    "fixed_sjf",
    "detector_only",
    "rule_fairness",
    "rule_wait",
    "q_recovery",
)


def _short(names: Sequence[str]) -> List[str]:
    return [name.replace("fixed_", "").replace("_", " ") for name in names]


def plot_healing_figures(
    figures_dir: Path,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    detection: pd.DataFrame,
    audit: pd.DataFrame,
    probe: pd.DataFrame,
) -> Dict[str, Path]:
    """Render the study figures and return their paths.

    Args:
        figures_dir: Directory the PNG files are written to.
        metrics: Final-test metric rows (one row per arm/seed/workload).
        comparisons: Paired bootstrap comparison table.
        detection: Per-arm detection and verdict summary.
        audit: Counterfactual replay audit rows.
        probe: Action-effect probe rows (may be empty).

    Returns:
        Mapping of artifact name to written path.
    """
    import matplotlib.pyplot as plt

    from visualization.style import apply_style

    apply_style()

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Mean waiting time per workload family for the main arms.
    arms = [arm for arm in _FIGURE_ARMS if arm in set(metrics["arm"])]
    families = sorted(metrics["family"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    width = 0.8 / max(1, len(arms))
    for index, arm in enumerate(arms):
        subset = metrics[metrics["arm"] == arm]
        means = [
            float(subset[subset["family"] == family]["avg_waiting_time"].astype(float).mean())
            for family in families
        ]
        offsets = [position + index * width for position in range(len(families))]
        axes[0].bar(offsets, means, width=width, label=arm)
    axes[0].set_xticks([position + 0.4 - width / 2 for position in range(len(families))])
    axes[0].set_xticklabels(families, rotation=30, ha="right")
    axes[0].set_ylabel("Mean waiting time")
    axes[0].set_title("Mean waiting time by workload family (final test)")
    axes[0].legend(fontsize=8)

    overall = (
        metrics[metrics["arm"].isin(arms)]
        .groupby("arm")["avg_waiting_time"]
        .mean()
        .reindex(arms)
    )
    p95 = metrics[metrics["arm"].isin(arms)].groupby("arm")["p95_waiting_time"].mean().reindex(arms)
    positions = range(len(arms))
    axes[1].bar([p - 0.2 for p in positions], overall.values, width=0.4, label="mean waiting")
    axes[1].bar([p + 0.2 for p in positions], p95.values, width=0.4, label="P95 waiting")
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(_short(list(overall.index)), rotation=20, ha="right")
    axes[1].set_title("Pooled mean and tail waiting time")
    axes[1].legend()
    figure_path = figures_dir / "healing_waiting_time.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=DPI)
    plt.close(fig)
    paths["figure_waiting_time"] = figure_path

    # 2. Recovery outcomes and the audit of interventions.
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    detection = detection[detection["split"] == "final_test"]
    arms_with_incidents = [
        arm
        for arm in _FIGURE_ARMS
        if arm in set(detection["arm"]) and arm not in ("fixed_fcfs", "fixed_sjf")
    ]
    bottom = [0.0] * len(arms_with_incidents)
    for column, colour in (
        ("accepted", "#2a7f62"),
        ("rejected", "#c1554a"),
        ("inconclusive", "#d9b04c"),
        ("held", "#8a8f98"),
    ):
        values = [
            float(detection[detection["arm"] == arm][column].iloc[0]) for arm in arms_with_incidents
        ]
        axes[0].bar(arms_with_incidents, values, bottom=bottom, label=column, color=colour)
        bottom = [left + value for left, value in zip(bottom, values)]
    axes[0].set_xticks(range(len(arms_with_incidents)))
    axes[0].set_xticklabels(_short(arms_with_incidents), rotation=20, ha="right")
    axes[0].set_ylabel("Incidents (final test)")
    axes[0].set_title("Recovery outcomes by arm")
    axes[0].legend(fontsize=8)

    if not audit.empty:
        final_audit = audit[audit["split"] == "final_test"]
        audited_arms = [arm for arm in arms_with_incidents if arm in set(final_audit["arm"])]
        gains = [
            final_audit[final_audit["arm"] == arm]["matched_gain"].astype(float).mean()
            for arm in audited_arms
        ]
        axes[1].bar(audited_arms, gains, color=color_for("primary"))
        axes[1].axhline(0.0, color="#222222", linewidth=0.8)
        axes[1].set_xticks(range(len(audited_arms)))
        axes[1].set_xticklabels(_short(audited_arms), rotation=20, ha="right")
        axes[1].set_ylabel("Mean matched-counterfactual gain (waiting-time units)")
        axes[1].set_title("Marginal value of each intervention (matched hold)")
    figure_path = figures_dir / "healing_recovery_outcomes.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=DPI)
    plt.close(fig)
    paths["figure_recovery_outcomes"] = figure_path

    # 3. Action-effect probe: measured reward against true effect.
    if not probe.empty:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
        # The authoritative reward column is ``controller_reward``; older artifacts
        # carry it as ``reward_band_units`` for backward compatibility.
        reward_col = "controller_reward" if "controller_reward" in probe.columns else "reward_band_units"
        by_action = probe.groupby("action_name")[
            [reward_col, "true_effect_per_process"]
        ].mean()
        axes[0].scatter(
            by_action[reward_col], by_action["true_effect_per_process"], color=color_for("primary")
        )
        for name, row in by_action.iterrows():
            axes[0].annotate(name, (row[reward_col], row["true_effect_per_process"]), fontsize=8)
        axes[0].axvline(
            float(probe[reward_col].mean()), color="#222222", linewidth=0.8, linestyle="--"
        )
        axes[0].set_xlabel("Controller reward (band units, hold level dashed)")
        axes[0].set_ylabel("True effect (per-process waiting-time reduction)")
        axes[0].set_title("Reward versus ground truth per action (validation)")

        diagnoses = sorted(probe["diagnosis"].unique().tolist())
        actions = sorted(probe["action_name"].unique().tolist())
        width = 0.8 / max(1, len(actions))
        for index, action in enumerate(actions):
            means = [
                float(
                    probe[(probe["diagnosis"] == diagnosis) & (probe["action_name"] == action)][
                        "true_effect_per_process"
                    ].mean()
                )
                for diagnosis in diagnoses
            ]
            axes[1].bar(
                [position + index * width for position in range(len(diagnoses))],
                means,
                width=width,
                label=action,
            )
        axes[1].set_xticks([position + 0.4 for position in range(len(diagnoses))])
        axes[1].set_xticklabels(diagnoses, rotation=25, ha="right")
        axes[1].set_ylabel("True effect per process")
        axes[1].set_title("True effect of each action by diagnosis")
        axes[1].legend(fontsize=8)
        figure_path = figures_dir / "healing_action_effects.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=DPI)
        plt.close(fig)
        paths["figure_action_effects"] = figure_path

    del comparisons
    return paths
