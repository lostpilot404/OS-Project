"""Figures for the offline policy-selection study.

Every figure is drawn from the generated ``decisions.csv`` / ``training_history.csv``
values; none of them hard-code results.  Matplotlib is imported lazily so the study can
run without figures (``--no-figures``) without importing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

__all__ = ["create_study_figures"]

_ARM_COLORS = {
    "ql_orig": "#4c72b0",
    "ql_risk": "#c44e52",
    "ql_risk_origstate": "#dd8452",
    "orig_fingerprint": "#937860",
    "ql_orig_fingerprint": "#937860",
    "best_fixed": "#55a868",
    "fingerprint_rule": "#8172b3",
    "random": "#ccb974",
}


def create_study_figures(
    decisions: pd.DataFrame,
    training_rows: pd.DataFrame,
    figures_dir: Path,
) -> List[Path]:
    """Create the study figures and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    test = decisions[decisions["split"] == "test"]
    arms = sorted(test["arm"].unique())

    def colors(names) -> List[str]:
        return [_ARM_COLORS.get(name, "#777777") for name in names]

    # 1. Mean declared-objective cost by arm and family.
    fig, ax = plt.subplots(figsize=(11, 5))
    families = sorted(test["family"].unique())
    width = 0.8 / len(arms)
    for index, arm in enumerate(arms):
        frame = test[test["arm"] == arm]
        means = [
            frame[frame["family"] == family]["risk_objective_cost"].mean()
            for family in families
        ]
        ax.bar(
            [x + index * width for x in range(len(families))],
            means,
            width=width,
            label=arm,
            color=_ARM_COLORS.get(arm, "#777777"),
        )
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(families))])
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("Mean declared-objective cost (lower is better)")
    ax.set_title("Held-out test split: declared-objective cost by arm and family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "study_cost_by_family.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 2. Regret by arm (mean with per-workload spread).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [test[test["arm"] == arm]["regret"].tolist() for arm in arms]
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(arms)
    ax.set_ylabel("Hindsight regret under the declared objective")
    ax.set_title("Held-out test split: selection regret by arm")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figures_dir / "study_regret_by_arm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 3. Selection mix per arm.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = [0.0] * len(arms)
    from config import ACTION_NAMES

    policy_colors = ["#4c72b0", "#c44e52", "#55a868", "#8172b3"]
    for policy, color in zip(ACTION_NAMES, policy_colors):
        shares = []
        for arm in arms:
            frame = test[test["arm"] == arm]
            shares.append(float((frame["policy_name"] == policy).mean()))
        ax.bar(arms, shares, bottom=bottom, label=policy, color=color)
        bottom = [b + s for b, s in zip(bottom, shares)]
    ax.set_ylabel("Share of selections")
    ax.set_title("Held-out test split: selected-policy mix by arm")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figures_dir / "study_selection_mix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 4. Training curves: mean terminal reward per episode across seeds.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for arm in sorted(training_rows["arm"].unique()):
        frame = training_rows[training_rows["arm"] == arm]
        curve = frame.groupby("episode")["objective_reward"].mean()
        window = max(1, len(curve) // 50)
        smoothed = curve.rolling(window, min_periods=1).mean()
        ax.plot(
            smoothed.index,
            smoothed.values,
            label=arm,
            color=_ARM_COLORS.get(arm, "#777777"),
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Terminal reward (smoothed mean across seeds)")
    ax.set_title("Training curves per arm (declared objective reward)")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "study_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 5. Both objectives side by side per arm (comparison, not replacement).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    risk_means = [test[test["arm"] == arm]["risk_objective_cost"].mean() for arm in arms]
    orig_means = [
        -test[test["arm"] == arm]["orig_objective_reward"].mean() for arm in arms
    ]
    x = range(len(arms))
    ax.bar([i - 0.2 for i in x], risk_means, width=0.4, label="risk_aware_v1 cost", color="#c44e52")
    ax.bar([i + 0.2 for i in x], orig_means, width=0.4, label="original objective cost (-reward)", color="#4c72b0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=30, ha="right")
    ax.set_ylabel("Objective cost (lower is better)")
    ax.set_title("Held-out test split: both declared objectives by arm")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "study_objectives_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths
