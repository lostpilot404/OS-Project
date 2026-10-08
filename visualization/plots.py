"""Figures for the offline workload-aware policy-selection experiment.

All values are derived from generated metric rows, decisions, training histories, and Q
values. The six figures contain only the four conventional policies and the learned
single-workload selector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ACTION_NAMES, SELECTOR_LABEL
from errors import ValidationError
from evaluation.comparison import ALL_METRICS, METRIC_LABELS, policy_selection_table
from experiments.evaluate import SELECTOR_REGIME, BASELINE_REGIME

__all__ = [
    "create_all_figures",
    "plot_metric_comparison",
    "plot_metric_by_family",
    "plot_policy_selection",
    "plot_reward_by_family",
    "plot_training_curve",
    "plot_state_space",
]

_SELECTOR_COLOR = "#c0392b"
_BASELINE_COLOR = "#2c3e50"


def _policy_order(policies: Sequence[str]) -> List[str]:
    """Return the four fixed actions followed by the offline policy selector."""
    ordered = [name for name in ACTION_NAMES if name in policies]
    if SELECTOR_LABEL in policies:
        ordered.append(SELECTOR_LABEL)
    return ordered


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    """Raise a project validation error if columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{where} is missing the columns {missing}")


def _headline_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Retain baselines once per unique workload and all model-specific decisions."""
    _require_columns(
        metrics,
        ["training_seed", "family", "repetition", "workload_fingerprint", "policy", "regime"],
        "metrics frame",
    )
    baseline = metrics[metrics["regime"] == BASELINE_REGIME].drop_duplicates(
        ["family", "repetition", "workload_fingerprint", "policy"]
    )
    selector = metrics[metrics["regime"] == SELECTOR_REGIME]
    if baseline.empty or selector.empty:
        raise ValidationError("headline figures require baseline and selector metric rows")
    return pd.concat([baseline, selector], ignore_index=True)


def _finish(fig, path: Path) -> Path:
    """Tighten, save and close one figure."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_comparison(
    metrics: pd.DataFrame,
    path: Path,
    metrics_to_show: Sequence[str] = ALL_METRICS,
) -> Path:
    """Plot the mean ± sample SD of each of the six reported metrics."""
    frame = _headline_rows(metrics)
    _require_columns(frame, ["policy", *metrics_to_show], "metrics frame")
    policies = _policy_order(list(frame["policy"].unique()))
    grouped = frame.groupby("policy", sort=False)[list(metrics_to_show)]
    means = grouped.mean().reindex(policies)
    errors = grouped.std(ddof=1).fillna(0.0).reindex(policies)

    columns = 3
    rows = int(np.ceil(len(metrics_to_show) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.4 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for axis, metric in zip(axes_list, metrics_to_show):
        values = means[metric]
        colors = [
            _SELECTOR_COLOR if name == SELECTOR_LABEL else _BASELINE_COLOR
            for name in values.index
        ]
        axis.bar(
            range(len(values)),
            values.to_numpy(),
            yerr=errors[metric].to_numpy(),
            capsize=3,
            color=colors,
        )
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(values.index, rotation=30, ha="right", fontsize=8)
        axis.set_title(f"{METRIC_LABELS[metric]} (mean ± SD)", fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    for axis in axes_list[len(metrics_to_show) :]:
        axis.axis("off")
    fig.suptitle("Paired workload results across evaluation instances and training seeds", fontsize=12)
    return _finish(fig, Path(path))


def plot_metric_by_family(
    metrics: pd.DataFrame,
    path: Path,
    metric: str = "avg_waiting_time",
) -> Path:
    """Plot a selected six-metric outcome by workload condition with SD bars."""
    frame = _headline_rows(metrics)
    _require_columns(frame, ["family", "policy", metric], "metrics frame")
    policies = _policy_order(list(frame["policy"].unique()))
    families = list(dict.fromkeys(frame["family"].tolist()))
    grouped = frame.groupby(["family", "policy"], sort=False)[metric]
    means = grouped.mean().unstack("policy").reindex(index=families, columns=policies)
    errors = grouped.std(ddof=1).fillna(0.0).unstack("policy").reindex(
        index=families, columns=policies
    )

    positions = np.arange(len(families))
    width = 0.8 / len(policies)
    fig, axis = plt.subplots(figsize=(max(8.0, 1.6 * len(families)), 4.8))
    for offset, policy in enumerate(policies):
        axis.bar(
            positions + offset * width - 0.4 + width / 2,
            means[policy].to_numpy(),
            yerr=errors[policy].to_numpy(),
            capsize=2,
            width=width,
            label=policy,
            color=_SELECTOR_COLOR if policy == SELECTOR_LABEL else None,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=20, ha="right", fontsize=9)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.set_title(f"{METRIC_LABELS.get(metric, metric)} by workload condition (mean ± SD)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, Path(path))


def plot_policy_selection(decisions: pd.DataFrame, path: Path) -> Path:
    """Plot per-condition action shares across model-seed × workload decisions."""
    table = policy_selection_table(decisions)
    share_columns = [f"share_{name}" for name in ACTION_NAMES]
    families = list(table.index)
    fig, axis = plt.subplots(figsize=(max(8.0, 1.5 * len(families)), 4.6))
    bottom = np.zeros(len(families))
    for name, column in zip(ACTION_NAMES, share_columns):
        values = table[column].to_numpy()
        axis.bar(families, values, bottom=bottom, label=name)
        bottom += values
    axis.set_ylabel("Share of model decisions")
    axis.set_ylim(0, 1)
    axis.set_title("Learned policy selections by workload condition")
    axis.tick_params(axis="x", rotation=20, labelsize=9)
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, Path(path))


def plot_reward_by_family(decisions: pd.DataFrame, path: Path) -> Path:
    """Plot mean reward ± sample SD of model-seed × workload decisions."""
    _require_columns(decisions, ["family", "reward"], "decisions frame")
    grouped = decisions.groupby("family", sort=False)["reward"]
    means = grouped.mean()
    errors = grouped.std(ddof=1).fillna(0.0)
    fig, axis = plt.subplots(figsize=(max(7.5, 1.4 * len(means)), 4.4))
    axis.bar(
        means.index,
        means.to_numpy(),
        yerr=errors.to_numpy(),
        capsize=4,
        color=_SELECTOR_COLOR,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Reward (0 = four-policy mean reference)")
    axis.set_title("Evaluation reward by workload condition (mean ± sample SD)")
    axis.tick_params(axis="x", rotation=20, labelsize=9)
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, Path(path))


def plot_training_curve(histories: Sequence[object], path: Path, window: int = 25) -> Path:
    """Plot the cross-seed rolling reward distribution and configured epsilon schedule."""
    if not histories:
        raise ValidationError("training history contains no runs")
    reward_arrays = [np.asarray(history.rewards, dtype=float) for history in histories]
    epsilon_arrays = [np.asarray(history.epsilons, dtype=float) for history in histories]
    if any(array.size == 0 for array in reward_arrays):
        raise ValidationError("a training history contains no episodes")
    if len({array.size for array in reward_arrays}) != 1:
        raise ValidationError("training histories have different episode counts")
    rewards = np.column_stack(reward_arrays)
    epsilons = np.column_stack(epsilon_arrays)
    rolling = np.column_stack(
        [pd.Series(rewards[:, index]).rolling(window=window, min_periods=1).mean().to_numpy()
         for index in range(rewards.shape[1])]
    )
    mean_rolling = rolling.mean(axis=1)
    sd_rolling = rolling.std(axis=1, ddof=1) if rolling.shape[1] > 1 else np.zeros(rolling.shape[0])
    mean_epsilon = epsilons.mean(axis=1)

    episodes = np.arange(rewards.shape[0])
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.plot(episodes, mean_rolling, color=_SELECTOR_COLOR, linewidth=1.8, label=f"mean rolling reward ({window})")
    axis.fill_between(
        episodes,
        mean_rolling - sd_rolling,
        mean_rolling + sd_rolling,
        color=_SELECTOR_COLOR,
        alpha=0.18,
        label="±1 SD across training seeds",
    )
    axis.axhline(0.0, color="black", linewidth=1, linestyle=":")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Reward")
    axis.set_title("Training reward across independent seeds")
    twin = axis.twinx()
    twin.plot(episodes, mean_epsilon, color="#2980b9", linewidth=1.4, linestyle="--", label="epsilon probability")
    twin.set_ylabel("Epsilon (probability of uniform random action)")
    twin.set_ylim(0, 1.05)
    lines, labels = axis.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axis.legend(lines + twin_lines, labels + twin_labels, fontsize=8, loc="lower right")
    axis.grid(alpha=0.3)
    return _finish(fig, Path(path))


def plot_state_space(q_table_rows: Sequence[Dict[str, object]], path: Path) -> Path:
    """Show greedy-action shares across training models that visited each state."""
    if not q_table_rows:
        raise ValidationError("Q-table is empty")
    states = sorted({int(row["state_index"]) for row in q_table_rows})
    seeds = sorted({int(row["training_seed"]) for row in q_table_rows})
    actions_by_state: Dict[int, List[str]] = {state: [] for state in states}
    for row in q_table_rows:
        if int(row["visit_count"]) > 0:
            actions_by_state[int(row["state_index"])].append(str(row["greedy_action"]))
    visited_states = [state for state in states if actions_by_state[state]]
    if not visited_states:
        raise ValidationError("no state received a training update")

    matrix = np.zeros((len(visited_states), len(ACTION_NAMES)), dtype=float)
    coverage = []
    for row_index, state in enumerate(visited_states):
        actions = actions_by_state[state]
        coverage.append(len(actions))
        for action_index, action_name in enumerate(ACTION_NAMES):
            matrix[row_index, action_index] = actions.count(action_name) / len(actions)

    fig, axis = plt.subplots(figsize=(8.8, max(4.5, 0.22 * len(visited_states))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axis.set_yticks(range(len(visited_states)))
    axis.set_yticklabels(
        [f"{state} ({count}/{len(seeds)})" for state, count in zip(visited_states, coverage)],
        fontsize=7,
    )
    axis.set_ylabel("State index (models that visited / total models)")
    axis.set_xticks(range(len(ACTION_NAMES)))
    axis.set_xticklabels(ACTION_NAMES, fontsize=9)
    axis.set_title("Greedy policy share by visited state across independent models")
    fig.colorbar(image, ax=axis, label="Share among models that visited state")
    return _finish(fig, Path(path))


def create_all_figures(
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    histories: Sequence[object],
    q_table_rows: Sequence[Dict[str, object]],
    figures_dir: Path,
) -> List[Path]:
    """Create the six figures used by the core experiment."""
    figures_dir = Path(figures_dir)
    return [
        plot_metric_comparison(metrics, figures_dir / "metric_comparison.png"),
        plot_metric_by_family(metrics, figures_dir / "metric_by_family.png"),
        plot_policy_selection(decisions, figures_dir / "policy_selection.png"),
        plot_reward_by_family(decisions, figures_dir / "reward_by_family.png"),
        plot_training_curve(histories, figures_dir / "training_curve.png"),
        plot_state_space(q_table_rows, figures_dir / "state_space.png"),
    ]
