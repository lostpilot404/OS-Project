"""Matplotlib figures for the measured experiment results.

Every figure is built from the frames produced by :mod:`experiments.evaluate`; no figure
contains a value that was not measured.  Figures are written to disk (the project is a
command-line study, not a GUI) using the non-interactive ``Agg`` backend, so the code
works on a headless machine.

The figures are:

* ``metric_comparison.png`` -- the six project metrics, per scheduler, across all
  evaluated workloads;
* ``metric_by_family.png`` -- mean waiting time per workload condition and scheduler;
* ``policy_selection.png`` -- how often the agent selected each policy, per condition;
* ``reward_by_family.png`` -- the adaptive scheduler's reward per condition, which shows
  where the learned policy is above or below the four-policy average;
* ``training_curve.png`` -- per-episode reward and exploration rate during training;
* ``state_space.png`` -- which discretised states occurred and which action is greedy
  there;
* ``round_robin_quantum.png`` -- classic Round Robin against Round Robin with the learned
  quantum multiplier (the direct measurement of the quantum controller).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import ACTION_NAMES, ADAPTIVE_LABEL  # noqa: E402
from errors import ValidationError  # noqa: E402
from evaluation.comparison import METRIC_LABELS, policy_selection_table  # noqa: E402
from experiments.evaluate import (  # noqa: E402
    ADAPTIVE_CLASSIC_REGIME,
    BASELINE_REGIME,
    ROUND_ROBIN_LEARNED_QUANTUM_LABEL,
    ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
)

__all__ = [
    "create_all_figures",
    "plot_round_robin_quantum",
    "plot_metric_comparison",
    "plot_metric_by_family",
    "plot_policy_selection",
    "plot_reward_by_family",
    "plot_training_curve",
    "plot_state_space",
]

#: Metrics shown in the six-panel comparison figure.
_COMPARISON_METRICS: Sequence[str] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "throughput",
    "context_switches",
)

#: Colour assigned to the adaptive scheduler so it is recognisable in every figure.
_ADAPTIVE_COLOR = "#c0392b"
_BASELINE_COLOR = "#2c3e50"


def _policy_order(policies: Sequence[str]) -> List[str]:
    """Return the policies in project order, with the adaptive scheduler last."""
    ordered = [name for name in ACTION_NAMES if name in policies]
    if ADAPTIVE_LABEL in policies:
        ordered.append(ADAPTIVE_LABEL)
    return ordered


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    """Raise if any required column is missing from ``frame``."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{where} is missing the columns {missing}")


def _finish(fig, path: Path, close: bool = True):
    """Tighten the layout, save the figure and optionally close it."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    if close:
        plt.close(fig)
    return path


def plot_metric_comparison(
    metrics: pd.DataFrame, path: Path, metrics_to_show: Sequence[str] = _COMPARISON_METRICS
) -> Path:
    """Plot the mean of every project metric per scheduler.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        path: Output file path (``.png``).
        metrics_to_show: The metrics to plot, one panel each.

    Returns:
        The written path.
    """
    _require_columns(metrics, ["policy", *metrics_to_show], "metrics frame")
    policies = _policy_order(list(metrics["policy"].unique()))
    means = metrics.groupby("policy")[list(metrics_to_show)].mean().reindex(policies)

    columns = 3
    rows = int(np.ceil(len(metrics_to_show) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for axis, metric in zip(axes_list, metrics_to_show):
        values = means[metric]
        colors = [_ADAPTIVE_COLOR if name == ADAPTIVE_LABEL else _BASELINE_COLOR for name in values.index]
        axis.bar(range(len(values)), values.to_numpy(), color=colors)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(values.index, rotation=30, ha="right", fontsize=8)
        axis.set_title(f"{METRIC_LABELS.get(metric, metric)} (mean)", fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    for axis in axes_list[len(metrics_to_show) :]:
        axis.axis("off")
    fig.suptitle("Measured scheduler comparison (mean over all evaluated workloads)", fontsize=12)
    return _finish(fig, path)


def plot_metric_by_family(
    metrics: pd.DataFrame, path: Path, metric: str = "avg_waiting_time"
) -> Path:
    """Plot one metric per workload condition and scheduler.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        path: Output file path.
        metric: The metric to plot.

    Returns:
        The written path.
    """
    _require_columns(metrics, ["family", "policy", metric], "metrics frame")
    policies = _policy_order(list(metrics["policy"].unique()))
    families = sorted(metrics["family"].unique())
    frame = (
        metrics.groupby(["family", "policy"])[metric].mean().unstack("policy").reindex(
            index=families, columns=policies
        )
    )

    positions = np.arange(len(families))
    width = 0.8 / len(policies)
    fig, axis = plt.subplots(figsize=(max(8.0, 1.6 * len(families)), 4.6))
    for offset, policy in enumerate(policies):
        axis.bar(
            positions + offset * width - 0.4 + width / 2,
            frame[policy].to_numpy(),
            width=width,
            label=policy,
            color=_ADAPTIVE_COLOR if policy == ADAPTIVE_LABEL else None,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=20, ha="right", fontsize=9)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.set_title(f"{METRIC_LABELS.get(metric, metric)} by workload condition (mean)", fontsize=12)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, path)


def plot_policy_selection(decisions: pd.DataFrame, path: Path) -> Path:
    """Plot how often the adaptive agent selected each policy, per condition.

    Args:
        decisions: The decision rows of an evaluation run.
        path: Output file path.

    Returns:
        The written path.
    """
    table = policy_selection_table(decisions)
    share_columns = [f"share_{name}" for name in ACTION_NAMES]
    families = list(table.index)

    fig, axis = plt.subplots(figsize=(max(8.0, 1.5 * len(families)), 4.6))
    bottom = np.zeros(len(families))
    for name, column in zip(ACTION_NAMES, share_columns):
        values = table[column].to_numpy()
        axis.bar(families, values, bottom=bottom, label=name)
        bottom += values
    axis.set_ylabel("Share of decisions")
    axis.set_ylim(0, 1)
    axis.set_title("Policy selected by the Q-learning agent, per workload condition", fontsize=12)
    axis.tick_params(axis="x", rotation=20, labelsize=9)
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, path)


def plot_reward_by_family(decisions: pd.DataFrame, path: Path) -> Path:
    """Plot the adaptive scheduler's reward per workload condition.

    The reward is zero when the chosen policy matches the average of the four
    conventional policies on the same workload, so the bars show measured performance
    relative to that reference.

    Args:
        decisions: The decision rows of an evaluation run.
        path: Output file path.

    Returns:
        The written path.
    """
    _require_columns(decisions, ["family", "reward"], "decisions frame")
    grouped = decisions.groupby("family")["reward"]
    means = grouped.mean()
    errors = grouped.std(ddof=0).fillna(0.0)

    fig, axis = plt.subplots(figsize=(max(7.5, 1.4 * len(means)), 4.4))
    axis.bar(means.index, means.to_numpy(), yerr=errors.to_numpy(), capsize=4, color=_ADAPTIVE_COLOR)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Reward (0 = average conventional policy)")
    axis.set_title("Adaptive scheduler reward by workload condition (mean +/- spread)", fontsize=12)
    axis.tick_params(axis="x", rotation=20, labelsize=9)
    axis.grid(axis="y", alpha=0.3)
    return _finish(fig, path)


def plot_training_curve(history, path: Path, window: int = 25) -> Path:
    """Plot per-episode training reward and the exploration rate.

    Args:
        history: The :class:`experiments.train.TrainingHistory` of the training run.
        path: Output file path.
        window: Width of the rolling mean applied to the reward curve.

    Returns:
        The written path.
    """
    rewards = np.asarray(history.rewards, dtype=float)
    epsilons = np.asarray(history.epsilons, dtype=float)
    if rewards.size == 0:
        raise ValidationError("training history contains no episodes")
    rolling = (
        pd.Series(rewards).rolling(window=window, min_periods=1).mean().to_numpy()
    )

    fig, axis = plt.subplots(figsize=(9.0, 4.6))
    axis.plot(rewards, color="#bdc3c7", linewidth=0.8, label="reward per episode")
    axis.plot(rolling, color=_ADAPTIVE_COLOR, linewidth=1.8, label=f"rolling mean ({window})")
    axis.axhline(0.0, color="black", linewidth=1, linestyle=":")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Reward")
    axis.set_title("Training: reward per episode (0 = average conventional policy)", fontsize=12)

    twin = axis.twinx()
    twin.plot(epsilons, color="#2980b9", linewidth=1.4, linestyle="--", label="epsilon")
    twin.set_ylabel("Exploration rate (epsilon)")
    twin.set_ylim(0, 1.05)

    lines, labels = axis.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    axis.legend(lines + twin_lines, labels + twin_labels, fontsize=8, loc="lower right")
    axis.grid(alpha=0.3)
    return _finish(fig, path)


def plot_state_space(q_table_rows: Sequence[Dict[str, object]], path: Path) -> Path:
    """Plot the learned Q-table: states against actions.

    Args:
        q_table_rows: Rows as produced by
            :meth:`experiments.train.TrainingResult.q_table_rows`.
        path: Output file path.

    Returns:
        The written path.
    """
    if not q_table_rows:
        raise ValidationError("Q-table is empty")
    states = [int(row["state_index"]) for row in q_table_rows]
    values = np.array(
        [[float(row[f"q_{name}"]) for name in ACTION_NAMES] for row in q_table_rows]
    )
    visits = np.array([int(row["visit_count"]) for row in q_table_rows])
    visited = visits > 0

    fig, axis = plt.subplots(figsize=(8.6, 4.6))
    if visited.any():
        image = axis.imshow(
            values[visited],
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        axis.set_yticks(range(int(visited.sum())))
        axis.set_yticklabels([str(state) for state, used in zip(states, visited) if used], fontsize=7)
        axis.set_ylabel("State index (visited states only)")
        fig.colorbar(image, ax=axis, label="Q-value")
    else:  # pragma: no cover - only reachable with an untrained agent
        axis.text(0.5, 0.5, "no state was visited during training", ha="center")
    axis.set_xticks(range(len(ACTION_NAMES)))
    axis.set_xticklabels(ACTION_NAMES, fontsize=9)
    axis.set_title("Learned Q-values per (state, policy)", fontsize=12)
    return _finish(fig, path)


def plot_round_robin_quantum(metrics: pd.DataFrame, path: Path) -> Path:
    """Plot classic Round Robin against Round Robin with the learned quantum.

    This is the direct measurement of the quantum controller: both variants schedule the
    same workloads, and only the quantum differs.

    Args:
        metrics: Metric rows covering the baseline regime and the learned-quantum regime.
        path: Output file path.

    Returns:
        The written path.
    """
    panels = (
        ("avg_waiting_time", "Mean waiting time"),
        ("avg_turnaround_time", "Mean turnaround time"),
        ("avg_response_time", "Mean response time"),
        ("context_switches", "Context switches per workload"),
    )
    _require_columns(metrics, ["policy", *[metric for metric, _ in panels]], "metrics frame")
    variants = ["Round Robin", ROUND_ROBIN_LEARNED_QUANTUM_LABEL]
    available = [name for name in variants if name in set(metrics["policy"])]
    if len(available) < 2:
        raise ValidationError(
            "the learned-quantum comparison needs both Round Robin variants; found "
            f"{available}"
        )
    means = metrics[metrics["policy"].isin(available)].groupby("policy")[
        [metric for metric, _ in panels]
    ].mean().reindex(available)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8))
    for axis, (metric, title) in zip(np.atleast_1d(axes), panels):
        values = means[metric]
        axis.bar(
            range(len(values)),
            values.to_numpy(),
            color=["#2c3e50", "#16a085"],
        )
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(["classic", "learned"], fontsize=9)
        axis.set_title(f"{title}\n(mean over all workloads)", fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("Round Robin: classic quantum vs learned quantum multiplier", fontsize=12)
    return _finish(fig, path)


def create_all_figures(
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    history,
    q_table_rows: Sequence[Dict[str, object]],
    figures_dir: Path,
) -> List[Path]:
    """Create every figure of the study.

    The main comparison figures use the baseline regime and the adaptive regime that
    keeps classic Round Robin; the quantum controller has its own figure so that the
    extension never mixes into the headline comparison.

    Args:
        metrics: The tidy metric frame of the evaluation run.
        decisions: The decision rows of the evaluation run.
        history: The training history.
        q_table_rows: Rows of the trained Q-table.
        figures_dir: Directory the figures are written to.

    Returns:
        The list of written paths, in creation order.
    """
    figures_dir = Path(figures_dir)
    headline = metrics[
        metrics["regime"].isin([BASELINE_REGIME, ADAPTIVE_CLASSIC_REGIME])
    ]
    written = [
        plot_metric_comparison(headline, figures_dir / "metric_comparison.png"),
        plot_metric_by_family(headline, figures_dir / "metric_by_family.png"),
        plot_policy_selection(decisions, figures_dir / "policy_selection.png"),
        plot_reward_by_family(decisions, figures_dir / "reward_by_family.png"),
        plot_training_curve(history, figures_dir / "training_curve.png"),
        plot_state_space(q_table_rows, figures_dir / "state_space.png"),
    ]
    if ROUND_ROBIN_LEARNED_QUANTUM_REGIME in set(metrics["regime"]):
        quantum_rows = metrics[
            metrics["regime"].isin([BASELINE_REGIME, ROUND_ROBIN_LEARNED_QUANTUM_REGIME])
            & metrics["policy"].isin(["Round Robin", ROUND_ROBIN_LEARNED_QUANTUM_LABEL])
        ]
        written.append(
            plot_round_robin_quantum(quantum_rows, figures_dir / "round_robin_quantum.png")
        )
    return written
