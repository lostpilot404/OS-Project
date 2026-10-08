"""Comparison tables built from measured evaluation results.

Everything here reads the tidy metric frame produced by :mod:`experiments.evaluate` and
returns summary frames.  No value is ever entered by hand: every number in every table
comes from a scheduling run that actually happened.

Aggregation convention (documented, because two conventions are defensible)
---------------------------------------------------------------------------
Per-workload comparisons are averaged as *ratios* rather than as absolute differences,
because a ten-time-unit gap is negligible on one workload condition and decisive on
another.  Tables therefore report, for each baseline and each metric, the mean over
workloads of ``baseline / adaptive`` and of ``baseline - adaptive``, so that a ratio above
``1`` for a cost metric means the baseline was worse than the adaptive scheduler.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES
from errors import ValidationError
from experiments.evaluate import ADAPTIVE_REGIME, BASELINE_REGIME

__all__ = [
    "COST_METRICS",
    "BENEFIT_METRICS",
    "ALL_METRICS",
    "METRIC_LABELS",
    "policy_summary",
    "family_summary",
    "adaptive_ratio_table",
    "policy_selection_table",
    "state_occupancy_table",
    "oracle_agreement_table",
]

#: Metrics where lower is better.
COST_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "context_switches_per_process",
    "context_switches",
)

#: Metrics where higher is better.
BENEFIT_METRICS: Tuple[str, ...] = ("cpu_utilization", "throughput")

#: All metrics compared across schedulers, in report order.
ALL_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "throughput",
    "context_switches",
    "context_switches_per_process",
)

#: Display labels of the metrics.
METRIC_LABELS: Dict[str, str] = {
    "avg_waiting_time": "Mean waiting time",
    "avg_turnaround_time": "Mean turnaround time",
    "avg_response_time": "Mean response time",
    "cpu_utilization": "CPU utilization (%)",
    "throughput": "Throughput (processes/time unit)",
    "context_switches": "Context switches (per workload)",
    "context_switches_per_process": "Context switches per process",
}


def _check_frame(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Return the rows of one regime, raising if the frame is empty or malformed."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationError("metrics frame is empty")
    for column in ("family", "repetition", "policy", "regime", "workload_fingerprint"):
        if column not in frame.columns:
            raise ValidationError(f"metrics frame is missing the column {column!r}")
    subset = frame[frame["regime"] == regime]
    if subset.empty:
        raise ValidationError(f"metrics frame has no rows for regime {regime!r}")
    return subset


def policy_summary(
    metrics: pd.DataFrame, regime: str = BASELINE_REGIME, metrics_to_show: Sequence[str] = ALL_METRICS
) -> pd.DataFrame:
    """Return the mean of every metric per policy.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        regime: Which regime to summarise.
        metrics_to_show: Which metric columns to include.

    Returns:
        A frame indexed by policy with one column per metric.
    """
    subset = _check_frame(metrics, regime)
    return subset.groupby("policy", sort=False)[list(metrics_to_show)].mean()


def family_summary(
    metrics: pd.DataFrame, regime: str = BASELINE_REGIME, metrics_to_show: Sequence[str] = ALL_METRICS
) -> pd.DataFrame:
    """Return the mean of every metric per ``(family, policy)``.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        regime: Which regime to summarise.
        metrics_to_show: Which metric columns to include.

    Returns:
        A frame indexed by ``(family, policy)`` with one column per metric.
    """
    subset = _check_frame(metrics, regime)
    return subset.groupby(["family", "policy"], sort=False)[list(metrics_to_show)].mean()


def adaptive_ratio_table(
    metrics: pd.DataFrame,
    adaptive_regime: str = ADAPTIVE_REGIME,
    baselines: Iterable[str] = tuple(ACTION_NAMES),
    metrics_to_show: Sequence[str] = ALL_METRICS,
) -> pd.DataFrame:
    """Compare every baseline with the adaptive scheduler workload by workload.

    For each baseline and metric the table reports the mean of ``baseline - adaptive`` and
    of ``baseline / adaptive`` over the evaluated workloads.  For a cost metric a
    ``ratio > 1`` means the baseline was worse than the adaptive scheduler; for a benefit
    metric a ``ratio < 1`` means the baseline was worse.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        adaptive_regime: Which adaptive regime to compare against.
        baselines: Baseline policy names to compare.
        metrics_to_show: Which metric columns to include.

    Returns:
        A frame indexed by ``(policy, metric)`` with the columns ``mean_baseline``,
        ``mean_adaptive``, ``mean_difference``, ``mean_ratio``, ``workloads`` (paired
        workloads compared) and ``ratio_workloads`` (pairs with a non-zero adaptive
        value, i.e. the pairs a ratio can be formed from).
    """
    baselines_frame = _check_frame(metrics, BASELINE_REGIME)
    adaptive_frame = _check_frame(metrics, adaptive_regime)

    adaptive_by_workload = adaptive_frame.set_index(["family", "repetition"])[list(metrics_to_show)]
    rows: List[Dict[str, object]] = []
    for policy in baselines:
        baseline_rows = baselines_frame[baselines_frame["policy"] == policy]
        if baseline_rows.empty:
            raise ValidationError(f"no baseline rows for policy {policy!r}")
        joined = baseline_rows.set_index(["family", "repetition"])[list(metrics_to_show)].join(
            adaptive_by_workload, lsuffix="_baseline", rsuffix="_adaptive", how="inner"
        )
        if len(joined) != len(baseline_rows):
            raise ValidationError(
                f"baseline {policy!r} and the adaptive scheduler were not evaluated on the "
                "same workloads"
            )
        for metric in metrics_to_show:
            baseline_values = joined[f"{metric}_baseline"]
            adaptive_values = joined[f"{metric}_adaptive"]
            difference = (baseline_values - adaptive_values).mean()
            # A ratio needs a non-zero denominator; workloads where the adaptive value is
            # zero are reported through ratio_workloads instead of being silently dropped.
            safe_adaptive = adaptive_values.where(adaptive_values != 0.0, other=np.nan)
            ratios = baseline_values / safe_adaptive
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean_baseline": baseline_values.mean(),
                    "mean_adaptive": adaptive_values.mean(),
                    "mean_difference": difference,
                    "mean_ratio": float(ratios.mean(skipna=True)) if ratios.notna().any() else float("nan"),
                    "workloads": int(len(joined)),
                    "ratio_workloads": int(ratios.notna().sum()),
                }
            )
    return pd.DataFrame(rows).set_index(["policy", "metric"])


def oracle_agreement_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return how often the agent's choice matched the reward-argmax, per condition.

    The "oracle" is the reward-argmax over the four conventional policies measured on
    the very same workload: the best any single-decision policy could have achieved
    under the declared reward.  Agreement with it is the sharpest available measure of
    how well the learned Q-table reproduces the workload-conditional optimum.

    Args:
        decisions: The decision rows of an evaluation run.

    Returns:
        A frame indexed by ``family`` with ``workloads``, ``matches``, ``agreement_rate``
        and the mean reward gap between the chosen and the oracle policy.
    """
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise ValidationError("decisions frame is empty")
    for column in ("matches_oracle", "chosen_minus_oracle_reward", "policy_name",
                   "oracle_policy_name"):
        if column not in decisions.columns:
            raise ValidationError(f"decisions frame is missing the column {column!r}")
    grouped = decisions.groupby("family")
    table = pd.DataFrame(
        {
            "workloads": grouped.size(),
            "matches": grouped["matches_oracle"].sum(),
            "agreement_rate": grouped["matches_oracle"].mean(),
            "mean_reward_gap_to_oracle": grouped["chosen_minus_oracle_reward"].mean(),
        }
    )
    return table


def policy_selection_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarise which policy the agent selected, per workload condition.

    Args:
        decisions: The decision rows of an evaluation result.

    Returns:
        A frame indexed by ``family`` with the selection count and share of every policy
        plus ``mean_reward`` and ``workloads``.

    Raises:
        ValidationError: If the decisions frame is empty or missing columns.
    """
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise ValidationError("decisions frame is empty")
    for column in ("family", "policy_name", "reward"):
        if column not in decisions.columns:
            raise ValidationError(f"decisions frame is missing the column {column!r}")

    frame = pd.DataFrame(index=pd.Index(sorted(decisions["family"].unique()), name="family"))
    totals = pd.Series(0, index=frame.index, dtype=int)
    for name in ACTION_NAMES:
        column = f"count_{name}"
        frame[column] = (
            decisions[decisions["policy_name"] == name].groupby("family").size()
        ).reindex(frame.index).fillna(0).astype(int)
        totals = totals + frame[column]
    for name in ACTION_NAMES:
        frame[f"share_{name}"] = (frame[f"count_{name}"] / totals).fillna(0.0)
    frame["workloads"] = totals
    frame["mean_reward"] = decisions.groupby("family")["reward"].mean()
    return frame


def state_occupancy_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return how many distinct states and workloads each condition occupied.

    Args:
        decisions: The decision rows of an evaluation result.

    Returns:
        A frame indexed by ``family`` with ``workloads``, ``distinct_states``, how many of
        those states the agent had visited during training (state coverage), the mean
        observed state values and the mean reward.
    """
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise ValidationError("decisions frame is empty")
    for column in ("state_index", "state_visit_count", "reward"):
        if column not in decisions.columns:
            raise ValidationError(f"decisions frame is missing the column {column!r}")
    grouped = decisions.groupby("family")
    visited = decisions.assign(_visited=decisions["state_visit_count"] > 0).groupby("family")
    table = pd.DataFrame(
        {
            "workloads": grouped.size(),
            "distinct_states": grouped["state_index"].nunique(),
            "states_visited_in_training": visited["state_index"].nunique(),
            "workloads_in_unvisited_states": grouped["state_visit_count"].apply(
                lambda counts: int((counts == 0).sum())
            ),
            "mean_burst_profile": grouped["state_burst_profile"].mean(),
            "mean_burst_dispersion": grouped["state_burst_dispersion"].mean(),
            "mean_long_job_share": grouped["state_long_job_share"].mean(),
            "mean_arrival_concentration": grouped["state_arrival_concentration"].mean(),
            "mean_offered_load": grouped["state_offered_load"].mean(),
            "mean_priority_spread": grouped["state_priority_spread"].mean(),
            "mean_priority_burst_alignment": grouped["state_priority_burst_alignment"].mean(),
            "mean_reward": grouped["reward"].mean(),
        }
    )
    return table


def best_policy_per_family(
    metrics: pd.DataFrame, regime: str = BASELINE_REGIME, metric: str = "avg_waiting_time"
) -> pd.DataFrame:
    """Return the conventional policy with the lowest mean value of ``metric`` per family.

    Args:
        metrics: The tidy metric frame of an evaluation run.
        regime: Which regime to summarise.
        metric: The cost metric used to rank the conventional policies.

    Returns:
        A frame indexed by ``family`` with the winning ``policy`` and its mean ``value``.
    """
    if metric not in ALL_METRICS:
        raise ValidationError(f"unknown metric {metric!r}")
    subset = _check_frame(metrics, regime)
    means = subset.groupby(["family", "policy"], sort=False)[metric].mean().reset_index()
    winners = (
        means.sort_values(["family", metric])
        .groupby("family", sort=False)
        .first()
        .rename(columns={"policy": "best_policy", metric: "value"})
    )
    return winners
