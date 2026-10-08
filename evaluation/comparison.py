"""Paired result tables for the four baselines and learned offline selector."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES
from errors import ValidationError
from experiments.evaluate import SELECTOR_REGIME, BASELINE_REGIME

__all__ = [
    "COST_METRICS",
    "BENEFIT_METRICS",
    "ALL_METRICS",
    "METRIC_LABELS",
    "policy_summary",
    "family_summary",
    "selector_ratio_table",
    "policy_selection_table",
    "state_occupancy_table",
    "best_policy_per_family",
]

COST_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "context_switches",
)
BENEFIT_METRICS: Tuple[str, ...] = ("cpu_utilization", "throughput")
ALL_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "throughput",
    "context_switches",
)
METRIC_LABELS: Dict[str, str] = {
    "avg_waiting_time": "Mean waiting time",
    "avg_turnaround_time": "Mean turnaround time",
    "avg_response_time": "Mean response time",
    "cpu_utilization": "CPU utilization (%)",
    "throughput": "Throughput (processes/time unit)",
    "context_switches": "Context switches per workload",
}

_PAIR_KEYS = ("training_seed", "family", "repetition")
_WORKLOAD_KEYS = ("family", "repetition", "workload_fingerprint")


def _check_frame(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Validate a tidy metric frame and return the rows of one regime."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationError("metrics frame is empty")
    required = (*_PAIR_KEYS, "policy", "regime", "workload_fingerprint", *ALL_METRICS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValidationError(f"metrics frame is missing columns {missing}")
    subset = frame[frame["regime"] == regime]
    if subset.empty:
        raise ValidationError(f"metrics frame has no rows for regime {regime!r}")
    return subset


def _observations(subset: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Avoid counting the same baseline workload once per training seed."""
    if regime == BASELINE_REGIME:
        return subset.drop_duplicates([*_WORKLOAD_KEYS, "policy"])
    return subset


def _describe(frame: pd.DataFrame, group_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Return clearly named means and sample standard deviations."""
    grouped = frame.groupby(list(group_columns), sort=False)[list(metrics)]
    means = grouped.mean().add_suffix("_mean")
    standard_deviations = grouped.std(ddof=1).fillna(0.0).add_suffix("_sd")
    counts = grouped.size().rename("observations")
    return pd.concat([means, standard_deviations, counts], axis=1)


def policy_summary(
    metrics: pd.DataFrame,
    regime: str = BASELINE_REGIME,
    metrics_to_show: Sequence[str] = ALL_METRICS,
) -> pd.DataFrame:
    """Report means and sample SDs by policy.

    Baseline rows are deduplicated to the 1-per-workload observations. Selector rows retain
    all model-seed × workload pairs, exposing both workload and training-seed variability.
    """
    subset = _observations(_check_frame(metrics, regime), regime)
    return _describe(subset, ["policy"], metrics_to_show)


def family_summary(
    metrics: pd.DataFrame,
    regime: str = BASELINE_REGIME,
    metrics_to_show: Sequence[str] = ALL_METRICS,
) -> pd.DataFrame:
    """Report mean and sample SD by ``(family, policy)`` under the same deduplication."""
    subset = _observations(_check_frame(metrics, regime), regime)
    return _describe(subset, ["family", "policy"], metrics_to_show)


def selector_ratio_table(
    metrics: pd.DataFrame,
    selector_regime: str = SELECTOR_REGIME,
    baselines: Iterable[str] = tuple(ACTION_NAMES),
    metrics_to_show: Sequence[str] = ALL_METRICS,
) -> pd.DataFrame:
    """Compute paired baseline/selector comparisons at model-seed × workload level.

    ``mean_ratio`` is the arithmetic mean of per-pair ``baseline / selector`` ratios; the
    table also exposes means, differences, pair count, and the count of unique workloads so
    duplicated baseline rows are never mislabeled as distinct workloads.
    """
    baselines_frame = _check_frame(metrics, BASELINE_REGIME)
    selector_frame = _check_frame(metrics, selector_regime)
    rows: List[Dict[str, object]] = []
    join_keys = list(_PAIR_KEYS)
    unique_count = int(
        metrics[metrics["regime"] == BASELINE_REGIME]
        .drop_duplicates(list(_WORKLOAD_KEYS))[list(_WORKLOAD_KEYS)]
        .shape[0]
    )

    for policy in baselines:
        baseline_rows = baselines_frame[baselines_frame["policy"] == policy]
        if baseline_rows.empty:
            raise ValidationError(f"no baseline rows for policy {policy!r}")
        baseline = baseline_rows.set_index(join_keys)[list(metrics_to_show)]
        selector = selector_frame.set_index(join_keys)[list(metrics_to_show)]
        if baseline.index.has_duplicates or selector.index.has_duplicates:
            raise ValidationError("metric frame has duplicate model/workload policy rows")
        joined = baseline.join(selector, lsuffix="_baseline", rsuffix="_selector", how="inner")
        if len(joined) != len(baseline_rows) or len(joined) != len(selector_frame):
            raise ValidationError(
                f"baseline {policy!r} and selector rows are not paired one-to-one"
            )
        for metric in metrics_to_show:
            baseline_values = joined[f"{metric}_baseline"]
            selector_values = joined[f"{metric}_selector"]
            safe_selector = selector_values.where(selector_values != 0.0, other=np.nan)
            ratios = baseline_values / safe_selector
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean_baseline": float(baseline_values.mean()),
                    "mean_selector": float(selector_values.mean()),
                    "mean_difference": float((baseline_values - selector_values).mean()),
                    "mean_ratio": float(ratios.mean(skipna=True)) if ratios.notna().any() else float("nan"),
                    "model_workload_pairs": int(len(joined)),
                    "unique_workloads": unique_count,
                    "ratio_pairs": int(ratios.notna().sum()),
                }
            )
    return pd.DataFrame(rows).set_index(["policy", "metric"])


def policy_selection_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Count learned choices per condition without calling repeated rows workloads."""
    required = {
        "training_seed",
        "family",
        "repetition",
        "workload_fingerprint",
        "policy_name",
        "reward",
        "state_seen_in_training",
    }
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise ValidationError("decisions frame is empty")
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValidationError(f"decisions frame is missing columns {missing}")

    rows: List[Dict[str, object]] = []
    for family, group in decisions.groupby("family", sort=False):
        counts = group["policy_name"].value_counts()
        total = len(group)
        record: Dict[str, object] = {
            "family": family,
            "model_decisions": total,
            "unique_workloads": int(group.drop_duplicates(list(_WORKLOAD_KEYS)).shape[0]),
            "training_seeds": int(group["training_seed"].nunique()),
            "mean_reward": float(group["reward"].mean()),
            "sd_reward": float(group["reward"].std(ddof=1)) if total > 1 else 0.0,
            "unseen_state_decisions": int((~group["state_seen_in_training"]).sum()),
            "state_seen_rate": float(group["state_seen_in_training"].mean()),
        }
        for name in ACTION_NAMES:
            count = int(counts.get(name, 0))
            record[f"count_{name}"] = count
            record[f"share_{name}"] = count / total if total else 0.0
        rows.append(record)
    return pd.DataFrame(rows).set_index("family")


def state_occupancy_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique workload states and model coverage by condition."""
    required = {
        "training_seed",
        "family",
        "repetition",
        "workload_fingerprint",
        "state_index",
        "state_burst_profile",
        "state_burst_dispersion",
        "state_offered_load",
        "state_priority_spread",
        "state_seen_in_training",
    }
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        raise ValidationError("decisions frame is empty")
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValidationError(f"decisions frame is missing columns {missing}")

    unique = decisions.drop_duplicates(list(_WORKLOAD_KEYS))
    unique_grouped = unique.groupby("family", sort=False)
    model_grouped = decisions.groupby("family", sort=False)
    table = pd.DataFrame(
        {
            "unique_workloads": unique_grouped.size(),
            "distinct_states": unique_grouped["state_index"].nunique(),
            "model_decisions": model_grouped.size(),
            "training_seeds": model_grouped["training_seed"].nunique(),
            "state_seen_rate": model_grouped["state_seen_in_training"].mean(),
            "unseen_state_decisions": (~decisions["state_seen_in_training"]).groupby(
                decisions["family"]
            ).sum(),
            "mean_burst_profile": unique_grouped["state_burst_profile"].mean(),
            "mean_burst_dispersion": unique_grouped["state_burst_dispersion"].mean(),
            "mean_offered_load": unique_grouped["state_offered_load"].mean(),
            "mean_priority_spread": unique_grouped["state_priority_spread"].mean(),
        }
    )
    return table


def best_policy_per_family(
    metrics: pd.DataFrame,
    regime: str = BASELINE_REGIME,
    metric: str = "avg_waiting_time",
) -> pd.DataFrame:
    """Return the conventional policy with the lowest family mean for ``metric``."""
    if metric not in ALL_METRICS:
        raise ValidationError(f"unknown metric {metric!r}")
    subset = _observations(_check_frame(metrics, regime), regime)
    means = subset.groupby(["family", "policy"], sort=False)[metric].mean().reset_index()
    winners = (
        means.sort_values(["family", metric, "policy"])
        .groupby("family", sort=False)
        .first()
        .rename(columns={"policy": "best_policy", metric: "value"})
    )
    return winners
