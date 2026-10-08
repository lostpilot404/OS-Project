"""One consistent reward for offline single-workload policy selection.

On each workload, compute the arithmetic-mean reference of exactly four policies: FCFS,
SJF, Round Robin, and Priority. For each metric, use the chosen-policy/reference ratio,
clipped above at ``reference_clip`` (a zero reference maps to ratio 1). Lower is better
for mean waiting, turnaround and response times and context switches per process; higher
is better for CPU utilization and throughput. The scalar reward is

``sum(w_cost * (1 - ratio_cost)) + sum(w_benefit * (ratio_benefit - 1))``.

The six configured weights sum to one. A reward of zero means the selected result matches
the four-policy arithmetic mean on every metric; a positive reward means the weighted net
is better under this objective, not that every metric improved. This same function is
used for training and interpretation of the evaluation outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from config import ACTION_NAMES, RewardConfig
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics

__all__ = [
    "COST_METRICS",
    "BENEFIT_METRICS",
    "RewardBreakdown",
    "compute_reward",
]

#: Metrics that are better when lower, in the order of :meth:`config.RewardConfig.cost_weights`.
COST_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "context_switches_per_process",
)

#: Metrics that are better when higher, in the order of :meth:`config.RewardConfig.benefit_weights`.
BENEFIT_METRICS: Tuple[str, ...] = ("cpu_utilization", "throughput")

#: All metrics that take part in the reward, costs first.
REWARD_METRICS: Tuple[str, ...] = COST_METRICS + BENEFIT_METRICS


@dataclass(frozen=True)
class RewardBreakdown:
    """The reward of one decision together with the terms that produced it.

    Attributes:
        reward: The scalar reward.
        reference_values: Reference metric values (cost metrics then benefit metrics).
        chosen_values: The chosen run's metric values, same order.
        normalised_terms: The clipped, scale-free terms, same order.
    """

    reward: float
    reference_values: Tuple[float, ...]
    chosen_values: Tuple[float, ...]
    normalised_terms: Tuple[float, ...]


def compute_reward(
    config: RewardConfig,
    attempts: Mapping[int, WorkloadMetrics],
    chosen_action: int,
) -> RewardBreakdown:
    """Reward an action given the results of all four conventional policies.

    Args:
        config: Reward weights and clipping.
        attempts: Metrics of every conventional policy on the *same* workload, keyed by
            action index.  All four actions must be present.
        chosen_action: Action the agent selected.

    Returns:
        The reward and its constituent terms.  A policy that matches the four-policy mean
        on every metric scores exactly ``0``.

    Raises:
        ValidationError: If the configuration, the attempts mapping or the workload
            identity is invalid.
    """
    _check_reward_config(config)
    expected_actions = set(range(len(ACTION_NAMES)))
    supplied_actions = set(attempts)
    if supplied_actions != expected_actions:
        missing = sorted(expected_actions - supplied_actions)
        unexpected = sorted(supplied_actions - expected_actions)
        raise ValidationError(
            f"attempts must contain exactly the four actions; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if chosen_action not in attempts:
        raise ValidationError(f"chosen_action {chosen_action} has no metrics in attempts")

    fingerprints = {metrics.workload_fingerprint for metrics in attempts.values()}
    if len(fingerprints) != 1:
        raise ValidationError(
            "all four policies must be evaluated on the same workload, got fingerprints "
            f"{sorted(fingerprints)}"
        )

    reference = _reference_values(attempts.values())
    chosen = tuple(getattr(attempts[chosen_action], metric) for metric in REWARD_METRICS)
    terms = _normalised_terms(config, chosen, reference)
    return RewardBreakdown(
        reward=_weighted_reward(config, terms),
        reference_values=reference,
        chosen_values=chosen,
        normalised_terms=terms,
    )


def _check_reward_config(config: RewardConfig) -> None:
    """Raise :class:`ValidationError` if ``config`` is not a :class:`config.RewardConfig`."""
    if not isinstance(config, RewardConfig):
        raise ValidationError(f"config must be a RewardConfig, got {type(config).__name__}")


def _reference_values(attempts: Sequence[WorkloadMetrics]) -> Tuple[float, ...]:
    """Return the mean of every reward metric over the given runs."""
    count = len(attempts)
    return tuple(
        sum(getattr(metrics, metric) for metrics in attempts) / count
        for metric in REWARD_METRICS
    )


def _normalised_terms(
    config: RewardConfig, values: Tuple[float, ...], reference: Tuple[float, ...]
) -> Tuple[float, ...]:
    """Return the clipped ``value / reference`` term of every reward metric."""
    return tuple(
        _normalise(value, reference_value, config.reference_clip)
        for value, reference_value in zip(values, reference)
    )


def _normalise(value: float, reference: float, clip: float) -> float:
    """Return the clipped ratio ``value / reference``, or ``1`` if the reference is 0."""
    if reference <= 0.0:
        return 1.0
    return min(clip, value / reference)


def _weighted_reward(config: RewardConfig, terms: Tuple[float, ...]) -> float:
    """Combine normalised terms into the scalar reward using the configured weights.

    Cost terms contribute ``weight * (1 - term)`` and benefit terms ``weight * (term - 1)``,
    so that a run matching the reference exactly scores ``0``.
    """
    cost_terms = terms[: len(COST_METRICS)]
    benefit_terms = terms[len(COST_METRICS) :]
    costs = sum(
        weight * (1.0 - term) for weight, term in zip(config.cost_weights(), cost_terms)
    )
    benefits = sum(
        weight * (term - 1.0) for weight, term in zip(config.benefit_weights(), benefit_terms)
    )
    return costs + benefits
