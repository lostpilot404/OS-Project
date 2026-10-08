"""Reward function of the policy-selection agent.

Reward definition
-----------------
The reward expresses the project objective -- improving scheduling performance -- as a
single scalar.  It is defined *relative to the four conventional policies evaluated on
the same workload*, so that it is invariant to the overall difficulty of a workload and
so that the agent cannot be rewarded for a workload being easy.

For a workload, let ``reference_m`` be the mean of metric ``m`` over the four
conventional policies and ``chosen_m`` the value of metric ``m`` under the policy the
agent chose.  Every metric becomes a scale-free term ``term_m = chosen_m / reference_m``,
so ``term_m = 1`` means "exactly as good as the average conventional policy".  Terms are
clipped to ``reward.reference_clip`` to bound the influence of any single metric, and a
term whose reference mean is zero is defined as ``1`` (neutral); this makes the reward of
an empty workload exactly ``0``.

Each metric contributes its *deviation from the average conventional policy*:

``reward = w_util * (util_term - 1) + w_throughput * (throughput_term - 1)
           + w_waiting * (1 - waiting_term) + w_turnaround * (1 - turnaround_term)
           + w_response * (1 - response_term) + w_switches * (1 - switches_term)``

so the reward is exactly ``0`` when the chosen policy performs like the average of the
four conventional policies, positive when it performs better and negative when it
performs worse.  Ranking policies does not depend on this centring (adding the constant
``-1`` to every term cannot change an argmax); the deviation form is used because it makes
the reward readable.  All weights live in :class:`config.RewardConfig` and sum to 1, so a
cost metric and a benefit metric of equal weight influence the reward equally.  Context
switches are scaled per process before comparison, because only their count relative to
the number of processes is comparable across workload sizes.

The reference for the Round-Robin quantum controller is classic Round Robin on the same
workload (see :func:`compute_reward_against_reference`).
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
    "compute_reward_against_reference",
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
    missing = [action for action in range(len(ACTION_NAMES)) if action not in attempts]
    if missing:
        raise ValidationError(f"attempts is missing actions {missing}")
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


def compute_reward_against_reference(
    config: RewardConfig,
    reference: WorkloadMetrics,
    candidate: WorkloadMetrics,
) -> float:
    """Reward a candidate run against one reference run of the same workload.

    Used by the Round-Robin quantum controller, whose reference is classic Round Robin
    (quantum multiplier 1.0) on the same workload.

    Args:
        config: Reward weights and clipping.
        reference: Metrics of the reference run.
        candidate: Metrics of the run being scored.

    Returns:
        The scalar reward; ``0`` means "as good as the reference".

    Raises:
        ValidationError: If the two runs are not on the same workload.
    """
    _check_reward_config(config)
    if reference.workload_fingerprint != candidate.workload_fingerprint:
        raise ValidationError(
            "reference and candidate must be evaluated on the same workload, got "
            f"{reference.workload_fingerprint} and {candidate.workload_fingerprint}"
        )
    reference_values = tuple(getattr(reference, metric) for metric in REWARD_METRICS)
    candidate_values = tuple(getattr(candidate, metric) for metric in REWARD_METRICS)
    return _weighted_reward(config, _normalised_terms(config, candidate_values, reference_values))


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
