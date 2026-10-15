"""Transparent objectives for offline policy selection.

Two objectives live side by side; the risk-aware objective never silently replaces the
original one.  Both are computed **after** an action is fixed (as the training reward or
as post-hoc counterfactual evidence) and both are reported together in the study
artifacts, so the comparison in ``docs/EVALUATION_RESULTS.md`` is measured, not
asserted.

Original objective (``original``)
---------------------------------
The preserved six-metric reward of :mod:`rl.reward`: for each metric, the chosen-policy
value over the arithmetic mean of FCFS, SJF, Round Robin and Priority on the same
workload, with cost metrics (mean waiting, turnaround, response, context switches per
process) and benefit metrics (CPU utilisation, throughput) combined with fixed weights.
Higher reward is better; the equivalent lower-is-better cost is ``-reward``.

Risk-aware objective (``risk_aware_v1``)
----------------------------------------
A mean-tail cost (lower is better)

``J = w_waiting * n(mean waiting)
    + w_response * n(mean response)
    + w_switch * n(context switches per process)
    + w_tail * n(p95 per-process waiting time)``

with normalisation ``n(x) = clip(x / reference, 0, reference_clip)`` where ``reference``
is the arithmetic mean of the metric over the four fixed policies on the same workload
(a zero reference maps to ``n = 1``).  Default weights are ``0.35 / 0.25 / 0.15 /
0.25`` and ``reference_clip = 2`` (see :class:`config.RiskObjectiveConfig`).

Risk criterion
--------------
The fourth term is the explicit risk criterion: the nearest-rank 95th percentile of
per-process waiting times.  A quarter of the objective weight penalises tail waiting, so
a policy whose mean is good because a few jobs suffer extreme waiting is penalised.  The
objective is therefore mean-tail and risk-averse with respect to waiting-time outliers.

Interpretation of the scale
---------------------------
Because the weights sum to one and every term is a clipped ratio, a policy that matches
the four-policy reference exactly on every metric scores ``J = 1``; ``J < 1`` means the
weighted normalised net is better than the four-policy average.  The reward fed to
Q-learning is ``-J`` (higher is better).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from config import ACTION_NAMES, RewardConfig, RiskObjectiveConfig
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from rl.reward import REWARD_METRICS, compute_reward
from workload.models import ScheduleResult

__all__ = [
    "OBJECTIVE_ORIGINAL",
    "OBJECTIVE_RISK_AWARE",
    "OBJECTIVE_NAMES",
    "RISK_OBJECTIVE_METRICS",
    "PolicyOutcome",
    "ObjectiveBreakdown",
    "outcome_from_result",
    "percentile_waiting_time",
    "compute_objective",
    "compute_all_objectives",
]

#: The preserved original six-metric reward objective.
OBJECTIVE_ORIGINAL = "original"

#: The risk-aware mean-tail objective documented in this module.
OBJECTIVE_RISK_AWARE = "risk_aware_v1"

#: Every implemented objective name.
OBJECTIVE_NAMES: Tuple[str, ...] = (OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE)

#: Metrics of the risk-aware objective, in term order.
RISK_OBJECTIVE_METRICS: Tuple[str, ...] = (
    "avg_waiting_time",
    "avg_response_time",
    "context_switches_per_process",
    "p95_waiting_time",
)


def percentile_waiting_time(result: ScheduleResult, quantile: float) -> float:
    """Nearest-rank quantile of the per-process waiting times of a schedule.

    For ``n`` processes the index is ``max(0, ceil(quantile * n) - 1)`` of the ascending
    waiting times, matching :meth:`workload.models.Workload.percentile_burst_time`.

    Raises:
        ValidationError: If ``result`` is not a ``ScheduleResult`` or ``quantile`` is
            not in ``(0, 1)``.
    """
    if not isinstance(result, ScheduleResult):
        raise ValidationError(f"result must be a ScheduleResult, got {type(result).__name__}")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ValidationError(f"quantile must be a number, got {quantile!r}")
    if not 0.0 < float(quantile) < 1.0:
        raise ValidationError(f"quantile must lie in (0, 1), got {quantile!r}")
    waitings = sorted(outcome.waiting_time for outcome in result.outcomes)
    if not waitings:
        return 0.0
    index = max(0, math.ceil(float(quantile) * len(waitings)) - 1)
    return float(waitings[index])


@dataclass(frozen=True)
class PolicyOutcome:
    """Metrics of one policy run plus the tail statistic of the risk objective.

    Attributes:
        metrics: The six standard metrics of the run.
        p95_waiting_time: Nearest-rank 95th percentile of per-process waiting times
            (the declared risk statistic of ``risk_aware_v1``).
    """

    metrics: WorkloadMetrics
    p95_waiting_time: float

    @property
    def workload_fingerprint(self) -> str:
        """Fingerprint of the workload this outcome was measured on."""
        return self.metrics.workload_fingerprint

    def as_dict(self) -> Dict[str, object]:
        """Flat table row: metric fields plus the tail statistic."""
        return {**self.metrics.as_dict(), "p95_waiting_time": self.p95_waiting_time}


def outcome_from_result(result: ScheduleResult) -> PolicyOutcome:
    """Compute the policy outcome (metrics + p95 waiting) of one schedule.

    Raises:
        ValidationError: If ``result`` is not a ``ScheduleResult``.
    """
    return PolicyOutcome(
        metrics=compute_metrics(result),
        p95_waiting_time=percentile_waiting_time(result, 0.95),
    )


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """The objective cost of one decision with every constituent term.

    Attributes:
        objective: Objective name (:data:`OBJECTIVE_ORIGINAL` or
            :data:`OBJECTIVE_RISK_AWARE`).
        action: Action that was scored.
        cost: Lower-is-better objective value.  For ``original`` this is ``-reward``;
            for ``risk_aware_v1`` it is ``J``.
        reward: Higher-is-better value used as the Q-learning training signal
            (``reward == -cost`` for ``risk_aware_v1``; the original reward for
            ``original``).
        metric_names: Metric of every entry of ``terms``/``chosen_values``.
        terms: Normalised term of every metric (ratios, clipped).
        reference_values: Four-policy arithmetic mean of every metric on this workload.
        chosen_values: Chosen policy's raw metric values.
    """

    objective: str
    action: int
    cost: float
    reward: float
    metric_names: Tuple[str, ...]
    terms: Tuple[float, ...]
    reference_values: Tuple[float, ...]
    chosen_values: Tuple[float, ...]

    def as_dict(self) -> Dict[str, object]:
        """Flat record of the breakdown with per-metric terms."""
        record: Dict[str, object] = {
            "objective": self.objective,
            "action": self.action,
            "objective_cost": self.cost,
            "objective_reward": self.reward,
        }
        for name, term, chosen, reference in zip(
            self.metric_names, self.terms, self.chosen_values, self.reference_values
        ):
            record[f"term_{name}"] = term
            record[f"chosen_{name}"] = chosen
            record[f"reference_{name}"] = reference
        return record


def compute_objective(
    objective: str,
    attempts: Mapping[int, PolicyOutcome],
    chosen_action: int,
    reward_config: RewardConfig | None = None,
    risk_config: RiskObjectiveConfig | None = None,
) -> ObjectiveBreakdown:
    """Score one action under the named objective.

    Args:
        objective: :data:`OBJECTIVE_ORIGINAL` or :data:`OBJECTIVE_RISK_AWARE`.
        attempts: Outcomes of all four fixed policies on the *same* workload, keyed by
            action index.  Used only after the action is fixed (training reward or
            post-hoc evaluation); never a selector input.
        chosen_action: Action that was selected.
        reward_config: Weights of the original objective (defaults apply when omitted).
        risk_config: Weights of the risk-aware objective (defaults apply when omitted).

    Returns:
        The objective breakdown of ``chosen_action``.

    Raises:
        ValidationError: On an unknown objective, malformed attempts, or a chosen
            action without outcomes.
    """
    if objective not in OBJECTIVE_NAMES:
        raise ValidationError(
            f"objective must be one of {list(OBJECTIVE_NAMES)}, got {objective!r}"
        )
    _check_attempts(attempts)
    if chosen_action not in attempts:
        raise ValidationError(f"chosen_action {chosen_action} has no outcomes in attempts")

    if objective == OBJECTIVE_ORIGINAL:
        return _original_breakdown(
            reward_config or RewardConfig(), attempts, chosen_action
        )
    return _risk_breakdown(risk_config or RiskObjectiveConfig(), attempts, chosen_action)


def compute_all_objectives(
    attempts: Mapping[int, PolicyOutcome],
    reward_config: RewardConfig | None = None,
    risk_config: RiskObjectiveConfig | None = None,
) -> Dict[str, Dict[int, ObjectiveBreakdown]]:
    """Score every action under both objectives (post-hoc comparison evidence).

    Returns:
        ``{objective_name: {action: breakdown}}``.

    Raises:
        ValidationError: If ``attempts`` is malformed.
    """
    _check_attempts(attempts)
    return {
        objective: {
            action: compute_objective(objective, attempts, action, reward_config, risk_config)
            for action in sorted(attempts)
        }
        for objective in OBJECTIVE_NAMES
    }


# --------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------


def _check_attempts(attempts: Mapping[int, PolicyOutcome]) -> None:
    if not isinstance(attempts, Mapping):
        raise ValidationError(f"attempts must be a mapping, got {type(attempts).__name__}")
    expected = set(range(len(ACTION_NAMES)))
    supplied = set(attempts)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValidationError(
            f"attempts must contain exactly the four actions; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for action, outcome in attempts.items():
        if not isinstance(outcome, PolicyOutcome):
            raise ValidationError(
                f"attempt {action} must be a PolicyOutcome, got {type(outcome).__name__}"
            )
    fingerprints = {outcome.workload_fingerprint for outcome in attempts.values()}
    if len(fingerprints) != 1:
        raise ValidationError(
            "all four policies must be evaluated on the same workload, got fingerprints "
            f"{sorted(fingerprints)}"
        )


def _risk_reference_values(attempts: Sequence[PolicyOutcome]) -> Tuple[float, ...]:
    return tuple(
        sum(_risk_value(outcome, metric) for outcome in attempts) / len(attempts)
        for metric in RISK_OBJECTIVE_METRICS
    )


def _risk_value(outcome: PolicyOutcome, metric: str) -> float:
    if metric == "p95_waiting_time":
        return outcome.p95_waiting_time
    return float(getattr(outcome.metrics, metric))


def _normalise(value: float, reference: float, clip: float) -> float:
    if reference <= 0.0:
        return 1.0
    return min(clip, value / reference)


def _risk_breakdown(
    config: RiskObjectiveConfig,
    attempts: Mapping[int, PolicyOutcome],
    chosen_action: int,
) -> ObjectiveBreakdown:
    chosen = attempts[chosen_action]
    reference = _risk_reference_values(list(attempts.values()))
    chosen_values = tuple(_risk_value(chosen, metric) for metric in RISK_OBJECTIVE_METRICS)
    terms = tuple(
        _normalise(value, ref, config.reference_clip)
        for value, ref in zip(chosen_values, reference)
    )
    cost = sum(weight * term for weight, term in zip(config.weights(), terms))
    return ObjectiveBreakdown(
        objective=OBJECTIVE_RISK_AWARE,
        action=chosen_action,
        cost=cost,
        reward=-cost,
        metric_names=RISK_OBJECTIVE_METRICS,
        terms=terms,
        reference_values=reference,
        chosen_values=chosen_values,
    )


def _original_breakdown(
    config: RewardConfig,
    attempts: Mapping[int, PolicyOutcome],
    chosen_action: int,
) -> ObjectiveBreakdown:
    breakdown = compute_reward(
        config,
        {action: outcome.metrics for action, outcome in attempts.items()},
        chosen_action,
    )
    return ObjectiveBreakdown(
        objective=OBJECTIVE_ORIGINAL,
        action=chosen_action,
        cost=-breakdown.reward,
        reward=breakdown.reward,
        metric_names=REWARD_METRICS,
        terms=breakdown.normalised_terms,
        reference_values=breakdown.reference_values,
        chosen_values=breakdown.chosen_values,
    )
