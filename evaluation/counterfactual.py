"""Post-decision counterfactual evaluation and hindsight regret.

The offline selector fixes its action from workload features and learned Q-values only
(:mod:`rl.selector`).  This module runs **after** that decision: it executes all four
fixed policies on the identical workload, scores every policy under both declared
objectives, and computes hindsight selection regret

``regret(objective) = cost(selected) - min_a cost(a)``

with the lower-is-better costs of :mod:`rl.objective` (``J`` for ``risk_aware_v1``,
``-reward`` for ``original``).  These outcomes are evaluation evidence; they are never
passed back into the decision function.  Training uses the same four runs as the reward
signal, but only after the action of the episode is fixed.

Isolation contract
------------------
* :meth:`rl.selector.WorkloadPolicySelector.select` receives no policies and cannot
  execute any schedule.
* This module receives an already-fixed action only inside :func:`selection_regret`,
  purely to score it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from config import ACTION_NAMES
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics
from rl.objective import (
    OBJECTIVE_NAMES,
    ObjectiveBreakdown,
    PolicyOutcome,
    compute_all_objectives,
    outcome_from_result,
)
from scheduler.base import SchedulingPolicy
from workload.models import Workload

__all__ = [
    "CounterfactualReport",
    "RegretRecord",
    "run_counterfactuals",
    "selection_regret",
]


@dataclass(frozen=True)
class RegretRecord:
    """Hindsight regret of one selection under one objective.

    Attributes:
        objective: Objective the regret is measured under.
        chosen_action: Action that was selected.
        chosen_cost: Declared-objective cost of the selected action.
        best_action: Hindsight-best action on this workload (lowest cost).
        best_cost: Cost of the hindsight-best action.
        regret: ``chosen_cost - best_cost`` (>= 0 up to floating-point error).
        is_hindsight_best: Whether the selection matched the hindsight-best action.
    """

    objective: str
    chosen_action: int
    chosen_cost: float
    best_action: int
    best_cost: float
    regret: float
    is_hindsight_best: bool

    def as_dict(self) -> Dict[str, object]:
        """Flat record of the regret."""
        return {
            "regret_objective": self.objective,
            "regret_chosen_action": self.chosen_action,
            "regret_chosen_cost": self.chosen_cost,
            "regret_best_action": self.best_action,
            "regret_best_cost": self.best_cost,
            "regret": self.regret,
            "is_hindsight_best": self.is_hindsight_best,
        }


@dataclass(frozen=True)
class CounterfactualReport:
    """Outcomes of all four fixed policies on one workload, scored post hoc.

    Attributes:
        workload_name: Name of the workload.
        workload_fingerprint: Content digest proving every policy saw the same input.
        outcomes: Outcome of every policy, keyed by action index.
        breakdowns: ``{objective: {action: breakdown}}`` for both objectives.
    """

    workload_name: str
    workload_fingerprint: str
    outcomes: Mapping[int, PolicyOutcome]
    breakdowns: Mapping[str, Mapping[int, ObjectiveBreakdown]]

    @property
    def metrics(self) -> Dict[int, WorkloadMetrics]:
        """Metrics of every policy, keyed by action index."""
        return {action: outcome.metrics for action, outcome in self.outcomes.items()}

    def cost(self, objective: str, action: int) -> float:
        """Lower-is-better declared-objective cost of one action.

        Raises:
            ValidationError: If the objective or action is unknown.
        """
        self._check_objective_action(objective, action)
        return self.breakdowns[objective][action].cost

    def best_action(self, objective: str) -> int:
        """Hindsight-best action under one objective (ties go to the lowest index)."""
        self._check_objective(objective)
        costs = {action: breakdown.cost for action, breakdown in self.breakdowns[objective].items()}
        return min(sorted(costs), key=lambda action: costs[action])

    def regret(self, objective: str, chosen_action: int) -> RegretRecord:
        """Hindsight regret of ``chosen_action`` under ``objective``."""
        return selection_regret(self, objective, chosen_action)

    def as_rows(self) -> Tuple[Dict[str, object], ...]:
        """One flat row per policy: metrics, tail statistic and both objectives."""
        rows = []
        for action in sorted(self.outcomes):
            row: Dict[str, object] = {
                "workload_name": self.workload_name,
                "workload_fingerprint": self.workload_fingerprint,
                "action": action,
                "policy_name": ACTION_NAMES[action],
            }
            row.update(self.outcomes[action].as_dict())
            for objective in OBJECTIVE_NAMES:
                breakdown = self.breakdowns[objective][action]
                prefix = "risk_" if objective != OBJECTIVE_NAMES[0] else "orig_"
                row[f"{prefix}objective_cost"] = breakdown.cost
                row[f"{prefix}objective_reward"] = breakdown.reward
                for name, term in zip(breakdown.metric_names, breakdown.terms):
                    row[f"{prefix}term_{name}"] = term
            rows.append(row)
        return tuple(rows)

    def _check_objective(self, objective: str) -> None:
        if objective not in OBJECTIVE_NAMES:
            raise ValidationError(
                f"objective must be one of {list(OBJECTIVE_NAMES)}, got {objective!r}"
            )

    def _check_objective_action(self, objective: str, action: int) -> None:
        self._check_objective(objective)
        if action not in self.outcomes:
            raise ValidationError(f"action {action} has no counterfactual outcome")


def run_counterfactuals(
    workload: Workload,
    policies: Sequence[SchedulingPolicy],
    reward_config=None,
    risk_config=None,
) -> CounterfactualReport:
    """Run all four fixed policies on ``workload`` and score every outcome.

    Call this only after a selection is fixed (training reward or post-hoc
    evaluation).  All policies must map to the four action indices in order and must
    report the exact workload fingerprint.

    Args:
        workload: The already-selected workload.
        policies: The four fixed policies, in action order.
        reward_config: Weights of the original objective (defaults apply).
        risk_config: Weights of the risk-aware objective (defaults apply).

    Returns:
        The counterfactual report with both objectives for every policy.

    Raises:
            ValidationError: If the workload or policies are invalid.
    """
    if not isinstance(workload, Workload):
        raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
    if len(policies) != len(ACTION_NAMES):
        raise ValidationError(
            f"expected exactly {len(ACTION_NAMES)} policies, got {len(policies)}"
        )
    outcomes: Dict[int, PolicyOutcome] = {}
    for action, policy in enumerate(policies):
        if not isinstance(policy, SchedulingPolicy):
            raise ValidationError(
                f"policy at index {action} must be a SchedulingPolicy, "
                f"got {type(policy).__name__}"
            )
        if policy.name != ACTION_NAMES[action]:
            raise ValidationError(
                f"policy at index {action} must be {ACTION_NAMES[action]!r}, "
                f"got {policy.name!r}"
            )
        outcome = outcome_from_result(policy.run(workload))
        if outcome.workload_fingerprint != workload.fingerprint:
            raise ValidationError(
                f"policy {policy.name!r} reported fingerprint "
                f"{outcome.workload_fingerprint} for workload {workload.fingerprint}"
            )
        outcomes[action] = outcome
    breakdowns = compute_all_objectives(outcomes, reward_config, risk_config)
    return CounterfactualReport(
        workload_name=workload.name,
        workload_fingerprint=workload.fingerprint,
        outcomes=outcomes,
        breakdowns=breakdowns,
    )


def selection_regret(
    report: CounterfactualReport,
    objective: str,
    chosen_action: int,
) -> RegretRecord:
    """Compute hindsight selection regret from an existing counterfactual report.

    Args:
        report: Post-decision counterfactual outcomes.
        objective: Objective the regret is measured under.
        chosen_action: The action the selector committed to.

    Returns:
        The regret record (non-negative up to floating-point error).

    Raises:
        ValidationError: If the report, objective or action is invalid.
    """
    if not isinstance(report, CounterfactualReport):
        raise ValidationError(
            f"report must be a CounterfactualReport, got {type(report).__name__}"
        )
    chosen_cost = report.cost(objective, chosen_action)
    best_action = report.best_action(objective)
    best_cost = report.cost(objective, best_action)
    regret = chosen_cost - best_cost
    return RegretRecord(
        objective=objective,
        chosen_action=chosen_action,
        chosen_cost=chosen_cost,
        best_action=best_action,
        best_cost=best_cost,
        regret=regret,
        is_hindsight_best=abs(regret) <= 1e-12,
    )
