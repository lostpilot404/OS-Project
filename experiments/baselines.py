"""Predeclared non-learning selection baselines.

Three baselines are compared against the Q-learning selectors.  None of them learns
from evaluation data and none may look at counterfactual outcomes at selection time:

* ``best_fixed`` -- the single policy with the lowest mean declared-objective cost over
  the **training** workloads (chosen once, before any evaluation workload is
  generated; see :func:`best_fixed_action`);
* ``fingerprint_rule`` -- a deterministic decision list over workload-fingerprint
  features, fixed in code before the study runs (:func:`fingerprint_rule_action`);
* ``random`` -- uniform choice over the four policies from a seeded generator
  (:func:`random_action`).

The hindsight oracle (best policy per workload) is **not** a baseline: it is used only
to compute regret and is always labelled as an upper bound in the reports.

Predeclared fingerprint rule
----------------------------
1. If ``first_arrival_burst_ratio >= 4`` and ``burst_cv >= 0.5``: **Round Robin** --
   a long head-of-line job plus many short jobs is the head-of-line blocking regime.
2. Else if ``priority_burst_correlation >= 0.6``: **Priority** -- priority order already
   tracks short-job order.
3. Else if ``burst_cv <= 0.25``: **FCFS** -- near-uniform bursts leave little for
   short-job ordering to gain.
4. Else: **SJF** -- burst variability makes short-job ordering worthwhile.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from config import ACTION_NAMES
from errors import ValidationError
from workload.fingerprint import WorkloadFingerprint

__all__ = [
    "RULE_BRANCHES",
    "best_fixed_action",
    "fingerprint_rule_action",
    "random_action",
]

#: Human-readable branches of the predeclared rule, in evaluation order.
RULE_BRANCHES: Tuple[str, ...] = (
    "first_arrival_burst_ratio >= 4 and burst_cv >= 0.5 -> Round Robin",
    "priority_burst_correlation >= 0.6 -> Priority",
    "burst_cv <= 0.25 -> FCFS",
    "otherwise -> SJF",
)


def best_fixed_action(mean_costs: Mapping[int, float]) -> int:
    """Return the action with the lowest mean training cost (ties: lowest index).

    Args:
        mean_costs: Mean declared-objective cost per action over training workloads
            only.  Callers must never pass evaluation/test costs.

    Returns:
        The best fixed action index.

    Raises:
        ValidationError: If the mapping does not cover all four actions.
    """
    expected = set(range(len(ACTION_NAMES)))
    if set(mean_costs) != expected:
        raise ValidationError(
            f"mean_costs must cover actions {sorted(expected)}, got {sorted(mean_costs)}"
        )
    return min(sorted(mean_costs), key=lambda action: mean_costs[action])


def fingerprint_rule_action(
    fingerprint: WorkloadFingerprint,
) -> Tuple[int, str, Dict[str, float]]:
    """Apply the predeclared decision list to a workload fingerprint.

    Args:
        fingerprint: Workload characteristics (process table only).

    Returns:
        ``(action, branch, evidence)`` where ``branch`` is the matched entry of
        :data:`RULE_BRANCHES` and ``evidence`` holds the compared feature values.

    Raises:
        ValidationError: If ``fingerprint`` is not a ``WorkloadFingerprint``.
    """
    if not isinstance(fingerprint, WorkloadFingerprint):
        raise ValidationError(
            f"fingerprint must be a WorkloadFingerprint, got {type(fingerprint).__name__}"
        )
    values = fingerprint.as_dict()
    if values["first_arrival_burst_ratio"] >= 4.0 and values["burst_cv"] >= 0.5:
        return (
            2,
            RULE_BRANCHES[0],
            {
                "first_arrival_burst_ratio": values["first_arrival_burst_ratio"],
                "burst_cv": values["burst_cv"],
            },
        )
    if values["priority_burst_correlation"] >= 0.6:
        return (
            3,
            RULE_BRANCHES[1],
            {"priority_burst_correlation": values["priority_burst_correlation"]},
        )
    if values["burst_cv"] <= 0.25:
        return (
            0,
            RULE_BRANCHES[2],
            {"burst_cv": values["burst_cv"]},
        )
    return (
        1,
        RULE_BRANCHES[3],
        {"burst_cv": values["burst_cv"]},
    )


def random_action(rng: np.random.Generator) -> Tuple[int, Dict[str, float]]:
    """Choose uniformly among the four policies from ``rng``.

    Returns:
        ``(action, evidence)`` with the draw probability recorded.

    Raises:
        ValidationError: If ``rng`` is not a ``numpy.random.Generator``.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValidationError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    action = int(rng.integers(0, len(ACTION_NAMES)))
    return action, {"draw_probability": 1.0 / len(ACTION_NAMES)}
