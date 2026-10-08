"""Evaluation of the trained adaptive scheduler against the four conventional policies.

Protocol
--------
For every workload condition and every repetition, one workload is generated from the
evaluation seed stream, and **all** schedulers -- FCFS, SJF, Round Robin, Priority and the
adaptive scheduler -- are evaluated on that same workload object.  The workload
fingerprint is carried into every result row, which makes "the baselines saw identical
workloads" a checkable property rather than an assumption.

Three guarantees are enforced while evaluating, and the run fails loudly if any of them is
violated:

1. no evaluation workload was used during training (fingerprints are disjoint);
2. evaluation never writes to the Q-table (the table is compared before and after);
3. every decision of the adaptive scheduler is made greedily (``learned`` is ``False``).

The evaluation is a held-out measurement: the workloads come from the evaluation master
seed, which :meth:`config.ExperimentConfig.validate` requires to differ from both the
training seed and the pre-training verification seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES, ADAPTIVE_LABEL, ExperimentConfig, derive_seed
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from rl.reward import compute_reward
from experiments.train import TrainingResult
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = [
    "EvaluationWorkload",
    "EvaluationResult",
    "build_evaluation_workloads",
    "evaluate",
    "ADAPTIVE_REGIME",
    "BASELINE_REGIME",
]

#: Regime label of the conventional policies.
BASELINE_REGIME = "baseline"
#: Regime label of the adaptive scheduler (the single regime: the greedy Q-table policy).
ADAPTIVE_REGIME = "adaptive"

#: Seed-derivation tag of the evaluation workload stream.
_WORKLOAD_STREAM_TAG = 0


@dataclass(frozen=True)
class EvaluationWorkload:
    """One workload of the evaluation set.

    Attributes:
        family: Workload condition the workload belongs to.
        repetition: 0-based repetition index within the condition.
        workload: The generated workload.
    """

    family: str
    repetition: int
    workload: Workload

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the workload."""
        return self.workload.fingerprint


@dataclass(frozen=True)
class EvaluationResult:
    """Measurements of one evaluation run.

    Attributes:
        metrics: One row per ``(family, repetition, policy, regime)``.
        decisions: One row per adaptive decision.
        families: Workload conditions evaluated, in configuration order.
        repetitions: Repetitions per condition.
        evaluation_seed: Master seed of the evaluation workload stream.
        training_seed: Master seed the agent was trained with.
    """

    metrics: pd.DataFrame
    decisions: pd.DataFrame
    families: Tuple[str, ...]
    repetitions: int
    evaluation_seed: int
    training_seed: int

    def regime(self, regime: str) -> pd.DataFrame:
        """Return the metric rows of one regime.

        Raises:
            ValidationError: If the regime is not present in the results.
        """
        subset = self.metrics[self.metrics["regime"] == regime]
        if subset.empty:
            raise ValidationError(f"no rows for regime {regime!r}")
        return subset

    def summary(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary of the run."""
        states = self.decisions["state_index"]
        visited = self.decisions["state_visit_count"] > 0
        distinct_states = int(states.nunique())
        visited_states = int(states[visited].nunique())
        return {
            "families": list(self.families),
            "repetitions": self.repetitions,
            "evaluation_seed": self.evaluation_seed,
            "training_seed": self.training_seed,
            "rows": int(len(self.metrics)),
            "regimes": sorted(self.metrics["regime"].unique().tolist()),
            "policies": sorted(self.metrics["policy"].unique().tolist()),
            "distinct_workloads": int(self.metrics["workload_fingerprint"].nunique()),
            "distinct_states_evaluated": distinct_states,
            "evaluated_states_visited_during_training": visited_states,
            "evaluated_states_not_visited_during_training": distinct_states - visited_states,
            "decisions_in_unvisited_states": int((~visited).sum()),
            "oracle_agreement_workloads": int(len(self.decisions)),
            "oracle_agreement_matches": int(self.decisions["matches_oracle"].sum()),
            "oracle_agreement_rate": float(self.decisions["matches_oracle"].mean()),
            "mean_chosen_minus_oracle_reward": float(
                self.decisions["chosen_minus_oracle_reward"].mean()
            ),
        }


def build_evaluation_workloads(
    config: ExperimentConfig, generator: WorkloadGenerator
) -> List[EvaluationWorkload]:
    """Generate the evaluation workloads.

    Workloads are generated from the *evaluation* seed, which
    :meth:`config.ExperimentConfig.validate` requires to differ from the training seed.

    Args:
        config: The experiment configuration.
        generator: The workload generator.

    Returns:
        One workload per ``(family, repetition)``, in configuration order.
    """
    workloads: List[EvaluationWorkload] = []
    for family_index, family in enumerate(config.families):
        for repetition in range(config.evaluation.repetitions):
            seed = derive_seed(
                config.evaluation.seed, _WORKLOAD_STREAM_TAG, family_index, repetition
            )
            workloads.append(
                EvaluationWorkload(
                    family=family.name,
                    repetition=repetition,
                    workload=generator.generate(family.name, seed),
                )
            )
    return workloads


def evaluate(
    config: ExperimentConfig,
    training: TrainingResult,
    generator: WorkloadGenerator | None = None,
) -> EvaluationResult:
    """Evaluate the four conventional policies and the trained adaptive scheduler.

    Args:
        config: The experiment configuration the agent was trained with.
        training: The training result (trained agent and scheduler).
        generator: Optional pre-built workload generator.

    Returns:
        The evaluation result, with one metric row per scheduler and workload.

    Raises:
        ValidationError: If any of the three guarantees documented in the module
            docstring is violated.
    """
    if training.config is not config and training.config != config:
        raise ValidationError("the training result was produced with a different configuration")
    generator = generator or WorkloadGenerator(config.families)
    workloads = build_evaluation_workloads(config, generator)

    training_fingerprints = training.training_fingerprints
    leaked = sorted({w.fingerprint for w in workloads} & training_fingerprints)
    if leaked:
        raise ValidationError(
            f"{len(leaked)} evaluation workloads were already used during training: {leaked[:3]}"
        )

    policies = {cls.action_index: cls(config.scheduler) for cls in POLICY_CLASSES}
    rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []

    q_before = np.array(training.agent.q_table, copy=True)
    visits_before = np.array(training.agent.visit_counts, copy=True)

    for item in workloads:
        workload = item.workload
        attempts: Dict[int, WorkloadMetrics] = {}
        for action_index in sorted(policies):
            metrics = compute_metrics(policies[action_index].run(workload))
            attempts[action_index] = metrics
            rows.append(
                _metrics_row(
                    family=item.family,
                    repetition=item.repetition,
                    policy=ACTION_NAMES[action_index],
                    regime=BASELINE_REGIME,
                    metrics=metrics,
                )
            )

        # The reward-argmax over the four conventional policies on this very workload:
        # the best any single-decision policy could achieve under the declared reward.
        # Ties are broken towards the lowest action index, exactly like the agent's
        # greedy tie-break, so the comparison is apples-to-apples.
        rewards = [
            compute_reward(config.reward, attempts, action).reward
            for action in range(len(ACTION_NAMES))
        ]
        oracle_action = max(range(len(ACTION_NAMES)), key=lambda a: (rewards[a], -a))

        decision = training.scheduler.run_evaluation(workload)
        _check_greedy(decision.learned)
        rows.append(
            _metrics_row(
                family=item.family,
                repetition=item.repetition,
                policy=ADAPTIVE_LABEL,
                regime=ADAPTIVE_REGIME,
                metrics=decision.metrics,
            )
        )
        decision_rows.append(
            {
                **decision.as_row(),
                "family": item.family,
                "repetition": item.repetition,
                "regime": ADAPTIVE_REGIME,
                "oracle_action": oracle_action,
                "oracle_policy_name": ACTION_NAMES[oracle_action],
                "oracle_reward": rewards[oracle_action],
                "chosen_minus_oracle_reward": decision.reward - rewards[oracle_action],
                "matches_oracle": decision.action == oracle_action,
            }
        )

    _check_q_table_untouched(q_before, np.array(training.agent.q_table))
    _check_visit_counts_untouched(visits_before, np.array(training.agent.visit_counts))

    metrics_frame = pd.DataFrame(rows)
    decisions_frame = pd.DataFrame(decision_rows)
    _check_identical_workloads(metrics_frame)

    return EvaluationResult(
        metrics=metrics_frame,
        decisions=decisions_frame,
        families=tuple(family.name for family in config.families),
        repetitions=config.evaluation.repetitions,
        evaluation_seed=config.evaluation.seed,
        training_seed=config.training.seed,
    )


def _metrics_row(
    family: str,
    repetition: int,
    policy: str,
    regime: str,
    metrics: WorkloadMetrics,
) -> Dict[str, object]:
    """Build one tidy result row from a metrics object."""
    row = metrics.as_dict()
    row.pop("policy_name")
    row.pop("workload_name")
    return {
        "family": family,
        "repetition": repetition,
        "policy": policy,
        "regime": regime,
        **row,
    }


def _check_greedy(learned: bool) -> None:
    """Raise if an evaluation decision reported that it updated the Q-table."""
    if learned:
        raise ValidationError("evaluation updated the Q-table; evaluation must be greedy only")


def _check_q_table_untouched(before: np.ndarray, after: np.ndarray) -> None:
    """Raise if the Q-table changed during evaluation."""
    if not np.array_equal(before, after):
        raise ValidationError("the Q-table changed during evaluation")


def _check_visit_counts_untouched(before: np.ndarray, after: np.ndarray) -> None:
    """Raise if the visit counts changed during evaluation."""
    if not np.array_equal(before, after):
        raise ValidationError("the state visit counts changed during evaluation")


def _check_identical_workloads(metrics: pd.DataFrame) -> None:
    """Raise if the schedulers of one workload did not all see the same workload."""
    grouped = metrics.groupby(["family", "repetition"], sort=False)["workload_fingerprint"].nunique()
    inconsistent = grouped[grouped > 1]
    if not inconsistent.empty:
        raise ValidationError(
            "some schedulers were evaluated on different workloads: "
            f"{inconsistent.to_dict()}"
        )
