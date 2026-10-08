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

Two adaptive regimes are measured: the headline one uses classic Round Robin with the
fixed configured quantum, and the secondary one lets the Round Robin action use the
learned quantum multiplier (only when the quantum controller is enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES, ADAPTIVE_LABEL, ExperimentConfig, derive_seed
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from experiments.train import TrainingResult
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = [
    "EvaluationWorkload",
    "EvaluationResult",
    "build_evaluation_workloads",
    "evaluate",
    "ADAPTIVE_LEARNED_REGIME",
    "BASELINE_REGIME",
]

#: Regime label of the conventional policies.
BASELINE_REGIME = "baseline"
#: Regime label of the adaptive scheduler using classic Round Robin.
ADAPTIVE_CLASSIC_REGIME = "adaptive_classic_quantum"
#: Regime label of the adaptive scheduler using the learned quantum.
ADAPTIVE_LEARNED_REGIME = "adaptive_learned_quantum"
#: Regime label of the direct measurement of the learned quantum (Round Robin forced).
ROUND_ROBIN_LEARNED_QUANTUM_REGIME = "round_robin_learned_quantum"
#: Policy label of the direct measurement of the learned quantum.
ROUND_ROBIN_LEARNED_QUANTUM_LABEL = "Round Robin (learned quantum)"

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
        return {
            "families": list(self.families),
            "repetitions": self.repetitions,
            "evaluation_seed": self.evaluation_seed,
            "training_seed": self.training_seed,
            "rows": int(len(self.metrics)),
            "regimes": sorted(self.metrics["regime"].unique().tolist()),
            "policies": sorted(self.metrics["policy"].unique().tolist()),
            "distinct_workloads": int(self.metrics["workload_fingerprint"].nunique()),
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
    include_learned_quantum: bool = True,
) -> EvaluationResult:
    """Evaluate the four conventional policies and the trained adaptive scheduler.

    Args:
        config: The experiment configuration the agent was trained with.
        training: The training result (trained agent, controller and scheduler).
        generator: Optional pre-built workload generator.
        include_learned_quantum: Whether to also measure the adaptive scheduler with the
            controller's learned quantum.  Ignored when the controller is disabled.

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
    controller_before = (
        np.array(training.controller.agent.q_table, copy=True)
        if training.controller is not None
        else None
    )

    use_learned = bool(include_learned_quantum and training.controller is not None)

    for item in workloads:
        workload = item.workload
        for action_index in sorted(policies):
            metrics = compute_metrics(policies[action_index].run(workload))
            rows.append(
                _metrics_row(
                    family=item.family,
                    repetition=item.repetition,
                    policy=ACTION_NAMES[action_index],
                    regime=BASELINE_REGIME,
                    metrics=metrics,
                )
            )

        classic = training.scheduler.run_evaluation(workload, use_learned_quantum=False)
        _check_greedy(classic.learned)
        rows.append(
            _metrics_row(
                family=item.family,
                repetition=item.repetition,
                policy=ADAPTIVE_LABEL,
                regime=ADAPTIVE_CLASSIC_REGIME,
                metrics=classic.metrics,
            )
        )
        decision_rows.append({**classic.as_row(), "family": item.family,
                              "repetition": item.repetition, "regime": ADAPTIVE_CLASSIC_REGIME})

        if training.controller is not None:
            # Direct measurement of the quantum controller: Round Robin is run with the
            # learned multiplier regardless of the action the agent chooses, so the
            # controller's effect is measurable even when Round Robin is never selected.
            controlled_metrics, _ = training.scheduler.run_round_robin_with_learned_quantum(
                workload
            )
            rows.append(
                _metrics_row(
                    family=item.family,
                    repetition=item.repetition,
                    policy=ROUND_ROBIN_LEARNED_QUANTUM_LABEL,
                    regime=ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
                    metrics=controlled_metrics,
                )
            )

        if use_learned:
            learned = training.scheduler.run_evaluation(workload, use_learned_quantum=True)
            _check_greedy(learned.learned)
            rows.append(
                _metrics_row(
                    family=item.family,
                    repetition=item.repetition,
                    policy=ADAPTIVE_LABEL,
                    regime=ADAPTIVE_LEARNED_REGIME,
                    metrics=learned.metrics,
                )
            )
            decision_rows.append({**learned.as_row(), "family": item.family,
                                  "repetition": item.repetition, "regime": ADAPTIVE_LEARNED_REGIME})

    _check_q_table_untouched(q_before, np.array(training.agent.q_table), "policy agent")
    if controller_before is not None:
        assert training.controller is not None  # narrows the type for the reader
        _check_q_table_untouched(
            controller_before, np.array(training.controller.agent.q_table), "quantum controller"
        )

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


def _check_q_table_untouched(before: np.ndarray, after: np.ndarray, name: str) -> None:
    """Raise if a Q-table changed during evaluation."""
    if not np.array_equal(before, after):
        raise ValidationError(f"the {name} Q-table changed during evaluation")


def _check_identical_workloads(metrics: pd.DataFrame) -> None:
    """Raise if the schedulers of one workload did not all see the same workload."""
    grouped = metrics.groupby(["family", "repetition"], sort=False)["workload_fingerprint"].nunique()
    inconsistent = grouped[grouped > 1]
    if not inconsistent.empty:
        raise ValidationError(
            "some schedulers were evaluated on different workloads: "
            f"{inconsistent.to_dict()}"
        )
