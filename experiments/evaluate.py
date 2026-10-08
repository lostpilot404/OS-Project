"""Read-only evaluation of each trained offline selector.

For every unique ``(family, repetition)`` workload, the four conventional schedulers and
the greedy learned selector receive the identical workload object. Workload fingerprints
make this pairing auditable. Evaluation workload fingerprints are checked against the
training history, and the policy-agent Q-table is compared before and after evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES, SELECTOR_LABEL, ExperimentConfig, derive_seed
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics
from experiments.train import TrainingResult
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = [
    "EvaluationWorkload",
    "EvaluationResult",
    "build_evaluation_workloads",
    "evaluate",
    "BASELINE_REGIME",
    "SELECTOR_REGIME",
]

BASELINE_REGIME = "baseline"
SELECTOR_REGIME = "selector"
_WORKLOAD_STREAM_TAG = 0


@dataclass(frozen=True)
class EvaluationWorkload:
    """One unique evaluation workload and its design coordinates."""

    family: str
    repetition: int
    workload: Workload

    @property
    def fingerprint(self) -> str:
        """Stable content fingerprint used to verify pairing and data separation."""
        return self.workload.fingerprint


@dataclass(frozen=True)
class EvaluationResult:
    """Paired metrics and decisions for one independent trained model."""

    metrics: pd.DataFrame
    decisions: pd.DataFrame
    families: Tuple[str, ...]
    repetitions: int
    evaluation_seed: int
    training_seeds: Tuple[int, ...]

    def regime(self, regime: str) -> pd.DataFrame:
        """Return rows for ``baseline`` or ``selector``."""
        subset = self.metrics[self.metrics["regime"] == regime]
        if subset.empty:
            raise ValidationError(f"no rows for regime {regime!r}")
        return subset

    def summary(self) -> Dict[str, object]:
        """Return counts distinguishing unique workloads from repeated model rows."""
        unique = self.metrics.drop_duplicates(
            ["family", "repetition", "workload_fingerprint"]
        )
        return {
            "families": list(self.families),
            "repetitions_per_family": self.repetitions,
            "evaluation_seed": self.evaluation_seed,
            "training_seeds": list(self.training_seeds),
            "training_models": len(self.training_seeds),
            "metric_rows": int(len(self.metrics)),
            "metric_rows_per_model": int(len(self.metrics) / max(1, len(self.training_seeds))),
            "selector_decision_rows": int(len(self.decisions)),
            "unique_workloads": int(len(unique)),
            "unique_workload_fingerprints": int(unique["workload_fingerprint"].nunique()),
            "regimes": sorted(self.metrics["regime"].unique().tolist()),
            "policies": sorted(self.metrics["policy"].unique().tolist()),
        }


def build_evaluation_workloads(
    config: ExperimentConfig, generator: WorkloadGenerator
) -> List[EvaluationWorkload]:
    """Generate one reproducible workload per configured family and repetition."""
    workloads: List[EvaluationWorkload] = []
    for family_index, family in enumerate(config.families):
        for repetition in range(config.evaluation.repetitions):
            seed = derive_seed(config.evaluation.seed, _WORKLOAD_STREAM_TAG, family_index, repetition)
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
    """Evaluate one trained model with the four fixed baselines and greedy selector.

    Baseline outputs are intentionally repeated for each independent training seed so
    every model's selector decisions have explicitly paired reference rows. The unique
    workloads are deduplicated only for workload-count summaries and ``workloads.csv``.
    """
    if training.config != config:
        raise ValidationError("the training result was produced with a different configuration")
    if training.history.seed != config.training.seed:
        raise ValidationError("training history seed differs from the requested model seed")
    generator = generator or WorkloadGenerator(config.families)
    workloads = build_evaluation_workloads(config, generator)

    training_fingerprints = training.training_fingerprints
    leaked = sorted({item.fingerprint for item in workloads} & training_fingerprints)
    if leaked:
        raise ValidationError(
            f"{len(leaked)} evaluation workloads were already used during training: {leaked[:3]}"
        )

    rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    q_before = np.array(training.agent.q_table, copy=True)

    for item in workloads:
        workload = item.workload
        decision = training.scheduler.run_evaluation(workload)
        if decision.learned or decision.explored or decision.epsilon != 0.0:
            raise ValidationError("evaluation must be deterministic, greedy, and read-only")
        if set(decision.reference_metrics) != set(range(len(ACTION_NAMES))):
            raise ValidationError("evaluation did not run exactly the four reference policies")
        for action_index, metrics in sorted(decision.reference_metrics.items()):
            rows.append(
                _metrics_row(
                    training_seed=training.history.seed,
                    family=item.family,
                    repetition=item.repetition,
                    policy=ACTION_NAMES[action_index],
                    regime=BASELINE_REGIME,
                    metrics=metrics,
                )
            )
        rows.append(
            _metrics_row(
                training_seed=training.history.seed,
                family=item.family,
                repetition=item.repetition,
                policy=SELECTOR_LABEL,
                regime=SELECTOR_REGIME,
                metrics=decision.metrics,
            )
        )
        decision_rows.append(
            {
                **decision.as_row(),
                "training_seed": training.history.seed,
                "family": item.family,
                "repetition": item.repetition,
                "regime": SELECTOR_REGIME,
            }
        )

    _check_q_table_untouched(q_before, np.array(training.agent.q_table), "policy agent")
    metrics_frame = pd.DataFrame(rows)
    decisions_frame = pd.DataFrame(decision_rows)
    _check_identical_workloads(metrics_frame)

    return EvaluationResult(
        metrics=metrics_frame,
        decisions=decisions_frame,
        families=tuple(family.name for family in config.families),
        repetitions=config.evaluation.repetitions,
        evaluation_seed=config.evaluation.seed,
        training_seeds=(training.history.seed,),
    )


def _metrics_row(
    training_seed: int,
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
        "training_seed": training_seed,
        "family": family,
        "repetition": repetition,
        "policy": policy,
        "regime": regime,
        **row,
    }


def _check_q_table_untouched(before: np.ndarray, after: np.ndarray, name: str) -> None:
    """Raise if a Q-table changed during evaluation."""
    if not np.array_equal(before, after):
        raise ValidationError(f"the {name} Q-table changed during evaluation")


def _check_identical_workloads(metrics: pd.DataFrame) -> None:
    """Require one shared workload fingerprint across exactly five methods per pair."""
    keys = ["training_seed", "family", "repetition"]
    grouped = metrics.groupby(keys, sort=False)
    fingerprint_counts = grouped["workload_fingerprint"].nunique()
    row_counts = grouped.size()
    invalid = fingerprint_counts[(fingerprint_counts != 1) | (row_counts != 5)]
    if not invalid.empty:
        raise ValidationError(
            "each model/workload must have five policy rows with one identical fingerprint: "
            f"{invalid.to_dict()}"
        )
    expected = set(ACTION_NAMES) | {SELECTOR_LABEL}
    for key, frame in grouped:
        if set(frame["policy"]) != expected:
            raise ValidationError(f"unexpected policy set for workload {key}: {set(frame['policy'])}")
