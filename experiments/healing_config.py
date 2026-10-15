"""Frozen design of the self-healing scheduling study.

The study compares fixed schedulers, a detection-only control, two predeclared recovery
rules and a learned recovery controller on *identical* workloads, and audits every
recovery decision with a counterfactual replay.

Splits and seeds
----------------
* **Training** (healing agents only): one independent policy per seed, five seeds
  ``7101..7105``, ``training_episodes_per_seed`` sequential traces each cycling through
  six workload families. ``poisson_arrivals`` is excluded, so distribution shift is
  measured rather than trained on.
* **Validation**: ``validation_seed``, ``validation_repetitions`` traces per family. Used
  for reporting, for the illustrative demo, and for the development decisions recorded in
  ``docs/DESIGN_AND_CHOICES.md``. No final-test data informed any of them.
* **Final test**: ``final_test_seed``, ``final_test_repetitions`` traces per family. The
  seed is declared in this file **before** the frozen run of the study and is never used for
  any development decision. Six earlier streams are retired as evidence and are not reused:
  the ``9401`` stream, which was used while the harness was built; the ``20501`` stream, which
  was touched by an early end-to-end smoke run before the detector's congestion gate was
  finalised; and the ``20719``, ``21617``, ``72601`` and ``21977`` streams, each of which was
  declared as a held-out seed, evaluated once and then *examined* while the system was still
  changing (phase-aware residual boundaries with accepted/rejected/inconclusive verdicts, the
  truncated-window rollback rule, the matched-counterfactual audit and its denominators, and,
  for ``21977``, the replacement of the free closure fit by the structural phase baseline with
  a physically observed design row). Reporting any of them again would mean quoting a held-out
  number that a later version of the system was built against; ``23819`` is the stream declared
  for the frozen run of the repaired verification rule, and every retired stream is refused as
  a development seed by :meth:`HealingExperimentConfig.__post_init__`.

Workload seeds are derived with split-specific tags, so training, validation and final
test cannot share a workload instance even in principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from config import (
    ACTION_FCFS,
    ACTION_SJF,
    QLearningConfig,
    SchedulerConfig,
    WorkloadFamilyConfig,
    build_default_config,
)
from errors import ConfigurationError
from healing.config import HealingConfig

__all__ = ["HealingArm", "HealingExperimentConfig", "default_arms"]

#: Split tags keep the derived workload streams disjoint by construction.
SPLIT_TAGS = {"training": 51, "validation": 67, "final_test": 83}

ARM_FIXED = "fixed"
ARM_HEALING = "healing"
POLICY_KIND_HOLD = "hold_only"
POLICY_KIND_RULE_FAIRNESS = "rule_fairness"
POLICY_KIND_RULE_WAITING = "rule_waiting_time"
POLICY_KIND_Q = "q_learning"


def _default_families() -> Tuple[WorkloadFamilyConfig, ...]:
    return build_default_config().families


@dataclass(frozen=True)
class HealingArm:
    """One evaluated method.

    Attributes:
        name: Method name used in every artifact.
        kind: ``"fixed"`` for a standalone scheduler, ``"healing"`` for the control loop.
        policy_kind: Which recovery policy the healing loop uses.
        initial_policy: Configuration in force at the start of a trace (healing arms only).
        training_seed: Seed of the learned policy, or ``None``.
        description: One-line description stored in the design artifact.
    """

    name: str
    kind: str
    policy_kind: str
    initial_policy: int = ACTION_FCFS
    training_seed: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (ARM_FIXED, ARM_HEALING):
            raise ConfigurationError(f"unknown arm kind {self.kind!r}")
        if not self.name:
            raise ConfigurationError("arm name must be non-empty")
        if self.kind == ARM_HEALING and self.policy_kind not in (
            POLICY_KIND_HOLD,
            POLICY_KIND_RULE_FAIRNESS,
            POLICY_KIND_RULE_WAITING,
            POLICY_KIND_Q,
        ):
            raise ConfigurationError(f"unknown healing policy kind {self.policy_kind!r}")


@dataclass(frozen=True)
class HealingExperimentConfig:
    """Predeclared design of the self-healing study."""

    families: Tuple[WorkloadFamilyConfig, ...] = field(default_factory=_default_families)
    training_families: Tuple[str, ...] = field(
        default_factory=lambda: tuple(
            family.name for family in build_default_config().families
            if family.name != "poisson_arrivals"
        )
    )
    training_seeds: Tuple[int, ...] = (7101, 7102, 7103, 7104, 7105)
    training_episodes_per_seed: int = 1200
    validation_seed: int = 8201
    #: Declared on 2026-10-15, *before* the frozen run of the structural phase baseline (the
    #: declared design row is the physically observed ``trigger_load, ready_count,
    #: log1p(visible_work)``; the previous free closure fit and the decision-counter variant
    #: were both measured on train/validation and set aside), and never evaluated with any
    #: earlier version of the system.
    final_test_seed: int = 23819
    #: Streams that were already used to make development decisions and must not come back.
    #: 20501 and 20719 were consumed by the earlier single-threshold verification rule; 21617
    #: was used to check the repaired loop end to end during development; 72601 was touched by
    #: audit/coverage development; and 21977 was examined while the structural phase baseline
    #: was being chosen. Their tables are development evidence, never held-out results, and the
    #: guard below makes that structural rather than a promise.
    retired_final_test_seeds: Tuple[int, ...] = (20501, 20719, 21617, 72601, 21977)
    validation_repetitions: int = 20
    final_test_repetitions: int = 30
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 104729
    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(round_robin_quantum=4, switching_cost=1)
    )
    q_learning: QLearningConfig = field(
        default_factory=lambda: QLearningConfig(
            learning_rate=0.1,
            # Undiscounted episodic return over the recovery decisions of one trace.
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay_per_episode=0.997,
            initial_value=0.0,
        )
    )
    healing: HealingConfig = field(default_factory=HealingConfig)
    action_probe_repetitions: int = 4
    action_probe_families: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(family.name for family in self.families)
        if not names or len(names) != len(set(names)):
            raise ConfigurationError("the study needs unique workload families")
        if not self.training_families:
            raise ConfigurationError("training_families must not be empty")
        unknown = set(self.training_families) - set(names)
        if unknown:
            raise ConfigurationError(f"unknown training families: {sorted(unknown)}")
        if self.training_episodes_per_seed < 1:
            raise ConfigurationError("training_episodes_per_seed must be >= 1")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ConfigurationError("training_seeds must be non-empty and unique")
        split_seeds = (self.validation_seed, self.final_test_seed)
        if self.validation_seed == self.final_test_seed:
            raise ConfigurationError("validation and final-test seeds must differ")
        if self.final_test_seed in self.retired_final_test_seeds:
            raise ConfigurationError(
                f"final-test seed {self.final_test_seed} was already used: pick a fresh stream"
            )
        if set(self.retired_final_test_seeds) & set(self.training_seeds):
            raise ConfigurationError("a retired stream cannot be reused as a training seed")
        if set(split_seeds) & set(self.training_seeds):
            raise ConfigurationError("training and evaluation master seeds must differ")
        if any(seed < 0 for seed in (*self.training_seeds, *split_seeds)):
            raise ConfigurationError("master seeds must be non-negative")
        if self.validation_repetitions < 1 or self.final_test_repetitions < 1:
            raise ConfigurationError("evaluation repetitions must be >= 1")
        if self.bootstrap_replicates < 100:
            raise ConfigurationError("bootstrap_replicates must be >= 100")
        if self.action_probe_repetitions < 1:
            raise ConfigurationError("action_probe_repetitions must be >= 1")

    @property
    def validation_families(self) -> Tuple[str, ...]:
        """Every declared family is evaluated, including the training-held-out one."""
        return tuple(family.name for family in self.families)


def default_arms(config: HealingExperimentConfig) -> Tuple[HealingArm, ...]:
    """Return the predeclared arm list of the study.

    The four fixed schedulers are run through their untouched standalone implementations.
    ``detector_only`` runs the whole healing loop but never intervenes, which is the
    control that separates "detecting" from "acting". The rule and learned arms start from
    the same naive configuration (FCFS), so the comparison measures the *recovery* rather
    than the starting point; the two ``*_from_sjf`` arms re-run the strongest recovery
    methods from the strongest fixed configuration to test whether the healing loop damages
    an already good scheduler.
    """
    arms = [
        HealingArm("fixed_fcfs", ARM_FIXED, "", description="standalone FCFS, no adaptation"),
        HealingArm("fixed_sjf", ARM_FIXED, "", description="standalone SJF, no adaptation"),
        HealingArm(
            "fixed_round_robin", ARM_FIXED, "", description="standalone Round Robin, no adaptation"
        ),
        HealingArm("fixed_priority", ARM_FIXED, "", description="standalone Priority, no adaptation"),
        HealingArm(
            "detector_only",
            ARM_HEALING,
            POLICY_KIND_HOLD,
            description="full monitoring/detection/diagnosis loop that never intervenes",
        ),
        HealingArm(
            "rule_fairness",
            ARM_HEALING,
            POLICY_KIND_RULE_FAIRNESS,
            description="predeclared fairness-oriented diagnosis rule",
        ),
        HealingArm(
            "rule_wait",
            ARM_HEALING,
            POLICY_KIND_RULE_WAITING,
            description="predeclared waiting-time-oriented diagnosis rule",
        ),
        HealingArm(
            "rule_wait_from_sjf",
            ARM_HEALING,
            POLICY_KIND_RULE_WAITING,
            initial_policy=ACTION_SJF,
            description="waiting-time rule starting from the strongest fixed policy",
        ),
    ]
    for seed in config.training_seeds:
        arms.append(
            HealingArm(
                "q_recovery",
                ARM_HEALING,
                POLICY_KIND_Q,
                training_seed=seed,
                description=f"tabular Q-learning recovery, independent seed {seed}",
            )
        )
    for seed in config.training_seeds:
        arms.append(
            HealingArm(
                "q_recovery_from_sjf",
                ARM_HEALING,
                POLICY_KIND_Q,
                initial_policy=ACTION_SJF,
                training_seed=seed,
                description=(
                    f"tabular Q-learning recovery starting from SJF, independent seed {seed}"
                ),
            )
        )
    return tuple(arms)
