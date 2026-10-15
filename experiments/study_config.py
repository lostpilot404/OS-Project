"""Frozen design of the offline policy-selection study.

Everything a reader needs to reproduce the study is declared here: training seeds and
episode count, the preserved held-out test split, the out-of-distribution families, the
compared arms and the predeclared non-learning rules.  Nothing in this module is tuned
on the test split.

Splits
------
* **Training** workloads are drawn on the fly from the six training families with
  ``derive_seed(seed, 0, episode)`` for each training seed (the preserved stream of
  :mod:`experiments.train`).
* **Validation** (``VALIDATION_SEED``) is an in-reporting split over all eight families;
  it is used for illustrative examples only, never for tuning.
* **Held-out test** (``TEST_SEED``, 8 families x 30 repetitions) is evaluated once, at
  the end; it includes the two out-of-distribution families that never appear in
  training.

Arms
----
Q-learning arms (``q_arm`` dictionaries name the state encoder and declared objective):

* ``ql_orig`` -- the original Q-learning selector (original 4-feature state + original
  six-metric reward), reproduced on the shared harness and pinned to the preserved
  legacy implementation by an equivalence test;
* ``ql_risk`` -- the risk-aware selector (workload-fingerprint state + ``risk_aware_v1``);
* ``ql_risk_origstate`` / ``ql_orig_fingerprint`` -- controlled ablations isolating the
  fingerprint features and the risk-aware objective.

Non-learning arms: ``best_fixed`` (best single policy on training workloads only),
``fingerprint_rule`` (predeclared deterministic decision list, see
:mod:`experiments.baselines`) and ``random`` (seeded uniform choice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from config import (
    ExperimentConfig,
    WorkloadFamilyConfig,
    build_default_config,
)

__all__ = [
    "STUDY_NAME",
    "TRAINING_EPISODES",
    "TRAINING_SEEDS",
    "VALIDATION_SEED",
    "VALIDATION_REPETITIONS",
    "TEST_SEED",
    "TEST_REPETITIONS",
    "RANDOM_SEEDS",
    "PRIMARY_OBJECTIVE",
    "POOR_REGRET_THRESHOLD",
    "TRAINING_FAMILIES",
    "HELD_OUT_FAMILIES",
    "Q_ARMS",
    "BASELINE_ARMS",
    "ARM_LABELS",
    "StudyDesign",
    "heavy_tail_stress_family",
    "build_study_config",
    "build_study_design",
]

#: Name of the study in generated artifacts.
STUDY_NAME = "risk_aware_offline_policy_selection"

#: Training episodes per arm and seed (matches the legacy default).
TRAINING_EPISODES = 1200

#: Independent training seeds (identical stream derivation to the legacy pipeline).
TRAINING_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

#: Master seed of the validation split (reporting/examples only).
VALIDATION_SEED = 5501

#: Workloads per family in the validation split.
VALIDATION_REPETITIONS = 16

#: Master seed of the preserved held-out test split (evaluated once, at the end).
TEST_SEED = 9501

#: Workloads per family in the held-out test split.
TEST_REPETITIONS = 30

#: Seeds of the random-selection baseline (one per Q-training seed).
RANDOM_SEEDS: Tuple[int, ...] = (7701, 7702, 7703, 7704, 7705)

#: The declared primary objective for selection, regret and reporting.
PRIMARY_OBJECTIVE = "risk_aware_v1"

#: Regret above which a selection counts as "poor" in the robustness tables.
POOR_REGRET_THRESHOLD = 0.1

#: Families used for training (the training family cycle).
TRAINING_FAMILIES: Tuple[str, ...] = (
    "short_jobs",
    "long_jobs",
    "mixed",
    "cpu_bursty",
    "priority_skewed",
    "staggered_interactive",
)

#: Families never used for training: distribution-shift conditions.
HELD_OUT_FAMILIES: Tuple[str, ...] = ("poisson_arrivals", "heavy_tail_stress")

#: Q-learning arms: name -> (state encoder kind, declared objective).
Q_ARMS: Dict[str, Tuple[str, str]] = {
    "ql_orig": ("legacy", "original"),
    "ql_risk": ("fingerprint", "risk_aware_v1"),
    "ql_risk_origstate": ("legacy", "risk_aware_v1"),
    "ql_orig_fingerprint": ("fingerprint", "original"),
}

#: Non-learning baseline arms.
BASELINE_ARMS: Tuple[str, ...] = ("best_fixed", "fingerprint_rule", "random")

#: Display label of every arm.
ARM_LABELS: Dict[str, str] = {
    "ql_orig": "Original Q-learning selector",
    "ql_risk": "Risk-aware Q-learning selector",
    "ql_risk_origstate": "Ablation: risk objective, original state",
    "ql_orig_fingerprint": "Ablation: original objective, fingerprint state",
    "best_fixed": "Best fixed policy (training-only)",
    "fingerprint_rule": "Fingerprint rule (predeclared)",
    "random": "Random policy (seeded)",
}


@dataclass(frozen=True)
class StudyDesign:
    """Frozen coordinates of one study run."""

    training_episodes: int
    training_seeds: Tuple[int, ...]
    validation_seed: int
    validation_repetitions: int
    test_seed: int
    test_repetitions: int
    random_seeds: Tuple[int, ...]
    primary_objective: str
    poor_regret_threshold: float
    training_families: Tuple[str, ...]
    held_out_families: Tuple[str, ...]
    q_arms: Dict[str, Tuple[str, str]]
    baseline_arms: Tuple[str, ...]


def build_study_design() -> StudyDesign:
    """Return the frozen study design."""
    return StudyDesign(
        training_episodes=TRAINING_EPISODES,
        training_seeds=TRAINING_SEEDS,
        validation_seed=VALIDATION_SEED,
        validation_repetitions=VALIDATION_REPETITIONS,
        test_seed=TEST_SEED,
        test_repetitions=TEST_REPETITIONS,
        random_seeds=RANDOM_SEEDS,
        primary_objective=PRIMARY_OBJECTIVE,
        poor_regret_threshold=POOR_REGRET_THRESHOLD,
        training_families=TRAINING_FAMILIES,
        held_out_families=HELD_OUT_FAMILIES,
        q_arms=dict(Q_ARMS),
        baseline_arms=BASELINE_ARMS,
    )


def heavy_tail_stress_family() -> WorkloadFamilyConfig:
    """Out-of-distribution stress family (never used for training).

    25 processes (the training families all use 15), 90% short bursts in 1-3 with rare
    long bursts up to 50, slow Poisson arrivals (rate 0.2) and priorities skewed to the
    high-priority band.  Every quantity stays inside the configured state ranges.
    """
    return WorkloadFamilyConfig(
        name="heavy_tail_stress",
        description=(
            "Out-of-distribution stress: 25 processes, 90% short bursts (1-3) with rare "
            "long bursts (4-50), Poisson arrivals at rate 0.2, 60% high-priority band."
        ),
        num_processes=25,
        burst_distribution="bimodal",
        burst_time_min=1,
        burst_time_max=50,
        short_burst_max=3,
        short_burst_fraction=0.9,
        arrival_pattern="poisson",
        arrival_rate=0.2,
        priority_pattern="high_priority_skewed",
        high_priority_cutoff=2,
        high_priority_fraction=0.6,
    )


def build_study_config() -> ExperimentConfig:
    """Experiment configuration with all eight study families.

    The six training families come from :func:`config.build_default_config`;
    ``poisson_arrivals`` is the preserved held-out family and ``heavy_tail_stress`` the
    new distribution-shift family.  The training family cycle excludes both.
    """
    base = build_default_config()
    families = tuple(base.families) + (heavy_tail_stress_family(),)
    config = ExperimentConfig(
        name=STUDY_NAME,
        families=families,
        training=base.training,
        evaluation=base.evaluation,
    )
    config.validate()
    cycle = config.training.family_cycle
    for held_out in HELD_OUT_FAMILIES:
        assert held_out not in cycle, f"held-out family {held_out} leaked into training"
    return config
