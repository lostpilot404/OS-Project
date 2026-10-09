"""Frozen design for the separate causal runtime-adaptation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from config import (
    QLearningConfig,
    SchedulerConfig,
    WorkloadFamilyConfig,
    build_default_config,
)
from errors import ConfigurationError

__all__ = ["RuntimeExperimentConfig"]


def _default_families() -> tuple[WorkloadFamilyConfig, ...]:
    return build_default_config().families


def _default_training_families() -> tuple[str, ...]:
    return tuple(
        family.name
        for family in build_default_config().families
        if family.name != "poisson_arrivals"
    )


@dataclass(frozen=True)
class RuntimeExperimentConfig:
    """Predeclared training/validation/final-test design for online policy switching."""

    families: Tuple[WorkloadFamilyConfig, ...] = field(default_factory=_default_families)
    training_families: Tuple[str, ...] = field(default_factory=_default_training_families)
    training_episodes_per_seed: int = 1200
    training_seeds: Tuple[int, ...] = (7101, 7102, 7103, 7104, 7105)
    validation_seed: int = 8201
    # Fresh held-out seed for the remediated queue/reward/state design.
    final_test_seed: int = 19301
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
            # Episodic undiscounted waiting-cost objective: sum of interval rewards.
            discount_factor=1.0,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay_per_episode=0.997,
            initial_value=0.0,
        )
    )

    def __post_init__(self) -> None:
        family_names = tuple(family.name for family in self.families)
        if not family_names or len(family_names) != len(set(family_names)):
            raise ConfigurationError("runtime experiment needs unique workload families")
        if not self.training_families:
            raise ConfigurationError("training_families must not be empty")
        if len(set(self.training_families)) != len(self.training_families):
            raise ConfigurationError("training_families must be unique")
        unknown = set(self.training_families) - set(family_names)
        if unknown:
            raise ConfigurationError(f"unknown training families: {sorted(unknown)}")
        if self.training_episodes_per_seed < 1:
            raise ConfigurationError("training_episodes_per_seed must be >= 1")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ConfigurationError("training_seeds must be non-empty and unique")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.training_seeds):
            raise ConfigurationError("training seeds must be non-negative integers")
        split_seeds = (self.validation_seed, self.final_test_seed)
        if any(isinstance(seed, bool) or seed < 0 for seed in split_seeds):
            raise ConfigurationError("validation/test seeds must be non-negative integers")
        if self.validation_seed == self.final_test_seed:
            raise ConfigurationError("validation and final-test seeds must differ")
        if set(split_seeds) & set(self.training_seeds):
            raise ConfigurationError("training, validation, and final-test master seeds must differ")
        if self.validation_repetitions < 1 or self.final_test_repetitions < 1:
            raise ConfigurationError("validation/test repetitions must be >= 1")
        if self.bootstrap_replicates < 100:
            raise ConfigurationError("bootstrap_replicates must be >= 100")
        if self.bootstrap_seed < 0:
            raise ConfigurationError("bootstrap_seed must be >= 0")

    @property
    def validation_families(self) -> Tuple[str, ...]:
        """Validation covers all declared families, including the training-held-out one."""
        return tuple(family.name for family in self.families)

    @property
    def final_test_families(self) -> Tuple[str, ...]:
        """Final test covers the same predeclared condition set as validation."""
        return self.validation_families
