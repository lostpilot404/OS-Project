"""Synthetic workload generation.

One generator serves every workload condition.  A condition is described by a
:class:`config.WorkloadFamilyConfig` and turned into a :class:`workload.models.Workload`
by :meth:`WorkloadGenerator.generate`.

All randomness flows through a ``numpy.random.Generator`` seeded with
:func:`config.derive_seed`, so a workload is reproducible from its family name and seed
alone and nothing about the process ordering depends on the platform.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from config import WorkloadFamilyConfig
from errors import ValidationError
from workload.models import Process, Workload

__all__ = ["WorkloadGenerator"]


class WorkloadGenerator:
    """Generates synthetic workloads from workload-family descriptions."""

    def __init__(self, families: Sequence[WorkloadFamilyConfig]) -> None:
        """
        Args:
            families: The workload conditions this generator can produce.

        Raises:
            ValidationError: If the family list is empty or contains duplicate names.
        """
        if not families:
            raise ValidationError("at least one workload family is required")
        names = [family.name for family in families]
        if len(set(names)) != len(names):
            raise ValidationError(f"workload family names must be unique, got {names}")
        self._families: Dict[str, WorkloadFamilyConfig] = {f.name: f for f in families}

    @property
    def family_names(self) -> tuple:
        """Names of the configured workload families, in configuration order."""
        return tuple(self._families)

    def family(self, name: str) -> WorkloadFamilyConfig:
        """Return the configuration of a family.

        Raises:
            ValidationError: If the family is unknown.
        """
        try:
            return self._families[name]
        except KeyError as error:
            raise ValidationError(
                f"unknown workload family {name!r}; known families: {list(self._families)}"
            ) from error

    def generate(self, family_name: str, seed: int) -> Workload:
        """Generate one workload of the given family.

        Args:
            family_name: Which configured family to draw from.
            seed: Seed for this workload; the same seed always yields the same workload.

        Returns:
            The generated workload with pids ``1..num_processes`` in generation order.

        Raises:
            ValidationError: If the family is unknown or the seed is negative.
        """
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValidationError(f"seed must be a non-negative integer, got {seed!r}")
        family = self.family(family_name)
        rng = np.random.default_rng(seed)
        count = family.num_processes
        bursts = self._draw_burst_times(family, rng)
        arrivals = self._draw_arrival_times(family, rng)
        priorities = self._draw_priorities(family, rng)
        processes = [
            Process(
                pid=index + 1,
                arrival_time=int(arrivals[index]),
                burst_time=int(bursts[index]),
                priority=int(priorities[index]),
            )
            for index in range(count)
        ]
        return Workload(
            name=family.name,
            processes=processes,
            description=family.description,
        )

    # -- drawing rules ----------------------------------------------------------------
    @staticmethod
    def _draw_burst_times(family: WorkloadFamilyConfig, rng: np.random.Generator) -> np.ndarray:
        """Draw one burst time per process.

        ``uniform`` draws from ``[burst_time_min, burst_time_max]``.  ``bimodal`` draws
        from the short mode ``[burst_time_min, short_burst_max]`` with probability
        ``short_burst_fraction`` and from the long mode
        ``[short_burst_max + 1, burst_time_max]`` otherwise.
        """
        count = family.num_processes
        if family.burst_distribution == "uniform":
            return rng.integers(family.burst_time_min, family.burst_time_max + 1, size=count)
        short = rng.random(count) < family.short_burst_fraction
        bursts = rng.integers(
            family.short_burst_max + 1, family.burst_time_max + 1, size=count
        )
        bursts[short] = rng.integers(
            family.burst_time_min, family.short_burst_max + 1, size=int(short.sum())
        )
        return bursts

    @staticmethod
    def _draw_arrival_times(family: WorkloadFamilyConfig, rng: np.random.Generator) -> np.ndarray:
        """Draw one arrival time per process.

        ``uniform`` draws independent arrival times from ``[0, arrival_window]``, so
        simultaneous arrivals occur naturally.  ``poisson`` builds a Poisson process with
        mean rate ``arrival_rate``: the first arrival is at time 0 and each following
        inter-arrival time is exponential; the (real-valued) arrival times are floored to
        whole time units.
        """
        count = family.num_processes
        if family.arrival_pattern == "uniform":
            return rng.integers(0, family.arrival_window + 1, size=count)
        inter_arrivals = rng.exponential(1.0 / family.arrival_rate, size=count)
        cumulative = np.concatenate(([0.0], np.cumsum(inter_arrivals)[:-1]))
        return np.floor(cumulative).astype(np.int64)

    @staticmethod
    def _draw_priorities(family: WorkloadFamilyConfig, rng: np.random.Generator) -> np.ndarray:
        """Draw one priority per process.

        ``uniform`` draws from ``[priority_min, priority_max]``.  ``high_priority_skewed``
        draws from ``[priority_min, high_priority_cutoff]`` with probability
        ``high_priority_fraction`` and from the full range otherwise.
        """
        count = family.num_processes
        if family.priority_pattern == "uniform":
            return rng.integers(family.priority_min, family.priority_max + 1, size=count)
        high = rng.random(count) < family.high_priority_fraction
        priorities = rng.integers(
            family.priority_min, family.priority_max + 1, size=count
        )
        priorities[high] = rng.integers(
            family.priority_min, family.high_priority_cutoff + 1, size=int(high.sum())
        )
        return priorities

    def __repr__(self) -> str:
        return f"WorkloadGenerator(families={list(self._families)})"
