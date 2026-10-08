"""Workload generation: the rules that make the conditions genuinely different.

The generator is the component that decides whether the experiment can show
workload-aware selection at all, so its rules are tested directly: the batch-head
arrival pattern (long background jobs released at the head of the window plus a
stream of short jobs), the configurable long-mode bound, and the burst-aligned
priority pattern (importance tracks job size).
"""

from __future__ import annotations

import pytest

from config import WorkloadFamilyConfig
from errors import ValidationError
from workload.generator import WorkloadGenerator
from workload.models import Workload


def _family(**overrides) -> WorkloadFamilyConfig:
    """A small bimodal family with a batch head, plus overrides."""
    base = dict(
        name="f",
        description="test",
        num_processes=12,
        burst_distribution="bimodal",
        burst_time_min=1,
        burst_time_max=100,
        short_burst_max=2,
        short_burst_fraction=0.75,
        long_burst_min=60,
        arrival_pattern="batch_head",
        arrival_window=40,
        head_window=3,
        priority_pattern="uniform",
    )
    base.update(overrides)
    return WorkloadFamilyConfig(**base)


class TestBatchHeadArrivals:
    def test_long_jobs_are_released_at_the_head_of_the_window(self) -> None:
        generator = WorkloadGenerator((_family(),))
        workload = generator.generate("f", 5)
        long_jobs = [p for p in workload.processes if p.burst_time >= 60]
        short_jobs = [p for p in workload.processes if p.burst_time < 60]
        assert long_jobs and short_jobs
        assert all(0 <= p.arrival_time <= 3 for p in long_jobs)
        assert all(0 <= p.arrival_time <= 40 for p in short_jobs)
        # At least one long job must be released before the short-job stream gets going,
        # otherwise the workload would not exercise the blocking that distinguishes the
        # interactive conditions.
        assert min(p.arrival_time for p in long_jobs) <= max(
            p.arrival_time for p in short_jobs
        )

    def test_long_mode_respects_long_burst_min(self) -> None:
        generator = WorkloadGenerator((_family(),))
        for seed in range(10):
            workload = generator.generate("f", seed)
            for process in workload.processes:
                if process.burst_time > 2:
                    assert 60 <= process.burst_time <= 100

    def test_long_burst_min_defaults_to_the_short_mode_bound_plus_one(self) -> None:
        family = _family(long_burst_min=0)
        assert family.long_mode_min() == 3
        assert _family().long_mode_min() == 60

    def test_batch_head_requires_a_bimodal_burst_distribution(self) -> None:
        with pytest.raises(ValidationError, match="batch_head"):
            WorkloadFamilyConfig(
                name="bad",
                description="x",
                num_processes=4,
                burst_distribution="uniform",
                arrival_pattern="batch_head",
            )

    def test_head_window_must_not_exceed_the_arrival_window(self) -> None:
        with pytest.raises(ValidationError, match="head_window"):
            _family(head_window=100)

    def test_invalid_long_burst_min_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="long_burst_min"):
            _family(long_burst_min=2)  # not above the short mode
        with pytest.raises(ValidationError, match="long_burst_min"):
            _family(long_burst_min=101)  # above the burst range
        with pytest.raises(ValidationError, match="long_burst_min"):
            _family(long_burst_min=60, burst_distribution="uniform")


class TestBurstAlignedPriorities:
    def test_shortest_job_carries_the_highest_priority(self) -> None:
        family = _family(
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            long_burst_min=0,
            arrival_pattern="uniform",
            priority_pattern="burst_aligned",
            priority_min=1,
            priority_max=5,
        )
        generator = WorkloadGenerator((family,))
        workload = generator.generate("f", 11)
        by_burst = sorted(workload.processes, key=lambda p: (p.burst_time, p.pid))
        priorities = [p.priority for p in by_burst]
        # Priorities never decrease as the burst time grows: importance tracks size.
        assert priorities == sorted(priorities)
        assert min(p.priority for p in workload.processes) == 1
        assert max(p.priority for p in workload.processes) <= 5

    def test_aligned_priorities_are_deterministic_in_the_bursts(self) -> None:
        family = _family(
            burst_distribution="uniform",
            long_burst_min=0,
            arrival_pattern="uniform",
            priority_pattern="burst_aligned",
        )
        generator = WorkloadGenerator((family,))
        first = generator.generate("f", 3)
        second = generator.generate("f", 3)
        assert first == second
        assert [p.priority for p in first.processes] == [
            p.priority for p in second.processes
        ]

    def test_unknown_priority_pattern_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="priority_pattern"):
            _family(priority_pattern="nope")


class TestExistingPatternsStillWork:
    def test_uniform_batch_release(self) -> None:
        family = WorkloadFamilyConfig(
            name="batch", description="x", num_processes=6,
            burst_distribution="uniform", burst_time_min=1, burst_time_max=4,
            arrival_pattern="uniform", arrival_window=0,
        )
        workload = WorkloadGenerator((family,)).generate("batch", 1)
        assert all(p.arrival_time == 0 for p in workload.processes)

    def test_uniform_staggered_release(self) -> None:
        family = WorkloadFamilyConfig(
            name="stream", description="x", num_processes=6,
            burst_distribution="uniform", burst_time_min=1, burst_time_max=4,
            arrival_pattern="uniform", arrival_window=30,
        )
        workload = WorkloadGenerator((family,)).generate("stream", 1)
        assert all(0 <= p.arrival_time <= 30 for p in workload.processes)
        assert len({p.arrival_time for p in workload.processes}) > 1

    def test_poisson_arrivals_are_spread_over_time(self) -> None:
        family = WorkloadFamilyConfig(
            name="poisson", description="x", num_processes=8,
            burst_distribution="uniform", burst_time_min=1, burst_time_max=20,
            arrival_pattern="poisson", arrival_rate=0.5,
        )
        workload = WorkloadGenerator((family,)).generate("poisson", 4)
        assert all(p.arrival_time >= 0 for p in workload.processes)
        assert workload.arrival_span > 0
        assert len({p.arrival_time for p in workload.processes}) > 1
        # Workloads are stored in arrival order, independently of generation order.
        arrivals = [p.arrival_time for p in workload.processes]
        assert arrivals == sorted(arrivals)

    def test_high_priority_skewed_draws_mostly_from_the_high_band(self) -> None:
        family = WorkloadFamilyConfig(
            name="skewed", description="x", num_processes=40,
            burst_distribution="uniform", burst_time_min=1, burst_time_max=50,
            arrival_pattern="uniform", arrival_window=20,
            priority_pattern="high_priority_skewed",
            high_priority_cutoff=2, high_priority_fraction=0.7,
        )
        workload = WorkloadGenerator((family,)).generate("skewed", 2)
        high = sum(1 for p in workload.processes if p.priority <= 2)
        assert high > len(workload.processes) // 2

    def test_reproducibility_and_seed_validation(self) -> None:
        generator = WorkloadGenerator((_family(),))
        assert generator.generate("f", 9) == generator.generate("f", 9)
        assert generator.generate("f", 9) != generator.generate("f", 10)
        with pytest.raises(ValidationError, match="seed"):
            generator.generate("f", -1)


class TestGeneratorValidation:
    def test_empty_family_list_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            WorkloadGenerator(())

    def test_duplicate_names_are_rejected(self) -> None:
        family = _family()
        with pytest.raises(ValidationError, match="unique"):
            WorkloadGenerator((family, family))

    def test_unknown_family_is_rejected(self) -> None:
        generator = WorkloadGenerator((_family(),))
        with pytest.raises(ValidationError, match="unknown workload family"):
            generator.generate("nope", 0)

    def test_family_lookup_round_trip(self) -> None:
        family = _family(name="mine")
        generator = WorkloadGenerator((family,))
        assert generator.family("mine") is family
        assert generator.family_names == ("mine",)

    def test_workload_carries_the_family_metadata(self) -> None:
        family = _family(name="meta", description="a description")
        workload = WorkloadGenerator((family,)).generate("meta", 0)
        assert isinstance(workload, Workload)
        assert workload.name == "meta"
        assert workload.description == "a description"
        assert len(workload.processes) == family.num_processes
