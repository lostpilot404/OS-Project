"""Synthetic workload generation for the declared offline experiment conditions."""

from __future__ import annotations

import pytest

from config import WorkloadFamilyConfig
from errors import ConfigurationError, ValidationError
from workload.generator import WorkloadGenerator


def test_staggered_interactive_generates_one_long_job_and_staggered_short_jobs() -> None:
    family = WorkloadFamilyConfig(
        name="interactive",
        description="one long process with staggered short jobs",
        num_processes=15,
        burst_distribution="staggered_interactive",
        burst_time_min=25,
        burst_time_max=50,
        short_burst_max=3,
        arrival_pattern="staggered",
        staggered_gap_min=4,
        staggered_gap_max=6,
        priority_min=1,
        priority_max=1,
    )
    workload = WorkloadGenerator((family,)).generate(family.name, seed=2024)

    assert workload.processes[0].arrival_time == 0
    assert 25 <= workload.processes[0].burst_time <= 50
    assert all(1 <= process.burst_time <= 3 for process in workload.processes[1:])
    arrivals = [process.arrival_time for process in workload.processes]
    assert arrivals == sorted(arrivals)
    assert all(4 <= right - left <= 6 for left, right in zip(arrivals, arrivals[1:]))


def test_bimodal_bursts_follow_the_declared_short_fraction() -> None:
    family = WorkloadFamilyConfig(
        name="short_jobs",
        description="short-job-heavy mixture",
        num_processes=1000,
        burst_distribution="bimodal",
        burst_time_min=1,
        burst_time_max=50,
        short_burst_max=5,
        short_burst_fraction=0.8,
    )
    generator = WorkloadGenerator((family,))
    workload = generator.generate(family.name, seed=15)
    bursts = [process.burst_time for process in workload.processes]
    short_share = sum(burst <= 5 for burst in bursts) / len(bursts)

    assert all(1 <= burst <= 50 for burst in bursts)
    assert 0.75 <= short_share <= 0.85
    assert workload == generator.generate(family.name, seed=15)


def test_uniform_and_poisson_families_respect_their_time_ranges() -> None:
    uniform = WorkloadFamilyConfig(
        name="uniform",
        description="uniform bursts and arrivals",
        num_processes=30,
        burst_distribution="uniform",
        burst_time_min=2,
        burst_time_max=8,
        arrival_pattern="uniform",
        arrival_window=12,
    )
    poisson = WorkloadFamilyConfig(
        name="poisson",
        description="Poisson arrivals",
        num_processes=30,
        burst_distribution="uniform",
        burst_time_min=1,
        burst_time_max=20,
        arrival_pattern="poisson",
        arrival_rate=0.5,
    )
    generator = WorkloadGenerator((uniform, poisson))
    uniform_workload = generator.generate("uniform", seed=10)
    poisson_workload = generator.generate("poisson", seed=11)

    assert all(2 <= process.burst_time <= 8 for process in uniform_workload.processes)
    assert all(0 <= process.arrival_time <= 12 for process in uniform_workload.processes)
    poisson_arrivals = [process.arrival_time for process in poisson_workload.processes]
    assert poisson_arrivals == sorted(poisson_arrivals)
    assert poisson_arrivals[0] == 0
    assert poisson_workload.arrival_span > 0


def test_priority_skew_uses_the_declared_high_priority_band() -> None:
    family = WorkloadFamilyConfig(
        name="priority_skewed",
        description="high-priority-skewed priorities",
        num_processes=100,
        burst_distribution="uniform",
        burst_time_min=1,
        burst_time_max=50,
        arrival_pattern="uniform",
        priority_pattern="high_priority_skewed",
        priority_min=1,
        priority_max=5,
        high_priority_cutoff=2,
        high_priority_fraction=0.7,
    )
    workload = WorkloadGenerator((family,)).generate(family.name, seed=2)
    assert sum(process.priority <= 2 for process in workload.processes) > 50


def test_family_lookup_and_seeded_generation_are_reproducible() -> None:
    family = WorkloadFamilyConfig(name="small", description="small", num_processes=8)
    generator = WorkloadGenerator((family,))
    assert generator.family("small") is family
    assert generator.family_names == ("small",)
    assert generator.generate("small", 4) == generator.generate("small", 4)
    assert generator.generate("small", 4) != generator.generate("small", 5)


def test_invalid_generator_inputs_are_project_validation_errors() -> None:
    family = WorkloadFamilyConfig(name="small", description="small", num_processes=3)
    with pytest.raises(ValidationError, match="at least one"):
        WorkloadGenerator(())
    with pytest.raises(ValidationError, match="unique"):
        WorkloadGenerator((family, family))
    with pytest.raises(ValidationError, match="unknown workload family"):
        WorkloadGenerator((family,)).generate("missing", 0)
    with pytest.raises(ValidationError, match="non-negative integer"):
        WorkloadGenerator((family,)).generate("small", -1)
    with pytest.raises(ValidationError, match="non-negative integer"):
        WorkloadGenerator((family,)).generate("small", True)


def test_interactive_distribution_requires_staggered_arrivals() -> None:
    with pytest.raises(ConfigurationError, match="require staggered arrivals"):
        WorkloadFamilyConfig(
            name="invalid",
            description="inconsistent pattern",
            num_processes=4,
            burst_distribution="staggered_interactive",
            burst_time_min=10,
            burst_time_max=20,
            short_burst_max=3,
            arrival_pattern="uniform",
        )
