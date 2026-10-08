"""Shared helpers for the test suite."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from config import SchedulerConfig
from workload.models import Process, Workload

#: A process row: ``(pid, arrival_time, burst_time, priority)``.
Row = Tuple[int, int, int, int]

#: Default scheduler configuration used by the scheduler tests
#: (quantum 4, costless switching, lower priority number = higher priority).
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()


def workload_from_rows(name: str, rows: Sequence[Row], allow_empty: bool = False) -> Workload:
    """Build a :class:`Workload` from ``(pid, arrival, burst, priority)`` rows."""
    return Workload(
        name=name,
        processes=[Process(*row) for row in rows],
        allow_empty=allow_empty,
    )


def timing(result) -> List[Tuple[int, int, int, int, int]]:
    """Return ``(pid, start, completion, waiting, turnaround)`` per process."""
    return [
        (o.pid, o.start_time, o.completion_time, o.waiting_time, o.turnaround_time)
        for o in result.outcomes
    ]


def trace(result) -> List[Tuple[int, int, int]]:
    """Return the execution trace as ``(pid, start, end)`` triples."""
    return [(s.pid, s.start_time, s.end_time) for s in result.trace]


def responses(result) -> List[Tuple[int, int]]:
    """Return ``(pid, response_time)`` per process."""
    return [(o.pid, o.response_time) for o in result.outcomes]


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, with an explicit error for an empty sequence."""
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)
