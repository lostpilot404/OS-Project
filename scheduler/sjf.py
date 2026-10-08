"""Shortest Job First (SJF) scheduling, non-preemptive.

The CPU is given to the ready process with the smallest burst time.  The non-preemptive
interpretation is the standard one and the one used throughout this project: once a
process is dispatched it runs to completion.  The preemptive variant (Shortest Remaining
Time First) is *not* implemented.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

from config import ACTION_SJF, SchedulerConfig
from scheduler.base import NonPreemptiveReadyQueuePolicy
from workload.models import Process

__all__ = ["SJF"]


class SJF(NonPreemptiveReadyQueuePolicy):
    """Non-preemptive Shortest Job First.

    Selection rule: smallest ``(burst_time, arrival_time, pid)`` among the ready
    processes.  Equal burst times are broken by arrival time and then by process id.
    """

    name: ClassVar[str] = "SJF"
    action_index: ClassVar[int] = ACTION_SJF
    preemptive: ClassVar[bool] = False

    @staticmethod
    def _selection_key(process: Process, config: SchedulerConfig) -> Tuple[float, ...]:
        """Return ``(burst_time, arrival_time, pid)``: shortest burst first."""
        return (process.burst_time, process.arrival_time, process.pid)
