"""First-Come, First-Served (FCFS) scheduling.

Non-preemptive.  The CPU is given to the ready process that arrived first; ties between
simultaneous arrivals are broken by the smaller process id, which makes the policy
deterministic.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

from config import ACTION_FCFS, SchedulerConfig
from scheduler.base import NonPreemptiveReadyQueuePolicy
from workload.models import Process

__all__ = ["FCFS"]


class FCFS(NonPreemptiveReadyQueuePolicy):
    """First-Come, First-Served.

    Selection rule: smallest ``(arrival_time, pid)`` among the ready processes -- i.e.
    strict arrival order, with simultaneous arrivals ordered by process id.
    """

    name: ClassVar[str] = "FCFS"
    action_index: ClassVar[int] = ACTION_FCFS
    preemptive: ClassVar[bool] = False

    @staticmethod
    def _selection_key(process: Process, config: SchedulerConfig) -> Tuple[float, ...]:
        """Return ``(arrival_time, pid)``: strict arrival order."""
        return (process.arrival_time, process.pid)
