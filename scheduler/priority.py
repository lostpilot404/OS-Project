"""Priority scheduling, non-preemptive.

The CPU is given to the ready process with the highest priority.  The non-preemptive
interpretation is the standard one and the one used throughout this project: once a
process is dispatched it runs to completion, so a high-priority arrival never preempts
the running process.

Priority ordering is defined explicitly: by default a **lower** priority number means a
**higher** priority (priority 1 is the most important).  The sense can be inverted with
``SchedulerConfig.lower_priority_number_is_higher_priority = False``.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

from config import ACTION_PRIORITY, SchedulerConfig
from scheduler.base import NonPreemptiveReadyQueuePolicy
from workload.models import Process

__all__ = ["Priority"]


class Priority(NonPreemptiveReadyQueuePolicy):
    """Non-preemptive priority scheduling.

    Selection rule: smallest ``(priority_key, arrival_time, pid)`` among the ready
    processes, where ``priority_key`` is the priority number itself when a lower number
    means higher priority, and its negation otherwise.  Equal priorities are broken by
    arrival time and then by process id.
    """

    name: ClassVar[str] = "Priority"
    action_index: ClassVar[int] = ACTION_PRIORITY
    preemptive: ClassVar[bool] = False

    @staticmethod
    def _selection_key(process: Process, config: SchedulerConfig) -> Tuple[float, ...]:
        """Return ``(priority_key, arrival_time, pid)``: highest priority first."""
        priority_key = (
            process.priority
            if config.lower_priority_number_is_higher_priority
            else -process.priority
        )
        return (priority_key, process.arrival_time, process.pid)
