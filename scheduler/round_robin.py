"""Round Robin (RR) scheduling.

Preemptive, quantum driven.  The ready queue is FIFO and every process may run for at
most one quantum per turn.  The quantum is a configuration value
(:attr:`config.SchedulerConfig.round_robin_quantum`), never a constant buried in the
code.

Arrival convention (documented, and covered by tests): when a process is preempted, the
processes that arrived at or before the end of its slice are placed **ahead** of it in
the queue.  A process arriving exactly at the end of a slice is therefore considered
ready at that instant.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, Deque, Dict, Sequence

from config import ACTION_ROUND_ROBIN
from scheduler.base import SchedulingPolicy, Timeline
from workload.models import ExecutionSlice, Process, Workload

__all__ = ["RoundRobin"]


def _admit_arrivals(
    processes: Sequence[Process],
    next_index: int,
    current_time: int,
    ready: Deque[int],
) -> int:
    """Move every process that has arrived by ``current_time`` into the ready queue.

    Args:
        processes: All processes, sorted by ``(arrival_time, pid)``.
        next_index: Index of the first process not yet admitted.
        current_time: The current simulation time.
        ready: The ready queue of *pids*, appended to in place.

    Returns:
        The updated ``next_index``.
    """
    total = len(processes)
    while next_index < total and processes[next_index].arrival_time <= current_time:
        ready.append(processes[next_index].pid)
        next_index += 1
    return next_index


class RoundRobin(SchedulingPolicy):
    """Round Robin with a configurable time quantum.

    Selection rule: FIFO ready queue, and each process runs for
    ``min(quantum, remaining_burst_time)`` per turn.  Simultaneous arrivals are enqueued
    in process-id order.
    """

    name: ClassVar[str] = "Round Robin"
    action_index: ClassVar[int] = ACTION_ROUND_ROBIN
    preemptive: ClassVar[bool] = True

    def _schedule(self, workload: Workload) -> Sequence[ExecutionSlice]:
        quantum = self._config.round_robin_quantum
        processes = workload.processes
        # Remaining burst times are keyed by pid: pids are arbitrary identifiers, not
        # 0-based indices, so they must never be used to index a list.
        remaining: Dict[int, int] = {process.pid: process.burst_time for process in processes}
        ready: Deque[int] = deque()
        timeline = Timeline()
        total = len(processes)
        next_index = 0
        current_time = 0
        previous_pid: int | None = None

        while next_index < total or ready:
            if not ready:
                # Nothing ready: idle until the next arrival.
                current_time = processes[next_index].arrival_time

            next_index = _admit_arrivals(processes, next_index, current_time, ready)
            if not ready:  # pragma: no cover - defensive, unreachable after the jump above
                continue

            pid = ready.popleft()
            start = self._switched_start(current_time, previous_pid, pid)
            run_time = min(quantum, remaining[pid])
            end = start + run_time
            timeline.execute(pid, start, end)
            remaining[pid] -= run_time
            current_time = end

            # Arrivals during the slice precede the preempted process (documented rule).
            next_index = _admit_arrivals(processes, next_index, current_time, ready)
            if remaining[pid] > 0:
                ready.append(pid)
            previous_pid = pid

        return timeline.slices
