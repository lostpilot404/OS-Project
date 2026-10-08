"""Shared scheduler infrastructure.

This module holds the pieces that all four policies need and that must therefore exist
in exactly one place:

* :class:`Timeline` -- accumulates execution slices and merges consecutive slices of the
  same process, so a policy that keeps the CPU on a process never records a spurious
  context switch;
* :class:`SchedulingPolicy` -- the interface every policy implements (``run``), plus the
  optional per-switch cost handling;
* :class:`NonPreemptiveReadyQueuePolicy` -- the simulation engine shared by the three
  non-preemptive policies, which differ only in their ready-set selection rule.

None of this code contains any reinforcement-learning logic: policies turn a workload
into a trace, nothing more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List, Sequence, Tuple

from config import SchedulerConfig
from errors import ValidationError
from workload.models import ExecutionSlice, Process, ScheduleResult, Workload, build_schedule_result

__all__ = ["Timeline", "SchedulingPolicy", "NonPreemptiveReadyQueuePolicy"]


class Timeline:
    """Ordered collection of execution slices with automatic merging.

    ``execute`` extends the previous slice when the same process keeps the CPU, which is
    what makes :func:`workload.models._count_running_process_changes` count *actual*
    changes of the running process rather than dispatches.
    """

    def __init__(self) -> None:
        self._slices: List[ExecutionSlice] = []

    def execute(self, pid: int, start_time: int, end_time: int) -> None:
        """Record that ``pid`` ran during ``[start_time, end_time)``.

        Args:
            pid: The running process.
            start_time: Start of the interval.
            end_time: End of the interval (exclusive).

        Raises:
            ValidationError: If the interval is empty or runs backwards in time.
        """
        if end_time <= start_time:
            raise ValidationError(
                f"cannot record an execution slice of length {end_time - start_time} "
                f"for pid {pid}"
            )
        if self._slices:
            last = self._slices[-1]
            if last.pid == pid and last.end_time == start_time:
                self._slices[-1] = ExecutionSlice(pid=pid, start_time=last.start_time, end_time=end_time)
                return
        self._slices.append(ExecutionSlice(pid=pid, start_time=start_time, end_time=end_time))

    @property
    def slices(self) -> Tuple[ExecutionSlice, ...]:
        """The recorded slices, in chronological order."""
        return tuple(self._slices)

    def __len__(self) -> int:
        return len(self._slices)


class SchedulingPolicy(ABC):
    """Base class of the four conventional CPU scheduling policies.

    Subclasses implement :meth:`_schedule`, which produces execution slices for a
    workload; :meth:`run` wraps that trace into a validated
    :class:`workload.models.ScheduleResult`.
    """

    #: Policy name used in results, tables and plots.
    name: ClassVar[str]
    #: Index of the corresponding Q-learning action.
    action_index: ClassVar[int]
    #: Whether the policy may take the CPU away from a running process.
    preemptive: ClassVar[bool]

    def __init__(self, config: SchedulerConfig) -> None:
        """
        Args:
            config: Scheduler configuration (quantum, switch cost, priority sense).
        """
        if not isinstance(config, SchedulerConfig):
            raise ValidationError(f"config must be a SchedulerConfig, got {type(config).__name__}")
        self._config = config

    @property
    def config(self) -> SchedulerConfig:
        """The configuration this policy was built with."""
        return self._config

    def run(self, workload: Workload) -> ScheduleResult:
        """Schedule ``workload`` and return the validated result.

        Args:
            workload: The workload to schedule.

        Returns:
            A :class:`workload.models.ScheduleResult` with per-process outcomes and the
            execution trace.

        Raises:
            ValidationError: If ``workload`` is not a :class:`workload.models.Workload`
                or the produced trace violates a scheduling invariant.
        """
        if not isinstance(workload, Workload):
            raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
        slices = self._schedule(workload)
        return build_schedule_result(
            workload=workload,
            slices=slices,
            policy_name=self.name,
            switching_cost=self._config.switching_cost,
        )

    @abstractmethod
    def _schedule(self, workload: Workload) -> Sequence[ExecutionSlice]:
        """Produce the execution slices for ``workload``."""

    def _switched_start(self, current_time: int, previous_pid: int | None, pid: int) -> int:
        """Return the time at which ``pid`` may start, charging a switch when needed.

        The first dispatch of a run is not a context switch, and re-dispatching the same
        process is not one either (see the context-switch definition in the README).
        """
        if previous_pid is None or previous_pid == pid:
            return current_time
        return current_time + self._config.switching_cost

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NonPreemptiveReadyQueuePolicy(SchedulingPolicy):
    """Simulation engine for non-preemptive policies driven by a ready-set key.

    A process runs to completion once dispatched.  Whenever the CPU becomes free the
    engine admits every process that has arrived by the current time and dispatches the
    ready process with the smallest selection key.  The three non-preemptive policies
    differ *only* in that key:

    * FCFS -- ``(arrival_time, pid)``
    * SJF -- ``(burst_time, arrival_time, pid)``
    * Priority -- ``(priority, arrival_time, pid)``

    Tie-breaking is therefore explicit and total, which makes the output deterministic.

    Subclasses implement :meth:`_selection_key`.
    """

    preemptive: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def _selection_key(process: Process, config: SchedulerConfig) -> Tuple[float, ...]:
        """Return the sort key selecting the next process from the ready set."""

    def _schedule(self, workload: Workload) -> Sequence[ExecutionSlice]:
        processes = workload.processes
        timeline = Timeline()
        ready: List[Process] = []
        total = len(processes)
        next_index = 0
        current_time = 0
        previous_pid: int | None = None

        while next_index < total or ready:
            # Admit every process that has arrived by the current time.
            while next_index < total and processes[next_index].arrival_time <= current_time:
                ready.append(processes[next_index])
                next_index += 1

            if not ready:
                # Nothing to run: the CPU idles until the next arrival.
                current_time = processes[next_index].arrival_time
                continue

            chosen = min(ready, key=lambda p: self._selection_key(p, self._config))
            ready.remove(chosen)
            start = self._switched_start(current_time, previous_pid, chosen.pid)
            end = start + chosen.burst_time
            timeline.execute(chosen.pid, start, end)
            current_time = end
            previous_pid = chosen.pid

        return timeline.slices
