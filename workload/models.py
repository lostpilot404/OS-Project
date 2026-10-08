"""Process / workload data model and the result of a scheduling run.

Time is measured in abstract integer time units.  The model deliberately contains only
the four attributes named by the project brief (process id, arrival time, burst time,
priority) plus the derived quantities of a *completed* schedule.

No I/O, no blocking, no deadlines and no preemption costs beyond the optional
:attr:`config.SchedulerConfig.switching_cost` are modelled; see
``docs/DESIGN_AND_CHOICES.md``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterator, List, Sequence, Tuple

from errors import ValidationError

__all__ = [
    "Process",
    "Workload",
    "ExecutionSlice",
    "ProcessOutcome",
    "ScheduleResult",
    "build_schedule_result",
]


def _require_integer(value: object, field_name: str) -> int:
    """Return ``value`` as a plain ``int``, rejecting non-integer input.

    Raises:
        ValidationError: If the value is not an integral number.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Process:
    """A synthetic process competing for a single CPU.

    Attributes:
        pid: Unique, non-negative process identifier.
        arrival_time: Time at which the process becomes ready (>= 0).
        burst_time: Total CPU time the process requires (> 0).
        priority: Scheduling priority (>= 0).  By default a *lower* number means a
            *higher* priority; the convention is configurable through
            :attr:`config.SchedulerConfig.lower_priority_number_is_higher_priority`.
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int

    def __post_init__(self) -> None:
        pid = _require_integer(self.pid, "pid")
        arrival_time = _require_integer(self.arrival_time, "arrival_time")
        burst_time = _require_integer(self.burst_time, "burst_time")
        priority = _require_integer(self.priority, "priority")
        if pid < 0:
            raise ValidationError(f"pid must be >= 0, got {pid}")
        if arrival_time < 0:
            raise ValidationError(f"arrival_time must be >= 0, got {arrival_time}")
        if burst_time <= 0:
            raise ValidationError(f"burst_time must be > 0, got {burst_time}")
        if priority < 0:
            raise ValidationError(f"priority must be >= 0, got {priority}")
        object.__setattr__(self, "pid", pid)
        object.__setattr__(self, "arrival_time", arrival_time)
        object.__setattr__(self, "burst_time", burst_time)
        object.__setattr__(self, "priority", priority)


class Workload:
    """An immutable, canonically ordered collection of processes.

    Processes are stored sorted by ``(arrival_time, pid)``, which makes attribute order
    independent of input order and therefore keeps every scheduler deterministic.

    Args:
        name: Identifier of the workload condition this instance belongs to.
        processes: The processes of the workload.  Must be non-empty when ``allow_empty``
            is ``False``; an empty workload is accepted for edge-case testing.
        description: Optional human-readable description.
        allow_empty: Whether an empty process collection is acceptable.

    Raises:
        ValidationError: If the processes are invalid, pids are duplicated, or an empty
            collection is passed while ``allow_empty`` is ``False``.
    """

    __slots__ = ("_name", "_description", "_processes", "_fingerprint")

    def __init__(
        self,
        name: str,
        processes: Sequence[Process],
        description: str = "",
        allow_empty: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValidationError("workload name must be a non-empty string")
        if not processes and not allow_empty:
            raise ValidationError(f"workload {name!r} must contain at least one process")
        checked: List[Process] = []
        for process in processes:
            if not isinstance(process, Process):
                raise ValidationError(
                    f"workload {name!r} contains a non-Process entry: {process!r}"
                )
            checked.append(process)
        pids = [p.pid for p in checked]
        if len(set(pids)) != len(pids):
            duplicates = sorted({pid for pid in pids if pids.count(pid) > 1})
            raise ValidationError(f"workload {name!r} has duplicate pids: {duplicates}")
        checked.sort(key=lambda p: (p.arrival_time, p.pid))
        self._name = name
        self._description = description
        self._processes: Tuple[Process, ...] = tuple(checked)
        digest = hashlib.sha256(
            ";".join(
                f"{p.pid}:{p.arrival_time}:{p.burst_time}:{p.priority}" for p in self._processes
            ).encode("ascii")
        ).hexdigest()[:16]
        self._fingerprint = digest

    # -- basic container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._processes)

    def __iter__(self) -> Iterator[Process]:
        return iter(self._processes)

    def __getitem__(self, index: int) -> Process:
        return self._processes[index]

    def __repr__(self) -> str:
        return f"Workload(name={self._name!r}, processes={len(self._processes)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Workload):
            return NotImplemented
        return self._fingerprint == other._fingerprint and self._name == other._name

    def __hash__(self) -> int:
        return hash((self._name, self._fingerprint))

    # -- accessors ----------------------------------------------------------------
    @property
    def name(self) -> str:
        """Identifier of the workload condition."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable description of the workload."""
        return self._description

    @property
    def processes(self) -> Tuple[Process, ...]:
        """The processes, ordered by ``(arrival_time, pid)``."""
        return self._processes

    @property
    def size(self) -> int:
        """Number of processes."""
        return len(self._processes)

    @property
    def total_burst_time(self) -> int:
        """Sum of the burst times of all processes."""
        return sum(p.burst_time for p in self._processes)

    @property
    def mean_burst_time(self) -> float:
        """Mean burst time, or ``0.0`` for an empty workload."""
        if not self._processes:
            return 0.0
        return self.total_burst_time / len(self._processes)

    @property
    def median_burst_time(self) -> float:
        """Conventional median burst time (mean of middle values when count is even)."""
        if not self._processes:
            return 0.0
        bursts = sorted(p.burst_time for p in self._processes)
        middle = len(bursts) // 2
        if len(bursts) % 2:
            return float(bursts[middle])
        return (bursts[middle - 1] + bursts[middle]) / 2.0

    def percentile_burst_time(self, fraction: float) -> float:
        """Return the burst time at ``fraction`` of the sorted burst-time list.

        Uses the nearest-rank definition: for a non-empty list of size ``n`` and fraction
        ``p``, the zero-based index is ``max(0, ceil(p*n) - 1)``. The result therefore
        remains on the integer grid of the generated bursts.

        Args:
            fraction: Quantile in ``[0, 1]``.

        Returns:
            The burst time at the requested quantile, or ``0.0`` for an empty workload.
        """
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not math.isfinite(float(fraction))
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValidationError(f"fraction must be a finite number in [0, 1], got {fraction!r}")
        if not self._processes:
            return 0.0
        bursts = sorted(p.burst_time for p in self._processes)
        index = max(0, math.ceil(float(fraction) * len(bursts)) - 1)
        return float(bursts[index])

    @property
    def max_arrival_time(self) -> int:
        """Latest arrival time, or ``0`` for an empty workload."""
        if not self._processes:
            return 0
        return max(p.arrival_time for p in self._processes)

    @property
    def min_arrival_time(self) -> int:
        """Earliest arrival time, or ``0`` for an empty workload."""
        if not self._processes:
            return 0
        return min(p.arrival_time for p in self._processes)

    @property
    def arrival_span(self) -> int:
        """Difference between the latest and the earliest arrival time."""
        return self.max_arrival_time - self.min_arrival_time

    @property
    def fingerprint(self) -> str:
        """Deterministic 16-hex-character digest of the process table.

        Used to prove that every scheduler was evaluated on byte-identical workloads.
        """
        return self._fingerprint

    def ready_at(self, time: int) -> Tuple[Process, ...]:
        """Return the processes that have arrived by ``time``.

        Args:
            time: The time instant to test against.

        Returns:
            The subset of processes with ``arrival_time <= time``.
        """
        return tuple(p for p in self._processes if p.arrival_time <= time)


@dataclass(frozen=True)
class ExecutionSlice:
    """One contiguous interval during which a single process owned the CPU.

    Attributes:
        pid: Process that was running.
        start_time: Time the slice started.
        end_time: Time the slice ended (exclusive).
    """

    pid: int
    start_time: int
    end_time: int

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            raise ValidationError(
                f"execution slice must have end_time > start_time, got "
                f"({self.start_time}, {self.end_time})"
            )

    @property
    def duration(self) -> int:
        """Length of the slice in time units."""
        return self.end_time - self.start_time


@dataclass(frozen=True)
class ProcessOutcome:
    """Timing results of one process in a completed schedule.

    The three metric definitions used by the project are derived here, once, so that no
    other module can re-derive them differently:

    * ``turnaround_time = completion_time - arrival_time``
    * ``waiting_time = turnaround_time - burst_time``
    * ``response_time = start_time - arrival_time``
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int
    start_time: int
    completion_time: int

    @property
    def turnaround_time(self) -> int:
        """Completion time minus arrival time."""
        return self.completion_time - self.arrival_time

    @property
    def waiting_time(self) -> int:
        """Time spent ready but not running: turnaround time minus burst time."""
        return self.turnaround_time - self.burst_time

    @property
    def response_time(self) -> int:
        """Time from arrival to first execution."""
        return self.start_time - self.arrival_time


@dataclass(frozen=True)
class ScheduleResult:
    """The complete, inspectable outcome of running one policy on one workload.

    Attributes:
        policy_name: Name of the policy that produced the schedule.
        workload_name: Name of the workload condition.
        workload_fingerprint: Fingerprint of the scheduled workload.
        outcomes: Per-process timing outcomes, ordered by pid.
        trace: Ordered execution slices.
        total_elapsed_time: Makespan -- completion time of the last process.
        cpu_busy_time: Sum of the execution-slice durations.
        switching_overhead: Total time spent switching between processes.
        context_switches: Number of changes of the running process.
    """

    policy_name: str
    workload_name: str
    workload_fingerprint: str
    outcomes: Tuple[ProcessOutcome, ...]
    trace: Tuple[ExecutionSlice, ...]
    total_elapsed_time: int
    cpu_busy_time: int
    switching_overhead: int
    context_switches: int

    @property
    def idle_time(self) -> int:
        """Time during which no process was running and no switch was in progress."""
        return self.total_elapsed_time - self.cpu_busy_time - self.switching_overhead

    def outcome(self, pid: int) -> ProcessOutcome:
        """Return the outcome of ``pid``.

        Raises:
            ValidationError: If the process is not part of the schedule.
        """
        for outcome in self.outcomes:
            if outcome.pid == pid:
                return outcome
        raise ValidationError(f"pid {pid} is not part of schedule {self.policy_name!r}")

    @property
    def starts_at(self) -> Dict[int, int]:
        """Mapping of pid to first execution time."""
        return {o.pid: o.start_time for o in self.outcomes}

    @property
    def completion_times(self) -> Dict[int, int]:
        """Mapping of pid to completion time."""
        return {o.pid: o.completion_time for o in self.outcomes}


def build_schedule_result(
    workload: Workload,
    slices: Sequence[ExecutionSlice],
    policy_name: str,
    switching_cost: int,
) -> ScheduleResult:
    """Assemble and validate a :class:`ScheduleResult` from raw execution slices.

    All four schedulers funnel their raw trace through this function, so the trace
    invariants are checked in exactly one place:

    * slices are strictly ordered and never overlap;
    * each process is given exactly its burst time, and never starts before it arrives;
    * the CPU is never idle while a process is waiting (work conservation), except for
      the configured per-switch cost.

    Args:
        workload: The scheduled workload.
        slices: Execution slices in chronological order.
        policy_name: Name of the producing policy.
        switching_cost: Time charged per context switch, from
            :attr:`config.SchedulerConfig.switching_cost`.

    Returns:
        The validated schedule result.

    Raises:
        ValidationError: If the trace violates any invariant above.
    """
    if switching_cost < 0:
        raise ValidationError(f"switching_cost must be >= 0, got {switching_cost}")

    ordered = list(slices)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_time < previous.end_time:
            raise ValidationError(
                f"execution slices overlap or are out of order: {previous} then {current}"
            )

    executed: Dict[int, int] = {}
    first_start: Dict[int, int] = {}
    last_completion: Dict[int, int] = {}
    known = {p.pid: p for p in workload.processes}
    for slice_ in ordered:
        if slice_.pid not in known:
            raise ValidationError(f"execution slice references unknown pid {slice_.pid}")
        process = known[slice_.pid]
        if slice_.start_time < process.arrival_time:
            raise ValidationError(
                f"pid {slice_.pid} is scheduled at {slice_.start_time} but only arrives at "
                f"{process.arrival_time}"
            )
        executed[slice_.pid] = executed.get(slice_.pid, 0) + slice_.duration
        first_start.setdefault(slice_.pid, slice_.start_time)
        last_completion[slice_.pid] = slice_.end_time

    for process in workload.processes:
        got = executed.get(process.pid, 0)
        if got != process.burst_time:
            raise ValidationError(
                f"pid {process.pid} executed {got} time units but has burst "
                f"{process.burst_time}"
            )

    context_switches = _count_running_process_changes(ordered)
    switching_overhead = context_switches * switching_cost
    cpu_busy_time = sum(slice_.duration for slice_ in ordered)
    total_elapsed_time = max((slice_.end_time for slice_ in ordered), default=0)

    # Work conservation: beyond one optional switch cost, the CPU must never be idle
    # while some process is pending.  A process is pending somewhere inside a time window
    # when it has arrived before the window ends and completes after the window starts.
    for previous, current in zip(ordered, ordered[1:]):
        gap = current.start_time - previous.end_time
        if gap <= switching_cost:
            continue
        window_start = previous.end_time + switching_cost
        pending = [
            p.pid
            for p in workload.processes
            if p.arrival_time < current.start_time and last_completion[p.pid] > window_start
        ]
        if pending:
            raise ValidationError(
                f"CPU idle from {window_start} to {current.start_time} although pids "
                f"{pending} were pending"
            )

    outcomes = tuple(
        ProcessOutcome(
            pid=process.pid,
            arrival_time=process.arrival_time,
            burst_time=process.burst_time,
            priority=process.priority,
            start_time=first_start[process.pid],
            completion_time=last_completion[process.pid],
        )
        for process in sorted(workload.processes, key=lambda p: p.pid)
    )

    idle_time = total_elapsed_time - cpu_busy_time - switching_overhead
    if idle_time < 0:
        raise ValidationError(
            f"schedule reports negative idle time ({idle_time}); cpu_busy_time="
            f"{cpu_busy_time}, makespan={total_elapsed_time}, overhead={switching_overhead}"
        )

    return ScheduleResult(
        policy_name=policy_name,
        workload_name=workload.name,
        workload_fingerprint=workload.fingerprint,
        outcomes=outcomes,
        trace=tuple(ordered),
        total_elapsed_time=total_elapsed_time,
        cpu_busy_time=cpu_busy_time,
        switching_overhead=switching_overhead,
        context_switches=context_switches,
    )


def _count_running_process_changes(slices: Sequence[ExecutionSlice]) -> int:
    """Count how often the identity of the running process changes.

    Schedulers merge consecutive slices of the same process (see
    :class:`scheduler.base.Timeline`), so this is the number of adjacent slice pairs
    with different pids.  A single slice -- or an empty trace -- counts as zero.

    Args:
        slices: Chronological execution slices.

    Returns:
        The number of context switches.
    """
    switches = 0
    for previous, current in zip(slices, slices[1:]):
        if previous.pid != current.pid:
            switches += 1
    return switches
