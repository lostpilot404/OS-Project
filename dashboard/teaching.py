"""The small three-process worked example, executed by the real scheduler classes.

The dashboard's teaching panel explains *why* four policies order the same jobs
differently. Its numbers must therefore come from ``scheduler/`` -- the same code the
simulator runs -- and not from a table typed into the HTML. This module executes the
documented example with the four conventional policies, verifies the result against the
hand calculation printed in ``docs/TEACHER_LIVE_DEMO.md`` (section C) and exposes the
traces for drawing.

Configuration matches the hand calculation: round-robin quantum 2, switching cost 0.
The live self-healing demo uses quantum 4 and switching cost 1; that difference is shown
on the page rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from config import SchedulerConfig
from evaluation.metrics import compute_metrics
from scheduler import POLICY_CLASSES
from workload.models import ExecutionSlice, Process, ScheduleResult, Workload

from dashboard.model import DashboardError, Interval, Outcome

__all__ = [
    "TEACHING_QUANTUM",
    "TEACHING_SWITCH_COST",
    "TeachingPolicy",
    "TeachingPanel",
    "build_teaching_panel",
]

#: Quantum used by the hand-calculated example in the teacher guide.
TEACHING_QUANTUM = 2
#: Switching cost used by the hand-calculated example (none, for hand arithmetic).
TEACHING_SWITCH_COST = 0

#: The documented hand calculation: per-process waiting times and completion times, in
#: pid order (P1, P2, P3), for each of the four policies.
_HAND_WAITING: Dict[str, Tuple[int, int, int]] = {
    "FCFS": (0, 5, 8),
    "SJF": (4, 1, 0),
    "Priority": (3, 0, 8),
    "Round Robin": (4, 5, 4),
}
_HAND_COMPLETION: Dict[str, Tuple[int, int, int]] = {
    "FCFS": (5, 8, 9),
    "SJF": (9, 4, 1),
    "Priority": (8, 3, 9),
    "Round Robin": (9, 8, 5),
}


@dataclass(frozen=True)
class TeachingPolicy:
    """One executed policy on the three-process teaching workload."""

    name: str
    intervals: Tuple[Interval, ...]
    outcomes: Tuple[Outcome, ...]
    mean_waiting_time: float
    mean_turnaround_time: float
    mean_response_time: float
    context_switches: int
    elapsed_time: int
    hand_verified: bool
    throughput: float = 0.0
    cpu_utilization: float = 100.0

    @property
    def service_order(self) -> str:
        """The order in which processes were first dispatched."""
        seen: list[str] = []
        for interval in self.intervals:
            label = f"P{interval.pid}"
            if label not in seen:
                seen.append(label)
        return " -> ".join(seen)

    @property
    def domain_end(self) -> int:
        return max(self.elapsed_time, max((row.end for row in self.intervals), default=0))


@dataclass(frozen=True)
class TeachingPanel:
    """The four-policy comparison for the worked example."""

    processes: Tuple[Process, ...]
    quantum: int
    switching_cost: int
    policies: Tuple[TeachingPolicy, ...]

    @property
    def hand_verified(self) -> bool:
        return all(policy.hand_verified for policy in self.policies)

    @property
    def best_by_waiting(self) -> str:
        return min(self.policies, key=lambda policy: policy.mean_waiting_time).name


def _intervals_from_trace(trace: Tuple[ExecutionSlice, ...], policy: str) -> Tuple[Interval, ...]:
    return tuple(
        Interval(
            index=index,
            pid=slice_.pid,
            policy=policy,
            start=slice_.start_time,
            end=slice_.end_time,
            switch_before=0,
            idle_before=0,
            waiting_increment=None,
            completed_pids=(),
        )
        for index, slice_ in enumerate(trace)
    )


def _outcomes_from_result(result: ScheduleResult) -> Tuple[Outcome, ...]:
    return tuple(
        Outcome(
            pid=outcome.pid,
            arrival_time=outcome.arrival_time,
            burst_time=outcome.burst_time,
            priority=outcome.priority,
            start_time=outcome.start_time,
            completion_time=outcome.completion_time,
        )
        for outcome in result.outcomes
    )


def _execute(policy_class, workload: Workload, config: SchedulerConfig) -> TeachingPolicy:
    result = policy_class(config).run(workload)
    metrics = compute_metrics(result)
    outcomes = _outcomes_from_result(result)
    hand_verified = (
        tuple(outcome.waiting_time for outcome in outcomes) == _HAND_WAITING[policy_class.name]
        and tuple(outcome.completion_time for outcome in outcomes)
        == _HAND_COMPLETION[policy_class.name]
    )
    return TeachingPolicy(
        name=policy_class.name,
        intervals=_intervals_from_trace(result.trace, policy_class.name),
        outcomes=outcomes,
        mean_waiting_time=metrics.avg_waiting_time,
        mean_turnaround_time=metrics.avg_turnaround_time,
        mean_response_time=metrics.avg_response_time,
        context_switches=metrics.context_switches,
        elapsed_time=metrics.total_elapsed_time,
        hand_verified=hand_verified,
        throughput=metrics.throughput,
        cpu_utilization=metrics.cpu_utilization,
    )


def build_teaching_panel() -> TeachingPanel:
    """Run the documented three-process example through the four real policies.

    Returns:
        The executed traces and metrics, each flagged with whether it reproduces the
        hand calculation in the teacher guide.

    Raises:
        DashboardError: If the schedulers were reordered or the documented hand
            calculation no longer reproduces, so a stale guide cannot silently disagree
            with the chart.
    """
    config = SchedulerConfig(
        round_robin_quantum=TEACHING_QUANTUM, switching_cost=TEACHING_SWITCH_COST
    )
    workload = Workload(
        "teacher-hand-example",
        [Process(1, 0, 5, 2), Process(2, 0, 3, 1), Process(3, 0, 1, 3)],
    )
    policies = tuple(_execute(policy_class, workload, config) for policy_class in POLICY_CLASSES)
    panel = TeachingPanel(
        processes=tuple(workload.processes),
        quantum=TEACHING_QUANTUM,
        switching_cost=TEACHING_SWITCH_COST,
        policies=policies,
    )
    if not panel.hand_verified:
        detail = ", ".join(
            f"{policy.name}: waiting {[row.waiting_time for row in policy.outcomes]}, "
            f"completion {[row.completion_time for row in policy.outcomes]}"
            for policy in panel.policies
        )
        raise DashboardError(
            "the three-process worked example no longer reproduces the hand calculation in "
            f"docs/TEACHER_LIVE_DEMO.md section C ({detail}); update the guide before "
            "presenting the dashboard"
        )
    return panel
