"""Scheduling performance metrics.

The six metrics required by the project brief are computed here, from a validated
:class:`workload.models.ScheduleResult`.  The definitions are the standard textbook ones
and are implemented in exactly one place:

===================  ====================================================================
Metric               Definition
===================  ====================================================================
Waiting time         ``turnaround_time - burst_time``                                     (per process, averaged)
Turnaround time      ``completion_time - arrival_time``                                   (per process, averaged)
Response time        ``first_execution_time - arrival_time``                              (per process, averaged)
CPU utilisation      ``100 * cpu_busy_time / total_elapsed_time``                         (percent)
Throughput           ``completed_processes / total_elapsed_time``                         (processes per time unit)
Context switches     number of changes of the running process in the execution trace
===================  ====================================================================

For an empty workload every metric is ``0`` (the simulation elapsed no time at all).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from workload.models import ScheduleResult

__all__ = ["WorkloadMetrics", "compute_metrics"]


@dataclass(frozen=True)
class WorkloadMetrics:
    """Metrics of one scheduling run.

    Attributes:
        policy_name: Policy that produced the schedule.
        workload_name: Workload condition that was scheduled.
        workload_fingerprint: Fingerprint of the scheduled workload, so that identical
            inputs can be proven across schedulers.
        num_processes: Number of completed processes.
        total_waiting_time: Sum of the per-process waiting times.
        total_turnaround_time: Sum of the per-process turnaround times.
        total_response_time: Sum of the per-process response times.
        avg_waiting_time: Mean waiting time per process.
        avg_turnaround_time: Mean turnaround time per process.
        avg_response_time: Mean response time per process.
        cpu_utilization: Busy time as a percentage of the total elapsed time.
        throughput: Completed processes per time unit.
        context_switches: Number of changes of the running process.
        context_switches_per_process: Context switches divided by the number of processes.
        cpu_busy_time: Time the CPU spent executing processes.
        cpu_idle_time: Time the CPU spent idle.
        total_elapsed_time: Makespan of the schedule.
    """

    policy_name: str
    workload_name: str
    workload_fingerprint: str
    num_processes: int
    total_waiting_time: int
    total_turnaround_time: int
    total_response_time: int
    avg_waiting_time: float
    avg_turnaround_time: float
    avg_response_time: float
    cpu_utilization: float
    throughput: float
    context_switches: int
    context_switches_per_process: float
    cpu_busy_time: int
    cpu_idle_time: int
    total_elapsed_time: int

    def as_dict(self) -> Dict[str, object]:
        """Return the metrics as a plain dictionary (used to build result tables)."""
        return asdict(self)


def compute_metrics(result: ScheduleResult) -> WorkloadMetrics:
    """Compute all project metrics for a completed schedule.

    Args:
        result: The schedule produced by one policy on one workload.

    Returns:
        The metrics of that run.

    Raises:
        ValidationError: If ``result`` is not a :class:`workload.models.ScheduleResult`.
    """
    if not isinstance(result, ScheduleResult):
        raise ValueError(f"result must be a ScheduleResult, got {type(result).__name__}")

    outcomes = result.outcomes
    num_processes = len(outcomes)
    total_waiting = sum(o.waiting_time for o in outcomes)
    total_turnaround = sum(o.turnaround_time for o in outcomes)
    total_response = sum(o.response_time for o in outcomes)

    if num_processes:
        avg_waiting = total_waiting / num_processes
        avg_turnaround = total_turnaround / num_processes
        avg_response = total_response / num_processes
    else:
        avg_waiting = avg_turnaround = avg_response = 0.0

    elapsed = result.total_elapsed_time
    if elapsed > 0:
        cpu_utilization = 100.0 * result.cpu_busy_time / elapsed
        throughput = num_processes / elapsed
    else:
        cpu_utilization = 0.0
        throughput = 0.0

    switches_per_process = (
        result.context_switches / num_processes if num_processes else 0.0
    )

    return WorkloadMetrics(
        policy_name=result.policy_name,
        workload_name=result.workload_name,
        workload_fingerprint=result.workload_fingerprint,
        num_processes=num_processes,
        total_waiting_time=total_waiting,
        total_turnaround_time=total_turnaround,
        total_response_time=total_response,
        avg_waiting_time=avg_waiting,
        avg_turnaround_time=avg_turnaround,
        avg_response_time=avg_response,
        cpu_utilization=cpu_utilization,
        throughput=throughput,
        context_switches=result.context_switches,
        context_switches_per_process=switches_per_process,
        cpu_busy_time=result.cpu_busy_time,
        cpu_idle_time=result.idle_time,
        total_elapsed_time=elapsed,
    )
