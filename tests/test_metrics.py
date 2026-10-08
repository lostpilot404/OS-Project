"""Metric definitions, checked against hand-computed values."""

from __future__ import annotations

import pytest

from config import SchedulerConfig
from errors import ValidationError
from scheduler.fcfs import FCFS
from scheduler.round_robin import RoundRobin
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, workload_from_rows
from evaluation.metrics import WorkloadMetrics, compute_metrics


def _run_fcfs(rows, config: SchedulerConfig = DEFAULT_SCHEDULER_CONFIG) -> WorkloadMetrics:
    return compute_metrics(FCFS(config).run(workload_from_rows("m", rows)))


class TestTimeMetrics:
    def test_waiting_turnaround_and_response_of_a_known_schedule(self) -> None:
        # P1 0-24, P2 24-27, P3 27-30 (see test_fcfs).
        metrics = _run_fcfs([(1, 0, 24, 1), (2, 0, 3, 1), (3, 0, 3, 1)])
        assert metrics.total_waiting_time == 51
        assert metrics.total_turnaround_time == 81
        assert metrics.total_response_time == 51
        assert metrics.avg_waiting_time == pytest.approx(17.0)
        assert metrics.avg_turnaround_time == pytest.approx(27.0)
        assert metrics.avg_response_time == pytest.approx(17.0)

    def test_means_equal_totals_divided_by_the_process_count(self) -> None:
        metrics = _run_fcfs([(1, 0, 7, 1), (2, 3, 2, 1), (3, 4, 9, 1)])
        n = metrics.num_processes
        assert metrics.avg_waiting_time == pytest.approx(metrics.total_waiting_time / n)
        assert metrics.avg_turnaround_time == pytest.approx(metrics.total_turnaround_time / n)
        assert metrics.avg_response_time == pytest.approx(metrics.total_response_time / n)

    def test_single_process_has_zero_waiting_and_zero_response(self) -> None:
        metrics = _run_fcfs([(1, 3, 6, 1)])
        assert metrics.avg_waiting_time == 0.0
        assert metrics.avg_response_time == 0.0
        assert metrics.avg_turnaround_time == pytest.approx(6.0)

    def test_response_time_can_be_lower_than_waiting_time_when_preempted(self) -> None:
        # Quantum 2, P1(0, 8), P2(2, 6).  Hand computation:
        #   P1 0-2, P2 2-4, P1 4-6, P2 6-8, P1 8-10, P2 10-12, P1 12-14
        #   P1: waiting 6, response 0; P2: waiting 4, response 0
        #   -> both start immediately, so the mean response time is 0 while the mean
        #      waiting time is 5.  Waiting is *not* the same as response time.
        config = SchedulerConfig(round_robin_quantum=2)
        workload = workload_from_rows("rr", [(1, 0, 8, 1), (2, 2, 6, 1)])
        result = RoundRobin(config).run(workload)
        metrics = compute_metrics(result)
        assert result.outcome(2).response_time == 0
        assert result.outcome(2).start_time == 2
        assert result.outcome(1).waiting_time == 6
        assert result.outcome(2).waiting_time == 4
        assert metrics.avg_response_time == pytest.approx(0.0)
        assert metrics.avg_waiting_time == pytest.approx(5.0)
        assert metrics.context_switches == 6


class TestSystemMetrics:
    def test_cpu_utilisation_counts_busy_time_over_makespan(self) -> None:
        # Busy 8 of 13 time units.
        metrics = _run_fcfs([(1, 0, 5, 1), (2, 10, 3, 1)])
        assert metrics.cpu_busy_time == 8
        assert metrics.total_elapsed_time == 13
        assert metrics.cpu_idle_time == 5
        assert metrics.cpu_utilization == pytest.approx(100 * 8 / 13)

    def test_full_utilisation_when_no_idle_time(self) -> None:
        metrics = _run_fcfs([(1, 0, 4, 1), (2, 0, 6, 1)])
        assert metrics.cpu_utilization == pytest.approx(100.0)

    def test_throughput_is_processes_over_makespan(self) -> None:
        metrics = _run_fcfs([(1, 0, 4, 1), (2, 0, 6, 1)])
        assert metrics.throughput == pytest.approx(2 / 10)

    def test_context_switches_and_per_process_rate(self) -> None:
        metrics = _run_fcfs([(1, 0, 2, 1), (2, 0, 2, 1), (3, 0, 2, 1), (4, 0, 2, 1)])
        assert metrics.context_switches == 3
        assert metrics.context_switches_per_process == pytest.approx(0.75)

    def test_switching_overhead_reduces_utilisation_but_not_busy_time(self) -> None:
        config = SchedulerConfig(switching_cost=2)
        metrics = compute_metrics(
            FCFS(config).run(workload_from_rows("cost", [(1, 0, 3, 1), (2, 0, 3, 1)]))
        )
        assert metrics.cpu_busy_time == 6
        assert metrics.total_elapsed_time == 8
        assert metrics.cpu_idle_time == 0
        assert metrics.cpu_utilization == pytest.approx(75.0)


class TestEmptyAndIdentity:
    def test_all_six_required_metrics_match_independent_hand_calculation(self) -> None:
        # P1 runs 0-4; the CPU is idle 4-6; P2 runs 6-8. Both begin at arrival.
        # Per-process waits are (0, 0), turns are (4, 2), responses are (0, 0).
        # Busy=6, makespan=8, completed=2, and there is one process change.
        metrics = _run_fcfs([(1, 0, 4, 1), (2, 6, 2, 1)])
        assert metrics.avg_waiting_time == pytest.approx(0.0)
        assert metrics.avg_turnaround_time == pytest.approx(3.0)
        assert metrics.avg_response_time == pytest.approx(0.0)
        assert metrics.cpu_utilization == pytest.approx(75.0)
        assert metrics.throughput == pytest.approx(0.25)
        assert metrics.context_switches == 1
        # Per-process switches are a derived reward input, not an additional required metric.
        assert metrics.context_switches_per_process == pytest.approx(0.5)

    def test_empty_workload_metrics_are_zero(self) -> None:
        metrics = compute_metrics(
            FCFS(DEFAULT_SCHEDULER_CONFIG).run(
                workload_from_rows("empty", [], allow_empty=True)
            )
        )
        assert metrics.num_processes == 0
        assert metrics.avg_waiting_time == 0.0
        assert metrics.avg_turnaround_time == 0.0
        assert metrics.avg_response_time == 0.0
        assert metrics.cpu_utilization == 0.0
        assert metrics.throughput == 0.0
        assert metrics.context_switches == 0
        assert metrics.context_switches_per_process == 0.0
        assert metrics.total_elapsed_time == 0

    def test_metrics_carry_the_workload_identity(self) -> None:
        workload = workload_from_rows("identity", [(1, 0, 3, 1)])
        metrics = compute_metrics(FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload))
        assert metrics.policy_name == "FCFS"
        assert metrics.workload_name == "identity"
        assert metrics.workload_fingerprint == workload.fingerprint

    def test_metrics_convert_to_a_row_for_tables(self) -> None:
        metrics = _run_fcfs([(1, 0, 3, 1)])
        row = metrics.as_dict()
        assert row["avg_waiting_time"] == 0.0
        assert set(row) == set(WorkloadMetrics.__dataclass_fields__)

    def test_invalid_argument_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a ScheduleResult"):
            compute_metrics({"avg_waiting_time": 1.0})  # type: ignore[arg-type]
