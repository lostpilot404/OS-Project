"""FCFS correctness, verified against hand-computed schedules."""

from __future__ import annotations

import pytest

from config import SchedulerConfig
from errors import ValidationError
from evaluation.metrics import compute_metrics
from scheduler.fcfs import FCFS
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, responses, timing, trace, workload_from_rows


class TestHandComputedSchedules:
    def test_three_processes_arriving_together(self) -> None:
        # Hand computation (all arrive at t=0, FCFS order = pid order):
        #   P1 0-24 (wait 0), P2 24-27 (wait 24), P3 27-30 (wait 27)
        #   mean waiting time = (0 + 24 + 27) / 3 = 17
        workload = workload_from_rows(
            "batch", [(1, 0, 24, 1), (2, 0, 3, 1), (3, 0, 3, 1)]
        )
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert timing(result) == [
            (1, 0, 24, 0, 24),
            (2, 24, 27, 24, 27),
            (3, 27, 30, 27, 30),
        ]
        metrics = compute_metrics(result)
        assert metrics.avg_waiting_time == pytest.approx(17.0)
        assert metrics.avg_turnaround_time == pytest.approx(27.0)
        assert metrics.avg_response_time == pytest.approx(17.0)
        assert metrics.context_switches == 2
        assert metrics.cpu_utilization == pytest.approx(100.0)
        assert metrics.throughput == pytest.approx(3 / 30)

    def test_arrival_order_beats_pid_order(self) -> None:
        # P2 arrives first, so FCFS runs P2 before P1 despite the larger pid.
        #   P2 0-4 (wait 0), P3 4-6 (wait 4), P1 6-9 (wait 4)
        workload = workload_from_rows(
            "idle-start", [(1, 2, 3, 1), (2, 0, 4, 1), (3, 0, 2, 1)]
        )
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert trace(result) == [(2, 0, 4), (3, 4, 6), (1, 6, 9)]
        assert timing(result) == [
            (1, 6, 9, 4, 7),
            (2, 0, 4, 0, 4),
            (3, 4, 6, 4, 6),
        ]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(8 / 3)

    def test_cpu_idles_until_the_next_arrival(self) -> None:
        # P1 0-5, CPU idle 5-10, P2 10-13.  Busy 8 of 13 time units.
        workload = workload_from_rows("gap", [(1, 0, 5, 1), (2, 10, 3, 1)])
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert trace(result) == [(1, 0, 5), (2, 10, 13)]
        assert result.idle_time == 5
        metrics = compute_metrics(result)
        assert metrics.cpu_utilization == pytest.approx(100 * 8 / 13)
        assert metrics.throughput == pytest.approx(2 / 13)
        assert metrics.context_switches == 1


class TestEdgeCases:
    def test_single_process(self) -> None:
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("one", [(1, 4, 6, 1)])
        )
        assert timing(result) == [(1, 4, 10, 0, 6)]
        assert responses(result) == [(1, 0)]
        assert result.context_switches == 0

    def test_empty_workload(self) -> None:
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("empty", [], allow_empty=True)
        )
        assert result.outcomes == ()
        assert result.total_elapsed_time == 0
        assert compute_metrics(result).avg_waiting_time == 0.0

    def test_long_burst_time(self) -> None:
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("long", [(1, 0, 10_000, 1), (2, 1, 1, 1)])
        )
        assert result.total_elapsed_time == 10_001
        # P2 arrives at t=1 and starts at t=10_000, so it waits 9_999 time units.
        assert result.outcome(2).start_time == 10_000
        assert result.outcome(2).waiting_time == 9_999

    def test_simultaneous_arrivals_are_ordered_by_pid(self) -> None:
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("ties", [(3, 5, 1, 1), (1, 5, 1, 1), (2, 5, 1, 1)])
        )
        assert [s.pid for s in result.trace] == [1, 2, 3]

    def test_deterministic_for_repeated_runs(self) -> None:
        workload = workload_from_rows(
            "det", [(1, 0, 7, 3), (2, 2, 4, 1), (3, 5, 9, 2)]
        )
        first = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        second = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert first == second

    def test_invalid_workload_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a Workload"):
            FCFS(DEFAULT_SCHEDULER_CONFIG).run([(1, 0, 3, 1)])  # type: ignore[arg-type]


class TestSwitchingCost:
    def test_switching_cost_delays_later_dispatches(self) -> None:
        # switch cost 2: P1 0-3, switch 3-5, P2 5-7.
        config = SchedulerConfig(switching_cost=2)
        result = FCFS(config).run(workload_from_rows("cost", [(1, 0, 3, 1), (2, 0, 2, 1)]))
        assert trace(result) == [(1, 0, 3), (2, 5, 7)]
        assert result.context_switches == 1
        assert result.switching_overhead == 2
        assert result.idle_time == 0
        assert result.outcome(2).start_time == 5

    def test_zero_switching_cost_keeps_the_trace_contiguous(self) -> None:
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("free", [(1, 0, 3, 1), (2, 0, 2, 1)])
        )
        assert trace(result) == [(1, 0, 3), (2, 3, 5)]
        assert result.switching_overhead == 0
