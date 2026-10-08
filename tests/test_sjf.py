"""Non-preemptive SJF correctness, verified against hand-computed schedules."""

from __future__ import annotations

import pytest

from evaluation.metrics import compute_metrics
from scheduler.sjf import SJF
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, timing, trace, workload_from_rows


class TestHandComputedSchedules:
    def test_four_processes_arriving_together(self) -> None:
        # Hand computation (all arrive at t=0; bursts P1=6, P2=8, P3=7, P4=3).
        # SJF runs the shortest job first, so the order is P4, P1, P3, P2:
        #   P4 0-3, P1 3-9, P3 9-16, P2 16-24
        #   waiting times 0, 3, 9, 16 -> mean = 28 / 4 = 7
        workload = workload_from_rows(
            "sjf-batch", [(1, 0, 6, 1), (2, 0, 8, 1), (3, 0, 7, 1), (4, 0, 3, 1)]
        )
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert trace(result) == [(4, 0, 3), (1, 3, 9), (3, 9, 16), (2, 16, 24)]
        assert timing(result) == [
            (1, 3, 9, 3, 9),
            (2, 16, 24, 16, 24),
            (3, 9, 16, 9, 16),
            (4, 0, 3, 0, 3),
        ]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(7.0)

    def test_arrivals_during_execution(self) -> None:
        # Hand computation: P1 runs 0-7; then bursts are P2=4, P3=1, P4=4 so P3 runs
        # 7-8, and the equal-burst pair is broken by arrival time (P2 arrived at 2,
        # P4 at 5): P2 8-12, P4 12-16.
        #   waiting times: P1 0, P2 6, P3 3, P4 7 -> mean = 16 / 4 = 4
        workload = workload_from_rows(
            "sjf-online", [(1, 0, 7, 1), (2, 2, 4, 1), (3, 4, 1, 1), (4, 5, 4, 1)]
        )
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert trace(result) == [(1, 0, 7), (3, 7, 8), (2, 8, 12), (4, 12, 16)]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(4.0)


class TestTieBreaking:
    def test_equal_bursts_broken_by_arrival_then_pid(self) -> None:
        # Hand computation:
        #   t=0: ready {1, 5}, both burst 4, both arrival 0 -> smaller pid runs: P1 0-4
        #   t=4: ready {5 (burst 4, arrival 0), 3 (burst 4, arrival 4), 4 (burst 2)}
        #        -> shortest burst is pid 4: P4 4-6
        #        -> then two burst-4 jobs, earlier arrival wins: P5 6-10, P3 10-14
        workload = workload_from_rows(
            "ties", [(5, 0, 4, 1), (1, 0, 4, 1), (3, 4, 4, 1), (4, 4, 2, 1)]
        )
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert [s.pid for s in result.trace] == [1, 4, 5, 3]

    def test_equal_bursts_and_equal_arrivals(self) -> None:
        workload = workload_from_rows("same", [(7, 2, 5, 1), (3, 2, 5, 1), (5, 2, 5, 1)])
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert [s.pid for s in result.trace] == [3, 5, 7]

    def test_priority_is_irrelevant_to_sjf(self) -> None:
        first = workload_from_rows("a", [(1, 0, 5, 9), (2, 0, 3, 1)])
        second = workload_from_rows("b", [(1, 0, 5, 1), (2, 0, 3, 9)])
        assert trace(SJF(DEFAULT_SCHEDULER_CONFIG).run(first)) == trace(
            SJF(DEFAULT_SCHEDULER_CONFIG).run(second)
        )


class TestNonPreemptiveSemantics:
    def test_running_process_is_never_preempted_by_a_shorter_arrival(self) -> None:
        # P1 is long but starts first; the short P2 arrives later and must wait.
        #   P1 0-10, P2 10-11
        workload = workload_from_rows("np", [(1, 0, 10, 1), (2, 1, 1, 1)])
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        assert trace(result) == [(1, 0, 10), (2, 10, 11)]
        assert result.outcome(2).waiting_time == 9

    def test_shortest_job_first_minimises_mean_waiting_for_a_batch(self) -> None:
        # All processes arrive together, so non-preemptive SJF is optimal for the mean
        # waiting time; FCFS on the same input is strictly worse.
        from scheduler.fcfs import FCFS

        workload = workload_from_rows(
            "batch", [(1, 0, 9, 1), (2, 0, 2, 1), (3, 0, 7, 1), (4, 0, 1, 1)]
        )
        sjf_wait = compute_metrics(SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)).avg_waiting_time
        fcfs_wait = compute_metrics(FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)).avg_waiting_time
        assert sjf_wait < fcfs_wait


class TestEdgeCases:
    def test_single_process(self) -> None:
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("one", [(1, 2, 3, 1)]))
        assert timing(result) == [(1, 2, 5, 0, 3)]

    def test_empty_workload(self) -> None:
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("empty", [], allow_empty=True)
        )
        assert result.outcomes == ()

    def test_idle_periods_between_arrivals(self) -> None:
        workload = workload_from_rows("idle", [(1, 0, 2, 1), (2, 20, 9, 1), (3, 21, 1, 1)])
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(workload)
        # CPU free at 2, nothing ready until 20: P2 runs (only candidate), then P3.
        assert trace(result) == [(1, 0, 2), (2, 20, 29), (3, 29, 30)]
        assert result.idle_time == 18

    def test_long_burst_time(self) -> None:
        result = SJF(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("long", [(1, 0, 5_000, 1), (2, 0, 4_999, 1)])
        )
        assert [s.pid for s in result.trace] == [2, 1]

    def test_deterministic_for_repeated_runs(self) -> None:
        workload = workload_from_rows("det", [(1, 0, 7, 3), (2, 2, 4, 1), (3, 5, 9, 2)])
        assert SJF(DEFAULT_SCHEDULER_CONFIG).run(workload) == SJF(
            DEFAULT_SCHEDULER_CONFIG
        ).run(workload)
