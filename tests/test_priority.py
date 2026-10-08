"""Non-preemptive priority scheduling, verified against hand-computed schedules."""

from __future__ import annotations

import pytest

from config import SchedulerConfig
from errors import ValidationError
from evaluation.metrics import compute_metrics
from scheduler.priority import Priority
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, timing, trace, workload_from_rows


class TestHandComputedSchedules:
    def test_five_processes_arriving_together(self) -> None:
        # Hand computation (lower number = higher priority):
        #   order P2(prio 1) 0-1, P5(prio 2) 1-6, P1(prio 3) 6-16, P3(prio 4) 16-18,
        #   P4(prio 5) 18-19
        #   waiting times 0, 1, 6, 16, 18 -> mean = 41 / 5 = 8.2
        rows = [
            (1, 0, 10, 3),
            (2, 0, 1, 1),
            (3, 0, 2, 4),
            (4, 0, 1, 5),
            (5, 0, 5, 2),
        ]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("prio", rows))
        assert trace(result) == [(2, 0, 1), (5, 1, 6), (1, 6, 16), (3, 16, 18), (4, 18, 19)]
        assert timing(result) == [
            (1, 6, 16, 6, 16),
            (2, 0, 1, 0, 1),
            (3, 16, 18, 16, 18),
            (4, 18, 19, 18, 19),
            (5, 1, 6, 1, 6),
        ]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(8.2)

    def test_high_priority_process_arriving_later_waits_its_turn(self) -> None:
        # Non-preemptive: P1 (low priority) already runs, so P2 must wait for it.
        rows = [(1, 0, 6, 9), (2, 1, 2, 1)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("np", rows))
        assert trace(result) == [(1, 0, 6), (2, 6, 8)]
        assert result.outcome(2).waiting_time == 5

    def test_priority_order_is_not_the_same_as_arrival_order(self) -> None:
        rows = [(1, 0, 5, 3), (2, 0, 5, 1), (3, 0, 5, 2)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("p", rows))
        assert [s.pid for s in result.trace] == [2, 3, 1]


class TestTieBreaking:
    def test_equal_priorities_broken_by_arrival_then_pid(self) -> None:
        # Hand computation: at t=0 the ready set is {1, 5} with equal priority -> smaller
        # pid first; pid 3 arrives later with the same priority, so it runs last.
        rows = [(5, 0, 2, 2), (1, 0, 2, 2), (3, 1, 2, 2)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("ties", rows))
        assert [s.pid for s in result.trace] == [1, 5, 3]

    def test_lower_number_wins_by_default(self) -> None:
        rows = [(1, 0, 3, 5), (2, 0, 3, 1)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("sense", rows))
        assert [s.pid for s in result.trace] == [2, 1]

    def test_higher_number_can_win_when_configured(self) -> None:
        rows = [(1, 0, 3, 5), (2, 0, 3, 1)]
        config = SchedulerConfig(lower_priority_number_is_higher_priority=False)
        result = Priority(config).run(workload_from_rows("sense", rows))
        assert [s.pid for s in result.trace] == [1, 2]

    def test_reversed_sense_reproduces_the_hand_computed_order(self) -> None:
        # Same input as the 8.2 case, but with the inverted priority sense:
        #   order P4 0-1, P3 1-3, P1 3-13, P5 13-18, P2 18-19
        #   waiting times 0, 1, 3, 13, 18 -> mean = 35 / 5 = 7.0
        rows = [
            (1, 0, 10, 3),
            (2, 0, 1, 1),
            (3, 0, 2, 4),
            (4, 0, 1, 5),
            (5, 0, 5, 2),
        ]
        config = SchedulerConfig(lower_priority_number_is_higher_priority=False)
        result = Priority(config).run(workload_from_rows("prio-inv", rows))
        assert [s.pid for s in result.trace] == [4, 3, 1, 5, 2]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(7.0)


class TestEdgeCases:
    def test_single_process(self) -> None:
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("one", [(1, 5, 4, 2)])
        )
        assert timing(result) == [(1, 5, 9, 0, 4)]

    def test_empty_workload(self) -> None:
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(
            workload_from_rows("empty", [], allow_empty=True)
        )
        assert result.outcomes == ()
        assert compute_metrics(result).throughput == 0.0

    def test_equal_priorities_and_equal_bursts(self) -> None:
        rows = [(2, 0, 3, 1), (1, 0, 3, 1), (3, 0, 3, 1)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("flat", rows))
        assert [s.pid for s in result.trace] == [1, 2, 3]

    def test_idle_period_before_the_first_arrival(self) -> None:
        rows = [(1, 7, 3, 1), (2, 7, 2, 2)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("idle", rows))
        assert trace(result) == [(1, 7, 10), (2, 10, 12)]
        assert result.idle_time == 7

    def test_same_priority_uses_arrival_order_over_pid(self) -> None:
        rows = [(1, 0, 3, 1), (2, 4, 2, 1), (3, 2, 2, 1)]
        result = Priority(DEFAULT_SCHEDULER_CONFIG).run(workload_from_rows("arr", rows))
        assert [s.pid for s in result.trace] == [1, 3, 2]

    def test_deterministic_for_repeated_runs(self) -> None:
        rows = [(1, 0, 7, 3), (2, 2, 4, 1), (3, 5, 9, 2)]
        workload = workload_from_rows("det", rows)
        assert Priority(DEFAULT_SCHEDULER_CONFIG).run(workload) == Priority(
            DEFAULT_SCHEDULER_CONFIG
        ).run(workload)

    def test_invalid_workload_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a Workload"):
            Priority(DEFAULT_SCHEDULER_CONFIG).run("not a workload")  # type: ignore[arg-type]
