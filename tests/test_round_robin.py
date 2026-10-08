"""Round Robin correctness, verified against hand-computed schedules."""

from __future__ import annotations

import pytest

from config import SchedulerConfig
from errors import ConfigurationError
from evaluation.metrics import compute_metrics
from scheduler.fcfs import FCFS
from scheduler.round_robin import RoundRobin
from tests.helpers import responses, timing, trace, workload_from_rows

QUANTUM_4 = SchedulerConfig(round_robin_quantum=4)


class TestHandComputedSchedules:
    def test_quantum_four_with_one_long_and_two_short_jobs(self) -> None:
        # Hand computation with quantum 4: P1(24), P2(3), P3(3), all arriving at 0.
        #   P1 0-4, P2 4-7 (done), P3 7-10 (done), then P1 repeatedly 10-14, 14-18,
        #   18-22, 22-26, 26-30; those slices merge into one 10-30 slice because P1 is
        #   the only ready process.
        #   waiting times: P1 6, P2 4, P3 7 -> mean = 17 / 3
        workload = workload_from_rows("rr", [(1, 0, 24, 1), (2, 0, 3, 1), (3, 0, 3, 1)])
        result = RoundRobin(QUANTUM_4).run(workload)
        assert trace(result) == [(1, 0, 4), (2, 4, 7), (3, 7, 10), (1, 10, 30)]
        assert timing(result) == [
            (1, 0, 30, 6, 30),
            (2, 4, 7, 4, 7),
            (3, 7, 10, 7, 10),
        ]
        assert responses(result) == [(1, 0), (2, 4), (3, 7)]
        metrics = compute_metrics(result)
        assert metrics.avg_waiting_time == pytest.approx(17 / 3)
        assert metrics.context_switches == 3

    def test_quantum_one_alternates_between_equal_jobs(self) -> None:
        workload = workload_from_rows("rr1", [(1, 0, 3, 1), (2, 0, 3, 1)])
        result = RoundRobin(SchedulerConfig(round_robin_quantum=1)).run(workload)
        assert trace(result) == [(1, 0, 1), (2, 1, 2), (1, 2, 3), (2, 3, 4), (1, 4, 5), (2, 5, 6)]
        assert timing(result) == [(1, 0, 5, 2, 5), (2, 1, 6, 3, 6)]
        assert result.context_switches == 5

    def test_burst_equal_to_quantum_finishes_in_one_turn(self) -> None:
        workload = workload_from_rows("exact", [(1, 0, 4, 1), (2, 0, 4, 1)])
        result = RoundRobin(QUANTUM_4).run(workload)
        assert trace(result) == [(1, 0, 4), (2, 4, 8)]
        assert result.context_switches == 1

    def test_burst_of_two_quanta_returns_to_the_queue_once(self) -> None:
        workload = workload_from_rows("double", [(1, 0, 8, 1), (2, 0, 3, 1)])
        result = RoundRobin(QUANTUM_4).run(workload)
        assert trace(result) == [(1, 0, 4), (2, 4, 7), (1, 7, 11)]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(3.5)


class TestArrivalConvention:
    def test_arrival_inside_a_slice_precedes_the_preempted_process(self) -> None:
        # Quantum 4, P1(0, 10), P2 arrives at 5 with burst 1:
        #   P1 0-4 (rem 6), queue [P1]; P1 4-8 (rem 2) -- P2 arrives during this slice, so
        #   P2 is queued ahead of the preempted P1; P2 8-9, P1 9-11.
        workload = workload_from_rows("arrival", [(1, 0, 10, 1), (2, 5, 1, 1)])
        result = RoundRobin(QUANTUM_4).run(workload)
        assert trace(result) == [(1, 0, 8), (2, 8, 9), (1, 9, 11)]
        # P1's two slices merge (0-4 and 4-8) because no other process was ready.
        assert timing(result) == [(1, 0, 11, 1, 11), (2, 8, 9, 3, 4)]

    def test_simultaneous_arrivals_enter_the_queue_in_pid_order(self) -> None:
        workload = workload_from_rows("sim", [(3, 0, 2, 1), (1, 0, 2, 1), (2, 0, 2, 1)])
        result = RoundRobin(SchedulerConfig(round_robin_quantum=1)).run(workload)
        assert [s.pid for s in result.trace] == [1, 2, 3, 1, 2, 3]


class TestIdleAndEdgeCases:
    def test_single_process_keeps_the_cpu_and_is_not_counted_as_switching(self) -> None:
        # Quantum 4 with a burst of 5: two allocations, one merged slice, zero switches.
        result = RoundRobin(QUANTUM_4).run(workload_from_rows("one", [(1, 3, 5, 1)]))
        assert trace(result) == [(1, 3, 8)]
        assert result.context_switches == 0
        assert timing(result) == [(1, 3, 8, 0, 5)]

    def test_empty_workload(self) -> None:
        result = RoundRobin(QUANTUM_4).run(
            workload_from_rows("empty", [], allow_empty=True)
        )
        assert result.outcomes == ()
        assert result.total_elapsed_time == 0

    def test_cpu_idles_between_arrivals(self) -> None:
        workload = workload_from_rows("idle", [(1, 0, 3, 1), (2, 10, 2, 1)])
        result = RoundRobin(SchedulerConfig(round_robin_quantum=2)).run(workload)
        assert trace(result) == [(1, 0, 3), (2, 10, 12)]
        assert result.idle_time == 7
        assert result.context_switches == 1

    def test_very_short_processes_with_quantum_one(self) -> None:
        workload = workload_from_rows("tiny", [(1, 0, 1, 1), (2, 0, 1, 1), (3, 0, 1, 1)])
        result = RoundRobin(SchedulerConfig(round_robin_quantum=1)).run(workload)
        assert trace(result) == [(1, 0, 1), (2, 1, 2), (3, 2, 3)]
        assert compute_metrics(result).avg_waiting_time == pytest.approx(1.0)

    def test_long_burst_time_with_small_quantum(self) -> None:
        workload = workload_from_rows("long", [(1, 0, 100, 1), (2, 0, 1, 1)])
        result = RoundRobin(SchedulerConfig(round_robin_quantum=3)).run(workload)
        assert trace(result)[0] == (1, 0, 3)
        assert trace(result)[1] == (2, 3, 4)
        assert result.outcome(1).completion_time == 101
        assert result.outcome(1).waiting_time == 1

    def test_quantum_larger_than_every_burst_behaves_like_fcfs(self) -> None:
        rows = [(1, 0, 2, 1), (2, 0, 3, 1), (3, 1, 4, 1)]
        round_robin = RoundRobin(SchedulerConfig(round_robin_quantum=50)).run(
            workload_from_rows("big-quantum", rows)
        )
        fcfs = FCFS(SchedulerConfig()).run(workload_from_rows("big-quantum", rows))
        assert timing(round_robin) == timing(fcfs)

    def test_deterministic_for_repeated_runs(self) -> None:
        workload = workload_from_rows("det", [(1, 0, 7, 3), (2, 2, 4, 1), (3, 5, 9, 2)])
        assert RoundRobin(QUANTUM_4).run(workload) == RoundRobin(QUANTUM_4).run(workload)


class TestQuantumValidationAndSwitchingCost:
    @pytest.mark.parametrize("quantum", [0, -1])
    def test_non_positive_quantum_is_rejected(self, quantum: int) -> None:
        with pytest.raises(ConfigurationError, match="round_robin_quantum"):
            SchedulerConfig(round_robin_quantum=quantum)

    def test_switching_cost_is_charged_on_each_change(self) -> None:
        config = SchedulerConfig(round_robin_quantum=2, switching_cost=1)
        workload = workload_from_rows("cost", [(1, 0, 4, 1), (2, 0, 2, 1)])
        result = RoundRobin(config).run(workload)
        # P1 0-2, switch, P2 3-5, switch, P1 6-8
        assert trace(result) == [(1, 0, 2), (2, 3, 5), (1, 6, 8)]
        assert result.context_switches == 2
        assert result.switching_overhead == 2
        assert result.idle_time == 0
