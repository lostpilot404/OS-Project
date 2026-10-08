"""Validation and behaviour of the process / workload data model."""

from __future__ import annotations

import pytest

from errors import ValidationError
from tests.helpers import Row, workload_from_rows
from workload.models import ExecutionSlice, Process, ProcessOutcome, Workload


class TestProcessValidation:
    def test_valid_process_is_accepted(self) -> None:
        process = Process(pid=1, arrival_time=0, burst_time=5, priority=2)
        assert (process.pid, process.arrival_time, process.burst_time, process.priority) == (
            1,
            0,
            5,
            2,
        )

    def test_zero_arrival_is_valid(self) -> None:
        assert Process(0, 0, 1, 0).arrival_time == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pid": -1, "arrival_time": 0, "burst_time": 1, "priority": 0},
            {"pid": 0, "arrival_time": -1, "burst_time": 1, "priority": 0},
            {"pid": 0, "arrival_time": 0, "burst_time": 0, "priority": 0},
            {"pid": 0, "arrival_time": 0, "burst_time": -4, "priority": 0},
            {"pid": 0, "arrival_time": 0, "burst_time": 1, "priority": -2},
        ],
    )
    def test_out_of_range_values_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            Process(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pid": 1.5, "arrival_time": 0, "burst_time": 1, "priority": 0},
            {"pid": 0, "arrival_time": 0.5, "burst_time": 1, "priority": 0},
            {"pid": 0, "arrival_time": 0, "burst_time": 2.5, "priority": 0},
            {"pid": 0, "arrival_time": 0, "burst_time": 1, "priority": 1.5},
            {"pid": True, "arrival_time": 0, "burst_time": 1, "priority": 0},
            {"pid": "1", "arrival_time": 0, "burst_time": 1, "priority": 0},
        ],
    )
    def test_non_integer_values_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            Process(**kwargs)

    def test_integer_like_inputs_are_normalised_to_int(self) -> None:
        process = Process(pid=1, arrival_time=2, burst_time=3, priority=4)
        assert all(
            type(value) is int
            for value in (process.pid, process.arrival_time, process.burst_time, process.priority)
        )


class TestWorkloadValidation:
    def test_canonical_ordering_by_arrival_then_pid(self) -> None:
        rows: list[Row] = [(5, 4, 3, 1), (2, 0, 3, 1), (9, 4, 1, 1), (1, 0, 2, 1)]
        workload = workload_from_rows("ordering", rows)
        assert [p.pid for p in workload.processes] == [1, 2, 5, 9]

    def test_empty_workload_rejected_by_default(self) -> None:
        with pytest.raises(ValidationError, match="at least one process"):
            workload_from_rows("empty", [], allow_empty=False)

    def test_empty_workload_allowed_when_requested(self) -> None:
        workload = workload_from_rows("empty", [], allow_empty=True)
        assert len(workload) == 0 and workload.size == 0

    def test_duplicate_pids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate pids"):
            workload_from_rows("dupes", [(1, 0, 3, 1), (1, 1, 3, 1)])

    def test_non_process_entries_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-Process"):
            Workload(name="bad", processes=[(1, 0, 3, 1)])  # type: ignore[list-item]

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty string"):
            Workload(name="", processes=[Process(1, 0, 1, 1)])


class TestWorkloadProperties:
    def test_aggregate_properties(self) -> None:
        workload = workload_from_rows("agg", [(1, 0, 4, 1), (2, 2, 8, 5), (3, 6, 6, 3)])
        assert workload.total_burst_time == 18
        assert workload.mean_burst_time == pytest.approx(6.0)
        assert workload.median_burst_time == pytest.approx(6.0)
        assert workload.arrival_span == 6
        assert workload.min_arrival_time == 0
        assert workload.max_arrival_time == 6

    def test_ready_at_returns_processes_that_have_arrived(self) -> None:
        workload = workload_from_rows("ready", [(1, 0, 1, 1), (2, 3, 1, 1), (3, 3, 1, 1)])
        assert [p.pid for p in workload.ready_at(0)] == [1]
        assert [p.pid for p in workload.ready_at(3)] == [1, 2, 3]
        assert [p.pid for p in workload.ready_at(2)] == [1]

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_percentile_rejects_invalid_fraction(self, fraction: float) -> None:
        workload = workload_from_rows("p", [(1, 0, 4, 1)])
        with pytest.raises(ValidationError):
            workload.percentile_burst_time(fraction)

    def test_empty_aggregates_are_zero(self) -> None:
        workload = workload_from_rows("empty", [], allow_empty=True)
        assert workload.total_burst_time == 0
        assert workload.mean_burst_time == 0.0
        assert workload.median_burst_time == 0.0
        assert workload.arrival_span == 0

    def test_fingerprint_is_stable_and_input_order_independent(self) -> None:
        rows: list[Row] = [(1, 0, 3, 1), (2, 1, 4, 2)]
        first = workload_from_rows("same", rows)
        second = workload_from_rows("same", list(reversed(rows)))
        assert first.fingerprint == second.fingerprint
        assert first == second

    def test_fingerprint_changes_with_content(self) -> None:
        first = workload_from_rows("a", [(1, 0, 3, 1)])
        second = workload_from_rows("a", [(1, 0, 3, 2)])
        assert first.fingerprint != second.fingerprint

    def test_iteration_and_indexing(self) -> None:
        workload = workload_from_rows("it", [(1, 0, 3, 1), (2, 1, 4, 2)])
        assert [p.pid for p in workload] == [1, 2]
        assert workload[1].pid == 2
        assert workload[0].burst_time == 3


class TestExecutionSliceAndOutcome:
    def test_slice_length_is_validated(self) -> None:
        with pytest.raises(ValidationError, match="end_time > start_time"):
            ExecutionSlice(pid=1, start_time=5, end_time=5)
        with pytest.raises(ValidationError):
            ExecutionSlice(pid=1, start_time=5, end_time=4)

    def test_slice_duration(self) -> None:
        assert ExecutionSlice(1, 2, 7).duration == 5

    def test_outcome_derives_the_three_time_metrics(self) -> None:
        outcome = ProcessOutcome(
            pid=1,
            arrival_time=2,
            burst_time=5,
            priority=3,
            start_time=10,
            completion_time=15,
        )
        assert outcome.turnaround_time == 13  # 15 - 2
        assert outcome.waiting_time == 8  # 13 - 5
        assert outcome.response_time == 8  # 10 - 2
