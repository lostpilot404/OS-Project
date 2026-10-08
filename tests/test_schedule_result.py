"""Validation and derived values of :func:`workload.models.build_schedule_result`."""

from __future__ import annotations

import pytest

from errors import ValidationError
from tests.helpers import workload_from_rows
from workload.models import ExecutionSlice, build_schedule_result


def _result(rows, slices, switching_cost: int = 0):
    workload = workload_from_rows("test", rows)
    return build_schedule_result(workload, slices, "Test", switching_cost)


class TestValidation:
    def test_overlapping_slices_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            _result(
                [(1, 0, 5, 1), (2, 0, 5, 1)],
                [ExecutionSlice(1, 0, 5), ExecutionSlice(2, 3, 8)],
            )

    def test_out_of_order_slices_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="overlap or are out of order"):
            _result(
                [(1, 0, 5, 1), (2, 0, 5, 1)],
                [ExecutionSlice(2, 5, 10), ExecutionSlice(1, 0, 5)],
            )

    def test_process_must_not_start_before_it_arrives(self) -> None:
        with pytest.raises(ValidationError, match="only arrives at"):
            _result([(1, 4, 5, 1)], [ExecutionSlice(1, 0, 5)])

    def test_executed_time_must_equal_the_burst_time(self) -> None:
        with pytest.raises(ValidationError, match="executed"):
            _result([(1, 0, 5, 1)], [ExecutionSlice(1, 0, 4)])

    def test_unknown_pid_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown pid"):
            _result([(1, 0, 5, 1)], [ExecutionSlice(1, 0, 5), ExecutionSlice(7, 5, 6)])

    def test_idling_while_work_is_pending_is_rejected(self) -> None:
        # Process 1 and 2 are both ready at time 0, yet the CPU idles from 5 to 8.
        with pytest.raises(ValidationError, match="CPU idle"):
            _result(
                [(1, 0, 5, 1), (2, 0, 3, 1)],
                [ExecutionSlice(1, 0, 5), ExecutionSlice(2, 8, 11)],
            )

    def test_idling_before_the_first_arrival_is_allowed(self) -> None:
        result = _result([(1, 6, 5, 1)], [ExecutionSlice(1, 6, 11)])
        assert result.idle_time == 6

    def test_gap_explained_by_the_switch_cost_is_allowed(self) -> None:
        result = _result(
            [(1, 0, 5, 1), (2, 0, 3, 1)],
            [ExecutionSlice(1, 0, 5), ExecutionSlice(2, 7, 10)],
            switching_cost=2,
        )
        assert result.switching_overhead == 2
        assert result.idle_time == 0

    def test_negative_switch_cost_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="switching_cost"):
            _result([(1, 0, 5, 1)], [ExecutionSlice(1, 0, 5)], switching_cost=-1)


class TestDerivedValues:
    def test_single_slice_has_no_context_switch(self) -> None:
        result = _result([(1, 3, 5, 1)], [ExecutionSlice(1, 3, 8)])
        assert result.context_switches == 0
        assert result.total_elapsed_time == 8
        assert result.cpu_busy_time == 5
        assert result.idle_time == 3

    def test_context_switches_count_changes_of_the_running_process(self) -> None:
        result = _result(
            [(1, 0, 4, 1), (2, 0, 3, 1)],
            [ExecutionSlice(1, 0, 2), ExecutionSlice(2, 2, 5), ExecutionSlice(1, 5, 7)],
        )
        assert result.context_switches == 2

    def test_empty_trace_yields_an_empty_result(self) -> None:
        workload = workload_from_rows("empty", [], allow_empty=True)
        result = build_schedule_result(workload, [], "Test", 0)
        assert result.outcomes == ()
        assert result.trace == ()
        assert result.total_elapsed_time == 0
        assert result.cpu_busy_time == 0
        assert result.context_switches == 0
        assert result.idle_time == 0

    def test_outcomes_and_trace_carry_the_workload_identity(self) -> None:
        result = _result([(1, 0, 5, 1)], [ExecutionSlice(1, 0, 5)])
        assert result.workload_name == "test"
        assert result.workload_fingerprint == workload_from_rows(
            "test", [(1, 0, 5, 1)]
        ).fingerprint
        assert result.outcome(1).completion_time == 5
        assert result.starts_at == {1: 0}
        assert result.completion_times == {1: 5}

    def test_outcome_lookup_of_unknown_pid_raises(self) -> None:
        result = _result([(1, 0, 5, 1)], [ExecutionSlice(1, 0, 5)])
        with pytest.raises(ValidationError, match="not part of schedule"):
            result.outcome(99)
