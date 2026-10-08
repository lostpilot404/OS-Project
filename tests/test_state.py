"""State construction, discretisation and encoding."""

from __future__ import annotations

import pytest

from config import StateConfig
from errors import ValidationError
from rl.state import LinearBinner, LogBinner, StateEncoder, StateSnapshot, observe_workload_state
from tests.helpers import Row, workload_from_rows


class TestObservation:
    def test_values_match_the_documented_formulas(self) -> None:
        # Sorted bursts [2, 4, 6, 8] have conventional median (4+6)/2 = 5.
        # Mean = 5; population std = sqrt(((2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2)/4) = sqrt(5).
        # priorities 1, 2, 2, 5 -> mean = 2.5, std = sqrt(2.25) = 1.5
        # arrivals 0, 2, 6, 10 -> span = 10, total burst = 20 -> offered load = 2.0
        rows: list[Row] = [(1, 0, 4, 1), (2, 2, 8, 2), (3, 6, 6, 2), (4, 10, 2, 5)]
        snapshot = observe_workload_state(workload_from_rows("obs", rows))
        assert snapshot.burst_profile == pytest.approx(5.0)
        assert snapshot.burst_dispersion == pytest.approx((5.0**0.5) / 5.0)
        assert snapshot.offered_load == pytest.approx(2.0)
        assert snapshot.priority_spread == pytest.approx(1.5)

    def test_batch_arrivals_use_a_span_of_one(self) -> None:
        # All arriving at t=0 -> span 0 -> the documented convention uses span 1.
        snapshot = observe_workload_state(
            workload_from_rows("batch", [(1, 0, 5, 1), (2, 0, 7, 1)])
        )
        assert snapshot.offered_load == pytest.approx(12.0)

    def test_zero_dispersion_for_equal_bursts(self) -> None:
        snapshot = observe_workload_state(
            workload_from_rows("equal", [(1, 0, 4, 1), (2, 1, 4, 1)])
        )
        assert snapshot.burst_dispersion == pytest.approx(0.0)
        assert snapshot.priority_spread == pytest.approx(0.0)

    def test_single_process(self) -> None:
        snapshot = observe_workload_state(workload_from_rows("one", [(1, 3, 9, 4)]))
        assert snapshot.burst_profile == pytest.approx(9.0)
        assert snapshot.burst_dispersion == pytest.approx(0.0)
        assert snapshot.offered_load == pytest.approx(9.0)
        assert snapshot.priority_spread == pytest.approx(0.0)

    def test_empty_workload_gives_zero_state(self) -> None:
        snapshot = observe_workload_state(workload_from_rows("empty", [], allow_empty=True))
        assert (snapshot.burst_profile, snapshot.burst_dispersion, snapshot.offered_load) == (
            0.0,
            0.0,
            0.0,
        )
        assert snapshot.priority_spread == 0.0

    def test_state_is_independent_of_input_order(self) -> None:
        rows: list[Row] = [(1, 0, 4, 1), (2, 5, 9, 3), (3, 2, 3, 2)]
        first = observe_workload_state(workload_from_rows("a", rows))
        second = observe_workload_state(workload_from_rows("a", list(reversed(rows))))
        assert first == second

    def test_state_does_not_depend_on_any_scheduling_outcome(self) -> None:
        # The observation function only accepts a workload; two workloads that differ
        # only in their *scheduling* behaviour (impossible here, since a workload carries
        # no scheduling result) must be indistinguishable.  The concrete check is that the
        # snapshot equals a recomputation from the workload description alone.
        workload = workload_from_rows("pure", [(1, 0, 4, 1), (2, 1, 6, 2)])
        first = observe_workload_state(workload)
        second = observe_workload_state(workload)
        assert first == second
        assert first.burst_profile == workload.median_burst_time

    def test_invalid_argument_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a Workload"):
            observe_workload_state([(1, 0, 4, 1)])  # type: ignore[arg-type]


class TestBinners:
    def test_linear_bins_split_the_range_evenly(self) -> None:
        binner = LinearBinner(0.0, 3.0, 3)
        assert binner.interior_edges == pytest.approx((1.0, 2.0))
        # A value exactly on an edge belongs to the upper bin (documented convention).
        assert [binner.index(value) for value in (0.0, 0.5, 1.0, 1.9, 2.0, 5.0)] == [0, 0, 1, 1, 2, 2]

    def test_log_bins_span_equal_factors(self) -> None:
        binner = LogBinner(1.0, 100.0, 2)
        assert binner.interior_edges == pytest.approx((10.0,))
        assert [binner.index(value) for value in (0.5, 1.0, 9.9, 10.0, 100.0, 1000.0)] == [
            0,
            0,
            0,
            1,
            1,
            1,
        ]

    def test_edges_are_open_at_both_ends(self) -> None:
        binner = LinearBinner(0.0, 4.0, 4)
        assert binner.lower_bounds()[0] == float("-inf")
        assert binner.upper_bound_of(binner.bins - 1) == float("inf")

    def test_describe_reports_the_edges(self) -> None:
        assert "2 bins" in LogBinner(1.0, 10.0, 2).describe()

    def test_invalid_ranges_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LinearBinner(2.0, 1.0, 3)
        with pytest.raises(ValidationError):
            LinearBinner(0.0, 1.0, 1)
        with pytest.raises(ValidationError):
            LogBinner(1.0, 10.0, 3).index("high")  # type: ignore[arg-type]


class TestEncoding:
    def test_state_space_size(self) -> None:
        encoder = StateEncoder(StateConfig())
        assert encoder.n_states == encoder.n_bins ** len(encoder.variables)
        assert encoder.n_states == 81

    def test_encode_decode_round_trip_for_every_state(self) -> None:
        encoder = StateEncoder(StateConfig())
        for index in range(encoder.n_states):
            bins = encoder.decode(index)
            assert len(bins) == len(encoder.variables)
            assert all(0 <= value < encoder.n_bins for value in bins.values())
            # Rebuild a snapshot that lies inside each reported bin and encode it again.
            snapshot = StateSnapshot(
                **{
                    variable: _midpoint(encoder, variable, bins[variable])
                    for variable in encoder.variables
                }
            )
            assert encoder.encode(snapshot) == index

    def test_encoding_is_monotonic_in_the_first_variable(self) -> None:
        encoder = StateEncoder(StateConfig())
        low = encoder.encode(StateSnapshot(1.0, 0.0, 1.0, 0.0))
        high = encoder.encode(StateSnapshot(50.0, 0.0, 1.0, 0.0))
        assert low < high

    def test_values_outside_the_ranges_clamp_to_the_outer_bins(self) -> None:
        encoder = StateEncoder(StateConfig())
        tiny = encoder.decode(encoder.encode(StateSnapshot(0.0, 0.0, 0.0, 0.0)))
        huge = encoder.decode(encoder.encode(StateSnapshot(10_000.0, 99.0, 10_000.0, 99.0)))
        assert all(value == 0 for value in tiny.values())
        assert all(value == encoder.n_bins - 1 for value in huge.values())

    def test_custom_variable_selection_and_bin_count(self) -> None:
        config = StateConfig(state_variables=("burst_profile", "offered_load"), bins_per_variable=5)
        encoder = StateEncoder(config)
        assert encoder.variables == ("burst_profile", "offered_load")
        assert encoder.n_states == 25
        assert encoder.encode(StateSnapshot(5.0, 5.0, 0.0, 0.0)) < encoder.n_states

    def test_unknown_variable_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown state variable"):
            StateConfig(state_variables=("burst_profile", "not_a_variable"))

    def test_duplicate_variables_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            StateConfig(state_variables=("burst_profile", "burst_profile"))

    def test_encode_rejects_wrong_types(self) -> None:
        encoder = StateEncoder(StateConfig())
        with pytest.raises(ValidationError, match="StateSnapshot"):
            encoder.encode((1.0, 2.0, 3.0, 4.0))  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="StateConfig"):
            StateEncoder("not a config")  # type: ignore[arg-type]

    def test_decode_rejects_out_of_range_indices(self) -> None:
        encoder = StateEncoder(StateConfig())
        with pytest.raises(ValidationError, match="state index"):
            encoder.decode(encoder.n_states)
        with pytest.raises(ValidationError, match="state index"):
            encoder.decode(-1)
        with pytest.raises(ValidationError, match="state index"):
            encoder.decode(1.5)  # type: ignore[arg-type]

    def test_binner_lookup_of_a_variable_outside_the_state_is_rejected(self) -> None:
        encoder = StateEncoder(StateConfig(state_variables=("burst_profile",)))
        with pytest.raises(ValidationError, match="not part of the state"):
            encoder.binner("priority_spread")

    def test_real_workloads_land_in_distinct_states_per_condition(self) -> None:
        from config import build_default_config
        from workload.generator import WorkloadGenerator

        config = build_default_config()
        generator = WorkloadGenerator(config.families)
        encoder = StateEncoder(config.state)
        states = {
            family.name: {
                encoder.encode(observe_workload_state(generator.generate(family.name, seed)))
                for seed in range(5)
            }
            for family in config.families
        }
        # The short-job and the CPU-bound conditions must not share a state.
        assert not (states["short_jobs"] & states["cpu_bursty"])
        assert not (states["short_jobs"] & states["poisson_arrivals"])


def _midpoint(encoder: StateEncoder, variable: str, bin_index: int) -> float:
    """Return a value inside the given bin of a variable."""
    binner = encoder.binner(variable)
    lower = 0.0 if bin_index == 0 else binner.interior_edges[bin_index - 1]
    upper = binner.interior_edges[bin_index] if bin_index < binner.bins - 1 else lower * 2 + 1.0
    if lower == float("-inf"):
        lower = 0.0
    return (lower + upper) / 2.0
