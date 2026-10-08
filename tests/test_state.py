"""State construction, discretisation and encoding."""

from __future__ import annotations

import pytest

from config import StateConfig
from errors import ValidationError
from rl.state import LinearBinner, LogBinner, StateEncoder, StateSnapshot, observe_workload_state
from tests.helpers import Row, workload_from_rows


class TestObservation:
    def test_values_match_the_documented_formulas(self) -> None:
        # bursts 4, 8, 6, 2 -> sorted [2, 4, 6, 8], median (nearest-rank) = 6.
        # mean = 5, population std = sqrt(((2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2)/4) = sqrt(5)
        # priorities 1, 2, 2, 5 -> mean = 2.5, std = sqrt(2.25) = 1.5
        # arrivals 0, 2, 6, 10 -> span = 10, total burst = 20 -> offered load = 2.0
        # long jobs (burst >= 16): none -> long_job_share = 0
        # modal arrival time: all distinct -> arrival_concentration = 1/4
        rows: list[Row] = [(1, 0, 4, 1), (2, 2, 8, 2), (3, 6, 6, 2), (4, 10, 2, 5)]
        snapshot = observe_workload_state(workload_from_rows("obs", rows))
        assert snapshot.burst_profile == pytest.approx(6.0)
        assert snapshot.burst_dispersion == pytest.approx((5.0**0.5) / 5.0)
        assert snapshot.offered_load == pytest.approx(2.0)
        assert snapshot.priority_spread == pytest.approx(1.5)
        assert snapshot.long_job_share == pytest.approx(0.0)
        assert snapshot.arrival_concentration == pytest.approx(0.25)

    def test_long_job_share_counts_the_burst_time_of_long_jobs(self) -> None:
        # threshold 16 by default: bursts 4 and 20 -> long share = 20 / 24
        rows: list[Row] = [(1, 0, 4, 1), (2, 1, 20, 1)]
        snapshot = observe_workload_state(workload_from_rows("long", rows))
        assert snapshot.long_job_share == pytest.approx(20.0 / 24.0)

    def test_long_job_threshold_is_configurable(self) -> None:
        rows: list[Row] = [(1, 0, 4, 1), (2, 1, 20, 1)]
        workload = workload_from_rows("long", rows)
        default = observe_workload_state(workload)
        lowered = observe_workload_state(workload, StateConfig(long_burst_threshold=4))
        assert default.long_job_share == pytest.approx(20.0 / 24.0)
        assert lowered.long_job_share == pytest.approx(1.0)

    def test_batch_arrivals_use_a_span_of_one(self) -> None:
        # All arriving at t=0 -> span 0 -> the documented convention uses span 1.
        snapshot = observe_workload_state(
            workload_from_rows("batch", [(1, 0, 5, 1), (2, 0, 7, 1)])
        )
        assert snapshot.offered_load == pytest.approx(12.0)
        # A batch release means every job shares the modal arrival time.
        assert snapshot.arrival_concentration == pytest.approx(1.0)

    def test_arrival_concentration_counts_the_modal_arrival_time(self) -> None:
        rows: list[Row] = [
            (1, 0, 5, 1),
            (2, 0, 7, 1),
            (3, 0, 3, 1),
            (4, 4, 2, 1),
        ]
        snapshot = observe_workload_state(workload_from_rows("conc", rows))
        assert snapshot.arrival_concentration == pytest.approx(0.75)

    def test_priority_burst_alignment_is_the_pearson_correlation(self) -> None:
        # priority number grows exactly with the burst time -> alignment = +1
        aligned: list[Row] = [(1, 0, 2, 1), (2, 1, 4, 2), (3, 2, 6, 3), (4, 3, 8, 4)]
        snapshot = observe_workload_state(workload_from_rows("aligned", aligned))
        assert snapshot.priority_burst_alignment == pytest.approx(1.0)
        # priority number grows against the burst time -> alignment = -1
        inverted: list[Row] = [(1, 0, 8, 1), (2, 1, 6, 2), (3, 2, 4, 3), (4, 3, 2, 4)]
        snapshot = observe_workload_state(workload_from_rows("inverted", inverted))
        assert snapshot.priority_burst_alignment == pytest.approx(-1.0)

    def test_alignment_is_zero_without_priority_or_burst_variation(self) -> None:
        constant_priority: list[Row] = [(1, 0, 4, 3), (2, 1, 9, 3)]
        snapshot = observe_workload_state(workload_from_rows("cp", constant_priority))
        assert snapshot.priority_burst_alignment == pytest.approx(0.0)
        constant_burst: list[Row] = [(1, 0, 4, 1), (2, 1, 4, 4)]
        snapshot = observe_workload_state(workload_from_rows("cb", constant_burst))
        assert snapshot.priority_burst_alignment == pytest.approx(0.0)

    def test_zero_dispersion_for_equal_bursts(self) -> None:
        snapshot = observe_workload_state(
            workload_from_rows("equal", [(1, 0, 4, 1), (2, 1, 4, 1)])
        )
        assert snapshot.burst_dispersion == pytest.approx(0.0)
        assert snapshot.priority_spread == pytest.approx(0.0)
        assert snapshot.long_job_share == pytest.approx(0.0)

    def test_single_process(self) -> None:
        snapshot = observe_workload_state(workload_from_rows("one", [(1, 3, 9, 4)]))
        assert snapshot.burst_profile == pytest.approx(9.0)
        assert snapshot.burst_dispersion == pytest.approx(0.0)
        assert snapshot.offered_load == pytest.approx(9.0)
        assert snapshot.priority_spread == pytest.approx(0.0)
        assert snapshot.arrival_concentration == pytest.approx(1.0)
        assert snapshot.priority_burst_alignment == pytest.approx(0.0)

    def test_empty_workload_gives_zero_state(self) -> None:
        snapshot = observe_workload_state(workload_from_rows("empty", [], allow_empty=True))
        assert (
            snapshot.burst_profile,
            snapshot.burst_dispersion,
            snapshot.long_job_share,
            snapshot.arrival_concentration,
            snapshot.offered_load,
            snapshot.priority_spread,
            snapshot.priority_burst_alignment,
        ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_state_is_independent_of_input_order(self) -> None:
        rows: list[Row] = [(1, 0, 4, 1), (2, 5, 9, 3), (3, 2, 3, 2)]
        first = observe_workload_state(workload_from_rows("a", rows))
        second = observe_workload_state(workload_from_rows("a", list(reversed(rows))))
        assert first == second

    def test_state_does_not_depend_on_any_scheduling_outcome(self) -> None:
        # The observation function only accepts a workload and a state configuration;
        # two workloads that differ only in their *scheduling* behaviour (impossible
        # here, since a workload carries no scheduling result) must be indistinguishable.
        # The concrete check is that the snapshot equals a recomputation from the
        # workload description alone.
        workload = workload_from_rows("pure", [(1, 0, 4, 1), (2, 1, 6, 2)])
        first = observe_workload_state(workload)
        second = observe_workload_state(workload)
        assert first == second
        assert first.burst_profile == workload.median_burst_time

    def test_invalid_arguments_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a Workload"):
            observe_workload_state([(1, 0, 4, 1)])  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="StateConfig"):
            observe_workload_state(
                workload_from_rows("x", [(1, 0, 4, 1)]), "not a config"  # type: ignore[arg-type]
            )


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
        assert encoder.n_states == 3**7

    def test_default_variables_are_the_documented_seven(self) -> None:
        encoder = StateEncoder(StateConfig())
        assert encoder.variables == (
            "burst_profile",
            "burst_dispersion",
            "long_job_share",
            "arrival_concentration",
            "offered_load",
            "priority_spread",
            "priority_burst_alignment",
        )

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
        low = encoder.encode(StateSnapshot(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        high = encoder.encode(StateSnapshot(50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        assert low < high

    def test_values_outside_the_ranges_clamp_to_the_outer_bins(self) -> None:
        encoder = StateEncoder(StateConfig())
        tiny = encoder.decode(encoder.encode(StateSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0)))
        huge = encoder.decode(encoder.encode(StateSnapshot(10_000.0, 99.0, 10.0, 9.0, 10_000.0, 99.0, 5.0)))
        assert all(value == 0 for value in tiny.values())
        assert all(value == encoder.n_bins - 1 for value in huge.values())

    def test_custom_variable_selection_and_bin_count(self) -> None:
        config = StateConfig(state_variables=("burst_profile", "offered_load"), bins_per_variable=5)
        encoder = StateEncoder(config)
        assert encoder.variables == ("burst_profile", "offered_load")
        assert encoder.n_states == 25
        assert (
            encoder.encode(StateSnapshot(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0)) < encoder.n_states
        )

    def test_unknown_variable_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown state variable"):
            StateConfig(state_variables=("burst_profile", "not_a_variable"))

    def test_duplicate_variables_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            StateConfig(state_variables=("burst_profile", "burst_profile"))

    def test_invalid_long_burst_threshold_is_rejected(self) -> None:
        from errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="long_burst_threshold"):
            StateConfig(long_burst_threshold=0)

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
        # The interactive conditions (staggered arrivals plus long background jobs) must
        # not share a state with the staggered conditions that contain no long jobs,
        # because those are exactly the conditions whose optimal policy differs.
        for interactive in ("interactive", "interactive_sparse"):
            assert not (states[interactive] & states["short_stream"])
            assert not (states[interactive] & states["quantum_sensitive"])
        # Batch and staggered short-job conditions must not share a state either.
        assert not (states["short_batch"] & states["short_stream"])


def _midpoint(encoder: StateEncoder, variable: str, bin_index: int) -> float:
    """Return a value inside the given bin of a variable.

    The lowest bin is open downwards and the highest bin is open upwards, so a
    representative value is placed one range-width beyond the configured bounds.
    """
    binner = encoder.binner(variable)
    lower = binner.lower_bounds()[bin_index]
    upper = binner.upper_bound_of(bin_index)
    width = binner.high - binner.low
    if lower == float("-inf"):
        lower = binner.low - width
    if upper == float("inf"):
        upper = binner.high + width
    return (lower + upper) / 2.0
