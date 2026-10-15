"""Tests for workload fingerprinting and its tabular state encoder.

Covers feature arithmetic on hand-built workloads, determinism, the declared
information boundary (fingerprint features are pure functions of the process table and
cannot import scheduler/metric/RL code), and the fingerprint state encoder.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from config import FingerprintStateConfig, KNOWN_FINGERPRINT_STATE_FEATURES
from errors import ValidationError
from rl.fingerprint_state import FingerprintStateEncoder
from workload.fingerprint import (
    FINGERPRINT_FEATURES,
    WorkloadFingerprint,
    fingerprint_workload,
)
from workload.models import Process, Workload

from tests.helpers import workload_from_rows


class TestFeatureArithmetic:
    def test_hand_computed_features(self) -> None:
        workload = workload_from_rows(
            "hand", [(1, 0, 2, 1), (2, 0, 4, 2), (3, 10, 6, 1)]
        )
        fingerprint = fingerprint_workload(workload)

        assert fingerprint.process_count == 3
        assert fingerprint.arrival_span == 10
        assert fingerprint.arrival_rate == pytest.approx(0.3)
        # gaps between sorted arrivals [0, 0, 10] are [0, 10]: mean 5, pop-std 5
        assert fingerprint.arrival_gap_cv == pytest.approx(1.0)
        assert fingerprint.burst_mean == pytest.approx(4.0)
        expected_cv = math.sqrt(8.0 / 3.0) / 4.0
        assert fingerprint.burst_cv == pytest.approx(expected_cv)
        # bursts [2, 4, 6]: q10=2, q50=4, q90=6 -> Bowley skew 0
        assert fingerprint.burst_skew == pytest.approx(0.0)
        # earliest arrivals (time 0) have bursts [2, 4] -> mean 3 over median burst 4
        assert fingerprint.first_arrival_burst_ratio == pytest.approx(0.75)
        assert fingerprint.priority_std == pytest.approx(math.sqrt(2.0) / 3.0)
        assert fingerprint.priority_unique_fraction == pytest.approx(2.0 / 3.0)
        # (priority, burst) = (1,2), (2,4), (1,6): covariance is exactly 0
        assert fingerprint.priority_burst_correlation == pytest.approx(0.0)

    def test_burst_skew_detects_long_tail(self) -> None:
        rows = [(index + 1, 0, burst, 1) for index, burst in enumerate([1, 1, 1, 1, 10])]
        fingerprint = fingerprint_workload(workload_from_rows("tail", rows))
        # q10 = 1, q50 = 1, q90 = 10 -> (10 + 1 - 2) / 9 = 1.0
        assert fingerprint.burst_skew == pytest.approx(1.0)

    def test_perfect_priority_burst_correlation(self) -> None:
        rows = [(1, 0, 2, 1), (2, 0, 4, 2), (3, 0, 6, 3)]
        fingerprint = fingerprint_workload(workload_from_rows("corr", rows))
        assert fingerprint.priority_burst_correlation == pytest.approx(1.0)

    def test_all_simultaneous_arrivals_have_zero_gap_cv(self) -> None:
        rows = [(1, 0, 3, 1), (2, 0, 5, 2)]
        fingerprint = fingerprint_workload(workload_from_rows("batch", rows))
        assert fingerprint.arrival_span == 0
        assert fingerprint.arrival_gap_cv == 0.0
        assert fingerprint.arrival_rate == pytest.approx(2.0)

    def test_single_process_degenerate_values(self) -> None:
        fingerprint = fingerprint_workload(workload_from_rows("solo", [(1, 4, 7, 3)]))
        assert fingerprint.process_count == 1
        assert fingerprint.arrival_span == 0
        assert fingerprint.arrival_gap_cv == 0.0
        assert fingerprint.burst_cv == 0.0
        assert fingerprint.burst_skew == 0.0
        assert fingerprint.priority_std == 0.0
        assert fingerprint.priority_unique_fraction == pytest.approx(1.0)
        assert fingerprint.priority_burst_correlation == 0.0

    def test_empty_workload_is_all_zero(self) -> None:
        workload = Workload(name="empty", processes=[], allow_empty=True)
        fingerprint = fingerprint_workload(workload)
        assert fingerprint.process_count == 0
        for name, value in fingerprint.as_dict().items():
            assert value == 0.0, name

    def test_features_cover_the_required_characteristics(self) -> None:
        assert FINGERPRINT_FEATURES[0] == "process_count"
        for required in (
            "process_count",
            "arrival_span",
            "arrival_rate",
            "arrival_gap_cv",
            "burst_cv",
            "burst_skew",
            "priority_std",
            "priority_unique_fraction",
        ):
            assert required in FINGERPRINT_FEATURES

    def test_as_dict_and_row_are_consistent(self) -> None:
        workload = workload_from_rows("row", [(1, 0, 2, 1), (2, 5, 8, 2)])
        fingerprint = fingerprint_workload(workload)
        values = fingerprint.as_dict()
        row = fingerprint.as_row()
        assert tuple(values) == FINGERPRINT_FEATURES
        for name in FINGERPRINT_FEATURES:
            assert row[name] == values[name]
        assert row["workload_fingerprint"] == workload.fingerprint


class TestDeterminismAndPurity:
    def test_same_process_table_gives_identical_features(self) -> None:
        rows = [(1, 0, 4, 1), (2, 3, 9, 4), (3, 3, 2, 2)]
        first = fingerprint_workload(workload_from_rows("a", rows))
        second = fingerprint_workload(workload_from_rows("b", rows))
        assert first.as_dict() == second.as_dict()
        assert first.digest == second.digest

    def test_features_are_invariant_to_input_order(self) -> None:
        rows = [(1, 0, 4, 1), (2, 3, 9, 4), (3, 3, 2, 2)]
        shuffled = [rows[2], rows[0], rows[1]]
        assert (
            fingerprint_workload(workload_from_rows("a", rows)).as_dict()
            == fingerprint_workload(workload_from_rows("b", shuffled)).as_dict()
        )

    def test_features_do_not_depend_on_workload_name_or_description(self) -> None:
        rows = [(1, 0, 4, 1), (2, 3, 9, 4)]
        plain = Workload(name="plain", processes=[Process(*row) for row in rows])
        described = Workload(
            name="described",
            processes=[Process(*row) for row in rows],
            description="completely different text",
        )
        assert fingerprint_workload(plain).as_dict() == fingerprint_workload(described).as_dict()

    def test_information_boundary_imports(self) -> None:
        """The fingerprint module must not import schedulers, metrics, or the RL stack."""
        source = Path("workload/fingerprint.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"scheduler", "evaluation", "rl", "experiments", "visualization"}
        assert not (imported & forbidden), f"boundary violation: {imported & forbidden}"
        assert imported == {"__future__", "math", "dataclasses", "typing", "errors", "workload"}

    def test_fingerprint_never_consults_schedulers_or_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural boundary: scheduler/metric entry points must not be called."""
        import evaluation.metrics as metrics_module
        import scheduler as scheduler_module

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("fingerprinting must not call schedulers or metrics")

        monkeypatch.setattr(metrics_module, "compute_metrics", explode)
        for name in ("FCFS", "SJF", "RoundRobin", "Priority"):
            monkeypatch.setattr(
                scheduler_module, name, explode, raising=False
            )
        workload = workload_from_rows("boundary", [(1, 0, 3, 1), (2, 2, 4, 2)])
        fingerprint = fingerprint_workload(workload)
        assert fingerprint.process_count == 2

    def test_rejects_non_workload(self) -> None:
        with pytest.raises(ValidationError):
            fingerprint_workload("not a workload")  # type: ignore[arg-type]


class TestFingerprintStateEncoder:
    def test_default_state_space_size(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        assert encoder.n_states == 81
        assert encoder.features == KNOWN_FINGERPRINT_STATE_FEATURES

    def test_encode_decode_roundtrip(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        workload = workload_from_rows("enc", [(1, 0, 2, 1), (2, 4, 10, 5), (3, 9, 3, 1)])
        fingerprint = fingerprint_workload(workload)
        index = encoder.encode(fingerprint)
        assert 0 <= index < encoder.n_states
        bins = encoder.decode(index)
        for feature, bin_index in bins.items():
            value = fingerprint.as_dict()[feature]
            binner = encoder.binner(feature)
            assert binner.index(value) == bin_index

    def test_outer_bins_are_unbounded(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        fingerprint = WorkloadFingerprint(
            workload_name="x",
            digest="0" * 16,
            process_count=10,
            arrival_span=1,
            arrival_rate=10.0,
            arrival_gap_cv=99.0,
            burst_mean=1.0,
            burst_cv=99.0,
            burst_skew=5.0,
            first_arrival_burst_ratio=1.0,
            priority_std=0.0,
            priority_unique_fraction=0.0,
            priority_burst_correlation=-5.0,
        )
        bins = encoder.decode(encoder.encode(fingerprint))
        assert bins["burst_cv"] == encoder.n_bins - 1
        assert bins["arrival_gap_cv"] == encoder.n_bins - 1
        assert bins["first_arrival_burst_ratio"] == 0  # ratio 1.0 is below the 4.0 edge
        assert bins["priority_burst_correlation"] == 0

    def test_higher_feature_values_give_monotone_indices(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())

        def make(burst_cv: float) -> WorkloadFingerprint:
            return WorkloadFingerprint(
                workload_name="x",
                digest="0" * 16,
                process_count=1,
                arrival_span=0,
                arrival_rate=1.0,
                arrival_gap_cv=0.0,
                burst_mean=1.0,
                burst_cv=burst_cv,
                burst_skew=0.0,
                first_arrival_burst_ratio=1.0,
                priority_std=0.0,
                priority_unique_fraction=1.0,
                priority_burst_correlation=0.0,
            )

        indices = [encoder.encode(make(value)) for value in (0.1, 0.8, 1.4)]
        assert indices == sorted(set(indices))

    def test_rejects_bad_configuration_and_input(self) -> None:
        with pytest.raises(ValidationError):
            FingerprintStateEncoder(object())  # type: ignore[arg-type]
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        with pytest.raises(ValidationError):
            encoder.encode("nope")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            encoder.decode(-1)
        with pytest.raises(ValidationError):
            encoder.binner("process_count")

    def test_describe_covers_every_state_feature(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        described = encoder.describe()
        assert set(described) == set(KNOWN_FINGERPRINT_STATE_FEATURES)
