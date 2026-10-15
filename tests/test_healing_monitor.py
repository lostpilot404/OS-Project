"""Tests of the causal monitor: window statistics, reference, band, telemetry and load."""

from __future__ import annotations

import pytest

from errors import ValidationError
from healing.config import HealingConfig
from healing.monitor import CausalMonitor


def _monitor(**overrides) -> CausalMonitor:
    return CausalMonitor(HealingConfig(**overrides))


def test_window_statistics_are_reported_exactly() -> None:
    """Window mean/median/max must be the arithmetic values of the observed waits."""
    monitor = _monitor(window_completions=4, min_window_completions=2)
    monitor.observe_completions([10, 20, 30])
    snapshot = monitor.snapshot()
    assert snapshot.window_size == 3
    assert snapshot.window_mean_wait == pytest.approx(20.0)
    assert snapshot.window_median_wait == pytest.approx(20.0)
    assert snapshot.window_max_wait == pytest.approx(30.0)
    monitor.observe_completions([40])
    snapshot = monitor.snapshot()
    assert snapshot.window_mean_wait == pytest.approx(25.0)
    assert snapshot.window_median_wait == pytest.approx(25.0)
    # The window is bounded, so the oldest sample leaves it.
    monitor.observe_completions([50])
    assert monitor.snapshot().window_size == 4
    assert monitor.snapshot().window_mean_wait == pytest.approx(35.0)


def test_thin_window_is_not_sufficient_telemetry_and_is_not_absorbed() -> None:
    """Below the telemetry floor the monitor reports insufficiency, not a verdict."""
    monitor = _monitor(window_completions=8, min_window_completions=4)
    monitor.observe_completions([5, 6])
    snapshot = monitor.snapshot()
    assert snapshot.telemetry_sufficient is False
    assert snapshot.reference_wait == pytest.approx(5.5)  # provisional, not yet a reference
    with pytest.raises(ValidationError, match="too thin"):
        monitor.absorb_into_reference(snapshot)
    with pytest.raises(ValidationError, match="no stable reference"):
        monitor.reference_band()


def test_reference_and_band_use_exponential_smoothing_with_a_floor() -> None:
    """The reference tracks normal windows and the band never falls below its floors."""
    monitor = _monitor(
        window_completions=4,
        min_window_completions=4,
        reference_ewma=0.5,
        band_ewma=0.5,
        min_band_fraction=0.25,
        min_band_absolute=1.0,
    )
    monitor.observe_completions([100, 100, 100, 100])
    first = monitor.snapshot()
    monitor.absorb_into_reference(first)
    reference, band = monitor.reference_band()
    assert reference == pytest.approx(100.0)
    assert band == pytest.approx(25.0)  # the declared floor, not zero

    # A much higher window pulls the reference only half-way (declared smoothing).
    monitor.observe_completions([200, 200, 200, 200])
    second = monitor.snapshot()
    monitor.absorb_into_reference(second)
    reference, band = monitor.reference_band()
    assert 100.0 < reference < 200.0
    assert band >= 0.25 * reference


def test_stale_or_invalid_absorption_is_rejected() -> None:
    """A snapshot of a different window must never be absorbed."""
    monitor = _monitor(window_completions=4, min_window_completions=4)
    monitor.observe_completions([10, 10, 10, 10])
    snapshot = monitor.snapshot()
    monitor.observe_completions([10])
    with pytest.raises(ValidationError, match="stale"):
        monitor.absorb_into_reference(snapshot)
    with pytest.raises(ValidationError, match="non-negative integer"):
        monitor.observe_completions([-1])
    with pytest.raises(ValidationError, match="non-negative integer"):
        monitor.observe_completions([True])


def test_load_statistic_is_the_waiting_accrual_rate() -> None:
    """The load level is ΔW/Δt, i.e. the mean number of waiting processes."""
    monitor = _monitor(load_window_intervals=3)
    monitor.observe_interval(accrued_wait=40, duration=10)
    monitor.observe_interval(accrued_wait=20, duration=10)
    assert monitor.load_level() == pytest.approx(3.0)
    assert monitor.load_level(intervals=1) == pytest.approx(2.0)
    with pytest.raises(ValidationError, match="duration"):
        monitor.observe_interval(accrued_wait=5, duration=0)
    with pytest.raises(ValidationError, match="accrued_wait"):
        monitor.observe_interval(accrued_wait=-3, duration=1)


def test_load_reference_can_be_frozen_so_the_incident_is_not_absorbed() -> None:
    """Freezing is what lets verification compare against the trace's normal load."""
    monitor = _monitor(band_ewma=0.5, min_band_absolute=0.5, min_band_fraction=0.1)
    monitor.observe_interval(accrued_wait=10, duration=10)  # rate 1.0, establishes reference
    assert monitor.load_reference == pytest.approx(1.0)
    monitor.observe_interval(accrued_wait=100, duration=10, allow_reference_update=False)
    assert monitor.load_reference == pytest.approx(1.0), "a frozen reference must not move"
    assert monitor.load_band() > 0.0
    monitor.observe_interval(accrued_wait=10, duration=10)
    assert monitor.load_reference < 1.0 + 0.5 * (monitor.load_level(1) - 1.0) + 1e-9


def test_load_band_is_floored_and_never_collapses() -> None:
    """A quiet trace must not drive the band (and thus every reward) toward zero."""
    monitor = _monitor(min_band_absolute=1.0, min_band_fraction=0.2, band_ewma=0.9)
    for _ in range(20):
        monitor.observe_interval(accrued_wait=50, duration=10)
    assert monitor.load_band() >= 0.2 * monitor.load_reference
    assert monitor.load_band() >= 1.0


def test_monitor_reports_pending_evidence() -> None:
    """Evaluations are driven by new completions, which the monitor must expose."""
    monitor = _monitor()
    assert monitor.has_pending_evidence is False
    monitor.observe_completions([1, 2])
    assert monitor.has_pending_evidence is True
    monitor.mark_evaluated()
    assert monitor.has_pending_evidence is False
    with pytest.raises(ValidationError):
        CausalMonitor("not-a-config")  # type: ignore[arg-type]
