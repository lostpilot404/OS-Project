"""Tests of degradation detection: reference, threshold, hysteresis, cooldown, work gate."""

from __future__ import annotations

import pytest

from errors import ValidationError
from healing.config import HealingConfig
from healing.detector import DegradationDetector
from healing.monitor import CausalMonitor


def _setup(**overrides):
    config = HealingConfig(**overrides)
    monitor = CausalMonitor(config)
    detector = DegradationDetector(config)
    return config, monitor, detector


def _evaluate(detector, monitor, **kwargs):
    arguments = {
        "ready_count": 4,
        "visible_remaining_work": 400,
        "observed_mean_burst": 10.0,
        "load_level": 3.0,
        "load_reference": 3.0,
        "decisions_since_conclusion": 10,
        "adaptation_enabled": True,
        "interventions_used": 0,
    }
    arguments.update(kwargs)
    return detector.evaluate(monitor, **arguments)


def test_thin_telemetry_abstains_without_triggering() -> None:
    """Not enough evidence must produce an explicit abstention, never a detection."""
    _, monitor, detector = _setup(window_completions=8, min_window_completions=4)
    monitor.observe_completions([10, 12])
    outcome = _evaluate(detector, monitor)
    assert outcome.triggered is False
    assert outcome.reason == "insufficient_telemetry"
    assert outcome.absorbable is False


def test_normal_window_is_absorbable_and_does_not_trigger() -> None:
    """A window inside the band establishes and updates the reference."""
    _, monitor, detector = _setup(window_completions=4, min_window_completions=4)
    monitor.observe_completions([100, 100, 100, 100])
    outcome = _evaluate(detector, monitor)
    assert outcome.reason == "within_band"
    assert outcome.absorbable is True
    assert detector.suspect_streak == 0


def test_hysteresis_requires_consecutive_suspect_evaluations() -> None:
    """One bad window is not an incident; the declared number of them is."""
    _, monitor, detector = _setup(
        window_completions=4,
        min_window_completions=4,
        hysteresis_evaluations=2,
        min_band_fraction=0.1,
    )
    monitor.observe_completions([100] * 4)
    monitor.absorb_into_reference(monitor.snapshot())  # reference 100, band 10

    monitor.observe_completions([200] * 4)
    first = _evaluate(detector, monitor)
    assert first.triggered is False
    assert first.reason == "hysteresis_pending"
    assert first.suspect_streak == 1

    monitor.observe_completions([200] * 4)
    second = _evaluate(detector, monitor)
    assert second.triggered is True
    assert second.reason == "degradation_detected"
    assert second.suspect_streak == 2
    assert second.snapshot is not None and second.snapshot.window_mean_wait == pytest.approx(200.0)


def test_hysteresis_counter_resets_on_a_normal_window() -> None:
    """A single normal window clears the suspicion, so isolated spikes cannot accumulate."""
    _, monitor, detector = _setup(
        window_completions=4, min_window_completions=4, min_band_fraction=0.1
    )
    monitor.observe_completions([100] * 4)
    monitor.absorb_into_reference(monitor.snapshot())
    monitor.observe_completions([300] * 4)
    assert _evaluate(detector, monitor).suspect_streak == 1
    monitor.observe_completions([100] * 4)
    outcome = _evaluate(detector, monitor)
    assert outcome.reason == "within_band"
    assert detector.suspect_streak == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"decisions_since_conclusion": 0}, "cooldown"),
        ({"adaptation_enabled": False}, "adaptation_disabled"),
        ({"interventions_used": 99}, "intervention_budget_exhausted"),
        ({"ready_count": 1}, "insufficient_ready_work"),
        ({"visible_remaining_work": 1}, "insufficient_remaining_work"),
        ({"observed_mean_burst": 250.0}, "insufficient_remaining_work"),
        ({"load_level": 1.0}, "load_below_reference"),
    ],
)
def test_every_suppression_path_has_its_own_reason(kwargs, expected) -> None:
    """Each declared gate reports itself, so 'no action' is never ambiguous in the ledger."""
    config, monitor, detector = _setup(
        window_completions=4,
        min_window_completions=4,
        min_band_fraction=0.1,
        hysteresis_evaluations=1,
        cooldown_decisions=3,
        min_ready_processes=2,
        min_remaining_work_factor=2.0,
        max_interventions_per_trace=5,
    )
    monitor.observe_completions([100] * 4)
    monitor.absorb_into_reference(monitor.snapshot())
    monitor.observe_completions([500] * 4)
    outcome = _evaluate(detector, monitor, **kwargs)
    assert outcome.triggered is False
    assert outcome.reason == expected
    assert outcome.absorbable is False, "a suspect window must never enter the reference"
    assert config.min_ready_processes == 2


def test_congestion_gate_only_blocks_when_the_queue_is_clearly_below_normal() -> None:
    """A degraded window during a drained backlog is not actionable; a degraded window at
    or above the trace's normal load is."""
    config, monitor, detector = _setup(
        window_completions=4,
        min_window_completions=4,
        min_band_fraction=0.1,
        hysteresis_evaluations=1,
        min_load_factor=1.0,
        min_remaining_work_factor=1.5,
    )
    monitor.observe_completions([100] * 4)
    monitor.absorb_into_reference(monitor.snapshot())
    for _ in range(8):
        monitor.observe_interval(accrued_wait=30, duration=10)  # 3.0 waiting processes
    monitor.observe_completions([300] * 4)
    below = _evaluate(detector, monitor, load_level=1.0, load_reference=monitor.load_reference)
    assert below.triggered is False and below.reason == "load_below_reference"
    assert detector.suspect_streak == 0, "a draining backlog must also clear the suspicion"
    at_level = _evaluate(detector, monitor, load_level=3.0, load_reference=monitor.load_reference)
    assert at_level.triggered is True
    assert config.min_load_factor == 1.0


def test_detector_requires_evidence_and_ready_work() -> None:
    """Invalid inputs are rejected instead of being silently treated as normal."""
    _, monitor, detector = _setup()
    with pytest.raises(ValidationError, match="at least one observed completion"):
        _evaluate(detector, monitor)
    monitor.observe_completions([10] * 4)
    with pytest.raises(ValidationError, match="at least one ready process"):
        _evaluate(detector, monitor, ready_count=0)
    with pytest.raises(ValidationError, match="decisions_since_conclusion"):
        _evaluate(detector, monitor, decisions_since_conclusion=-1)
    with pytest.raises(ValidationError, match="load_level"):
        _evaluate(detector, monitor, load_level=-1.0)
    with pytest.raises(ValidationError, match="load_reference"):
        _evaluate(detector, monitor, load_reference=-0.5)


def test_detection_is_deterministic() -> None:
    """The same evidence sequence always produces the same outcome sequence."""
    def run() -> list:
        _, monitor, detector = _setup(
            window_completions=4, min_window_completions=4, min_band_fraction=0.1
        )
        outcomes = []
        for batch in ([50] * 4, [90] * 4, [90] * 4, [50] * 4):
            monitor.observe_completions(batch)
            outcomes.append(_evaluate(detector, monitor))
            if monitor.has_pending_evidence and outcomes[-1].absorbable:
                monitor.absorb_into_reference(outcomes[-1].snapshot)
        return outcomes

    first = run()
    second = run()
    assert [outcome.reason for outcome in first] == [
        "within_band",
        "hysteresis_pending",
        "degradation_detected",
        "within_band",
    ]
    assert first == second
