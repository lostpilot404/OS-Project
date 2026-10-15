"""Tests of the evidence-backed diagnosis and the recovery action space."""

from __future__ import annotations

import pytest

from config import ACTION_FCFS, ACTION_PRIORITY, ACTION_ROUND_ROBIN, ACTION_SJF
from errors import ValidationError
from healing.actions import (
    GROW_QUANTUM,
    HEALING_ACTION_NAMES,
    HOLD,
    REVERT_TO_LAST_ACCEPTED,
    SET_PRIORITY,
    SET_ROUND_ROBIN,
    SET_SJF,
    SHRINK_QUANTUM,
    SchedulerSettings,
    apply_healing_action,
)
from healing.config import HealingConfig
from healing.diagnosis import (
    DIAGNOSIS_CATEGORIES,
    ReadyEvidence,
    build_ready_evidence,
    diagnose,
)
from healing.monitor import CausalMonitor
from scheduler.runtime import ObservedProcess, RuntimeObservation


def _observation(rows, *, current_time: int = 100) -> RuntimeObservation:
    """Build a syntactically valid observation from ``(pid, arrival, burst, remaining, prio)``."""
    return RuntimeObservation(
        current_time=current_time,
        ready_processes=tuple(
            ObservedProcess(
                pid=pid,
                arrival_time=arrival,
                burst_time=burst,
                remaining_burst=remaining,
                priority=priority,
            )
            for pid, arrival, burst, remaining, priority in rows
        ),
        completed_count=2,
        arrived_count=len(rows) + 2,
        observed_mean_burst=10.0,
        observed_priority_spread=2,
        mean_ready_arrival_age=float(current_time),
        previous_policy=ACTION_FCFS,
    )


def _snapshot(waits, *, reference: float, band: float):
    """Build a monitor snapshot with a directly chosen reference and band."""
    monitor = CausalMonitor(HealingConfig(window_completions=len(waits), min_window_completions=1))
    monitor.observe_completions(waits)
    snapshot = monitor.snapshot()
    object.__setattr__(snapshot, "reference_wait", reference)  # frozen dataclass, test-only
    object.__setattr__(snapshot, "band", band)
    return snapshot


def test_ready_evidence_summarizes_only_visible_work() -> None:
    """Head, median, ages and priority bounds describe the ready set exactly."""
    observation = _observation(
        [
            (1, 0, 40, 40, 5),
            (2, 80, 4, 4, 1),
            (3, 90, 6, 6, 3),
        ],
        current_time=100,
    )
    evidence = build_ready_evidence(
        observation, running_priority=5, lower_priority_number_is_higher_priority=True
    )
    assert evidence.ready_count == 3
    assert evidence.head_remaining == 40
    assert evidence.median_remaining == pytest.approx(6.0)
    assert evidence.max_arrival_age == 100
    assert evidence.max_remaining == 40
    assert evidence.min_remaining == 4
    assert evidence.priority_spread == 4
    assert evidence.visible_remaining_work == 50
    assert evidence.best_ready_priority == 1
    assert evidence.running_is_worse_than_best_ready is True


def test_priority_convention_is_respected() -> None:
    """With the inverted convention the same numbers must not be called an inversion."""
    observation = _observation([(1, 0, 10, 10, 1), (2, 0, 10, 10, 5)])
    higher_is_better = build_ready_evidence(
        observation, running_priority=1, lower_priority_number_is_higher_priority=False
    )
    assert higher_is_better.best_ready_priority == 5
    assert higher_is_better.running_is_worse_than_best_ready is True
    lower_is_better = build_ready_evidence(
        observation, running_priority=1, lower_priority_number_is_higher_priority=True
    )
    assert lower_is_better.best_ready_priority == 1
    assert lower_is_better.running_is_worse_than_best_ready is False


@pytest.mark.parametrize(
    ("rows", "waits", "expected"),
    [
        # A long FIFO head in front of short work: head-of-line blocking outranks the rest.
        ([(1, 0, 60, 60, 5), (2, 9, 3, 3, 1), (3, 9, 4, 4, 1)], [5, 5, 6, 7], "head_of_line_blocking"),
        # A very old ready job with no extreme head: fairness starvation.
        ([(1, 0, 5, 5, 1), (2, 0, 5, 5, 1)], [5, 6, 7, 8], "fairness_starvation"),
        # A few very long waits inside an otherwise normal window: tail dispersion.
        ([(1, 0, 5, 5, 1), (2, 0, 5, 5, 1)], [1, 1, 1, 400], "tail_dispersion"),
        # Many ready jobs and nothing else distinctive: congestion.
        (
            [(1, 0, 5, 5, 1), (2, 0, 5, 5, 1), (3, 0, 5, 5, 1), (4, 0, 5, 5, 1), (5, 0, 5, 5, 1)],
            [2, 2, 2, 2],
            "queue_congestion",
        ),
    ],
)
def test_categories_fire_on_their_own_evidence(rows, waits, expected) -> None:
    """Each declared category is reachable from its own observable pattern."""
    config = HealingConfig()
    observation = _observation(rows, current_time=10 if expected != "fairness_starvation" else 60)
    evidence = build_ready_evidence(
        observation, running_priority=None, lower_priority_number_is_higher_priority=True
    )
    snapshot = _snapshot(waits, reference=10.0, band=2.0)
    diagnosis = diagnose(evidence, snapshot, config)
    assert diagnosis.category == expected
    assert expected in diagnosis.matched
    assert diagnosis.statement.startswith(expected)
    assert dict(diagnosis.evidence)


def test_precedence_and_unexplained_fallback() -> None:
    """Precedence is fixed, and an inexplicable excess is reported as unexplained."""
    config = HealingConfig()
    # Several categories match at once: the declared order decides the primary.
    observation = _observation(
        [(1, 0, 80, 80, 5), (2, 0, 4, 4, 1), (3, 0, 5, 5, 1), (4, 0, 5, 5, 1)],
        current_time=200,
    )
    evidence = build_ready_evidence(
        observation, running_priority=5, lower_priority_number_is_higher_priority=True
    )
    snapshot = _snapshot([1, 1, 1, 500], reference=10.0, band=2.0)
    diagnosis = diagnose(evidence, snapshot, config)
    assert diagnosis.category == "head_of_line_blocking"
    assert set(diagnosis.matched) >= {
        "head_of_line_blocking",
        "fairness_starvation",
        "priority_ordering_anomaly",
        "tail_dispersion",
    }
    assert DIAGNOSIS_CATEGORIES.index(diagnosis.category) == min(
        DIAGNOSIS_CATEGORIES.index(name) for name in diagnosis.matched
    )

    plain = _observation([(1, 0, 5, 5, 3), (2, 0, 5, 5, 3)], current_time=1)
    plain_evidence = build_ready_evidence(
        plain, running_priority=None, lower_priority_number_is_higher_priority=True
    )
    quiet = diagnose(plain_evidence, _snapshot([10, 10, 10, 10], reference=10.0, band=2.0), config)
    assert quiet.category == "unexplained"
    assert quiet.matched == ()


def test_diagnosis_rejects_wrong_types() -> None:
    """The diagnoser and its evidence builder validate their inputs."""
    config = HealingConfig()
    with pytest.raises(ValidationError, match="RuntimeObservation"):
        build_ready_evidence("nope", running_priority=None, lower_priority_number_is_higher_priority=True)  # type: ignore[arg-type]
    empty = RuntimeObservation(
        current_time=0,
        ready_processes=(),
        completed_count=0,
        arrived_count=0,
        observed_mean_burst=0.0,
        observed_priority_spread=0,
        mean_ready_arrival_age=0.0,
        previous_policy=None,
    )
    with pytest.raises(ValidationError, match="ready process"):
        build_ready_evidence(empty, running_priority=None, lower_priority_number_is_higher_priority=True)
    with pytest.raises(ValidationError, match="ReadyEvidence"):
        diagnose("nope", _snapshot([1], reference=1.0, band=1.0), config)  # type: ignore[arg-type]


def test_healing_action_space_is_stable_and_bounded() -> None:
    """The declared action names and indices are part of the interface."""
    assert HEALING_ACTION_NAMES[0] == "hold"
    assert HEALING_ACTION_NAMES[SET_SJF] == "set_sjf"
    assert HEALING_ACTION_NAMES[GROW_QUANTUM] == "grow_quantum"
    assert len(HEALING_ACTION_NAMES) == 8
    with pytest.raises(ValidationError):
        apply_healing_action(99, current=SchedulerSettings(ACTION_FCFS, 4), last_accepted=None, min_quantum=1, max_quantum=16)


def test_actions_that_cannot_change_anything_collapse_to_hold() -> None:
    """No-op requests are reported as unchanged, so they are never counted as recovery."""
    current = SchedulerSettings(policy=ACTION_FCFS, quantum=4)
    hold = apply_healing_action(
        HOLD, current=current, last_accepted=None, min_quantum=1, max_quantum=16
    )
    assert hold.changed is False
    requested_sjf = apply_healing_action(
        SET_SJF, current=current, last_accepted=None, min_quantum=1, max_quantum=16
    )
    assert requested_sjf.changed is True and requested_sjf.settings.policy == ACTION_SJF
    same_sjf = apply_healing_action(
        SET_SJF,
        current=SchedulerSettings(ACTION_SJF, 4),
        last_accepted=None,
        min_quantum=1,
        max_quantum=16,
    )
    assert same_sjf.changed is False
    # The quantum only affects Round-Robin service.
    shrink_under_fcfs = apply_healing_action(
        SHRINK_QUANTUM, current=current, last_accepted=None, min_quantum=1, max_quantum=16
    )
    assert shrink_under_fcfs.changed is False and "Round-Robin" in shrink_under_fcfs.note
    shrink_under_rr = apply_healing_action(
        SHRINK_QUANTUM,
        current=SchedulerSettings(ACTION_ROUND_ROBIN, 4),
        last_accepted=None,
        min_quantum=1,
        max_quantum=16,
    )
    assert shrink_under_rr.changed is True and shrink_under_rr.settings.quantum == 2
    # Reverting is only possible when something different was accepted before.
    no_target = apply_healing_action(
        REVERT_TO_LAST_ACCEPTED, current=current, last_accepted=None, min_quantum=1, max_quantum=16
    )
    assert no_target.changed is False
    with_target = apply_healing_action(
        REVERT_TO_LAST_ACCEPTED,
        current=SchedulerSettings(ACTION_ROUND_ROBIN, 2),
        last_accepted=current,
        min_quantum=1,
        max_quantum=16,
    )
    assert with_target.changed is True and with_target.settings == current


def test_quantum_actions_respect_the_declared_bounds() -> None:
    """Shrinking stops at the minimum and growing stops at the maximum."""
    tiny = apply_healing_action(
        SHRINK_QUANTUM,
        current=SchedulerSettings(ACTION_ROUND_ROBIN, 1),
        last_accepted=None,
        min_quantum=1,
        max_quantum=16,
    )
    assert tiny.changed is False
    big = apply_healing_action(
        GROW_QUANTUM,
        current=SchedulerSettings(ACTION_ROUND_ROBIN, 16),
        last_accepted=None,
        min_quantum=1,
        max_quantum=16,
    )
    assert big.changed is False
    doubling = apply_healing_action(
        GROW_QUANTUM,
        current=SchedulerSettings(ACTION_ROUND_ROBIN, 3),
        last_accepted=None,
        min_quantum=1,
        max_quantum=16,
    )
    assert doubling.settings.quantum == 6
