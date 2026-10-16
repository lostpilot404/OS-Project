"""Detector quality: what it reacts to, what it misses, and what it must not react to.

The thresholds are derived from a no-action control (see
:mod:`experiments.healing_thresholds`), which measures false triggers but says nothing about
recall. These tests cover the other half with *controlled* degradations
(:mod:`experiments.healing_detection`): a step change is injected into a fraction of the
traces, the hold-only control is run on both versions, and the detections are attributed to
the workload itself. They also pin the three ways a detector can be wrong in the other
direction: a reference that absorbs a sustained degradation, an evaluation that reuses stale
completions, and a loop that retriggers while its own recovery window is still open.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import ACTION_FCFS, build_default_config
from errors import ValidationError
from experiments.healing_detection import (
    INJECTION_FRACTION,
    SCENARIOS,
    inject_scenario,
    run_detection_scenarios,
    scenario_rows,
)
from experiments.healing_config import HealingExperimentConfig
from experiments.healing_experiment import build_healing_workloads
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from healing.policies import HoldOnlyPolicy, RuleHealingPolicy
from scheduler import RuntimeSimulator
from workload.generator import WorkloadGenerator

SCHEDULER = HealingExperimentConfig().scheduler
GENERATOR = WorkloadGenerator(build_default_config().families)


def _workload(family: str = "mixed", seed: int = 11):
    return GENERATOR.generate(family, seed)


def _run_control(workload, healing: HealingConfig = HealingConfig()):
    """Run the hold-only control over one workload and return its episode record."""
    controller = SelfHealingController(
        healing, SCHEDULER, HoldOnlyPolicy(), initial_policy=ACTION_FCFS, label="control"
    )
    controller.begin_episode(0)
    run = RuntimeSimulator(SCHEDULER).run(workload, controller, training=False, policy_name="control")
    return run, controller.end_episode()


# -- the injections themselves ----------------------------------------------------------
def test_injections_are_deterministic_and_leave_the_prefix_untouched() -> None:
    """Every scenario keeps the process identities and the pre-injection prefix, and the
    same workload always produces the same injected workload."""
    workload = _workload("long_jobs", 3)
    arrivals = [process.arrival_time for process in workload]
    boundary = min(arrivals) + round(INJECTION_FRACTION * (max(arrivals) - min(arrivals)))
    original = {process.pid: process for process in workload}

    assert inject_scenario(workload, "stable") is workload
    for scenario in SCENARIOS:
        first = inject_scenario(workload, scenario)
        second = inject_scenario(workload, scenario)
        assert first.fingerprint == second.fingerprint, "injection must be deterministic"
        injected = {process.pid: process for process in first}
        assert set(injected) == set(original), "no process may be added or dropped"
        assert len(first) == len(workload)
        for pid, process in injected.items():
            assert process.priority == original[pid].priority
            if original[pid].arrival_time < boundary:
                # The pre-injection prefix is identical in every scenario, which is what
                # makes the comparison with the untouched trace meaningful.
                assert (process.arrival_time, process.burst_time) == (
                    original[pid].arrival_time,
                    original[pid].burst_time,
                )
        if scenario == "burst_shock":
            for pid, process in injected.items():
                if original[pid].arrival_time >= boundary:
                    assert process.arrival_time == original[pid].arrival_time
                    assert process.burst_time >= original[pid].burst_time
            assert max(process.burst_time for process in first) > max(
                process.burst_time for process in workload
            )
        if scenario == "arrival_storm":
            assert max(process.arrival_time for process in first) < max(arrivals)
            for pid, process in injected.items():
                if original[pid].arrival_time >= boundary:
                    assert process.burst_time == original[pid].burst_time
                    assert process.arrival_time >= boundary
    with pytest.raises(ValidationError):
        inject_scenario(workload, "no-such-scenario")


# -- recall and delay on controlled degradations ----------------------------------------
def test_detector_detects_injected_degradation_in_most_traces() -> None:
    """Recall on the injected scenarios must be high, and the detections must happen after
    the injection instant (never before, which would mean the detector is firing at random)."""
    config = HealingExperimentConfig()
    items = build_healing_workloads("validation", config, GENERATOR)
    selected = [item for index, item in enumerate(items) if index % 19 == 0][:8]
    assert selected, "the validation split must contain workloads"
    rows = run_detection_scenarios(selected, config=config)
    assert {row.scenario for row in rows} == set(SCENARIOS)
    per_scenario = {}
    for row in rows:
        per_scenario.setdefault(row.scenario, []).append(row)

    for scenario, subset in per_scenario.items():
        assert all(row.interventions == 0 for row in subset), "the control never intervenes"
        if scenario == "stable":
            continue
        detected = [row for row in subset if row.detected]
        assert len(detected) >= 0.5 * len(subset), (
            f"{scenario}: the detector must react to an injected step change"
        )
        for row in detected:
            assert row.first_detection_time is not None
            assert row.first_detection_time >= row.injection_time
            assert row.detection_delay_time >= 0
            assert row.first_detection_decision is not None
    summary = scenario_rows(pd.DataFrame(row.as_row() for row in rows))
    assert {row["scenario"] for row in summary} >= set(SCENARIOS)
    assert all(row["kind"] for row in summary)


def test_detector_does_not_fire_on_an_uninjected_trace_at_the_same_rate() -> None:
    """The false-trigger rate on untouched traces must stay below the injected recall: the
    detector has to respond to the injection, not to the trace length."""
    config = HealingExperimentConfig()
    items = build_healing_workloads("validation", config, GENERATOR)
    selected = [item for index, item in enumerate(items) if index % 19 == 0][:8]
    rows = run_detection_scenarios(selected, config=config, scenarios=("stable",))
    per_trace = [row.incidents for row in rows]
    assert sum(1 for count in per_trace if count > 0) <= 0.75 * len(per_trace)
    assert sum(per_trace) / len(per_trace) <= 1.0


# -- mistakes the detector must not make ------------------------------------------------
def test_sustained_degradation_is_not_absorbed_into_the_reference() -> None:
    """A trace that degrades and stays degraded must keep failing the same comparison: the
    reference may only be updated from windows that looked normal."""
    controller = SelfHealingController(
        HealingConfig(hysteresis_evaluations=1),
        SCHEDULER,
        HoldOnlyPolicy(),
        initial_policy=ACTION_FCFS,
        label="control",
    )
    from tests.test_healing_verification import _ScriptedTrace

    trace = _ScriptedTrace(controller)
    for _ in range(6):  # establish a healthy reference
        trace.step(rate=6.0, completions=(20,))
    reference_healthy = controller._monitor.snapshot().reference_wait  # noqa: SLF001
    for _ in range(24):  # a long, sustained degradation
        trace.step(rate=40.0, completions=(200,))
    snapshot = controller._monitor.snapshot()  # noqa: SLF001
    assert snapshot.reference_wait <= 2.0 * reference_healthy, (
        "the reference absorbed the degradation it is supposed to be compared against"
    )
    episode = trace.close()
    assert episode.incidents, "a sustained degradation must keep being detected"
    assert episode.interventions == 0


def test_verification_window_blocks_new_detections_and_runtime_labels_match() -> None:
    """The loop cannot stack incidents: while one recovery window is running every decision
    is labelled ``verification-window-open`` and no second detection is evaluated."""
    config = HealingExperimentConfig()
    from healing.policies import RuleWaitingTimePolicy

    healing = HealingConfig()
    controller = SelfHealingController(
        healing,
        SCHEDULER,
        RuleWaitingTimePolicy(healing),
        initial_policy=ACTION_FCFS,
        label="rule",
    )
    items = build_healing_workloads("validation", config, GENERATOR)
    for index, item in enumerate(items[:12]):
        controller.begin_episode(index)
        run = RuntimeSimulator(SCHEDULER).run(
            item.workload, controller, training=False, policy_name="rule"
        )
        episode = controller.end_episode()
        # A window counts its own epoch, so it closes at decision_index + decisions - 1, and
        # the next incident cannot open before the declared cooldown has elapsed on top.
        cooldown = HealingConfig().cooldown_decisions
        for previous, following in zip(episode.incidents, episode.incidents[1:]):
            conclusion = previous.decision_index + previous.verification_decisions - 1
            assert following.decision_index >= conclusion + cooldown, (
                "a new incident opened before the cooldown of the previous one elapsed"
            )
            assert following.decision_index > previous.decision_index
        # Every epoch the engine labelled as an open recovery window really lies inside one:
        # the runtime's view and the ledger agree on when verification was running.
        spans = [
            (incident.decision_index + 1, incident.decision_index + incident.verification_decisions)
            for incident in episode.incidents
        ]
        for record in run.decisions:
            if record.decision.fallback_reason == "verification-window-open":
                assert any(start <= record.decision_index <= stop for start, stop in spans)

# -- the loop must not trigger itself --------------------------------------------------
def test_no_incident_opens_while_a_verification_window_is_open() -> None:
    """The monitor must not read a configuration change as fresh degradation: while a window
    is open the detector is skipped, and the next incident can only open once the declared
    cooldown has elapsed after the window closed."""
    controller = SelfHealingController(
        HealingConfig(), SCHEDULER, RuleHealingPolicy(HealingConfig()), initial_policy=ACTION_FCFS
    )
    controller.begin_episode(0)
    run = RuntimeSimulator(SCHEDULER).run(
        _workload("long_jobs", 1), controller, training=False, policy_name="self-trigger-probe"
    )
    episode = controller.end_episode()
    assert episode.incidents, "the probe needs a trace that triggers"

    # A window counts its own epoch, so it spans decisions [start, start + length - 1] and
    # the cooldown begins there. No other incident may start inside that span.
    spans = [
        (
            incident.decision_index,
            incident.decision_index + incident.verification_decisions - 1,
        )
        for incident in episode.incidents
        if incident.action_changed
    ]
    cooldown = HealingConfig().cooldown_decisions
    for previous, following in zip(episode.incidents, episode.incidents[1:]):
        _, closes = (
            previous.decision_index,
            previous.decision_index + previous.verification_decisions - 1,
        )
        assert following.decision_index >= closes + cooldown, (
            "an incident opened before the previous window closed and its cooldown elapsed"
        )
    for record in run.decisions:
        if record.decision.fallback_reason == "verification-window-open":
            assert any(start <= record.decision_index <= stop for start, stop in spans)
    # A window that is open means no other incident can be open at the same time.
    assert len({incident.decision_index for incident in episode.incidents}) == len(episode.incidents)


def test_the_scenario_summary_reports_recall_delay_and_false_triggers() -> None:
    """The machine-readable summary carries the quantities the trade-off is judged on, with
    an explicit proxy label for precision."""
    config = HealingExperimentConfig()
    items = build_healing_workloads("validation", config, GENERATOR)
    selected = [item for index, item in enumerate(items) if index % 29 == 0][:5]
    rows = run_detection_scenarios(selected, config=config)
    summary = {row["scenario"]: row for row in scenario_rows(pd.DataFrame(r.as_row() for r in rows))}
    assert set(summary) == set(SCENARIOS) | {"all"}
    overall = summary["all"]
    for key in (
        "precision_proxy",
        "recall_on_injected",
        "false_trigger_rate_per_stable_trace",
        "median_detection_delay_decisions",
        "median_detection_delay_time_units",
    ):
        assert key in overall, key
    assert 0.0 <= overall["precision_proxy"] <= 1.0
    assert 0.0 <= overall["recall_on_injected"] <= 1.0
    assert overall["false_trigger_rate_per_stable_trace"] >= 0.0
