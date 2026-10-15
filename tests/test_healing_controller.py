"""End-to-end tests of the self-healing loop on the event-driven simulator.

These tests exercise the whole chain: causal monitor -> detector -> diagnosis -> recovery
policy -> verification window -> accept / roll back / safe fallback -> ledger, plus the
counterfactual replay audit used as evidence in the study.
"""

from __future__ import annotations

import pytest

from config import ACTION_FCFS, ACTION_ROUND_ROBIN, ACTION_SJF, SchedulerConfig, build_default_config
from errors import ValidationError
from healing.actions import SET_ROUND_ROBIN
from healing.config import HealingConfig
from healing.controller import (
    SelfHealingController,
    replay_action_at_incident,
    replay_hold_counterfactual,
)
from healing.policies import (
    HealingContextEncoder,
    HoldOnlyPolicy,
    QLearningHealingPolicy,
    RuleHealingPolicy,
    RuleWaitingTimePolicy,
)
from scheduler import FCFS, RuntimeSimulator
from workload.generator import WorkloadGenerator

SCHEDULER = SchedulerConfig(round_robin_quantum=4, switching_cost=1)
VERDICTS = {"accepted", "rolled_back", "held", "censored"}


def _workload(family: str = "mixed", seed: int = 3):
    return WorkloadGenerator(build_default_config().families).generate(family, seed)


def _controller(policy, **overrides) -> SelfHealingController:
    return SelfHealingController(
        HealingConfig(**overrides), SCHEDULER, policy, initial_policy=ACTION_FCFS, label="test-arm"
    )


def _total_waiting(result):
    """Sum of realized waiting times of a schedule result."""
    return sum(outcome.waiting_time for outcome in result.outcomes)


def _project(records):
    """Schedule-relevant projection of decision records (audit metadata excluded)."""
    return [
        (
            record.decision.action,
            record.decision.quantum,
            record.chosen_pid,
            record.start_time,
            record.end_time,
            record.waiting_cost_increment,
            record.completed_pids,
            record.applied_quantum,
        )
        for record in records
    ]


def _run(controller, workload, *, training: bool = False, simulator=None):
    engine = simulator or RuntimeSimulator(SCHEDULER)
    controller.begin_episode(0)
    run = engine.run(workload, controller, training=training, episode=0, policy_name="healing")
    episode = controller.end_episode()
    return run, episode


def test_hold_only_arm_reproduces_the_declared_fixed_policy_exactly() -> None:
    """The control arm must be indistinguishable from plain FCFS: the healing machinery
    cannot change the schedule when the policy never intervenes."""
    workload = _workload()
    run, episode = _run(_controller(HoldOnlyPolicy()), workload)
    reference = FCFS(SCHEDULER).run(workload)
    assert _total_waiting(run.result) == _total_waiting(reference)
    assert run.result.context_switches == reference.context_switches
    assert run.result.switching_overhead == reference.switching_overhead
    assert [(o.pid, o.start_time, o.completion_time, o.waiting_time) for o in run.result.outcomes] == [
        (o.pid, o.start_time, o.completion_time, o.waiting_time) for o in reference.outcomes
    ]
    assert run.policy_switch_count == 0
    assert episode.interventions == 0
    assert all(decision.decision.action == ACTION_FCFS for decision in run.decisions)


def test_full_loop_records_a_traceable_incident_ledger() -> None:
    """A rule arm produces incidents whose verdicts, actions and evidence are consistent
    with the simulator's own decision log."""
    workload = _workload("cpu_bursty", 5)
    controller = _controller(RuleHealingPolicy(HealingConfig()), hysteresis_evaluations=1)
    run, episode = _run(controller, workload)

    assert episode.incidents, "the fairness rule arm is expected to detect and act"
    for incident in episode.incidents:
        assert incident.verdict in VERDICTS
        assert incident.detector_reason == "degradation_detected"
        assert incident.window_mean_at_trigger > incident.trigger_threshold
        assert f",{incident.action_name}," in f",{incident.offered_actions},"
        assert incident.action_changed == (incident.settings_before != incident.settings_after)
        assert incident.verification_decisions >= 1
        record = run.decisions[incident.decision_index]
        assert record.decision.action == incident.settings_after.policy
        assert record.decision.quantum == incident.settings_after.quantum
        if incident.verdict == "accepted":
            assert incident.rollback_target is None
        elif incident.verdict == "rolled_back":
            assert incident.rollback_target is not None
            assert incident.closure_fraction < incident.required_closure_factor
    assert episode.interventions == sum(1 for i in episode.incidents if i.action_changed)
    assert episode.holds == sum(1 for i in episode.incidents if not i.action_changed)
    assert (
        episode.accepted + episode.rolled_back + episode.censored + episode.holds
        == len(episode.incidents)
    )


def test_verification_window_is_bounded_and_censored_at_trace_end() -> None:
    """A window that cannot close before the trace ends is recorded as unknown, and the
    declared maximum window length is never exceeded."""
    controller = _controller(
        RuleHealingPolicy(HealingConfig()),
        hysteresis_evaluations=1,
        verify_decisions=5,
        verify_completions=3,
        verify_max_decisions=14,
    )
    _, episode = _run(controller, _workload("long_jobs", 4))
    for incident in episode.incidents:
        assert incident.verification_decisions <= 14
        if incident.verdict == "censored":
            assert incident.action_changed is True


def test_rollback_is_applied_at_the_next_epoch() -> None:
    """After a failed verification the previous configuration is restored in the engine."""
    controller = _controller(
        RuleWaitingTimePolicy(HealingConfig()),
        hysteresis_evaluations=1,
        accept_closure_factor=10.0,
    )
    run, episode = _run(controller, _workload("mixed", 9))
    rolled = [incident for incident in episode.incidents if incident.verdict == "rolled_back"]
    if not rolled:
        pytest.skip("this trace produced no rolled-back intervention")
    incident = rolled[0]
    # The verdict is reached when the window closes, so the rollback is applied at the
    # first epoch at or after the recorded end of the verification window.
    after = [
        record for record in run.decisions if record.start_time >= incident.window_end_time
    ]
    assert after, "the trace must contain an epoch after the verification window"
    assert after[0].decision.action == incident.settings_before.policy
    assert after[0].decision.quantum == incident.settings_before.quantum


def test_repeated_failures_trigger_the_safe_fallback() -> None:
    """After the declared number of consecutive failures the controller stops adapting and
    pins the known-good configuration instead of oscillating."""
    controller = _controller(
        RuleWaitingTimePolicy(HealingConfig()),
        hysteresis_evaluations=1,
        accept_closure_factor=10.0,
        max_consecutive_failures=1,
    )
    run, episode = _run(controller, _workload("mixed", 9))
    assert episode.fallback_triggered is True
    fallback_incidents = [i for i in episode.incidents if i.fallback_triggered]
    assert fallback_incidents
    assert not [
        incident
        for incident in episode.incidents
        if incident.decision_index > fallback_incidents[0].decision_index
    ]
    # After the fallback every later decision uses exactly one pinned configuration: the
    # one the episode closes with.
    last = fallback_incidents[-1]
    tail = [record for record in run.decisions if record.start_time >= last.window_end_time]
    assert tail
    assert all(decision.decision.action == episode.settings_final.policy for decision in tail)
    assert all(decision.decision.quantum == episode.settings_final.quantum for decision in tail)


def test_counterfactual_replay_aligns_with_the_realized_prefix() -> None:
    """The audit's counterfactuals replay the recorded prefix exactly, so the only
    difference from the realized run is the intervention under test."""
    controller = _controller(RuleWaitingTimePolicy(HealingConfig()), hysteresis_evaluations=1)
    workload = _workload("cpu_bursty", 5)
    run, episode = _run(controller, workload)
    changed = [incident for incident in episode.incidents if incident.action_changed]
    if not changed:
        pytest.skip("this trace produced no configuration change to audit")
    incident = changed[0]

    hold = replay_hold_counterfactual(RuntimeSimulator(SCHEDULER), workload, run.decisions, incident)
    assert _project(hold.decisions[: incident.decision_index]) == _project(
        run.decisions[: incident.decision_index]
    ), "the replayed prefix must reproduce the realized run exactly"
    assert hold.decisions[incident.decision_index].decision.action == incident.settings_before.policy
    assert _total_waiting(hold.result) != _total_waiting(run.result)

    same = replay_action_at_incident(
        RuntimeSimulator(SCHEDULER),
        workload,
        run.decisions,
        incident,
        incident.action,
        config=controller._config,
    )
    # The counterfactual applies the action at the incident epoch and then holds, so it
    # agrees with the realized run exactly up to and including that epoch.
    assert _project(same.decisions[: incident.decision_index + 1]) == _project(
        run.decisions[: incident.decision_index + 1]
    )
    assert same.decisions[incident.decision_index].decision.action == incident.settings_after.policy

    other = SET_ROUND_ROBIN if incident.settings_after.policy != ACTION_ROUND_ROBIN else ACTION_SJF
    alternative = replay_action_at_incident(
        RuntimeSimulator(SCHEDULER),
        workload,
        run.decisions,
        incident,
        other,
        config=controller._config,
    )
    assert alternative.decisions[incident.decision_index].decision.action in (
        ACTION_ROUND_ROBIN,
        ACTION_SJF,
    )
    assert _total_waiting(alternative.result) != _total_waiting(same.result)


def test_learning_controller_trains_and_then_evaluates_read_only() -> None:
    """Training updates the table across episodes; evaluation leaves it untouched."""
    config = HealingConfig()
    policy = QLearningHealingPolicy(
        HealingContextEncoder(config), build_default_config().q_learning, seed=5
    )
    controller = _controller(policy, hysteresis_evaluations=1, max_interventions_per_trace=3)
    engine = RuntimeSimulator(SCHEDULER)
    workloads = [_workload("mixed", seed) for seed in (1, 2, 3)]
    for index, workload in enumerate(workloads):
        controller.begin_episode(index)
        engine.run(workload, controller, training=True, episode=index, policy_name="heal-q")
        controller.end_episode()
    table = policy.agent.q_table.copy()
    visits = policy.agent.state_action_visit_counts.sum()
    assert visits > 0, "the learning controller must have visited state-action pairs"

    controller.begin_episode(99)
    evaluation = engine.run(workloads[0], controller, training=False, episode=99, policy_name="heal-q")
    controller.end_episode()
    assert (policy.agent.q_table == table).all()
    assert policy.agent.state_action_visit_counts.sum() == visits
    assert all(decision.decision.action in range(4) for decision in evaluation.decisions)


def test_controller_rejects_invalid_lifecycle_and_arguments() -> None:
    """Lifecycle mistakes are rejected instead of silently producing meaningless logs."""
    controller = _controller(HoldOnlyPolicy())
    with pytest.raises(ValidationError, match="no episode is open"):
        controller.end_episode()
    with pytest.raises(ValidationError, match="RuntimeObservation"):
        controller.select_action(None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        controller.begin_episode(-1)
    with pytest.raises(ValidationError):
        _controller(HoldOnlyPolicy(), hysteresis_evaluations=0)
    with pytest.raises(ValidationError):
        SelfHealingController(HealingConfig(), SCHEDULER, object())  # type: ignore[arg-type]
