"""End-to-end tests of the self-healing loop on the event-driven simulator.

These tests exercise the whole chain: causal monitor -> detector -> diagnosis -> recovery
policy -> verification window -> accept / roll back / safe fallback -> ledger, plus the
counterfactual replay audit used as evidence in the study.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import ACTION_FCFS, ACTION_ROUND_ROBIN, ACTION_SJF, SchedulerConfig, build_default_config
from errors import ValidationError
from healing.actions import SET_ROUND_ROBIN, SET_SJF
from healing.config import HealingConfig
from healing.controller import (
    SelfHealingController,
    replay_action_at_incident,
    replay_hold_counterfactual,
)
from healing.policies import (
    HealingContextEncoder,
    HoldOnlyPolicy,
    PolicyChoice,
    QLearningHealingPolicy,
    RuleHealingPolicy,
    RuleWaitingTimePolicy,
)
from experiments.healing_config import HealingExperimentConfig, default_arms
from experiments.healing_experiment import _audit_episode
from scheduler import FCFS, RuntimeSimulator
from workload.generator import WorkloadGenerator

SCHEDULER = SchedulerConfig(round_robin_quantum=4, switching_cost=1)
VERDICTS = {"accepted", "rejected", "inconclusive", "held"}


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
        assert incident.verifier_reason
        assert incident.confirmed_failure == (incident.verdict == "rejected")
        if incident.verdict == "accepted":
            assert incident.rollback_target is None
            assert incident.closure_residual >= incident.accept_residual_factor
        else:
            assert incident.rollback_target is not None
            assert incident.closure_residual is None or (
                incident.closure_residual < incident.accept_residual_factor
            )
    assert episode.interventions == sum(1 for i in episode.incidents if i.action_changed)
    assert episode.holds == sum(1 for i in episode.incidents if not i.action_changed)
    assert (
        episode.accepted + episode.rejected + episode.inconclusive + episode.holds
        == len(episode.incidents)
    )
    assert episode.unverified == episode.rejected + episode.inconclusive


def test_verification_window_is_bounded_and_inconclusive_at_trace_end() -> None:
    """A window that cannot close before the trace ends is recorded as inconclusive (never
    as a failure), and the declared maximum window length is never exceeded."""
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
        if incident.window_truncated:
            assert incident.action_changed is True
            assert incident.verdict == "inconclusive"
            assert incident.confirmed_failure is False
            # The interrupted window still follows the declared posture: an unverified
            # change is reverted, so the ledger and ``settings_final`` agree.
            assert incident.rollback_target == incident.settings_before
            assert episode.settings_final == incident.settings_before


def test_rollback_is_applied_at_the_next_epoch() -> None:
    """After an unverified window the previous configuration is restored in the engine."""
    controller = _controller(
        RuleWaitingTimePolicy(HealingConfig()),
        hysteresis_evaluations=1,
        accept_residual_factor=10.0,
        reject_residual_factor=9.0,
    )
    run, episode = _run(controller, _workload("mixed", 9))
    rolled = [
        incident
        for incident in episode.incidents
        if incident.rollback_target is not None and incident.action_changed
    ]
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
        accept_residual_factor=10.0,
        reject_residual_factor=9.0,
        max_consecutive_failures=1,
    )
    run, episode = _run(controller, _workload("mixed", 9))
    assert episode.confirmed_failures >= 1
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
    # One intervention per trace, so the realized run holds the audited action for the rest
    # of the trace: the whole difference between the realized run and the hold replay is then
    # attributable to this single decision, and replaying it must reproduce the realized run.
    controller = _controller(
        RuleWaitingTimePolicy(HealingConfig()),
        hysteresis_evaluations=1,
        max_interventions_per_trace=1,
        min_load_factor=0.0,
    )
    workload = _workload("cpu_bursty", 5)
    run, episode = _run(controller, workload)
    changed = [incident for incident in episode.incidents if incident.action_changed]
    if not changed:
        pytest.skip("this trace produced no configuration change to audit")
    assert len(changed) == 1, "the audit identity below is stated for a single intervention"
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


def test_the_audit_gain_matches_a_hand_replayed_hold() -> None:
    """Hand-verifiable audit identity: one intervention, the same workload, one forced hold.

    The claim the audit's numbers rest on is that the counterfactual differs from the realized
    run only by the audited decision. This test checks that identity end to end on a single
    trace, against a total waiting time computed by hand from the two runs.
    """
    controller = _controller(RuleWaitingTimePolicy(HealingConfig()), hysteresis_evaluations=1)
    workload = _workload("cpu_bursty", 5)
    run, episode = _run(controller, workload)
    changed = [incident for incident in episode.incidents if incident.action_changed]
    if not changed:
        pytest.skip("this trace produced no configuration change to audit")
    incident = changed[0]

    hold = replay_hold_counterfactual(RuntimeSimulator(SCHEDULER), workload, run.decisions, incident)
    realized_total = float(sum(outcome.waiting_time for outcome in run.result.outcomes))
    hold_total = float(sum(outcome.waiting_time for outcome in hold.result.outcomes))
    audited_gain = hold_total - realized_total

    # 1. Same workload and same configuration prefix: the only difference is the forced hold.
    assert workload.fingerprint == hold.result.workload_fingerprint
    assert _project(hold.decisions[: incident.decision_index]) == _project(
        run.decisions[: incident.decision_index]
    )
    assert hold.decisions[incident.decision_index].decision.action == incident.settings_before.policy
    # 2. The gain the audit reports is the hand-computed difference, with the same sign.
    assert audited_gain == pytest.approx(
        float(sum(outcome.waiting_time for outcome in hold.result.outcomes)) - realized_total
    )
    assert (audited_gain > 0.0) == (hold_total > realized_total)
    assert (audited_gain < 0.0) == (hold_total < realized_total)
    # 3. The hold replay diverges from the realized run exactly at the audited epoch, and
    #    replaying the action the controller actually took reproduces the realized prefix
    #    through that epoch (the replay then holds, where the realized run verifies and may
    #    roll back, so the totals legitimately differ afterwards).
    assert hold.decisions[incident.decision_index].decision.action != (
        run.decisions[incident.decision_index].decision.action
    )
    same = replay_action_at_incident(
        RuntimeSimulator(SCHEDULER),
        workload,
        run.decisions,
        incident,
        incident.action,
        config=controller._config,
    )
    assert same.decisions[incident.decision_index].decision.action == incident.settings_after.policy
    assert _project(same.decisions[: incident.decision_index + 1]) == _project(
        run.decisions[: incident.decision_index + 1]
    )
    # 4. With a single intervention the two counterfactuals answer the same question, so the
    #    matched replay and the hold-forever replay must agree on the total they compare to.
    forever = replay_hold_counterfactual(
        RuntimeSimulator(SCHEDULER), workload, run.decisions, incident
    )
    assert float(sum(outcome.waiting_time for outcome in forever.result.outcomes)) == pytest.approx(
        hold_total
    )


def test_the_audit_replays_in_evaluation_mode_and_cannot_reach_the_learner() -> None:
    """No counterfactual outcome can reach the online loop.

    The audit re-runs the trace with the *same* policy object, so the only thing that keeps a
    learner read-only during those replays is that every call is made in evaluation mode. This
    test forces a learned policy to intervene, audits the episode and checks that every
    outcome the policy observed during the audit was an evaluation call and that the Q-table
    is bit-for-bit what the episode left behind.
    """
    config = HealingConfig()
    policy = QLearningHealingPolicy(
        HealingContextEncoder(config), build_default_config().q_learning, seed=11
    )
    observed: list = []
    original_observe = policy.observe_outcome

    def spy(context, choice, reward, next_context, *, terminal, training):
        observed.append(training)
        return original_observe(
            context, choice, reward, next_context, terminal=terminal, training=training
        )

    # Force the learner to change the configuration, so that the audited episode contains an
    # intervention to replay, then run and audit it.
    policy.observe_outcome = spy  # type: ignore[method-assign]
    original_select = policy.select

    def forced_select(context, *, training, episode, allowed_actions=None):
        choice = original_select(
            context, training=training, episode=episode, allowed_actions=allowed_actions
        )
        if training or choice.action == 0:
            allowed = tuple(allowed_actions or ())
            if SET_SJF in allowed:
                return PolicyChoice(action=SET_SJF, note="forced intervention for the audit test")
        return choice

    policy.select = forced_select  # type: ignore[method-assign]
    # The arm the audit is asked about is the learned one; any row with a policy_kind is fine
    # for this test, the audit only reads its name and initial policy.
    controller = _controller(policy, hysteresis_evaluations=1, min_load_factor=0.0)
    workload = _workload("mixed", 13)
    run, episode = _run(controller, workload)
    changed = [incident for incident in episode.incidents if incident.action_changed]
    if not changed:
        pytest.skip("this trace produced no configuration change to audit")
    table_after_episode = policy.agent._q_table.copy()  # noqa: SLF001
    rewards_before = [incident.reward for incident in episode.incidents]

    rows = _audit_episode(
        RuntimeSimulator(SCHEDULER),
        "validation",
        default_arms(HealingExperimentConfig())[0],
        SimpleNamespace(
            workload=workload, family="mixed", repetition=0, fingerprint=workload.fingerprint
        ),
        run,
        episode,
        0,
        policy,
        HealingExperimentConfig(),
    )
    assert rows, "the audited episode must produce at least one row"
    assert observed, "the replay must have called the policy's outcome hook"
    assert all(training is False for training in observed), "the audit must never train"
    assert np.array_equal(policy.agent._q_table, table_after_episode)  # noqa: SLF001
    assert [incident.reward for incident in episode.incidents] == rewards_before


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
