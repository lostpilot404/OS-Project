"""Verification semantics of the healing loop: accepted, rejected, inconclusive.

Three layers are tested here.

* The **verifier** in isolation (:mod:`healing.verifier`): the two declared boundaries, the
  evidence floor, truncated windows, and the fact that the rule is phase-aware.
* The **controller on a scripted trace**: a synthetic decision/observation stream that gives
  exact control over the monitor's load statistic, so the reachable verdicts, the rollback,
  the safe fallback, the reward and the escalation counter can be checked deterministically.
* The **whole loop on real workloads**: the end-to-end check that no future arrival and no
  final metric can reach an online decision.

Acceptance criteria covered here (the list from ``docs/EVALUATION_RESULTS.md``):
a harmful intervention is rejected and rolled back; a beneficial intervention is accepted
when the evidence suffices; an ambiguous window is inconclusive and is not mislabelled
harmful; insufficient observations can never cause an invalid acceptance; rollback restores
the last *accepted* configuration; repeated confirmed failures trigger the safe fallback;
repeated detections and oscillation stay bounded; no future outcome leaks into an online
decision, a verdict or the reward; and every decision is explainable from recorded evidence.
"""

from __future__ import annotations

import pytest

from config import (
    ACTION_FCFS,
    ACTION_ROUND_ROBIN,
    ACTION_SJF,
    SchedulerConfig,
    build_default_config,
)
from errors import ValidationError
from healing.actions import SET_ROUND_ROBIN, SET_SJF
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from healing.policies import HealingContextEncoder, PolicyChoice, RuleHealingPolicy
from healing.verifier import (
    VERDICT_ACCEPTED,
    VERDICT_HELD,
    VERDICT_INCONCLUSIVE,
    VERDICT_REJECTED,
    TriggerPhase,
    VerificationEvidence,
    closure_residual,
    expected_closure,
    verify,
)
from scheduler import RuntimeSimulator
from scheduler.runtime import (
    ObservedProcess,
    RuntimeDecisionRecord,
    RuntimeObservation,
)
from workload.generator import WorkloadGenerator
from workload.models import Process, Workload

SCHEDULER = SchedulerConfig(round_robin_quantum=4, switching_cost=1)

#: A trigger phase of the size the study's own control shows (queue length ~10, reference
#: ~9, band ~3). Using the real scale keeps the phase-aware baseline inside its fitted range.
PHASE = TriggerPhase(
    excess_band_units=1.3,
    relative_band=0.15,
    load_band=3.0,
    trigger_load=11.5,
    reference_load=9.2,
)


def _evidence(
    closure, *, changed=True, completions=3, decisions=5, duration=20, truncated=False, phase=PHASE
):
    return VerificationEvidence(
        changed=changed,
        closure_fraction=closure,
        phase=phase,
        window_decisions=decisions,
        window_completions=completions,
        window_duration=duration,
        truncated=truncated,
    )


# --------------------------------------------------------------------------------------
# The verifier in isolation
# --------------------------------------------------------------------------------------
def test_the_phase_baseline_is_the_declared_linear_function_of_trigger_observables() -> None:
    """The expectation is built from declared quantities the controller can see at the
    trigger epoch: a linear prediction of the no-action post-window load, turned into a
    closure with the statistic's own denominator, and nothing else takes part in it."""
    config = HealingConfig()
    coefficients = config.post_load_baseline
    assert len(coefficients) == len(PHASE.features()) + 1
    predicted_post = coefficients[0] + sum(
        c * f for c, f in zip(coefficients[1:], PHASE.features())
    )
    denominator = max(
        PHASE.trigger_load - PHASE.reference_load, max(PHASE.load_band, 1e-9)
    )
    assert expected_closure(PHASE, config) == pytest.approx(
        (PHASE.trigger_load - predicted_post) / denominator
    )
    # No clock and no decision counter is a regressor: a phase that differs only in how long
    # the trace has been running, or in how many decisions the controller has taken, is the
    # same phase as far as the expectation is concerned.
    late = TriggerPhase(
        PHASE.excess_band_units,
        PHASE.relative_band,
        PHASE.load_band,
        PHASE.trigger_load,
        PHASE.reference_load,
        ready_count=PHASE.ready_count,
        visible_work=PHASE.visible_work,
        decision_index=PHASE.decision_index + 400,
        elapsed_time=PHASE.elapsed_time + 5000,
    )
    assert expected_closure(late, config) == pytest.approx(expected_closure(PHASE, config))
    # A trigger that stands out less from its own reference is expected to close less of the
    # gap, so the same closure means something different in the two phases.
    quiet = TriggerPhase(1.3, 0.15, 3.0, 9.3, 9.2)
    assert expected_closure(quiet, config) < expected_closure(PHASE, config)
    assert closure_residual(1.0, PHASE, config) == pytest.approx(
        1.0 - expected_closure(PHASE, config)
    )
    assert closure_residual(None, PHASE, config) is None


def test_the_two_boundaries_split_the_window_into_three_meaningful_outcomes() -> None:
    """Both boundaries are percentile points of the same control distribution, so the
    measured false-accept and false-reject rates on the control are each at most 5 %."""
    config = HealingConfig()
    expected = expected_closure(PHASE, config)
    accept = expected + config.accept_residual_factor
    reject = expected + config.reject_residual_factor
    assert reject < accept, "the declared boundaries must be ordered"
    cases = (
        (accept + 0.05, VERDICT_ACCEPTED, False),
        (accept - 0.05, VERDICT_INCONCLUSIVE, False),
        (0.5 * (accept + reject), VERDICT_INCONCLUSIVE, False),
        (reject + 0.05, VERDICT_INCONCLUSIVE, False),
        (reject - 0.05, VERDICT_REJECTED, True),
        (reject - 1.0, VERDICT_REJECTED, True),
    )
    for closure, verdict, failure in cases:
        outcome = verify(_evidence(closure), config)
        assert outcome.verdict == verdict, (closure, verdict, outcome.reason)
        assert outcome.confirmed_failure is failure
        assert outcome.residual == pytest.approx(closure - expected)
        assert outcome.reason


def test_a_hold_is_never_a_failure_and_never_needs_a_rollback() -> None:
    """Nothing was changed, so there is nothing to reject and nothing to undo -- whatever
    the window looks like."""
    config = HealingConfig()
    for closure in (-50.0, 0.0, 50.0, None):
        outcome = verify(_evidence(closure, changed=False), config)
        assert outcome.verdict == VERDICT_HELD
        assert outcome.confirmed_failure is False
        assert outcome.rollback_required is False
        assert not outcome.accepted and not outcome.inconclusive


def test_insufficient_evidence_can_never_be_accepted() -> None:
    """A window that closed its whole gap but saw too few completions, no duration at all or
    an interrupted interval is inconclusive: never accepted, never a confirmed failure."""
    config = HealingConfig()
    for evidence in (
        _evidence(50.0, completions=config.verify_completions - 1),
        _evidence(50.0, duration=0),
        _evidence(50.0, truncated=True),
        _evidence(None),
    ):
        outcome = verify(evidence, config)
        assert outcome.verdict == VERDICT_INCONCLUSIVE, evidence
        assert not outcome.accepted
        assert outcome.confirmed_failure is False
        assert outcome.reason == "insufficient_evidence_thin_telemetry" or (
            outcome.reason == "insufficient_evidence_window_truncated"
        )


def test_an_ambiguous_window_in_one_phase_can_be_confirmed_in_another() -> None:
    """The same statistic has different meanings in different phases: the boundary is on the
    residual, not on the raw closure."""
    config = HealingConfig()
    loud = TriggerPhase(1.3, 0.15, 3.0, 11.5, 9.2)
    calm = TriggerPhase(0.2, 0.10, 3.0, 9.3, 9.2)
    assert expected_closure(calm, config) < expected_closure(loud, config)
    closure = expected_closure(calm, config) + config.accept_residual_factor + 0.02
    assert verify(_evidence(closure, phase=calm), config).verdict == VERDICT_ACCEPTED
    assert verify(_evidence(closure, phase=loud), config).verdict != VERDICT_ACCEPTED


def test_evidence_validation_rejects_inconsistent_windows() -> None:
    with pytest.raises(ValidationError):
        verify(object(), HealingConfig())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _evidence(1.0, decisions=2, completions=5)
    with pytest.raises(ValidationError):
        _evidence(1.0, decisions=-1)
    with pytest.raises(ValidationError):
        _evidence(1.0, duration=-1)
    with pytest.raises(ValidationError):
        _evidence(1.0, phase=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# A scripted trace: deterministic control of the verification statistic
# --------------------------------------------------------------------------------------
class _ScriptedPolicy:
    """Policy stub: request a fixed action whenever the controller allows it."""

    def __init__(self, action: int) -> None:
        self._action = action
        self.choices = 0
        self.rewards = []

    def select(self, context, *, training, episode, allowed_actions=None) -> PolicyChoice:
        del context, training, episode
        self.choices += 1
        allowed = tuple(allowed_actions or ())
        action = self._action if not allowed or self._action in allowed else 0
        return PolicyChoice(action=action, note="scripted")

    def observe_outcome(self, context, choice, reward, next_context, *, terminal, training) -> None:
        del context, choice, next_context, terminal, training
        self.rewards.append(reward)


class _SequencePolicy(_ScriptedPolicy):
    """Policy stub that walks a declared action sequence (repeating the last entry)."""

    def __init__(self, *actions: int) -> None:
        super().__init__(actions[0])
        self._actions = tuple(actions)

    def select(self, context, *, training, episode, allowed_actions=None) -> PolicyChoice:
        del context, training, episode
        self.choices += 1
        allowed = tuple(allowed_actions or ())
        for action in self._actions:
            if not allowed or action in allowed:
                return PolicyChoice(action=action, note="scripted sequence")
        return PolicyChoice(action=0, note="scripted sequence exhausted")


class _ScriptedTrace:
    """Drives a controller with synthetic observations and records.

    The monitor's load statistic is the waiting-time accrual rate of a decision interval, so
    the caller controls it exactly through ``rate`` (waiting processes per unit time) and
    ``completions`` (observed waiting times revealed at the end of the interval).

    The ready set defaults to nine processes of ten units each because that is the population
    the declared phase baseline is calibrated on: the training control's ready-set size at a
    trigger has a median of nine and its visible work a median of about one hundred units. A
    synthetic trigger with three ready processes and a load of 11.5 belongs to no real trace
    -- no scheduling statistic stays at the level of nine waiting processes while only three
    exist -- and the phase baseline correctly says such a queue would drain by itself, so
    every window in such a trace would be judged against an expectation it cannot reach.
    """

    def __init__(self, controller: SelfHealingController, *, ready: int = 9, mean_burst: float = 10.0):
        self.controller = controller
        self._ready = ready
        self._mean_burst = mean_burst
        self._time = 0
        self._decisions = 0
        self._completed = 0
        self._arrived = ready
        controller.begin_episode(0)

    @property
    def window_open(self) -> bool:
        """Whether a verification window is currently open."""
        return self.controller._pending is not None  # noqa: SLF001 - test instrumentation

    def observation(self) -> RuntimeObservation:
        return RuntimeObservation(
            current_time=self._time,
            ready_processes=tuple(
                ObservedProcess(
                    pid=1000 + index,
                    arrival_time=0,
                    burst_time=int(self._mean_burst),
                    remaining_burst=int(self._mean_burst),
                    priority=2,
                )
                for index in range(self._ready)
            ),
            completed_count=self._completed,
            arrived_count=self._arrived,
            observed_mean_burst=self._mean_burst,
            observed_priority_spread=1,
            mean_ready_arrival_age=float(self._time),
            previous_policy=ACTION_FCFS,
        )

    def step(self, *, rate: float, duration: int = 4, completions=()):
        """Run one decision interval at the given waiting-accrual rate."""
        observation = self.observation()
        decision = self.controller.select_action(observation, training=False, episode=0)
        record = RuntimeDecisionRecord(
            decision_index=self._decisions,
            observation=observation,
            decision=decision,
            chosen_pid=observation.ready_processes[0].pid,
            start_time=self._time,
            end_time=self._time + duration,
            switch_overhead=0,
            waiting_cost_increment=int(round(rate * duration)),
            reward=0.0,
            next_state_index=None,
            terminal=False,
            policy_switch=False,
            completed_pids=tuple(2000 + index for index in range(len(completions))),
            completed_waiting_times=tuple(completions),
        )
        self.controller.observe_decision(record)
        self._time += duration
        self._decisions += 1
        self._completed += len(completions)
        self._arrived += len(completions)
        return decision

    def close(self):
        """Close the episode and return its record."""
        return self.controller.end_episode()


def _controller(policy, **overrides) -> SelfHealingController:
    values = {"hysteresis_evaluations": 1}
    values.update(overrides)
    return SelfHealingController(
        HealingConfig(**values), SCHEDULER, policy, initial_policy=ACTION_FCFS, label="scripted"
    )


def _warm_up(trace: _ScriptedTrace, *, steps: int = 5, wait: int = 10, rate: float = 8.0) -> None:
    """Establish the reference waiting time and the normal load level.

    At least ``min_window_completions`` intervals must complete before the detector can
    absorb anything into its reference, so the warm-up is longer than that floor.
    """
    for _ in range(steps):
        trace.step(rate=rate, completions=(wait,))


def _trigger(trace: _ScriptedTrace, *, rate: float = 12.0, wait: int = 60, cap: int = 30) -> None:
    """Raise the completion window above the detection threshold and stop with it open."""
    for _ in range(cap):
        if trace.window_open:
            return
        trace.step(rate=rate, completions=(wait,))
    raise AssertionError("the scripted trace never triggered a degradation event")


def _close_window(trace: _ScriptedTrace, *, rate: float, completions: int = 30, cap: int = 40) -> None:
    """Feed the verification window until it closes at the given accrual rate.

    The window closes at ``decision_index + verify_decisions - 1`` (its own epoch counts), so
    the loop stops when the controller reports no pending window instead of after a fixed
    number of steps.
    """
    for _ in range(cap):
        if not trace.window_open:
            return
        trace.step(rate=rate, completions=(completions,))
    raise AssertionError("the verification window never closed")


#: Window accrual rates measured to land in the three declared outcomes in this scripted
#: scenario. The phase of the trigger is identical in all of them (queue length 11.5 against
#: the trace's own reference of 11.47 in a band of 1.72 with nine ready processes holding
#: ninety units of work, which the declared baseline turns into an expected closure of
#: 3.139). Measured residuals against that expectation: 0.0 -> +2.15 and accepted, 4.5 ->
#: +0.06 and inconclusive, 12.0 -> -3.43 and rejected. Draining the queue to zero closes the
#: gap, holding the full load keeps the statistic inside the control's own range, and
#: accruing at the trigger rate throughout stays below the control's lower tail.
ACCEPTING_RATE = 0.0
AMBIGUOUS_RATE = 4.5
REJECTING_RATE = 12.0


def _run_scripted(*, window_rate: float, action: int = SET_SJF, **overrides):
    """Warm up, open exactly one incident, close its window at ``window_rate`` and return the
    trace and the episode."""
    trace = _ScriptedTrace(_controller(_ScriptedPolicy(action), **overrides))
    _warm_up(trace)
    _trigger(trace)
    _close_window(trace, rate=window_rate)
    return trace, trace.close()


def test_a_beneficial_window_is_accepted_and_keeps_the_configuration() -> None:
    """The safe path: a window that drains the queue completely is accepted, the new
    configuration becomes the known-good one and nothing is rolled back."""
    trace = _ScriptedTrace(_controller(_ScriptedPolicy(SET_SJF)))
    _warm_up(trace)
    _trigger(trace)
    _close_window(trace, rate=0.0)
    episode = trace.close()
    assert len(episode.incidents) == 1
    incident = episode.incidents[0]
    assert incident.verdict == VERDICT_ACCEPTED
    assert incident.confirmed_failure is False
    assert incident.rollback_target is None
    assert incident.settings_after.policy == ACTION_SJF
    assert episode.settings_final == incident.settings_after


def test_an_ambiguous_window_is_inconclusive_and_is_not_called_harmful() -> None:
    """A window that neither closes the gap nor behaves worse than the control's own range is
    inconclusive: no confirmed failure, no escalation and no rollback penalty."""
    trace, episode = _run_scripted(window_rate=AMBIGUOUS_RATE)
    incident = episode.incidents[0]
    assert incident.verdict == VERDICT_INCONCLUSIVE
    assert incident.confirmed_failure is False
    assert incident.consecutive_failures == 0
    assert episode.confirmed_failures == 0
    assert episode.fallback_triggered is False
    # Declared posture: an unverified change is still reverted, and the ledger says so.
    assert incident.rollback_target is not None
    assert incident.rollback_target == incident.settings_before
    assert incident.verifier_reason.startswith("inconclusive_")


def test_a_window_worse_than_the_control_range_is_rejected_and_rolled_back() -> None:
    """The failure path: the load does not fall at all, so the change is a confirmed failure
    and it is reverted to the last accepted configuration."""
    trace, episode = _run_scripted(window_rate=12.0)
    incident = episode.incidents[0]
    assert incident.verdict == VERDICT_REJECTED
    assert incident.confirmed_failure is True
    assert incident.consecutive_failures == 1
    assert incident.rollback_target == incident.settings_before
    assert incident.verifier_reason == "residual_below_control_lower_tail"


def test_consecutive_confirmed_failures_trigger_the_safe_fallback() -> None:
    """Two confirmed failures in a row pin the declared safe configuration for the rest of
    the trace."""
    trace = _ScriptedTrace(_controller(_ScriptedPolicy(SET_SJF), max_consecutive_failures=2))
    _warm_up(trace)
    for _ in range(2):
        _trigger(trace)
        _close_window(trace, rate=REJECTING_RATE)
        # Let the cooldown pass without opening a window.
        for _ in range(4):
            if trace.window_open:
                break
            trace.step(rate=12.0, completions=(60,))
    for _ in range(6):
        trace.step(rate=12.0, completions=(60,))
    episode = trace.close()
    rejected = [item for item in episode.incidents if item.verdict == VERDICT_REJECTED]
    assert len(rejected) >= 2, [item.verdict for item in episode.incidents]
    assert episode.confirmed_failures >= 2
    assert episode.fallback_triggered is True
    assert episode.settings_final.policy == HealingConfig().fallback_policy
    assert trace.controller._adaptation_enabled is False  # noqa: SLF001


def test_ambiguity_is_not_evidence_of_harm_and_never_counts_towards_the_fallback() -> None:
    """An inconclusive window is neither a success nor a failure: it cannot advance the
    escalation counter, whatever the declared fallback threshold is."""
    trace = _ScriptedTrace(
        _controller(_ScriptedPolicy(SET_SJF), max_consecutive_failures=1, min_load_factor=0.9)
    )
    _warm_up(trace)
    for _ in range(4):
        _trigger(trace)
        _close_window(trace, rate=AMBIGUOUS_RATE)
        for _ in range(4):
            if trace.window_open:
                break
            trace.step(rate=10.0, completions=(40,))
    episode = trace.close()
    assert episode.inconclusive >= 2, [item.verdict for item in episode.incidents]
    assert episode.confirmed_failures == 0
    assert episode.fallback_triggered is False
    assert all(item.verdict != VERDICT_REJECTED for item in episode.incidents)


def test_the_rollback_restores_the_last_accepted_configuration_not_the_previous_one() -> None:
    """The failure path after an accepted change: the rollback returns to the configuration
    that was actually verified, not to the one that happened to precede the failure."""
    trace = _ScriptedTrace(_controller(_SequencePolicy(SET_SJF, SET_ROUND_ROBIN)))
    _warm_up(trace)
    _trigger(trace)
    _close_window(trace, rate=ACCEPTING_RATE)
    accepted = trace.controller._episode_incidents[0]  # noqa: SLF001
    assert accepted.verdict == VERDICT_ACCEPTED
    assert accepted.settings_after.policy == ACTION_SJF
    assert trace.controller._last_accepted == accepted.settings_after  # noqa: SLF001
    _trigger(trace)
    _close_window(trace, rate=REJECTING_RATE)
    episode = trace.close()
    rejected = episode.incidents[-1]
    assert rejected.verdict == VERDICT_REJECTED
    assert rejected.settings_after.policy != accepted.settings_after.policy
    assert rejected.rollback_target == accepted.settings_after
    assert trace.controller._settings == accepted.settings_after  # noqa: SLF001


def test_the_declared_inconclusive_posture_is_the_only_thing_that_changes() -> None:
    """Both postures produce the same verdict, the same failure accounting and the same
    reward; they differ only in whether the unverified change is reverted."""
    outcomes = {}
    for posture in (True, False):
        trace, episode = _run_scripted(window_rate=AMBIGUOUS_RATE, inconclusive_rollback=posture)
        incident = episode.incidents[0]
        outcomes[posture] = (
            incident.verdict,
            incident.confirmed_failure,
            incident.consecutive_failures,
            incident.reward,
            incident.rollback_target is not None,
        )
    assert outcomes[True][:4] == outcomes[False][:4]
    assert outcomes[True][4] is True and outcomes[False][4] is False


def test_verdicts_and_rewards_do_not_depend_on_what_happens_after_the_window() -> None:
    """No future information reaches a verdict or a reward: two traces that share a prefix
    and then diverge completely produce identical incidents for that prefix."""
    recorded = []
    for future_rate, future_wait in ((0.0, 30), (60.0, 200)):
        trace = _ScriptedTrace(_controller(_ScriptedPolicy(SET_SJF)))
        _warm_up(trace)
        _trigger(trace)
        _close_window(trace, rate=AMBIGUOUS_RATE)
        first = trace.controller._episode_incidents[0]  # noqa: SLF001
        for _ in range(8):
            trace.step(rate=future_rate, completions=(future_wait,))
        episode = trace.close()
        assert episode.incidents[0] is first
        recorded.append(first.as_row())
    assert recorded[0] == recorded[1]


def test_oscillation_and_intervention_rate_stay_bounded() -> None:
    """The loop cannot flip the configuration at every epoch: the cooldown, the hysteresis
    and the declared per-trace budget bound how often it acts."""
    controller = _controller(
        _ScriptedPolicy(SET_SJF), max_interventions_per_trace=2, min_load_factor=0.9
    )
    trace = _ScriptedTrace(controller)
    for _ in range(40):
        # Sustained degradation: the detector keeps wanting to act.
        trace.step(rate=25.0, completions=(80,))
    episode = trace.close()
    assert episode.interventions <= 2, "the declared intervention budget must bind"
    indices = [item.decision_index for item in episode.incidents]
    assert indices == sorted(indices)
    for previous, following in zip(episode.incidents, episode.incidents[1:]):
        assert (
            following.decision_index - previous.decision_index
            >= HealingConfig().cooldown_decisions
        ), "incidents cannot be closer than the declared cooldown"


def test_the_ledger_explains_every_verdict_and_reward_from_recorded_evidence() -> None:
    """Every verdict must be recomputable from the incident's own row, and the reward must be
    the recorded statistic minus the declared boundary, the measured switch cost and -- only
    for a confirmed failure -- the declared rollback penalty."""
    config = HealingConfig()
    for rate in (ACCEPTING_RATE, AMBIGUOUS_RATE, REJECTING_RATE):
        _, episode = _run_scripted(window_rate=rate)
        incident = episode.incidents[0]
        residual = incident.closure_residual
        assert residual is not None
        if residual >= config.accept_residual_factor:
            expected = VERDICT_ACCEPTED
        elif residual <= config.reject_residual_factor:
            expected = VERDICT_REJECTED
        else:
            expected = VERDICT_INCONCLUSIVE
        assert incident.verdict == expected
        assert incident.confirmed_failure == (expected == VERDICT_REJECTED)
        band = max(config.min_band_absolute, config.min_band_fraction * incident.reference_at_trigger)
        penalty = (
            config.rollback_penalty * incident.reference_at_trigger / band
            if incident.confirmed_failure
            else 0.0
        )
        assert incident.reward == pytest.approx(
            residual - config.accept_residual_factor - incident.cost_band_units - penalty
        )


# --------------------------------------------------------------------------------------
# The whole loop on a real workload: no future information can reach a decision
# --------------------------------------------------------------------------------------
def _run_loop(workload):
    """Run the declared loop with the declared rule policy on a real workload."""
    engine = RuntimeSimulator(SCHEDULER)
    controller = SelfHealingController(
        HealingConfig(), SCHEDULER, RuleHealingPolicy(HealingConfig()), initial_policy=ACTION_FCFS
    )
    controller.begin_episode(0)
    run = engine.run(workload, controller, training=False, episode=0, policy_name="leakage-probe")
    episode = controller.end_episode()
    return run, episode


def test_a_later_arrival_cannot_change_an_incident_that_closed_before_it() -> None:
    """Real simulator, real controller: six heavy processes are appended *after* the first
    verification window. The incident that closed before they arrived must stay byte
    identical, while the schedule as a whole must become measurably worse -- otherwise the
    probe would be vacuous."""
    base = WorkloadGenerator(build_default_config().families).generate("mixed", 3)
    base_run, episode = _run_loop(base)
    assert episode.incidents, "the probe needs a trace that triggers"
    first = episode.incidents[0]
    assert first.window_end_time is not None
    horizon = first.window_end_time

    extra = [
        Process(9000 + index, horizon + 1 + 3 * index, 80, 1) for index in range(6)
    ]
    perturbed = Workload(
        f"{base.name}+future", list(base) + extra, description="future arrival probe"
    )
    perturbed_run, perturbed_episode = _run_loop(perturbed)

    assert first.as_row() == perturbed_episode.incidents[0].as_row(), (
        "an incident cannot depend on work that had not arrived yet"
    )
    def prefix(run):
        return [
            (record.start_time, record.end_time, record.chosen_pid)
            for record in run.decisions
            if record.end_time <= horizon
        ]

    def total(run):
        return sum(outcome.waiting_time for outcome in run.result.outcomes)

    assert prefix(base_run) == prefix(perturbed_run)
    assert total(perturbed_run) > total(base_run), "the appended work must matter"


def test_the_encoder_used_by_the_reward_is_the_declared_one() -> None:
    """The learning context is the declared encoder, so a reward can always be traced back to
    the state it was scored in."""
    config = HealingConfig()
    encoder = HealingContextEncoder(config)
    assert encoder.n_states == 6 * 3 * 3 * 4 * 3
    assert config.inconclusive_rollback is True, "the declared posture is rollback"
