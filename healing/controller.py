"""The self-healing controller: monitor, detect, diagnose, act, verify, roll back.

This module is the primary system of the project. One instance drives the event-driven
scheduler of :mod:`scheduler.runtime` for one complete workload trace:

1. **monitor** -- every completed process contributes its realized waiting time to a
   rolling statistic with a stable reference and dispersion band (:mod:`healing.monitor`);
2. **detect** -- a degradation event is declared only when the statistic exceeds the
   reference by the declared threshold for several consecutive evaluations, outside the
   cooldown, with enough visible remaining work (:mod:`healing.detector`);
3. **diagnose** -- the visible ready-set composition is turned into a primary cause with
   the evidence behind it (:mod:`healing.diagnosis`);
4. **act** -- the recovery policy (predeclared rule or tabular Q-learning) selects one of
   eight recovery actions and the configuration is applied to the very next dispatch;
5. **verify** -- the window's own load closure is compared with two declared boundaries
   derived from the no-action control (:mod:`healing.verifier`), which yields *accepted*,
   *rejected* or *inconclusive*;
6. **accept, roll back, or fall back** -- a verified improvement is accepted and becomes
   the new known-good configuration; a rejected or inconclusive change is rolled back to
   the last accepted configuration, and after repeated *rejected* interventions (confirmed
   failures -- ambiguity is not enough) it pins that configuration as a safe fallback and
   stops adapting for the remainder of the trace;
7. **record** -- every incident keeps its trigger evidence, diagnosis, action, verification
   result, realized switch overhead and outcome, so any claim can be traced to a log line.

Everything the controller sees is causal: :class:`scheduler.runtime.RuntimeObservation`
plus the completion notices of :meth:`observe_decision`. The controller never receives the
workload, the unarrived processes, the future event list or the finished schedule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import ACTION_FCFS, ACTION_ROUND_ROBIN, SchedulerConfig
from errors import ValidationError
from healing.actions import (
    GROW_QUANTUM,
    HEALING_ACTION_NAMES,
    HOLD,
    REVERT_TO_LAST_ACCEPTED,
    SET_FCFS,
    SET_PRIORITY,
    SET_ROUND_ROBIN,
    SET_SJF,
    SHRINK_QUANTUM,
    SchedulerSettings,
    apply_healing_action,
)
from healing.config import HealingConfig
from healing.detector import DegradationDetector, DetectionOutcome
from healing.diagnosis import (
    DIAGNOSIS_CATEGORIES,
    ReadyEvidence,
    build_ready_evidence,
    diagnose,
)
from healing.monitor import CausalMonitor, MonitorSnapshot
from healing.policies import (
    HealingContext,
    HealingContextEncoder,
    HealingPolicy,
    PolicyChoice,
)
from healing.reward import (
    closure_fraction_from_loads,
    controller_reward,
    cost_in_band_units,
)
from healing.verifier import (
    ALL_VERDICTS,
    VERDICT_ACCEPTED,
    VERDICT_HELD,
    VERDICT_INCONCLUSIVE,
    VERDICT_REJECTED,
    TriggerPhase,
    VerificationEvidence,
    verify,
)
from scheduler.runtime import (
    RuntimeActionDecision,
    RuntimeDecisionRecord,
    RuntimeObservation,
)

__all__ = [
    "available_actions",
    "replay_action_at_incident",
    "IncidentRecord",
    "EpisodeRecord",
    "SelfHealingController",
    "VERDICTS",
]

#: Possible outcomes of a recovery window: the three intervention verdicts of
#: :mod:`healing.verifier` plus ``held`` for an incident that changed nothing.
VERDICTS: Tuple[str, ...] = ALL_VERDICTS


@dataclass(frozen=True)
class IncidentRecord:
    """One concluded healing incident with its full evidence trail.

    The record is intentionally wide: a reader must be able to reconstruct *why* the
    system acted, *what* it changed, *what* it observed afterwards and *what that cost*
    without re-running the experiment.
    """

    episode_index: int
    incident_index: int
    decision_index: int
    simulated_time: int
    detector_reason: str
    suspicion_evaluations: int
    trigger_threshold: float
    reference_at_trigger: float
    band_at_trigger: float
    window_mean_at_trigger: float
    window_size_at_trigger: int
    window_median_at_trigger: float
    window_max_at_trigger: float
    excess_band_units: float
    ready_count_at_trigger: int
    visible_remaining_work_at_trigger: int
    diagnosis: str
    diagnosis_index: int
    diagnosis_matched: str
    diagnosis_statement: str
    context_state_index: int
    offered_actions: str
    action: int
    action_name: str
    action_changed: bool
    action_note: str
    request_note: str
    settings_before: SchedulerSettings
    settings_after: SchedulerSettings
    epsilon: float
    explored: bool
    verdict: str
    verifier_reason: str
    confirmed_failure: bool
    consecutive_failures: int
    window_truncated: bool
    forced_hold: bool
    verification_decisions: int
    verification_completions: int
    post_window_mean: Optional[float]
    improvement: Optional[float]
    improvement_band_units: Optional[float]
    pre_load_level: float
    trigger_load_level: float
    post_load_level: Optional[float]
    load_band_at_trigger: float
    load_improvement: Optional[float]
    load_improvement_band_units: Optional[float]
    closure_fraction: Optional[float]
    closure_residual: Optional[float]
    expected_closure: float
    required_closure_factor: float
    reject_closure_factor: float
    accept_residual_factor: float
    reject_residual_factor: float
    cost_band_units: float
    reward: Optional[float]
    rollback_target: Optional[SchedulerSettings]
    switch_overhead_units: int
    window_end_time: Optional[int]
    fallback_triggered: bool

    @property
    def is_intervention(self) -> bool:
        """Whether the system actually changed the live configuration."""
        return self.action_changed

    @property
    def is_unverified_intervention(self) -> bool:
        """In-run indicator: an intervention that was not verified (rejected or inconclusive).

        This is the signal the system itself can observe while running. The independent
        ground truth used by the study is the counterfactual replay audit
        (:mod:`experiments.healing_experiment`), which asks what would have happened had
        the same controller held instead of applying this action. The two agree poorly on
        purpose: the audit knows the future, the verifier does not (see
        ``docs/EVALUATION_RESULTS.md``).
        """
        return self.action_changed and self.verdict in (VERDICT_REJECTED, VERDICT_INCONCLUSIVE)

    def as_row(self) -> Dict[str, Any]:
        """Flatten the record into one CSV row."""
        row: Dict[str, Any] = {
            "episode_index": self.episode_index,
            "incident_index": self.incident_index,
            "decision_index": self.decision_index,
            "simulated_time": self.simulated_time,
            "detector_reason": self.detector_reason,
            "suspicion_evaluations": self.suspicion_evaluations,
            "trigger_threshold": self.trigger_threshold,
            "reference_at_trigger": self.reference_at_trigger,
            "band_at_trigger": self.band_at_trigger,
            "window_mean_at_trigger": self.window_mean_at_trigger,
            "window_size_at_trigger": self.window_size_at_trigger,
            "window_median_at_trigger": self.window_median_at_trigger,
            "window_max_at_trigger": self.window_max_at_trigger,
            "excess_band_units": self.excess_band_units,
            "ready_count_at_trigger": self.ready_count_at_trigger,
            "visible_remaining_work_at_trigger": self.visible_remaining_work_at_trigger,
            "diagnosis": self.diagnosis,
            "diagnosis_index": self.diagnosis_index,
            "diagnosis_matched": self.diagnosis_matched,
            "diagnosis_statement": self.diagnosis_statement,
            "context_state_index": self.context_state_index,
            "offered_actions": self.offered_actions,
            "action": self.action,
            "action_name": self.action_name,
            "action_changed": self.action_changed,
            "action_note": self.action_note,
            "request_note": self.request_note,
            "settings_before_policy": self.settings_before.policy_name,
            "settings_before_quantum": self.settings_before.quantum,
            "settings_after_policy": self.settings_after.policy_name,
            "settings_after_quantum": self.settings_after.quantum,
            "epsilon": self.epsilon,
            "explored": self.explored,
            "verdict": self.verdict,
            "verifier_reason": self.verifier_reason,
            "confirmed_failure": self.confirmed_failure,
            "consecutive_failures": self.consecutive_failures,
            "window_truncated": self.window_truncated,
            "forced_hold": self.forced_hold,
            "verification_decisions": self.verification_decisions,
            "verification_completions": self.verification_completions,
            "post_window_mean": self.post_window_mean,
            "improvement": self.improvement,
            "improvement_band_units": self.improvement_band_units,
            "pre_load_level": self.pre_load_level,
            "trigger_load_level": self.trigger_load_level,
            "post_load_level": self.post_load_level,
            "load_band_at_trigger": self.load_band_at_trigger,
            "load_improvement": self.load_improvement,
            "load_improvement_band_units": self.load_improvement_band_units,
            "closure_fraction": self.closure_fraction,
            "closure_residual": self.closure_residual,
            "expected_closure": self.expected_closure,
            "required_closure_factor": self.required_closure_factor,
            "reject_closure_factor": self.reject_closure_factor,
            "accept_residual_factor": self.accept_residual_factor,
            "reject_residual_factor": self.reject_residual_factor,
            "cost_band_units": self.cost_band_units,
            "reward": self.reward,
            "rollback_target_policy": (
                self.rollback_target.policy_name if self.rollback_target else ""
            ),
            "rollback_target_quantum": (
                self.rollback_target.quantum if self.rollback_target else ""
            ),
            "switch_overhead_units": self.switch_overhead_units,
            "window_end_time": self.window_end_time,
            "fallback_triggered": self.fallback_triggered,
        }
        return row


def available_actions(
    settings: SchedulerSettings,
    last_accepted: Optional[SchedulerSettings],
    config: HealingConfig,
) -> Tuple[int, ...]:
    """Return every action that can actually change the live configuration right now.

    ``HOLD`` is always available. A requested action outside this set is not a decision:
    it cannot alter a single scheduling decision, so it must not be offered to a learned
    controller and must never be counted as an intervention.
    """
    allowed = [HOLD]
    for action, policy in (
        (SET_FCFS, 0),
        (SET_SJF, 1),
        (SET_ROUND_ROBIN, 2),
        (SET_PRIORITY, 3),
    ):
        if policy != settings.policy:
            allowed.append(action)
    if settings.policy == ACTION_ROUND_ROBIN:
        if settings.quantum > config.min_quantum:
            allowed.append(SHRINK_QUANTUM)
        if settings.quantum < config.max_quantum:
            allowed.append(GROW_QUANTUM)
    if last_accepted is not None and last_accepted != settings:
        allowed.append(REVERT_TO_LAST_ACCEPTED)
    return tuple(allowed)


@dataclass(frozen=True)
class EpisodeRecord:
    """Healing bookkeeping of one complete trace (one workload)."""

    episode_index: int
    decisions: int
    incidents: Tuple[IncidentRecord, ...]
    interventions: int
    holds: int
    accepted: int
    rejected: int
    inconclusive: int
    unverified: int
    confirmed_failures: int
    fallback_triggered: bool
    interventions_used: int
    settings_initial: SchedulerSettings
    settings_final: SchedulerSettings
    detector_reasons: Tuple[Tuple[str, int], ...]

    def as_row(self) -> Dict[str, Any]:
        """Flatten the episode summary into one CSV row."""
        return {
            "episode_index": self.episode_index,
            "decisions": self.decisions,
            "incidents": len(self.incidents),
            "interventions": self.interventions,
            "holds": self.holds,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "inconclusive": self.inconclusive,
            "unverified": self.unverified,
            "confirmed_failures": self.confirmed_failures,
            "fallback_triggered": self.fallback_triggered,
            "interventions_used": self.interventions_used,
            "settings_initial_policy": self.settings_initial.policy_name,
            "settings_initial_quantum": self.settings_initial.quantum,
            "settings_final_policy": self.settings_final.policy_name,
            "settings_final_quantum": self.settings_final.quantum,
        }


@dataclass
class _PendingIncident:
    """Mutable state of an incident whose verification window is still open."""

    incident_index: int
    decision_index: int
    simulated_time: int
    detector_reason: str
    suspicion_evaluations: int
    snapshot: MonitorSnapshot
    threshold: float
    evidence: ReadyEvidence
    diagnosis_statement: str
    diagnosis_category: str
    diagnosis_index: int
    diagnosis_matched: str
    context: HealingContext
    choice: PolicyChoice
    offered_actions: str
    changed: bool
    forced_hold: bool
    action: int
    action_note: str
    settings_before: SchedulerSettings
    settings_after: SchedulerSettings
    pre_load_level: float = 0.0
    trigger_load_level: float = 0.0
    load_band_at_trigger: float = 0.0
    decisions: int = 0
    completions: int = 0
    switch_overhead: int = 0
    window_end_time: Optional[int] = None
    waiting: List[int] = field(default_factory=list)
    window_accrued_wait: int = 0
    window_duration: int = 0
    truncated: bool = False


class SelfHealingController:
    """Causal self-healing controller implementing the runtime controller protocol."""

    def __init__(
        self,
        config: HealingConfig,
        scheduler_config: SchedulerConfig,
        policy: HealingPolicy,
        *,
        initial_policy: int = ACTION_FCFS,
        initial_quantum: Optional[int] = None,
        label: str = "self-healing",
        forced_holds: Sequence[int] = (),
    ) -> None:
        """Create a controller.

        Args:
            config: Declared detection/verification thresholds.
            scheduler_config: Configuration of the event engine (priority convention and
                the default quantum used when ``initial_quantum`` is not given).
            policy: Recovery-action policy (rule, Q-learning or hold-only control).
            initial_policy: Policy in force at the start of every trace. All healing arms
                start from the same declared configuration; starting configurations are
                part of the experiment design, not of the controller.
            initial_quantum: Quantum in force at the start of every trace.
            label: Human-readable arm label used in logs and artifacts.
            forced_holds: Indices of incidents at which the controller must *not* act, even
                though the detector fired. This is the hook the offline counterfactual
                audit uses: re-running the whole controller with one incident forced to
                hold measures what that single intervention was worth under the same
                policy and the same workload. It is a declared property of the run (known
                before it starts) and it never lets the controller see an outcome it could
                not see online.

        Raises:
            ValidationError: If any argument is invalid.
        """
        if not isinstance(config, HealingConfig):
            raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
        if not isinstance(scheduler_config, SchedulerConfig):
            raise ValidationError("scheduler_config must be a SchedulerConfig")
        if initial_policy not in range(4):
            raise ValidationError(f"initial_policy must be an index in 0..3, got {initial_policy!r}")
        for method in ("select", "observe_outcome"):
            if not callable(getattr(policy, method, None)):
                raise ValidationError(
                    f"policy must define {method}(...), got {type(policy).__name__}"
                )
        quantum = scheduler_config.round_robin_quantum if initial_quantum is None else initial_quantum
        self._config = config
        self._scheduler_config = scheduler_config
        self._policy = policy
        self._encoder = HealingContextEncoder(config)
        self._initial_settings = SchedulerSettings(policy=initial_policy, quantum=quantum)
        self._label = label
        if isinstance(forced_holds, (int, str)) or not hasattr(forced_holds, "__iter__"):
            raise ValidationError("forced_holds must be an iterable of incident indices")
        self._forced_holds = frozenset(int(index) for index in forced_holds)
        if any(index < 0 for index in self._forced_holds):
            raise ValidationError("forced_holds indices must be >= 0")
        self._monitor = CausalMonitor(config)
        self._detector = DegradationDetector(config)
        # Optional uncertainty boundary: when enabled the controller sees
        # noisy burst estimates instead of exact simulator bursts.
        self._burst_estimator = None
        if config.burst_estimator_enabled:
            from healing.burst_estimator import BurstEstimator as _BurstEstimator

            self._burst_estimator = _BurstEstimator(
                alpha=config.burst_estimator_alpha,
                error_cv=config.burst_estimator_error_cv,
                error_bias=config.burst_estimator_bias,
                seed=42,
            )
        self._episodes: List[EpisodeRecord] = []
        self._episode_index = -1
        self._training = False
        self._reset_episode_state()

    # -- introspection ------------------------------------------------------------------
    @property
    def label(self) -> str:
        """Arm label of this controller."""
        return self._label

    @property
    def encoder(self) -> HealingContextEncoder:
        """The declared context encoder (used to dump the Q table and its coverage)."""
        return self._encoder

    @property
    def policy(self) -> HealingPolicy:
        """The recovery-action policy in use."""
        return self._policy

    @property
    def episodes(self) -> Tuple[EpisodeRecord, ...]:
        """All concluded episodes in order."""
        return tuple(self._episodes)

    @property
    def incidents(self) -> Tuple[IncidentRecord, ...]:
        """All concluded incidents from every episode, in order."""
        return tuple(
            incident for episode in self._episodes for incident in episode.incidents
        )

    def q_table_dump(self) -> Optional[Dict[str, Any]]:
        """Return the learned table and coverage when the policy is learned."""
        agent = getattr(self._policy, "agent", None)
        if agent is None:
            return None
        table = agent.q_table
        visits = agent.state_action_visit_counts
        return {
            "n_states": int(agent.n_states),
            "n_actions": int(agent.n_actions),
            "visited_states": int((agent.visit_counts > 0).sum()),
            "visited_state_action_pairs": int((visits > 0).sum()),
            "table": table,
            "state_action_visits": visits,
            "actions": HEALING_ACTION_NAMES,
        }

    # -- episode lifecycle ---------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        """Reset everything that is per-trace; learning state survives across episodes."""
        self._monitor.reset()
        self._detector.reset()
        if self._burst_estimator is not None:
            self._burst_estimator.reset()
        self._settings = self._initial_settings
        self._last_accepted = self._initial_settings
        self._pending: Optional[_PendingIncident] = None
        self._pending_rollback: Optional[SchedulerSettings] = None
        self._decisions_since_conclusion = self._config.cooldown_decisions
        self._consecutive_failures = 0
        self._adaptation_enabled = True
        self._interventions_used = 0
        self._fallback_triggered = False
        self._episode_decisions = 0
        self._episode_incidents: List[IncidentRecord] = []
        self._reason_counts: Counter[str] = Counter()
        self._running_priority: Optional[int] = None
        self._last_ready_count = 1
        self._last_work_ratio = 0.0

    def begin_episode(self, episode_index: int) -> None:
        """Start a new trace.

        Raises:
            ValidationError: If ``episode_index`` is negative or the previous episode was
                not closed.
        """
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValidationError(f"episode_index must be an integer >= 0, got {episode_index!r}")
        if self._pending is not None:
            raise ValidationError("previous episode must be closed before starting a new one")
        self._episode_index = episode_index
        self._reset_episode_state()

    def select_action(
        self,
        observation: RuntimeObservation,
        *,
        training: bool = False,
        episode: int = 0,
    ) -> RuntimeActionDecision:
        """Return the configuration the engine must use for the next dispatch.

        The decision is a full configuration ``(policy, quantum)``, not only a policy
        label, so the scheduler log alone describes the configuration trajectory and the
        counterfactual replay audit can reconstruct any prefix of a run exactly.
        """
        if not isinstance(observation, RuntimeObservation):
            raise ValidationError("SelfHealingController requires a RuntimeObservation")
        if not isinstance(training, bool):
            raise ValidationError("training must be a bool")
        if episode < 0:
            raise ValidationError("episode must be >= 0")
        if self._episode_index < 0:
            raise ValidationError("begin_episode must be called before selecting actions")
        # Uncertainty boundary: optionally replace exact bursts with estimates.
        effective_observation = observation
        if self._burst_estimator is not None:
            from healing.burst_estimator import perturb_observation as _perturb

            effective_observation = _perturb(observation, self._burst_estimator).observation
            # The simulator's ground-truth remaining work is still used for
            # scheduling correctness; the controller's *view* is noisy.

        self._training = training
        self._decisions_since_conclusion += 1
        self._last_ready_count = len(effective_observation.ready_processes)
        visible_now = sum(p.remaining_burst for p in effective_observation.ready_processes)
        self._last_work_ratio = visible_now / max(
            1.0,
            self._monitor.snapshot().reference_wait
            if self._monitor.completed_count
            else float(visible_now),
        )

        if self._pending_rollback is not None:
            # A failed intervention is undone at the first epoch after the verdict.
            self._settings = self._pending_rollback
            self._pending_rollback = None

        state_index: Optional[int] = None
        epsilon = 0.0
        explored = False
        reason: Optional[str] = None

        if self._pending is not None:
            reason = "verification-window-open"
        elif not self._adaptation_enabled:
            reason = "safe-fallback-active"
        elif not self._monitor.has_pending_evidence:
            # Evaluation costs nothing when no new completion has been observed, and a
            # decision epoch without completed work carries no new performance evidence.
            reason = "no_new_completions"
        else:
            visible_work = sum(p.remaining_burst for p in effective_observation.ready_processes)
            outcome = self._detector.evaluate(
                self._monitor,
                ready_count=len(effective_observation.ready_processes),
                visible_remaining_work=visible_work,
                observed_mean_burst=effective_observation.observed_mean_burst,
                load_level=self._monitor.load_level(),
                load_reference=self._monitor.load_reference,
                decisions_since_conclusion=self._decisions_since_conclusion,
                adaptation_enabled=self._adaptation_enabled,
                interventions_used=self._interventions_used,
            )
            self._monitor.mark_evaluated()
            self._reason_counts[outcome.reason] += 1
            if outcome.absorbable and outcome.snapshot is not None:
                self._monitor.absorb_into_reference(outcome.snapshot)
            if outcome.triggered and outcome.snapshot is not None:
                decision = self._open_incident(effective_observation, outcome, training=training, episode=episode)
                state_index = decision.state_index
                epsilon = decision.epsilon
                explored = decision.explored

        self._episode_decisions += 1
        return RuntimeActionDecision(
            action=self._settings.policy,
            state_index=state_index,
            epsilon=epsilon,
            explored=explored,
            fallback_reason=reason,
            quantum=self._settings.quantum,
        )

    def _open_incident(
        self,
        observation: RuntimeObservation,
        outcome: DetectionOutcome,
        *,
        training: bool,
        episode: int,
    ) -> RuntimeActionDecision:
        """Diagnose a detected event, choose an action, apply it and open verification."""
        snapshot: MonitorSnapshot = outcome.snapshot
        evidence = build_ready_evidence(
            observation,
            running_priority=self._running_priority,
            lower_priority_number_is_higher_priority=(
                self._scheduler_config.lower_priority_number_is_higher_priority
            ),
        )
        diagnosis = diagnose(evidence, snapshot, self._config)
        context = HealingContext(
            diagnosis_index=diagnosis.category_index,
            excess_band_units=snapshot.excess_band_units,
            ready_count=evidence.ready_count,
            current_policy=self._settings.policy,
            relative_band=snapshot.relative_band,
            remaining_work_ratio=(
                evidence.visible_remaining_work / max(snapshot.reference_wait, 1e-9)
            ),
        )
        allowed = available_actions(self._settings, self._last_accepted, self._config)
        forced_hold = len(self._episode_incidents) in self._forced_holds
        if forced_hold:
            # Counterfactual replay: the same controller, the same workload, the same
            # diagnosis -- but this one incident holds the configuration instead of acting.
            choice = PolicyChoice(action=HOLD, note="forced-hold-counterfactual")
        else:
            choice = self._policy.select(
                context, training=training, episode=episode, allowed_actions=allowed
            )
        applied = apply_healing_action(
            choice.action,
            current=self._settings,
            last_accepted=self._last_accepted,
            min_quantum=self._config.min_quantum,
            max_quantum=self._config.max_quantum,
        )
        changed = applied.changed
        settings_before = self._settings
        if changed:
            self._settings = applied.settings
            self._interventions_used += 1

        # Verification baseline: the trace's stable *normal* load and the load observed at
        # the trigger epoch. The verdict asks how much of the gap between the two the
        # recovery window closed; the trace's normal level is what a healthy trace looks
        # like, and the trigger level is what the incident actually looked like.
        pre_load_level = self._monitor.load_reference
        trigger_load_level = self._monitor.load_level()
        self._pending = _PendingIncident(
            incident_index=len(self._episode_incidents),
            decision_index=self._episode_decisions,
            simulated_time=observation.current_time,
            detector_reason=outcome.reason,
            suspicion_evaluations=outcome.suspect_streak,
            snapshot=snapshot,
            threshold=outcome.threshold,
            evidence=evidence,
            diagnosis_statement=diagnosis.statement,
            diagnosis_category=diagnosis.category,
            diagnosis_index=diagnosis.category_index,
            diagnosis_matched=",".join(diagnosis.matched),
            context=context,
            choice=choice,
            offered_actions=",".join(
                HEALING_ACTION_NAMES[action] for action in allowed
            ),
            changed=changed,
            forced_hold=forced_hold,
            action=choice.action,
            action_note=applied.note,
            settings_before=settings_before,
            settings_after=self._settings,
            pre_load_level=pre_load_level,
            trigger_load_level=trigger_load_level,
            load_band_at_trigger=self._monitor.load_band(),
        )
        return RuntimeActionDecision(
            action=self._settings.policy,
            state_index=self._encoder.encode(context),
            epsilon=choice.epsilon,
            explored=choice.explored,
            fallback_reason=None,
            quantum=self._settings.quantum,
        )

    def observe_decision(self, record: RuntimeDecisionRecord) -> None:
        """Receive one realized decision interval from the engine (always called)."""
        if not isinstance(record, RuntimeDecisionRecord):
            raise ValidationError("observe_decision requires a RuntimeDecisionRecord")
        if record.completed_waiting_times:
            self._monitor.observe_completions(record.completed_waiting_times)
        duration = record.end_time - record.observation.current_time
        self._monitor.observe_interval(
            record.waiting_cost_increment,
            duration,
            allow_reference_update=(
                self._pending is None and self._detector.suspect_streak == 0
            ),
        )
        for process in record.observation.ready_processes:
            if process.pid == record.chosen_pid:
                self._running_priority = process.priority
                break
        # Feed estimator with ground-truth completed bursts when uncertainty is enabled.
        if self._burst_estimator is not None and record.completed_pids:
            # Look up true burst lengths from the process table via the record's completed pids.
            # The observation that produced this decision still carries the true burst in
            # the engine's record; we map pid->burst from the ready_processes that were present
            # before dispatch. Simpler: use the waiting_time arithmetic to infer nothing;
            # instead infer burst from the slice duration when possible. For now, feed
            # completed waiting_time-adjacent bursts via the observation's burst fields:
            # the engine's observation before dispatch contains true bursts (even when
            # controller view was noisy), so we can read them from record.observation.
            # To avoid extra state, we estimate burst as max(1, waiting_time proxy) — but
            # since we have the true execution slice end-start, that equals service time
            # when no preemption. For non-preemptive policies this is the burst chunk.
            # As a safe fallback we observe the service duration as a burst sample.
            service = record.end_time - record.start_time - record.switch_overhead
            for pid in record.completed_pids:
                # Use service as approximate burst of the completed process;
                # for RR-preempted processes completion service == remaining burst,
                # which still reflects true burst size proportionally.
                if service > 0:
                    self._burst_estimator.observe_completed(int(service), bucket="global")

        pending = self._pending
        if pending is None:
            return
        pending.decisions += 1
        pending.switch_overhead += record.switch_overhead
        pending.window_end_time = record.end_time
        pending.window_accrued_wait += record.waiting_cost_increment
        pending.window_duration += duration
        if record.completed_waiting_times:
            pending.waiting.extend(record.completed_waiting_times)
            pending.completions += len(record.completed_waiting_times)
        config = self._config
        ready_to_close = (
            pending.decisions >= config.verify_decisions
            and pending.completions >= config.verify_completions
        )
        if ready_to_close or pending.decisions >= config.verify_max_decisions:
            self._conclude_pending(pending)

    def _band_scaled(self, absolute: float, snapshot: MonitorSnapshot) -> float:
        """Convert an absolute waiting-time quantity into dispersion band units."""
        return absolute / max(snapshot.band, 1e-9)

    def _emit_incident(
        self,
        pending: _PendingIncident,
        *,
        verdict: str,
        verifier_reason: str,
        confirmed_failure: bool,
        consecutive_failures: int,
        outcome: Any,
        improvement: Optional[float],
        post_mean: Optional[float],
        post_load: Optional[float],
        load_improvement: Optional[float],
        load_improvement_band: Optional[float],
        closure: Optional[float],
        reward: float,
        improvement_band_units: Optional[float],
        cost_band_units: float,
        rollback_target: Optional[SchedulerSettings],
        fallback: bool,
    ) -> IncidentRecord:
        """Build the immutable ledger record of one concluded incident."""
        snapshot = pending.snapshot
        record = IncidentRecord(
            episode_index=self._episode_index,
            incident_index=pending.incident_index,
            decision_index=pending.decision_index,
            simulated_time=pending.simulated_time,
            detector_reason=pending.detector_reason,
            suspicion_evaluations=pending.suspicion_evaluations,
            trigger_threshold=pending.threshold,
            reference_at_trigger=snapshot.reference_wait,
            band_at_trigger=snapshot.band,
            window_mean_at_trigger=snapshot.window_mean_wait,
            window_size_at_trigger=snapshot.window_size,
            window_median_at_trigger=snapshot.window_median_wait,
            window_max_at_trigger=snapshot.window_max_wait,
            excess_band_units=snapshot.excess_band_units,
            ready_count_at_trigger=pending.evidence.ready_count,
            visible_remaining_work_at_trigger=pending.evidence.visible_remaining_work,
            diagnosis=pending.diagnosis_category,
            diagnosis_index=pending.diagnosis_index,
            diagnosis_matched=pending.diagnosis_matched,
            diagnosis_statement=pending.diagnosis_statement,
            context_state_index=self._encoder.encode(pending.context),
            offered_actions=pending.offered_actions,
            action=pending.action,
            action_name=HEALING_ACTION_NAMES[pending.action],
            action_changed=pending.changed,
            action_note=pending.action_note,
            request_note=(pending.choice.note or ""),
            settings_before=pending.settings_before,
            settings_after=pending.settings_after,
            epsilon=pending.choice.epsilon,
            explored=pending.choice.explored,
            verdict=verdict,
            verifier_reason=verifier_reason,
            confirmed_failure=confirmed_failure,
            consecutive_failures=consecutive_failures,
            window_truncated=pending.truncated,
            forced_hold=pending.forced_hold,
            verification_decisions=pending.decisions,
            verification_completions=pending.completions,
            post_window_mean=post_mean,
            improvement=improvement,
            improvement_band_units=improvement_band_units,
            pre_load_level=pending.pre_load_level,
            trigger_load_level=pending.trigger_load_level,
            post_load_level=post_load,
            load_band_at_trigger=pending.load_band_at_trigger,
            load_improvement=load_improvement,
            load_improvement_band_units=load_improvement_band,
            closure_fraction=closure,
            closure_residual=outcome.residual,
            expected_closure=outcome.expected,
            required_closure_factor=outcome.required_closure,
            reject_closure_factor=outcome.reject_closure,
            accept_residual_factor=outcome.required_residual,
            reject_residual_factor=outcome.reject_residual,
            cost_band_units=cost_band_units,
            reward=reward,
            rollback_target=rollback_target,
            switch_overhead_units=pending.switch_overhead,
            window_end_time=pending.window_end_time,
            fallback_triggered=fallback,
        )
        self._episode_incidents.append(record)
        return record

    def _phase(self, pending: _PendingIncident) -> TriggerPhase:
        """Return the trigger phase the phase-aware verification baseline reads.

        Every quantity comes from the trigger snapshot or from the incident's own record of
        the trigger epoch, so the baseline stays a causal comparison with the situation the
        controller actually faced.
        """
        snapshot = pending.snapshot
        return TriggerPhase(
            excess_band_units=snapshot.excess_band_units,
            relative_band=snapshot.relative_band,
            load_band=pending.load_band_at_trigger,
            trigger_load=pending.trigger_load_level,
            reference_load=pending.pre_load_level,
            # Progress observables: the controller's own decision counter and the simulated
            # clock at the trigger epoch. Both are recorded in the incident ledger and in the
            # derivation artifact as evidence of when the trigger happened, and the declared
            # reference deliberately does not read them: a clock feature would make the
            # expectation depend on how far the trace has already run, and the candidate
            # comparison measured no decision-relevant gain from adding it.
            ready_count=int(pending.evidence.ready_count),
            visible_work=float(pending.evidence.visible_remaining_work),
            decision_index=pending.decision_index,
            elapsed_time=pending.simulated_time,
        )

    def _load_closure(self, pending: _PendingIncident) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(post_load, closure_fraction, gap_denominator)`` for one window.

        The closure statistic asks how much of the gap between the observed load at the
        trigger epoch and the trace's own normal load level was closed during the recovery
        window. Both ends of the comparison are observed before the verdict, so the
        statistic stays causal, and the denominator is floored by the trace's dispersion
        band so that a tiny gap cannot produce an enormous ratio.
        """
        if pending.window_duration <= 0:
            return None, None, None
        post_load = pending.window_accrued_wait / pending.window_duration
        gap = max(
            pending.trigger_load_level - pending.pre_load_level,
            max(pending.load_band_at_trigger, 1e-9),
        )
        if gap <= 0.0:
            return post_load, None, gap
        return post_load, (pending.trigger_load_level - post_load) / gap, gap

    def _conclude_pending(self, pending: _PendingIncident) -> None:
        """Judge one recovery window: accept, roll back, or register a hold.

        The verdict is produced by :func:`healing.verifier.verify` from the *load-closure*
        statistic: the fraction of the gap between the load observed at the trigger and the
        trace's normal load that the window closed. The previous rule (absolute improvement
        relative to the trace's normal level) had a measured no-action expectation of about
        two dispersion-band widths on the hold-only control, so it accepted 95 % of the
        interventions that changed nothing; the closure statistic is centred on the no-action
        behaviour of the same measurement, and both of its boundaries are its measured
        percentiles on that control. The learning reward is that same statistic minus the
        declared intervention cost -- and minus the rollback penalty only when the window
        *confirmed* a failure, never when the evidence was merely insufficient.
        """
        config = self._config
        self._pending = None
        self._decisions_since_conclusion = 0
        snapshot = pending.snapshot
        reference = max(snapshot.reference_wait, 1e-9)
        cost_band = self._band_scaled(config.intervention_cost * reference, snapshot)

        # Completion-based evidence (reported, not used for the verdict: see the module
        # docstring and docs/EVALUATION_RESULTS.md for the measured reason).
        post_mean: Optional[float] = None
        improvement: Optional[float] = None
        improvement_band: Optional[float] = None
        if pending.completions:
            post_mean = sum(pending.waiting) / len(pending.waiting)
            improvement = (snapshot.reference_wait - post_mean) / reference
            improvement_band = self._band_scaled(snapshot.reference_wait - post_mean, snapshot)

        # Load-based verdict evidence: reported relative improvement plus the statistic the
        # verdict actually uses.
        post_load, closure, _gap = self._load_closure(pending)
        load_improvement: Optional[float] = None
        load_improvement_band: Optional[float] = None
        pre_load = max(pending.pre_load_level, 1e-9)
        load_band = max(pending.load_band_at_trigger, 1e-9)
        if post_load is not None:
            load_improvement = (pending.pre_load_level - post_load) / pre_load
            load_improvement_band = (pending.pre_load_level - post_load) / load_band

        evidence = VerificationEvidence(
            changed=pending.changed,
            closure_fraction=closure,
            phase=self._phase(pending),
            window_decisions=pending.decisions,
            window_completions=pending.completions,
            window_duration=pending.window_duration,
        )
        outcome = verify(evidence, config)
        verdict = outcome.verdict
        # The learning reward is the verifier's own phase-aware score minus its declared
        # accept boundary, using the shared reward helper so the probe and the controller
        # cannot drift apart.
        rollback_target: Optional[SchedulerSettings] = None
        charged_cost = cost_band if pending.changed else 0.0
        rollback_penalty_band = self._band_scaled(
            config.rollback_penalty * reference, snapshot
        )
        reward = controller_reward(
            residual=outcome.residual,
            config=config,
            cost_band_units=charged_cost,
            rollback_penalty_band=rollback_penalty_band,
            changed=pending.changed,
            confirmed_failure=outcome.confirmed_failure,
            held=(outcome.verdict == VERDICT_HELD),
        )
        if outcome.verdict == VERDICT_HELD:
            pass
        elif outcome.accepted:
            self._last_accepted = pending.settings_after
            self._consecutive_failures = 0
        elif outcome.confirmed_failure:
            rollback_target = self._last_accepted
            self._consecutive_failures += 1
        else:
            # Inconclusive: the change is still undone (the loop never keeps a configuration
            # it could not verify) but no failure is charged, so ambiguity cannot drive the
            # safe fallback and cannot teach the controller that the action was harmful.
            if outcome.rollback_required:
                rollback_target = self._last_accepted

        fallback = False
        if rollback_target is not None:
            self._pending_rollback = rollback_target
            if self._consecutive_failures >= config.max_consecutive_failures:
                self._adaptation_enabled = False
                self._fallback_triggered = True
                fallback = True

        if not pending.forced_hold:
            next_context = self._context_for_learning(pending)
            self._policy.observe_outcome(
                pending.context,
                pending.choice,
                reward,
                next_context,
                terminal=False,
                training=self._training,
            )
        self._emit_incident(
            pending,
            verdict=verdict,
            verifier_reason=outcome.reason,
            confirmed_failure=outcome.confirmed_failure,
            consecutive_failures=self._consecutive_failures,
            outcome=outcome,
            reward=reward,
            improvement=improvement,
            post_mean=post_mean,
            closure=closure,
            improvement_band_units=improvement_band,
            post_load=post_load,
            load_improvement=load_improvement,
            load_improvement_band=load_improvement_band,
            cost_band_units=charged_cost,
            rollback_target=rollback_target,
            fallback=fallback,
        )

    def _context_for_learning(self, pending: _PendingIncident) -> HealingContext:
        """Return the successor context used by the learned controller's update.

        The situation described by the diagnosis is treated as persisting until the next
        detected event; the other components are the ones observable right now, after the
        recovery window, with the configuration that is actually in force.
        """
        try:
            snapshot = self._monitor.snapshot()
            excess = snapshot.excess_band_units
            relative_band = snapshot.relative_band
        except ValidationError:
            excess = pending.snapshot.excess_band_units
            relative_band = pending.snapshot.relative_band
        return HealingContext(
            diagnosis_index=pending.diagnosis_index,
            excess_band_units=excess,
            ready_count=self._last_ready_count,
            current_policy=self._settings.policy,
            relative_band=relative_band,
            remaining_work_ratio=self._last_work_ratio,
        )

    def end_episode(self) -> EpisodeRecord:
        """Close the current trace and return its healing summary.

        A verification window that is still open when the trace ends is recorded as an
        ``inconclusive`` interruption with ``window_truncated`` set: the ledger says
        explicitly that the evidence is incomplete instead of silently counting it as a
        success or a failure. Its partial evidence still forms the terminal learning
        update, and it carries neither the rollback penalty nor a failure count.
        """
        if self._episode_index < 0:
            raise ValidationError("no episode is open")
        if self._pending_rollback is not None:
            # A verdict reached on the last decision of the trace cannot wait for a next
            # epoch: the rollback is applied here so ``settings_final`` is the configuration
            # the episode actually leaves in force rather than the one that was rejected.
            self._settings = self._pending_rollback
            self._pending_rollback = None
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            snapshot = pending.snapshot
            reference = max(snapshot.reference_wait, 1e-9)
            cost_band = self._band_scaled(self._config.intervention_cost * reference, snapshot)
            post_mean: Optional[float] = None
            improvement: Optional[float] = None
            improvement_band: Optional[float] = None
            if pending.completions:
                post_mean = sum(pending.waiting) / len(pending.waiting)
                improvement = (snapshot.reference_wait - post_mean) / reference
                improvement_band = self._band_scaled(
                    snapshot.reference_wait - post_mean, snapshot
                )
            post_load, closure, _gap = self._load_closure(pending)
            load_improvement: Optional[float] = None
            load_improvement_band: Optional[float] = None
            if post_load is not None:
                pre_load = max(pending.pre_load_level, 1e-9)
                load_improvement = (pending.pre_load_level - post_load) / pre_load
                load_improvement_band = (pending.pre_load_level - post_load) / max(
                    pending.load_band_at_trigger, 1e-9
                )
            charged_cost = cost_band if pending.changed else 0.0
            pending.truncated = True
            evidence = VerificationEvidence(
                changed=pending.changed,
                closure_fraction=closure,
                phase=self._phase(pending),
                window_decisions=pending.decisions,
                window_completions=pending.completions,
                window_duration=pending.window_duration,
                truncated=bool(pending.changed),
            )
            outcome = verify(evidence, self._config)
            reward = controller_reward(
                residual=outcome.residual,
                config=self._config,
                cost_band_units=charged_cost,
                changed=pending.changed,
                confirmed_failure=False,
                held=(outcome.verdict == VERDICT_HELD),
            )
            if not pending.forced_hold:
                next_context = self._context_for_learning(pending)
                self._policy.observe_outcome(
                    pending.context, pending.choice, reward, next_context,
                    terminal=True, training=self._training,
                )
            # The interrupted window carries exactly the consequence its verdict implies:
            # an inconclusive change follows the declared posture (rollback) even when the
            # trace ends right here, so the ledger never claims a change was kept that the
            # declared rule would have reverted. Nothing runs after this epoch, so this is a
            # bookkeeping statement about ``settings_final``.
            truncated_rollback = self._last_accepted if outcome.rollback_required else None
            if truncated_rollback is not None:
                self._settings = truncated_rollback
            self._emit_incident(
                pending,
                verdict=outcome.verdict,
                verifier_reason=outcome.reason,
                confirmed_failure=outcome.confirmed_failure,
                consecutive_failures=self._consecutive_failures,
                outcome=outcome,
                improvement=improvement,
                post_mean=post_mean,
                post_load=post_load,
                load_improvement=load_improvement,
                load_improvement_band=load_improvement_band,
                closure=closure,
                reward=reward,
                improvement_band_units=improvement_band,
                cost_band_units=charged_cost,
                rollback_target=truncated_rollback,
                fallback=False,
            )

        incidents = tuple(self._episode_incidents)
        record = EpisodeRecord(
            episode_index=self._episode_index,
            decisions=self._episode_decisions,
            incidents=incidents,
            interventions=sum(1 for item in incidents if item.action_changed),
            holds=sum(1 for item in incidents if not item.action_changed),
            accepted=sum(1 for item in incidents if item.verdict == VERDICT_ACCEPTED),
            rejected=sum(1 for item in incidents if item.verdict == VERDICT_REJECTED),
            inconclusive=sum(
                1 for item in incidents if item.verdict == VERDICT_INCONCLUSIVE
            ),
            unverified=sum(1 for item in incidents if item.is_unverified_intervention),
            confirmed_failures=sum(1 for item in incidents if item.confirmed_failure),
            fallback_triggered=self._fallback_triggered,
            interventions_used=self._interventions_used,
            settings_initial=self._initial_settings,
            settings_final=self._settings,
            detector_reasons=tuple(sorted(self._reason_counts.items())),
        )
        self._episodes.append(record)
        return record

    # -- runtime controller protocol -----------------------------------------------------
    def observe_transition(
        self,
        decision: RuntimeActionDecision,
        reward: float,
        next_observation: Optional[RuntimeObservation],
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """No-op: the healing loop learns from its own verified windows, not from the
        engine's per-interval waiting-cost reward."""
        del decision, reward, next_observation, terminal, training

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of the arm configuration."""
        return {
            "label": self._label,
            "policy_kind": self._policy.name,
            "initial_policy": self._initial_settings.policy_name,
            "initial_quantum": self._initial_settings.quantum,
            "n_context_states": self._encoder.n_states,
            "n_actions": len(HEALING_ACTION_NAMES),
            "actions": list(HEALING_ACTION_NAMES),
            "diagnosis_categories": list(DIAGNOSIS_CATEGORIES),
            "healing_config": self._config.as_dict(),
        }


def replay_hold_counterfactual(
    simulator: Any,
    workload: Any,
    records: Sequence[RuntimeDecisionRecord],
    incident: IncidentRecord,
    *,
    policy_name: str = "counterfactual-hold",
) -> Any:
    """Re-run one trace holding at a recorded incident, for the recovery audit.

    The pre-incident prefix of the recorded configuration trajectory is replayed exactly,
    so the engine reaches byte-identical state at the incident epoch; from that epoch
    onward the configuration that was in force *before* the intervention is kept and no
    further recovery happens. The difference between this counterfactual's total waiting
    time and the realized run's is the marginal value of adapting from that point on.

    Args:
        simulator: The event engine (:class:`scheduler.runtime.RuntimeSimulator`).
        workload: The same workload object the realized run used.
        records: The realized run's decision records.
        incident: The incident to hold at.
        policy_name: Label stored in the counterfactual schedule result.

    Returns:
        The :class:`scheduler.runtime.RuntimeRun` of the counterfactual.
    """
    return _replay_from_incident(
        simulator, workload, records, incident, policy_name=policy_name
    )


def replay_action_at_incident(
    simulator: Any,
    workload: Any,
    records: Sequence[RuntimeDecisionRecord],
    incident: IncidentRecord,
    action: int,
    *,
    config: Optional[HealingConfig] = None,
    policy_name: str = "counterfactual-action",
) -> Any:
    """Re-run one trace applying a single candidate action at a recorded incident.

    This is the ground-truth probe of the study's action-effect analysis: the recorded
    prefix is replayed exactly (so the state at the incident is identical), the candidate
    action is applied at that epoch, and the configuration is then held for the rest of
    the trace. Comparing the resulting total waiting time with a hold-only replay measures
    what the action itself was worth.

    Args:
        simulator: The event engine.
        workload: The same workload object the realized run used.
        records: The realized run's decision records.
        incident: The incident the action is applied at.
        action: Healing action index to apply.
        config: Thresholds used to resolve the action's effect (defaults to the declared
            configuration).
        policy_name: Label stored in the counterfactual schedule result.

    Returns:
        The :class:`scheduler.runtime.RuntimeRun` of the counterfactual.
    """
    config = config or HealingConfig()
    applied = apply_healing_action(
        action,
        current=incident.settings_before,
        last_accepted=incident.settings_before,
        min_quantum=config.min_quantum,
        max_quantum=config.max_quantum,
    )
    return _replay_from_incident(
        simulator,
        workload,
        records,
        incident,
        override=applied.settings,
        policy_name=policy_name,
    )


def _replay_from_incident(
    simulator: Any,
    workload: Any,
    records: Sequence[RuntimeDecisionRecord],
    incident: IncidentRecord,
    *,
    override: Optional[SchedulerSettings] = None,
    policy_name: str,
) -> Any:
    """Replay the recorded configuration prefix, then hold (optionally after one change)."""
    if incident.decision_index < 0 or incident.decision_index > len(records):
        raise ValidationError("incident decision index is outside the recorded run")
    prefix = records[: incident.decision_index]
    hold_settings = incident.settings_before
    controller = _HoldReplayController(prefix, hold_settings, override=override)
    return simulator.run(workload, controller, training=False, policy_name=policy_name)


class _HoldReplayController:
    """Replays a recorded configuration prefix, then holds one configuration forever.

    ``override`` is applied once, at the first epoch after the replay prefix, which is how
    the action-effect probe applies a single candidate action to a recorded incident.
    """

    def __init__(
        self,
        prefix: Sequence[RuntimeDecisionRecord],
        hold_settings: SchedulerSettings,
        *,
        override: Optional[SchedulerSettings] = None,
    ) -> None:
        self._prefix = tuple(
            RuntimeActionDecision(
                action=record.decision.action,
                quantum=record.decision.quantum,
                fallback_reason="counterfactual-replay",
            )
            for record in prefix
        )
        self._hold = hold_settings
        self._override = override
        self._index = 0
        self._applied_override = False

    def select_action(
        self,
        observation: RuntimeObservation,
        *,
        training: bool = False,
        episode: int = 0,
    ) -> RuntimeActionDecision:
        """Replay the prefix, apply the override once (if any), then hold."""
        del training, episode
        if self._index < len(self._prefix):
            decision = self._prefix[self._index]
        elif self._override is not None and not self._applied_override:
            self._applied_override = True
            decision = RuntimeActionDecision(
                action=self._override.policy,
                quantum=self._override.quantum,
                fallback_reason="counterfactual-action",
            )
        else:
            decision = RuntimeActionDecision(
                action=self._hold.policy,
                quantum=self._hold.quantum,
                fallback_reason="counterfactual-hold",
            )
        self._index += 1
        return decision

    def observe_transition(
        self,
        decision: RuntimeActionDecision,
        reward: float,
        next_observation: Optional[RuntimeObservation],
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """No-op: the replay neither learns nor reacts."""
        del decision, reward, next_observation, terminal, training
