"""Online verification of a recovery intervention: accepted, rejected or inconclusive.

The verifier answers one question at the end of a recovery window: *does the evidence
recorded so far support keeping the new configuration?*  It is deliberately a separate
module, because the earlier design (a bare ``closure >= factor`` test inside the controller)
collapsed two different situations into the same verdict:

* an intervention whose observation window is **too thin, too short or too ambiguous to say
  anything**, and
* an intervention whose window shows that the configuration change **did not work**.

Both were recorded as ``rolled_back``/``censored`` and both were counted as failures by the
safe-fallback escalation, which is why the study reported a 90-98 % "false recovery" rate
that the counterfactual audit contradicted (see ``docs/EVALUATION_RESULTS.md``,
"Verification" section, and :mod:`experiments.healing_thresholds`).

What the rule compares
----------------------
The statistic is the *normalised load closure* of the window, i.e. the fraction of the gap
between the queue length observed at the trigger and the trace's own normal queue length
that the window closed::

    closure = (trigger_load - post_load) / max(trigger_load - reference_load, load_band)

All three loads are observed before the verdict, so the statistic is causal.  Its
*unconditional* distribution is however dominated by the trigger's own situation, which is
measured: a **structural** baseline of trigger-observable quantities explains most of the
no-action variance of the statistic on the training control.  Predicting the no-action
post-window load from the trigger's own observables and dividing by the statistic's own
denominator leaves an out-of-sample residual spread of 0.160, against 0.294 for the same
prediction restricted to the two queue levels, 0.288 for the earlier free fit of the closure
itself and 0.806 for the raw statistic; the declared row is the physically observed
``trigger_load``, ready-set size and visible work (``healing_verification_models.csv``).  The
rule therefore compares the
**residual** after that phase-aware baseline, not the raw statistic, and its two boundaries
are percentile points of the control residual distribution over the windows the rule can
actually judge, measured by :mod:`experiments.healing_thresholds`.  Both constants are
declared in :class:`healing.config.HealingConfig`; nothing here is fitted at run time, and
no quantity used by the rule refers to unarrived work, remaining work or the finished
schedule.

Three intervention outcomes are kept distinct:

``accepted``
    The residual reached the declared accept boundary (the control's 95th percentile,
    rounded outward), so the change is kept and becomes the new known-good configuration.
``rejected``
    The residual fell to or below the declared reject boundary (the control's 5th
    percentile, rounded outward): the window behaved worse than the trace's own phase
    predicts.  Only this outcome is a *confirmed failure*: it is what the safe-fallback
    escalation counts and what carries the rollback penalty in the learning reward.
``inconclusive``
    Anything in between, or a window that never reached the declared evidence floor (at
    least ``verify_completions`` completions, a positive duration, and a window that the
    end of the trace did not truncate).  An inconclusive intervention is still rolled back
    -- the loop never keeps a configuration it could not verify -- but it is *not* recorded
    as a failure, does not advance the fallback escalation and carries no rollback penalty,
    because the evidence does not establish harm.

Why this is a filter and not an estimator
-----------------------------------------
Development measurement on the validation split (matched counterfactual: the same
controller, one incident forced to hold) shows that an intervention moves the closure
statistic by about ``+0.15`` while the residual spread is ``0.16``: the rule can certify a
minority of genuinely useful interventions and is nearly silent about harmful ones, and it
is reported as a safety filter with a measured rate in
``docs/EVALUATION_RESULTS.md``, never as evidence that the recommended action was the best
one.  What it cannot do — and what no single five-decision window can do — is turn an
average improvement into a certificate for every intervention; that limit is quantified in
the same document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from errors import ValidationError
from healing.config import HealingConfig

__all__ = [
    "VERDICT_ACCEPTED",
    "VERDICT_HELD",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_REJECTED",
    "INTERVENTION_VERDICTS",
    "ALL_VERDICTS",
    "TriggerPhase",
    "VerificationEvidence",
    "VerificationVerdict",
    "expected_closure",
    "closure_residual",
    "verify",
]

#: The intervention was verified and becomes the new known-good configuration.
VERDICT_ACCEPTED = "accepted"
#: The window shows that the change did not work; confirmed failure.
VERDICT_REJECTED = "rejected"
#: The window does not support a confident verdict either way.
VERDICT_INCONCLUSIVE = "inconclusive"
#: No configuration change was made by this incident (control or hold decision).
VERDICT_HELD = "held"

#: The three outcomes of a real intervention, in the order they are reported.
INTERVENTION_VERDICTS = (VERDICT_ACCEPTED, VERDICT_REJECTED, VERDICT_INCONCLUSIVE)
#: Every verdict a ledger row can carry.
ALL_VERDICTS = INTERVENTION_VERDICTS + (VERDICT_HELD,)

#: Reasons attached to a verdict, so a ledger row explains itself.
REASON_ACCEPTED = "verified_phase_residual"
REASON_REJECTED = "residual_below_control_lower_tail"
REASON_HELD = "no_change_to_verify"
REASON_THIN_TELEMETRY = "insufficient_evidence_thin_telemetry"
REASON_TRUNCATED = "insufficient_evidence_window_truncated"
REASON_AMBIGUOUS = "inconclusive_residual_within_null_range"


@dataclass(frozen=True)
class TriggerPhase:
    """The quantities the phase-aware baseline reads, all observable at the trigger.

    The baseline predicts the queue level the window would have shown with **no
    intervention**, so its features are the ones that determine how fast this trace drains
    from where it stands: the queue level at the trigger, the trace's own normal level, how
    many processes are ready, and how much CPU work they still hold.

    Attributes:
        excess_band_units: Detection excess ``(window_mean - reference) / band`` of the
            completion window that fired the incident.
        relative_band: Waiting-time dispersion band divided by the reference waiting time.
        load_band: Dispersion band of the queue-length statistic at the trigger.
        trigger_load: Observed queue length (waiting accrual rate) at the trigger.
        reference_load: The trace's own normal queue length at the trigger.
        ready_count: Number of ready processes observed at the trigger epoch.
        visible_work: Remaining CPU work of the ready set, in simulated time units.
        decision_index: The controller's own decision counter at the trigger epoch.
        elapsed_time: Simulated time elapsed since the start of the trace at the trigger.
    """

    excess_band_units: float
    relative_band: float
    load_band: float
    trigger_load: float
    reference_load: float
    ready_count: int = 0
    visible_work: float = 0.0
    decision_index: int = 0
    elapsed_time: int = 0

    def features(self) -> Tuple[float, ...]:
        """Return the declared baseline regressors, in coefficient order.

        This is the design row of :data:`healing.config.HealingConfig.post_load_baseline`:
        the queue level at the trigger, how many processes are ready, and how much CPU work
        the ready set still holds (log scale). The trace's own normal level enters through
        the denominator of the closure, and adding it as a regressor was measured to add
        nothing (it is collinear with the trigger level and its coefficient changes sign
        between control halves); ``healing_verification_models.csv`` reports that comparison.
        Every entry is observed at the trigger epoch and none of them is a clock or a
        decision counter, so the expectation neither drifts with the length of a trace nor
        depends on how long the controller has been running. The decision counter and the
        elapsed time are still recorded in the ledger (see ``as_dict``) as phase evidence.
        """
        return (
            float(self.trigger_load),
            float(self.ready_count),
            math.log1p(max(float(self.visible_work), 0.0)),
        )

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger."""
        return {
            "phase_excess_band_units": self.excess_band_units,
            "phase_relative_band": self.relative_band,
            "phase_load_band": self.load_band,
            "phase_trigger_load": self.trigger_load,
            "phase_reference_load": self.reference_load,
            "phase_ready_count": self.ready_count,
            "phase_visible_work": self.visible_work,
            "phase_decision_index": self.decision_index,
            "phase_elapsed_time": self.elapsed_time,
        }

    def as_row(self) -> dict:
        """Flat dictionary with the regression row itself, for the derivation artifacts."""
        payload = self.as_dict()
        payload.update({f"feature_{index}": value for index, value in enumerate(self.features())})
        return payload


def expected_closure(phase: TriggerPhase, config: HealingConfig) -> float:
    """Return the closure a window in this phase would have shown with **no intervention**.

    The expectation is structural rather than a free fit of the closure statistic itself: the
    declared baseline predicts the *queue level* the window would have drained to with
    nothing changed, and that prediction is turned into a closure through the same formula
    the controller uses for the observed value::

        expected = (trigger_load - expected_post_load) / max(trigger_load - reference_load,
                                                             load_band)

    Keeping the statistics' own denominator (including its dispersion floor) inside the
    expectation is what makes the residual comparable across triggers whose load gap is
    smaller than their noise band; a free linear fit of the closure cannot follow that
    non-linearity and was measured to leave twice the residual spread.

    Raises:
        ValidationError: If ``phase`` or ``config`` has the wrong type.
    """
    if not isinstance(phase, TriggerPhase):
        raise ValidationError("expected_closure() requires a TriggerPhase")
    if not isinstance(config, HealingConfig):
        raise ValidationError("expected_closure() requires a HealingConfig")
    coefficients = config.post_load_baseline
    if len(coefficients) != len(phase.features()) + 1:
        raise ValidationError("post_load_baseline must hold the intercept and every feature")
    expected_post_load = coefficients[0] + sum(
        coefficient * feature for coefficient, feature in zip(coefficients[1:], phase.features())
    )
    denominator = max(
        float(phase.trigger_load) - float(phase.reference_load),
        max(float(phase.load_band), 1e-9),
    )
    return float((float(phase.trigger_load) - expected_post_load) / denominator)


def closure_residual(
    closure_fraction: Optional[float], phase: TriggerPhase, config: HealingConfig
) -> Optional[float]:
    """Return the phase-aware residual of a closure statistic (``None`` without evidence)."""
    if closure_fraction is None:
        return None
    return float(closure_fraction) - expected_closure(phase, config)


@dataclass(frozen=True)
class VerificationEvidence:
    """Everything the verifier is allowed to use, all of it observed before the verdict."""

    #: Whether the incident actually changed the live configuration.
    changed: bool
    #: Normalised load closure of the window, or ``None`` when no interval was observed.
    closure_fraction: Optional[float]
    #: Phase of the trigger the window is compared against.
    phase: TriggerPhase
    #: Decisions, completions and simulated time observed inside the window.
    window_decisions: int
    window_completions: int
    window_duration: int
    #: Whether the trace ended before the window could close normally.
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.window_decisions < 0 or self.window_completions < 0 or self.window_duration < 0:
            raise ValidationError("verification window counters must be non-negative")
        if self.window_completions > self.window_decisions + 1:
            raise ValidationError("a window cannot complete more processes than it dispatches")
        if not isinstance(self.phase, TriggerPhase):
            raise ValidationError("verification evidence requires a TriggerPhase")
        if self.closure_fraction is not None and not isinstance(self.closure_fraction, float):
            raise ValidationError("closure_fraction must be a float or None")

    @property
    def observed(self) -> float:
        """The closure statistic, with the declared neutral value when nothing was seen."""
        return 0.0 if self.closure_fraction is None else self.closure_fraction

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger."""
        payload = {
            "changed": self.changed,
            "closure_fraction": self.closure_fraction,
            "window_decisions": self.window_decisions,
            "window_completions": self.window_completions,
            "window_duration": self.window_duration,
            "truncated": self.truncated,
        }
        payload.update(self.phase.as_dict())
        return payload


@dataclass(frozen=True)
class VerificationVerdict:
    """Outcome of one verification window, with the numbers behind it."""

    verdict: str
    reason: str
    #: Phase-aware residual the verdict was taken on (``None`` without a closure).
    residual: Optional[float]
    #: Expected closure of this phase under the no-action control.
    expected: float
    #: Declared boundaries the residual was compared against.
    required_residual: float
    reject_residual: float
    #: The unconditional comparison, reported for the study's before/after analysis.
    required_closure: float
    reject_closure: float
    #: Whether the escalation counter may count this incident as a failure.
    confirmed_failure: bool
    #: Whether the live configuration must be rolled back to the last accepted one.
    rollback_required: bool
    #: Number of completions the window needed before it could be judged at all.
    required_completions: int

    @property
    def accepted(self) -> bool:
        """Whether the intervention was verified."""
        return self.verdict == VERDICT_ACCEPTED

    @property
    def inconclusive(self) -> bool:
        """Whether the evidence was insufficient or ambiguous."""
        return self.verdict == VERDICT_INCONCLUSIVE

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "residual": self.residual,
            "expected_closure": self.expected,
            "required_residual": self.required_residual,
            "reject_residual": self.reject_residual,
            "required_closure": self.required_closure,
            "reject_closure": self.reject_closure,
            "confirmed_failure": self.confirmed_failure,
            "rollback_required": self.rollback_required,
        }

    @property
    def rollback_required_for(self) -> str:
        """Short label of the consequence the verdict carries into the controller."""
        if not self.rollback_required:
            return "keep"
        return "rollback"


def verify(evidence: VerificationEvidence, config: HealingConfig) -> VerificationVerdict:
    """Decide what a closed (or truncated) recovery window proves.

    Args:
        evidence: The causal evidence recorded during the window.
        config: The declared thresholds (phase baseline, residual boundaries, evidence floor).

    Returns:
        A :class:`VerificationVerdict`.  ``rollback_required`` is true for both a rejected
        and an inconclusive *change*: the loop never keeps a configuration it could not
        verify.  Only :attr:`VerificationVerdict.confirmed_failure` distinguishes them.

    Raises:
        ValidationError: If ``evidence`` is not a :class:`VerificationEvidence`.
    """
    if not isinstance(evidence, VerificationEvidence):
        raise ValidationError("verify() requires a VerificationEvidence")
    expected = expected_closure(evidence.phase, config)
    residual = closure_residual(evidence.closure_fraction, evidence.phase, config)
    required = config.accept_residual_factor
    reject_below = config.reject_residual_factor
    floor = config.verify_completions

    def outcome(verdict: str, reason: str, confirmed_failure: bool) -> VerificationVerdict:
        return VerificationVerdict(
            verdict,
            reason,
            residual,
            expected,
            required,
            reject_below,
            config.accept_closure_factor,
            config.reject_closure_factor,
            confirmed_failure=confirmed_failure,
            # Only an actual change can need undoing: a hold has nothing to roll back. A
            # rejected change is always undone; an inconclusive one follows the declared
            # posture (``config.inconclusive_rollback``).
            rollback_required=evidence.changed
            and (verdict == VERDICT_REJECTED
                 or (verdict == VERDICT_INCONCLUSIVE and config.inconclusive_rollback)),
            required_completions=floor,
        )

    if not evidence.changed:
        return outcome(VERDICT_HELD, REASON_HELD, False)
    if evidence.truncated:
        # The trace ended while the window was open: the evidence is incomplete by
        # construction, so the intervention cannot be called a failure.
        return outcome(VERDICT_INCONCLUSIVE, REASON_TRUNCATED, False)
    if residual is None or evidence.window_duration <= 0:
        return outcome(VERDICT_INCONCLUSIVE, REASON_THIN_TELEMETRY, False)
    if evidence.window_completions < floor:
        # The window was closed by the decision cap before enough work completed.  No
        # acceptance is possible on this little evidence, and no failure is claimed either.
        return outcome(VERDICT_INCONCLUSIVE, REASON_THIN_TELEMETRY, False)

    if residual >= required:
        return outcome(VERDICT_ACCEPTED, REASON_ACCEPTED, False)
    if residual <= reject_below:
        return outcome(VERDICT_REJECTED, REASON_REJECTED, True)
    return outcome(VERDICT_INCONCLUSIVE, REASON_AMBIGUOUS, False)
