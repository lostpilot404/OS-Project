"""Frozen configuration of the self-healing scheduling loop.

Every threshold used by the monitor, detector, diagnoser, verification window, rollback
and safe fallback is declared here, in one place, with the value it takes in the recorded
study.  Nothing in the healing package hard-codes a threshold, so the declared design can
be read off this file instead of being reverse-engineered from the code.

The values were frozen before the held-out evaluation described in
``docs/METHODOLOGY.md``; the training and validation splits were used to choose them and
the final-test split was generated afterwards from a seed declared in
``experiments/healing_config.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import ACTION_FCFS
from errors import ConfigurationError

__all__ = ["HealingConfig"]


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be an integer >= 0, got {value!r}")
    return value


def _require_finite(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise ConfigurationError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class HealingConfig:
    """Thresholds of the monitor, detector, diagnoser and recovery state machine.

    Monitoring (causal, completion-based):

        window_completions: Number of most recent completed processes whose observed
            waiting times form the rolling performance statistic.
        reference_ewma, band_ewma: Exponential smoothing factors used to update the
            stable reference and its dispersion band while the system looks normal.
        min_band_fraction, min_band_absolute: Floors that stop the band collapsing to
            zero on a very regular trace (which would make every deviation an incident).
        min_window_completions: Telemetry floor: with fewer completions than this in the
            window the detector abstains instead of firing on thin evidence.
        load_window_intervals: Length of the accrual-rate (waiting-process count) window
            used by verification.

    Detection:

        threshold_band_multiplier: Suspect when ``window_mean > reference + k * band``.
        hysteresis_evaluations: Consecutive suspect evaluations required before a
            degradation event is declared.
        cooldown_decisions: Minimum number of decisions between two concluded incidents.
        min_ready_processes, min_remaining_work_factor: Work gate. An intervention is
            only attempted when at least this many processes are ready and the visible
            remaining work is at least this multiple of the observed mean burst, so a
            trace cannot be perturbed when little can still be improved. The gate compares
            CPU work with CPU work per job; comparing it with the waiting-time reference
            (an earlier version of the code) mixed two units and blocked almost every
            intervention.
        min_load_factor: Congestion gate. An incident is only acted on while the observed
            queue length is at least this multiple of the trace's own normal load level.
            Waiting-time windows are usually worst *while* a backlog drains, and without
            this gate the detector spends its budget on traces that need no help: measured
            on the training split, the hold-only control (no intervention at all) declared
            1.37 incidents per trace, most of them during a decaying backlog.
        max_interventions_per_trace: Overhead budget; afterwards adaptation is disabled
            for the remainder of the trace.

    Verification and safety:

        verify_decisions, verify_completions, verify_max_decisions: A recovery window
            closes after ``verify_decisions`` decisions and at least
            ``verify_completions`` completions, or at ``verify_max_decisions`` regardless.
        accept_closure_factor: The declared verification rule. An intervention is accepted
            only if the *normalised load closure* of the recovery window is at least this
            large, i.e. ``(trigger_load - post_load) / max(trigger_load - reference_load,
            load_band) >= accept_closure_factor``; otherwise the configuration is rolled
            back to the last accepted one. All three quantities are observed before the
            verdict, so the rule is causal and scale-free. The declared value is derived by
            :mod:`experiments.healing_thresholds` from the null distribution of the
            statistic on the *no-action control* over the training split: the control's
            95th percentile is the smallest value whose false-accept rate under "nothing
            changed" is at most 5 %, measured at 2.38 and declared as 2.4. A value above 1
            is expected and is itself a finding -- the trace drains after a trigger (the
            mean load falls from about 10 to about 6 waiting processes in the control)
            faster than any recovery action can change it, so the rule is deliberately
            strict: it never reports an unverified improvement as a success.
        max_consecutive_failures: Consecutive rollbacks that trigger the safe fallback.
        fallback_policy: Configuration pinned by the safe fallback when a previous
            accepted configuration is unavailable.

    Diagnosis (all comparisons use observable arrived work only):

        head_block_factor, head_block_min_ready: Head-of-line blocking evidence.
        starvation_age_factor: Fairness-starvation evidence.
        tail_dispersion_factor: Tail-dispersion evidence.
        congestion_ready_count: Queue-congestion evidence.

    Reward:

        intervention_cost: Cost charged for each configuration change, expressed as a
            fraction of the trigger reference waiting time and converted to dispersion
            band units when the reward is computed. It represents the real price of
            changing configuration under a live system (one context switch plus the risk
            of an unverified change), and is set to about one simulated switch out of the
            ~130 time units of normal waiting. Development measurement showed that a 5 %
            charge exceeded the achievable improvement of *any* action on these families
            (about 3 % of the reference waiting time for the best action), which made the
            learned controller indifferent to genuinely useful recoveries; the declared
            value is therefore 1 %, and the measured reward ranking of actions is reported
            against the true effect in ``docs/EVALUATION_RESULTS.md``.
        rollback_penalty: Additional charge, in the same units, when an intervention
            fails verification.

    Actions:

        min_quantum, max_quantum: Bounds of the Round-Robin quantum recovery actions.
    """

    # Monitoring
    window_completions: int = 8
    reference_ewma: float = 0.35
    band_ewma: float = 0.25
    min_band_fraction: float = 0.15
    min_band_absolute: float = 1.0
    min_window_completions: int = 4
    load_window_intervals: int = 8
    # Detection
    threshold_band_multiplier: float = 1.25
    hysteresis_evaluations: int = 2
    cooldown_decisions: int = 3
    min_ready_processes: int = 2
    min_remaining_work_factor: float = 1.5
    min_load_factor: float = 1.0
    max_interventions_per_trace: int = 6
    # Verification and safety
    verify_decisions: int = 5
    verify_completions: int = 3
    verify_max_decisions: int = 14
    accept_closure_factor: float = 2.4
    max_consecutive_failures: int = 2
    fallback_policy: int = ACTION_FCFS
    # Diagnosis
    head_block_factor: float = 4.0
    head_block_min_ready: int = 3
    starvation_age_factor: float = 3.0
    tail_dispersion_factor: float = 5.0
    congestion_ready_count: int = 5
    # Reward
    intervention_cost: float = 0.01
    rollback_penalty: float = 0.02
    # Action bounds
    min_quantum: int = 1
    max_quantum: int = 16

    def __post_init__(self) -> None:
        _require_positive_int(self.window_completions, "window_completions")
        _require_positive_int(self.min_window_completions, "min_window_completions")
        _require_positive_int(self.load_window_intervals, "load_window_intervals")
        if self.min_window_completions > self.window_completions:
            raise ConfigurationError(
                "min_window_completions cannot exceed the rolling window size"
            )
        for name in ("reference_ewma", "band_ewma"):
            value = _require_finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ConfigurationError(f"{name} must lie in (0, 1], got {value}")
        _require_finite(self.min_band_fraction, "min_band_fraction")
        _require_finite(self.min_band_absolute, "min_band_absolute")
        _require_finite(self.threshold_band_multiplier, "threshold_band_multiplier")
        if self.threshold_band_multiplier <= 0.0:
            raise ConfigurationError("threshold_band_multiplier must be > 0")
        _require_positive_int(self.hysteresis_evaluations, "hysteresis_evaluations")
        _require_non_negative_int(self.cooldown_decisions, "cooldown_decisions")
        _require_positive_int(self.min_ready_processes, "min_ready_processes")
        _require_finite(self.min_remaining_work_factor, "min_remaining_work_factor")
        if self.min_remaining_work_factor <= 0.0:
            raise ConfigurationError("min_remaining_work_factor must be > 0")
        _require_positive_int(self.max_interventions_per_trace, "max_interventions_per_trace")
        _require_positive_int(self.verify_decisions, "verify_decisions")
        _require_positive_int(self.verify_completions, "verify_completions")
        _require_positive_int(self.verify_max_decisions, "verify_max_decisions")
        if self.verify_max_decisions < self.verify_decisions:
            raise ConfigurationError("verify_max_decisions must be >= verify_decisions")
        _require_finite(self.accept_closure_factor, "accept_closure_factor")
        _require_finite(self.min_load_factor, "min_load_factor")
        _require_positive_int(self.max_consecutive_failures, "max_consecutive_failures")
        if self.fallback_policy not in range(4):
            raise ConfigurationError("fallback_policy must be a policy index in 0..3")
        _require_finite(self.head_block_factor, "head_block_factor")
        _require_positive_int(self.head_block_min_ready, "head_block_min_ready")
        _require_finite(self.starvation_age_factor, "starvation_age_factor")
        _require_finite(self.tail_dispersion_factor, "tail_dispersion_factor")
        _require_positive_int(self.congestion_ready_count, "congestion_ready_count")
        _require_finite(self.intervention_cost, "intervention_cost")
        _require_finite(self.rollback_penalty, "rollback_penalty")
        _require_positive_int(self.min_quantum, "min_quantum")
        _require_positive_int(self.max_quantum, "max_quantum")
        if self.max_quantum < self.min_quantum:
            raise ConfigurationError("max_quantum must be >= min_quantum")

    def as_dict(self) -> dict:
        """Return the full declared configuration as a plain dictionary."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
