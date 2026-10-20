"""Online burst-time estimator with a strict observable/evaluation boundary.

This module models the realism gap of burst knowledge.  In a real OS the
scheduler never sees true burst lengths; it sees predictions.

Information boundary (enforced by this module)
----------------------------------------------
Observable by controller (causal, may be used for decisions):
  * Arrival time, pid, priority of each ready process.
  * Cumulative CPU service already received by each pid, derived from
    the execution history the scheduler itself observed (start_time,
    end_time of every dispatched slice, including pre-empted Round-Robin
    quanta).  The scheduler tracks ``service_done[pid]`` as a sum of its own
    dispatched service durations — no simulator-internal ``remaining_burst``
    or ``burst_time`` is used.
  * Predicted total burst per process, produced by ``BurstEstimator``
    from *only* the service totals of already-completed processes
    (EWMA over completed-job totals, one EWMA per bucket plus a global
    fallback).  No future arrivals, no unfinished-job true bursts.

Evaluation-only (ground truth, never observable by controller):
  * True ``burst_time`` / ``remaining_burst`` stored in ``Workload`` and
    simulator state, exposed only via schedule results, audit, or the
    ``true_burst`` fields of ``EstimationError`` for post-hoc analysis.
    Keeping this separation is why ``perturb_observation`` takes no hidden
    burst values to compute remaining work.

The estimator is causal: predictions at time *t* use only completions
strictly before *t*.  When ``error_cv == 0`` the noisy prediction equals
the EWMA mean (still not ground truth unless the EWMA happens to match it;
documented as "zero added noise", not "unbiased or accurate").

Usage
-----
    estimator = BurstEstimator(alpha=0.5, error_cv=0.3, seed=42)
    # Per episode: estimator.reset()
    # On each dispatch: estimator.record_service(pid, service_duration)
    # On completion:    estimator.record_completion(pid)  # uses tracked total
    # At decision epoch:
    noisy_obs = perturb_observation(observation, estimator)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from config import derive_seed
from errors import ValidationError
from scheduler.runtime import ObservedProcess, RuntimeObservation

__all__ = [
    "BurstEstimator",
    "EstimationError",
    "NoisyBurstObservation",
    "perturb_observation",
]


@dataclass(frozen=True)
class EstimationError:
    """Prediction error for one process burst — evaluation-only ground truth.

    ``true_burst`` is never observable by the controller; it is kept only so
    post-hoc analysis can quantify estimator quality.  Controller decisions
    must not read this object.
    """

    pid: int
    true_burst: int
    predicted_burst: int
    absolute_error: int
    relative_error: float
    overestimated: bool

    @property
    def is_accurate_within(self) -> bool:
        """Whether relative error is within 25% (common literature threshold)."""
        return abs(self.relative_error) <= 0.25


@dataclass(frozen=True)
class NoisyBurstObservation:
    """Perturbed observation plus evaluation-only error metadata.

    ``observation`` is what the controller sees (predicted bursts, remaining
    estimated from *observable* service only).  ``errors`` contains ground
    truth for evaluation-only analysis and must not be used for decisions.
    """

    observation: RuntimeObservation
    errors: Tuple[EstimationError, ...]
    mean_relative_error: float
    max_relative_error: float


class BurstEstimator:
    """Online causal burst predictor with observable service tracking.

    The predictor maintains:
      * One EWMA per bucket (EWMA over *completed* total-burst samples,
        each sample = sum of observable service durations for that pid).
      * Per-pid cumulative service done so far (sum of observable dispatch
        durations for pids still in the system).  This is the *only* source
        for estimating remaining work: ``remaining = max(1, predicted_total -
        service_done[pid])``.  No hidden ``remaining_burst`` or true total
        is ever read to compute remaining.

    Noise is injected at prediction time to model estimator error:
    ``predicted_total = max(1, round(EWMA_mean * exp(N(bias, cv^2))))``
    where N is normal in log space.  ``error_cv == 0`` means "no added
    log-normal noise" (prediction equals EWMA mean); it does not imply
    predictions equal ground truth or are unbiased, which depends on workload
    history.

    Attributes:
        alpha: EWMA smoothing factor in (0, 1].
        error_cv: Coefficient of variation of multiplicative log-normal noise.
        error_bias: Mean log-error (negative = systematic underestimation).
        initial_guess: Prior prediction before any observation.
        seed: Base seed for reproducible noise draws.

    Observable vs evaluation-only:
        * Observable: ``predict()``, ``predict_total_noisy()``,
          ``record_service()``, ``service_done()``, ``reset()``.
        * Evaluation-only: ``observe_completed()`` with ground truth (used
          only by experiment harnesses that have post-hoc access to totals;
          the controller path uses ``record_completion()`` which derives the
          total from observable service history).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        error_cv: float = 0.3,
        error_bias: float = 0.0,
        initial_guess: int = 10,
        seed: int = 0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValidationError(f"alpha must be in (0, 1], got {alpha}")
        if error_cv < 0.0 or not math.isfinite(error_cv):
            raise ValidationError(f"error_cv must be finite and >= 0, got {error_cv}")
        if not math.isfinite(error_bias):
            raise ValidationError(f"error_bias must be finite, got {error_bias}")
        if not isinstance(initial_guess, int) or initial_guess < 1:
            raise ValidationError(f"initial_guess must be int >= 1, got {initial_guess!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValidationError(f"seed must be non-negative int, got {seed!r}")
        self.alpha = float(alpha)
        self.error_cv = float(error_cv)
        self.error_bias = float(error_bias)
        self.initial_guess = int(initial_guess)
        self.seed = int(seed)
        self._estimates: Dict[str, float] = {}
        self._global_mean: Optional[float] = None
        self._count: int = 0
        self._draw_counter: int = 0
        # Observable service tracking: pid -> cumulative CPU time dispatched.
        self._service_done: Dict[int, int] = {}

    def reset(self) -> None:
        """Clear all learned estimates and per-pid service totals (per-episode)."""
        self._estimates.clear()
        self._global_mean = None
        self._count = 0
        self._draw_counter = 0
        self._service_done.clear()

    # -- observable service tracking -----------------------------------------

    def record_service(self, pid: int, service: int) -> None:
        """Record observable CPU service dispatched to ``pid``.

        This is the *only* legitimate source for computing remaining work.
        Called by the controller's ``observe_decision`` hook which sees
        ``end_time - start_time - switch_overhead`` for each slice.

        Args:
            pid: Process id.
            service: Dispatched CPU time in this slice (>=1).

        Raises:
            ValidationError: If pid/service invalid.
        """
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            raise ValidationError(f"pid must be non-negative int, got {pid!r}")
        if isinstance(service, bool) or not isinstance(service, int) or service < 1:
            raise ValidationError(f"service must be int >=1, got {service!r}")
        self._service_done[pid] = self._service_done.get(pid, 0) + service

    def service_done(self, pid: int) -> int:
        """Return observable cumulative service done for ``pid`` (0 if never dispatched)."""
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            raise ValidationError(f"pid must be non-negative int, got {pid!r}")
        return self._service_done.get(pid, 0)

    def record_completion(self, pid: int, bucket: str = "global") -> None:
        """Mark ``pid`` completed: use its observable total as EWMA sample.

        The total sample is the sum of all observable service slices for that
        pid (tracked via ``record_service``).  This is what a real system
        observes: it knows how long the job ran, not a hidden ``burst_time``.
        After updating the EWMA, the per-pid tracker is cleared.

        Args:
            pid: Completed process id.
            bucket: EWMA bucket name.

        Raises:
            ValidationError: If pid not tracked or bucket invalid.
        """
        if not isinstance(bucket, str) or not bucket:
            raise ValidationError(f"bucket must be non-empty str, got {bucket!r}")
        if pid not in self._service_done:
            raise ValidationError(f"completion for untracked pid {pid}; record_service must be called first")
        total = self._service_done.pop(pid)
        # total is int >=1, reuse existing EWMA update path.
        self.observe_completed(total, bucket=bucket)

    # -- EWMA over completed totals (observable only via record_completion) --

    def observe_completed(self, burst_time: int, bucket: str = "global") -> None:
        """Feed back a completed-job total (observable sum of service slices).

        Prefer ``record_completion`` in controller code; this method exists for
        evaluation harnesses and tests that have post-hoc ground truth and want
        to seed the EWMA without per-pid tracking.  The controller's causal
        path does not use ground-truth ``burst_time`` directly to compute
        remaining work.

        Args:
            burst_time: Total CPU time observed for the completed job (>=1).
            bucket: EWMA bucket name.

        Raises:
            ValidationError: If burst_time/bucket invalid.
        """
        if not isinstance(burst_time, int) or burst_time < 1:
            raise ValidationError(f"burst_time must be int >= 1, got {burst_time!r}")
        if not isinstance(bucket, str) or not bucket:
            raise ValidationError(f"bucket must be non-empty str, got {bucket!r}")
        value = float(burst_time)
        if bucket in self._estimates:
            self._estimates[bucket] = self.alpha * value + (1 - self.alpha) * self._estimates[bucket]
        else:
            self._estimates[bucket] = value
        if self._global_mean is None:
            self._global_mean = value
        else:
            self._global_mean = (self._global_mean * self._count + value) / (self._count + 1)
        self._count += 1

    def predict(self, bucket: str = "global") -> int:
        """Predict total burst for next job in ``bucket`` (no noise, no ground truth)."""
        if bucket in self._estimates:
            return max(1, int(round(self._estimates[bucket])))
        if self._global_mean is not None:
            return max(1, int(round(self._global_mean)))
        return self.initial_guess

    def predict_with_noise(
        self, true_burst: Optional[int] = None, bucket: str = "global"
    ) -> Tuple[int, Optional[EstimationError]]:
        """Predict total burst for a job in ``bucket``, optionally with error metadata.

        The prediction itself uses *only* EWMA history and log-normal noise;
        ``true_burst`` is *never* used to compute the predicted value and may
        be None.  When ``true_burst`` is provided, an ``EstimationError`` with
        ground truth is returned for evaluation-only analysis; controller code
        must not read it for decisions.

        Args:
            true_burst: Optional ground truth for evaluation-only error calc.
                When None, error is None; prediction is still returned.
            bucket: EWMA bucket.

        Returns:
            (predicted_total, error_or_None) where predicted_total >=1.

        Raises:
            ValidationError: If true_burst is not None and invalid.
        """
        if true_burst is not None:
            if isinstance(true_burst, bool) or not isinstance(true_burst, int) or true_burst < 1:
                raise ValidationError(f"true_burst must be int >=1, got {true_burst!r}")
        base = self.predict(bucket)
        if self.error_cv == 0.0:
            predicted = base
        else:
            rng = np.random.default_rng(derive_seed(self.seed, self._draw_counter))
            self._draw_counter += 1
            log_pred = math.log(max(1, base)) + self.error_bias + rng.normal(0, self.error_cv)
            predicted = max(1, int(round(math.exp(log_pred))))
        error: Optional[EstimationError] = None
        if true_burst is not None:
            abs_err = abs(predicted - true_burst)
            rel_err = (predicted - true_burst) / true_burst if true_burst else 0.0
            error = EstimationError(
                pid=-1,
                true_burst=true_burst,
                predicted_burst=predicted,
                absolute_error=abs_err,
                relative_error=rel_err,
                overestimated=predicted > true_burst,
            )
        return predicted, error

    @property
    def global_mean(self) -> Optional[float]:
        return self._global_mean

    @property
    def buckets(self) -> Tuple[str, ...]:
        return tuple(self._estimates.keys())

    def summary(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "error_cv": self.error_cv,
            "error_bias": self.error_bias,
            "initial_guess": self.initial_guess,
            "seed": self.seed,
            "global_mean": self._global_mean,
            "buckets": dict(self._estimates),
            "completed_count": self._count,
            "tracked_pids": len(self._service_done),
        }


def perturb_observation(
    observation: RuntimeObservation,
    estimator: BurstEstimator,
    *,
    bucket: str = "global",
) -> NoisyBurstObservation:
    """Return controller-view observation with estimated bursts/remaining.

    Observable information used (controller-legitimate):
      * ``estimator.service_done(pid)`` — cumulative dispatched service per
        ready pid, tracked from the scheduler's own execution history.
        This is the *only* source for estimating remaining work.
      * ``estimator.predict()/predict_with_noise(bucket)`` — EWMA over
        *completed-job totals* (sums of observable service), never hidden
        ``burst_time``.

    Evaluation-only (not observable, not used for decisions):
      * ``proc.burst_time`` / ``proc.remaining_burst`` from the ground-truth
        observation are read *only* to populate ``EstimationError`` for
        post-hoc analysis.  The controller's observed ``burst_time`` and
        ``remaining_burst`` are derived solely from the two observable
        quantities above.

    Remaining-work estimation (strictly observable):
        predicted_total = noisy EWMA prediction (evaluation-only noise)
        service = estimator.service_done(pid)  # 0 if never dispatched
        predicted_remaining = max(1, predicted_total - service)
        (clamped below to 1, above to predicted_total; handles zero/negative
        and the case where service already exceeds prediction).

    Args:
        observation: Ground-truth simulator observation (contains hidden bursts;
            treated as evaluation-only source for error metadata).
        estimator: Causal estimator with per-pid service tracking.
        bucket: EWMA bucket name.

    Returns:
        NoisyBurstObservation with controller-visible noisy bursts and
        evaluation-only errors.

    Raises:
        ValidationError: If observation/estimator invalid.
    """
    if not isinstance(observation, RuntimeObservation):
        raise ValidationError(f"observation must be RuntimeObservation, got {type(observation).__name__}")
    if not isinstance(estimator, BurstEstimator):
        raise ValidationError(f"estimator must be BurstEstimator, got {type(estimator).__name__}")

    errors: list[EstimationError] = []
    noisy_processes: list[ObservedProcess] = []
    for proc in observation.ready_processes:
        # Observable prediction (no hidden burst used).
        # For evaluation-only error, pass true total if available, but the
        # prediction value itself does NOT depend on it.
        predicted_total, err = estimator.predict_with_noise(true_burst=proc.burst_time, bucket=bucket)
        if err is not None:
            # Fill pid for error reporting (evaluation-only).
            err = EstimationError(
                pid=proc.pid,
                true_burst=err.true_burst,
                predicted_burst=err.predicted_burst,
                absolute_error=err.absolute_error,
                relative_error=err.relative_error,
                overestimated=err.overestimated,
            )
            errors.append(err)
        else:
            # Should not happen when true_burst provided, but handle.
            errors.append(
                EstimationError(
                    pid=proc.pid,
                    true_burst=proc.burst_time,
                    predicted_burst=predicted_total,
                    absolute_error=abs(predicted_total - proc.burst_time),
                    relative_error=(predicted_total - proc.burst_time) / proc.burst_time if proc.burst_time else 0.0,
                    overestimated=predicted_total > proc.burst_time,
                )
            )

        # Observable remaining estimate: predicted total minus observable service.
        # This is the fix: no use of proc.remaining_burst or true burst ratio.
        service = estimator.service_done(proc.pid)
        # Service should never exceed predicted_total by much, but if it does
        # (e.g., predictor underestimated and job ran longer than predicted),
        # clamp remaining to minimum 1 to keep observation valid.
        predicted_remaining = predicted_total - service
        if predicted_remaining < 1:
            predicted_remaining = 1
        if predicted_remaining > predicted_total:
            predicted_remaining = predicted_total

        noisy_processes.append(
            ObservedProcess(
                pid=proc.pid,
                arrival_time=proc.arrival_time,
                burst_time=predicted_total,
                remaining_burst=predicted_remaining,
                priority=proc.priority,
            )
        )

    noisy_bursts = [p.burst_time for p in noisy_processes]
    noisy_obs = RuntimeObservation(
        current_time=observation.current_time,
        ready_processes=tuple(noisy_processes),
        completed_count=observation.completed_count,
        arrived_count=observation.arrived_count,
        observed_mean_burst=(sum(noisy_bursts) / len(noisy_bursts) if noisy_bursts else 0.0),
        observed_priority_spread=observation.observed_priority_spread,
        mean_ready_arrival_age=observation.mean_ready_arrival_age,
        previous_policy=observation.previous_policy,
    )
    mean_rel = sum(e.relative_error for e in errors) / len(errors) if errors else 0.0
    max_rel = max((abs(e.relative_error) for e in errors), default=0.0)
    return NoisyBurstObservation(
        observation=noisy_obs,
        errors=tuple(errors),
        mean_relative_error=mean_rel,
        max_relative_error=max_rel,
    )
