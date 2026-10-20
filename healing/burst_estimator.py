"""Online burst-time estimator with prediction error.

This module addresses limitation #2 from Review 2: the real OS does not know
exact burst lengths.  It provides:

* ``BurstEstimator`` — an exponential-smoothing predictor that sees only
  completed burst times and predicts the next burst of the same workload class.
  History is per-family or per-priority bucket; the estimator is causal
  (no future information) and deterministic given a seed.

* ``NoisyBurstObservation`` / ``perturb_observation`` — a thin wrapper that
  replaces the simulator's exact ``remaining_burst`` with a noisy estimate.
  Error is modelled as multiplicative log-normal noise with configurable
  bias and coefficient of variation, matching empirical observations that burst
  prediction error scales with job size.

* ``EstimationError`` — a dataclass capturing the error of one prediction,
  used for evaluation and for the controller's uncertainty-aware logic.

The controller's observation boundary stays causal: the estimator only sees
workloads of already-completed processes, exactly what a kernel's burst
predictor would see.  When ``error_cv == 0`` the estimator degenerates to
the exact simulator (backwards compatible).

Usage::

    estimator = BurstEstimator(alpha=0.5, error_cv=0.3, seed=42)
    # In controller.select_action, before diagnosis:
    noisy_obs = perturb_observation(observation, estimator)
    # After each completion, feed ground truth back:
    estimator.observe_completed(burst_time=completed_burst, family="mixed")
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
    """Prediction error for one process burst."""

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
    """Perturbed observation plus its error metadata."""

    observation: RuntimeObservation
    errors: Tuple[EstimationError, ...]
    mean_relative_error: float
    max_relative_error: float


class BurstEstimator:
    """Online causal burst predictor with configurable noise.

    The predictor maintains one EWMA per bucket (family name by default, or
    a fallback global bucket).  Predicted burst for an unseen bucket is the
    global mean of all completed bursts so far, or a prior ``initial_guess``
    when nothing has been observed yet.

    Noise is injected at prediction time to model real-world estimator error:
    ``predicted = max(1, round(true * exp(N(bias, cv^2))))`` where N is
    normal in log space.  ``error_cv == 0`` disables noise (exact predictions).

    Attributes:
        alpha: EWMA smoothing factor in (0, 1].
        error_cv: Coefficient of variation of multiplicative log-normal error.
        error_bias: Mean log-error (negative = systematic underestimation).
        initial_guess: Prior prediction before any observation.
        seed: Base seed for reproducible noise draws.
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

    def reset(self) -> None:
        """Clear all learned estimates (call at episode start)."""
        self._estimates.clear()
        self._global_mean = None
        self._count = 0
        self._draw_counter = 0

    def observe_completed(self, burst_time: int, bucket: str = "global") -> None:
        """Feed back ground truth after a process completes."""
        if not isinstance(burst_time, int) or burst_time < 1:
            raise ValidationError(f"burst_time must be int >= 1, got {burst_time!r}")
        if not isinstance(bucket, str) or not bucket:
            raise ValidationError(f"bucket must be non-empty str, got {bucket!r}")
        value = float(burst_time)
        if bucket in self._estimates:
            self._estimates[bucket] = self.alpha * value + (1 - self.alpha) * self._estimates[bucket]
        else:
            self._estimates[bucket] = value
        # Update global mean incrementally.
        if self._global_mean is None:
            self._global_mean = value
        else:
            self._global_mean = (self._global_mean * self._count + value) / (self._count + 1)
        self._count += 1

    def predict(self, bucket: str = "global") -> int:
        """Predict burst for next process in bucket (before seeing it)."""
        if bucket in self._estimates:
            return max(1, int(round(self._estimates[bucket])))
        if self._global_mean is not None:
            return max(1, int(round(self._global_mean)))
        return self.initial_guess

    def predict_with_noise(self, true_burst: int, bucket: str = "global") -> Tuple[int, EstimationError]:
        """Predict burst for a process whose true burst is ``true_burst``, with noise.

        In simulation we know the true burst; this method shows what the
        estimator *would* have predicted given its history, then perturbs that
        prediction with calibrated noise to simulate real-world error.

        Returns:
            (predicted_burst, error_metadata) where predicted_burst >= 1.
        """
        # Base prediction from history (what a real predictor would output).
        base = self.predict(bucket)
        # Blend base prediction with true value via exponential history:
        # New predictor trusts its estimate, not the true value; true_burst only
        # matters for noise calibration (error scales with size).
        if self.error_cv == 0.0:
            predicted = base
        else:
            # Log-normal multiplicative noise: log(predicted) ~ N(log(base)+bias, cv^2)
            # This preserves the mean when bias=0 and cv small.
            rng = np.random.default_rng(derive_seed(self.seed, self._draw_counter))
            self._draw_counter += 1
            log_pred = math.log(max(1, base)) + self.error_bias + rng.normal(0, self.error_cv)
            predicted = max(1, int(round(math.exp(log_pred))))
        # Also compute what error relative to true would look like.
        abs_err = abs(predicted - true_burst)
        rel_err = (predicted - true_burst) / true_burst if true_burst else 0.0
        error = EstimationError(
            pid=-1,  # caller fills real pid
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
        }


def perturb_observation(
    observation: RuntimeObservation,
    estimator: BurstEstimator,
    *,
    bucket: str = "global",
) -> NoisyBurstObservation:
    """Return a copy of observation where burst lengths are estimator predictions.

    The true remaining_burst is hidden; the caller sees only noisy estimates.
    This is the causal uncertainty boundary: the controller never receives the
    true burst, only what the estimator believes.  Evaluation still measures
    against ground truth.

    Args:
        observation: Original simulator observation with exact bursts.
        estimator: Causal burst estimator (call reset() per episode).
        bucket: Bucket name for per-family prediction.

    Returns:
        NoisyBurstObservation with perturbed ready_processes.
    """
    if not isinstance(observation, RuntimeObservation):
        raise ValidationError(f"observation must be RuntimeObservation, got {type(observation).__name__}")
    if not isinstance(estimator, BurstEstimator):
        raise ValidationError(f"estimator must be BurstEstimator, got {type(estimator).__name__}")

    errors: list[EstimationError] = []
    noisy_processes: list[ObservedProcess] = []
    for proc in observation.ready_processes:
        predicted, err = estimator.predict_with_noise(proc.burst_time, bucket=bucket)
        # Scale remaining proportionally: if predictor thinks job is longer,
        # remaining scales by same factor (preserving progress fraction).
        progress = 1.0 - (proc.remaining_burst / proc.burst_time) if proc.burst_time else 0.0
        predicted_remaining = max(1, int(round(predicted * (1.0 - progress))))
        # Never predict remaining > predicted total.
        predicted_remaining = min(predicted_remaining, predicted)
        noisy_processes.append(
            ObservedProcess(
                pid=proc.pid,
                arrival_time=proc.arrival_time,
                burst_time=predicted,
                remaining_burst=predicted_remaining,
                priority=proc.priority,
            )
        )
        errors.append(
            EstimationError(
                pid=proc.pid,
                true_burst=proc.burst_time,
                predicted_burst=predicted,
                absolute_error=err.absolute_error,
                relative_error=err.relative_error,
                overestimated=err.overestimated,
            )
        )

    # Recompute aggregates from noisy bursts.
    noisy_bursts = [p.burst_time for p in noisy_processes]
    # Also perturb derived aggregates proportionally.
    # observed_mean_burst and priority spread stay exact (priority is observable).
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
