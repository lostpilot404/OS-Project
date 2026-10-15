"""Causal performance monitoring for the self-healing scheduler.

The monitor is the *sensor* of the healing loop. It consumes only facts that the running
system has already produced:

* the observed waiting time of every process that has just **completed**
  (``completion_time - arrival_time - burst_time``, i.e. the standard waiting-time
  definition of :mod:`evaluation.metrics`, revealed only at completion), and
* the composition of the ready queue at a decision epoch, which the environment already
  exposes to any controller through :class:`scheduler.runtime.RuntimeObservation`.

Nothing else is visible: not the unarrived processes, not their burst times, not the
completed schedule and not the future. The rolling window therefore describes *realized*
behaviour, and the reference it is compared against is built from the same realized
behaviour during periods that looked normal.

Two statistics are tracked, because they answer two different questions:

``wait_window`` (detection)
    Waiting times of the most recently completed processes. This is the project's target
    metric and the right signal for *"is realized performance worse than normal?"*.
    It is **not** a valid before/after comparison at an intervention: the processes that
    complete immediately after a recovery action spent their wait during the incident
    itself, so the same statistic measured across the action mixes the incident's backlog
    into the "after" sample. Development measurement of that naive design rejected every
    single intervention (see ``docs/EVALUATION_RESULTS.md``, verification-design section).
``load_window`` (verification)
    Waiting-time accrual rate of each decision interval, ``ΔW / Δt``, which is exactly the
    mean number of processes waiting on the ready queue during that interval (each
    waiting process accrues one unit of waiting per unit of time). It is observed
    immediately, is not contaminated by a backlog cohort, and responds directly to a
    scheduler change: a configuration that drains the queue faster lowers it.

Reference update rule (the arithmetic is deliberately explicit, because this is where a
monitor silently self-masks a real degradation): a window is first judged against the
reference it did **not** create, and only afterwards -- and only when it did not look
suspect, no incident is open and no grace period is running -- is it absorbed into the
reference with exponential smoothing. A permanently degraded trace therefore keeps
failing the same comparison instead of quietly raising its own baseline, while slow
drift under normal conditions is tracked.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

from errors import ValidationError
from healing.config import HealingConfig

__all__ = ["MonitorSnapshot", "CausalMonitor"]


@dataclass(frozen=True)
class MonitorSnapshot:
    """Rolling performance evidence available at one decision epoch."""

    completed_count: int
    window_size: int
    window_mean_wait: float
    window_median_wait: float
    window_max_wait: float
    reference_wait: float
    band: float
    relative_band: float
    stable_samples: int
    telemetry_sufficient: bool
    new_completions_since_evaluation: int
    load_level: float
    load_reference: float
    load_band: float

    @property
    def excess_band_units(self) -> float:
        """How far the window mean sits above the reference, in band widths."""
        if self.band <= 0.0:
            return 0.0
        return (self.window_mean_wait - self.reference_wait) / self.band

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger and by artifacts."""
        payload = {
            "completed_count": self.completed_count,
            "window_size": self.window_size,
            "window_mean_wait": self.window_mean_wait,
            "window_median_wait": self.window_median_wait,
            "window_max_wait": self.window_max_wait,
            "reference_wait": self.reference_wait,
            "band": self.band,
            "relative_band": self.relative_band,
            "stable_samples": self.stable_samples,
            "telemetry_sufficient": self.telemetry_sufficient,
        }
        payload["excess_band_units"] = self.excess_band_units
        payload["load_level"] = self.load_level
        payload["load_reference"] = self.load_reference
        payload["load_band"] = self.load_band
        return payload


class CausalMonitor:
    """Rolling completed-work statistic plus its stable reference and dispersion band."""

    def __init__(self, config: HealingConfig) -> None:
        if not isinstance(config, HealingConfig):
            raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
        self._config = config
        self.reset()

    @property
    def config(self) -> HealingConfig:
        """The declared monitoring configuration."""
        return self._config

    def reset(self) -> None:
        """Drop all accumulated evidence (used at the start of every trace)."""
        self._window: Deque[float] = deque(maxlen=self._config.window_completions)
        self._completed = 0
        self._new_completions = 0
        self._reference: float | None = None
        self._band: float | None = None
        self._stable_samples = 0
        self._loads: Deque[float] = deque(maxlen=self._config.load_window_intervals)
        self._load_reference: float | None = None
        self._load_band: float | None = None
        self._load_total_wait = 0
        self._load_total_time = 0

    # -- load statistic (waiting-time accrual) ------------------------------------------
    def observe_interval(
        self, accrued_wait: int, duration: int, *, allow_reference_update: bool = True
    ) -> float:
        """Register one decision interval's waiting-time accrual.

        Args:
            accrued_wait: Waiting-time area accrued during the interval (the engine's own
                causal ``waiting_cost_increment``).
            duration: Length of the interval in simulated time units.
            allow_reference_update: Whether this interval may move the stable load
                reference. The caller freezes it while an incident is under investigation
                and while the wait statistic looks suspect, so the reference keeps
                describing the trace's *normal* load instead of absorbing the incident it
                is supposed to be compared against.

        Returns:
            The interval's accrual rate (mean number of waiting processes).

        Raises:
            ValidationError: If the interval or the accrual is invalid.
        """
        if isinstance(accrued_wait, bool) or not isinstance(accrued_wait, int) or accrued_wait < 0:
            raise ValidationError(f"accrued_wait must be a non-negative integer, got {accrued_wait!r}")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValidationError(f"interval duration must be an integer >= 1, got {duration!r}")
        if not isinstance(allow_reference_update, bool):
            raise ValidationError("allow_reference_update must be a bool")
        rate = accrued_wait / duration
        self._loads.append(rate)
        self._load_total_wait += accrued_wait
        self._load_total_time += duration
        if self._load_reference is None:
            self._load_reference = rate
            self._load_band = max(self._config.min_band_absolute, 0.5 * rate)
            return rate
        if allow_reference_update:
            deviation = abs(rate - self._load_reference)
            self._load_reference = (
                (1.0 - self._config.band_ewma) * self._load_reference
                + self._config.band_ewma * rate
            )
            self._load_band = (1.0 - self._config.band_ewma) * (self._load_band or 0.0) + (
                self._config.band_ewma * deviation
            )
        # The floor keeps the dispersion band -- and therefore every reward and every
        # acceptance threshold expressed in band units -- bounded away from zero, which
        # would otherwise let a single quiet trace produce enormous reward magnitudes.
        self._load_band = max(
            self._load_band or 0.0,
            self._config.min_band_absolute,
            self._config.min_band_fraction * self._load_reference,
        )
        return rate

    def load_level(self, intervals: int | None = None) -> float:
        """Mean accrual rate over the last ``intervals`` intervals.

        Raises:
            ValidationError: If no interval has been observed yet.
        """
        if not self._loads:
            raise ValidationError("no interval has been observed yet")
        count = len(self._loads) if intervals is None else max(1, min(intervals, len(self._loads)))
        values = list(self._loads)[-count:]
        return sum(values) / len(values)

    def load_band(self) -> float:
        """Current dispersion band of the accrual-rate statistic."""
        if self._load_band is None:
            raise ValidationError("no interval has been observed yet")
        return self._load_band

    @property
    def load_reference(self) -> float:
        """Exponentially smoothed accrual rate."""
        return 0.0 if self._load_reference is None else self._load_reference

    def accrued_so_far(self) -> Tuple[int, int]:
        """Return ``(total accrued waiting, total simulated time)`` since the trace start."""
        return self._load_total_wait, self._load_total_time

    def observe_completions(self, waiting_times: Iterable[int]) -> int:
        """Register the observed waiting times of processes that just finished.

        Args:
            waiting_times: Waiting times revealed by completions since the previous call.

        Returns:
            The number of samples added.

        Raises:
            ValidationError: If a waiting time is negative or not integral.
        """
        added = 0
        for value in waiting_times:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(
                    f"observed waiting time must be a non-negative integer, got {value!r}"
                )
            self._window.append(float(value))
            self._completed += 1
            added += 1
        self._new_completions += added
        return added

    @property
    def completed_count(self) -> int:
        """Number of completions observed so far in this trace."""
        return self._completed

    @property
    def has_pending_evidence(self) -> bool:
        """Whether new completions have arrived since the last detector evaluation."""
        return self._new_completions > 0

    def mark_evaluated(self) -> None:
        """Consume the pending-completion counter (called once per detector evaluation)."""
        self._new_completions = 0

    def snapshot(self) -> MonitorSnapshot:
        """Return the current window statistic, reference and band.

        Raises:
            ValidationError: If called before any completion has been observed.
        """
        if not self._window:
            raise ValidationError("monitor has no completed-work evidence yet")
        samples = sorted(self._window)
        count = len(samples)
        mean = sum(samples) / count
        median = samples[count // 2] if count % 2 else 0.5 * (
            samples[count // 2 - 1] + samples[count // 2]
        )
        reference = mean if self._reference is None else self._reference
        band = self._band if self._band is not None else 0.0
        load_level = self.load_level() if self._loads else 0.0
        load_band = self._load_band if self._load_band is not None else 0.0
        return MonitorSnapshot(
            completed_count=self._completed,
            window_size=count,
            window_mean_wait=mean,
            window_median_wait=median,
            window_max_wait=samples[-1],
            reference_wait=reference,
            band=band,
            relative_band=(band / reference) if reference > 0 else 0.0,
            stable_samples=self._stable_samples,
            telemetry_sufficient=(
                self._reference is not None and count >= self._config.min_window_completions
            ),
            new_completions_since_evaluation=self._new_completions,
            load_level=load_level,
            load_reference=(0.0 if self._load_reference is None else self._load_reference),
            load_band=load_band,
        )

    def absorb_into_reference(self, snapshot: MonitorSnapshot) -> None:
        """Blend a *normal-looking* window into the stable reference and its band.

        Args:
            snapshot: The snapshot the caller judged non-suspect. It must be the snapshot
                of the current window.

        Raises:
            ValidationError: If the snapshot is stale (it was not taken from the current
                completion count) or if the window is still too thin to be a reference.
        """
        if snapshot.completed_count != self._completed:
            raise ValidationError("cannot absorb a stale monitor snapshot")
        if len(self._window) < self._config.min_window_completions:
            raise ValidationError("rolling window is too thin to become a reference")

        config = self._config
        if self._reference is None:
            self._reference = snapshot.window_mean_wait
            self._band = max(
                config.min_band_absolute,
                config.min_band_fraction * self._reference,
            )
        else:
            deviation = abs(snapshot.window_mean_wait - self._reference)
            self._reference = (
                (1.0 - config.reference_ewma) * self._reference
                + config.reference_ewma * snapshot.window_mean_wait
            )
            band = (1.0 - config.band_ewma) * (self._band or 0.0) + config.band_ewma * deviation
            self._band = max(
                band,
                config.min_band_absolute,
                config.min_band_fraction * self._reference,
            )
        self._stable_samples += 1

    def reference_band(self) -> Tuple[float, float]:
        """Return the current ``(reference, band)`` pair.

        Raises:
            ValidationError: If no reference has been established yet.
        """
        if self._reference is None or self._band is None:
            raise ValidationError("no stable reference has been established yet")
        return self._reference, self._band
