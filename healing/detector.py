"""Degradation detection with a reference, thresholds, hysteresis and cooldown.

Detection is deliberately conservative and entirely causal. An evaluation happens only
when new completions have arrived, and a *degradation event* is declared only when all of
the following hold:

1. enough telemetry: the rolling window holds at least ``min_window_completions``
   completions (below that the detector abstains instead of firing on thin evidence);
2. a reference exists and the window mean exceeds ``reference + k * band``;
3. the excess persists for ``hysteresis_evaluations`` consecutive evaluations;
4. the cooldown since the previous concluded incident has elapsed;
5. adaptation is enabled and the per-trace intervention budget is not exhausted;
6. the queue is still at least as congested as the trace's own normal level: waiting-time
   windows are worst while a backlog drains, and a decayed backlog needs no help (the
   hold-only control otherwise declares most of its incidents there);
7. enough visible work remains for an intervention to matter: at least
   ``min_ready_processes`` processes are ready and their total remaining burst is at
   least ``min_remaining_work_factor`` times the observed mean burst of arrived work.

   The work gate compares *CPU work remaining* with *CPU work per job*, not with the
   waiting-time reference: the earlier version of this gate compared a burst-time total
   with a waiting-time level, which silently blocked every intervention on traces whose
   jobs were much shorter than their waiting times.

Every evaluation returns an explicit reason, including the suppressed ones, so incident
ledgers and tests can distinguish "nothing was wrong" from "something was wrong but the
system deliberately did not act".
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from errors import ValidationError
from healing.config import HealingConfig
from healing.monitor import CausalMonitor, MonitorSnapshot

__all__ = ["DetectionOutcome", "DegradationDetector"]


@dataclass(frozen=True)
class DetectionOutcome:
    """Result of one detector evaluation, including suppressed evaluations."""

    triggered: bool
    reason: str
    suspect_streak: int
    threshold: float
    absorbable: bool
    snapshot: Optional[MonitorSnapshot]
    ready_count: int
    visible_remaining_work: int
    observed_mean_burst: float
    load_level: float
    load_reference: float

    @property
    def excess_ratio(self) -> float:
        """Window mean as a multiple of the detection threshold (``nan`` without one)."""
        if self.snapshot is None or self.threshold <= 0.0:
            return float("nan")
        return self.snapshot.window_mean_wait / self.threshold

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger."""
        payload = {
            "triggered": self.triggered,
            "reason": self.reason,
            "suspect_streak": self.suspect_streak,
            "threshold": self.threshold,
            "ready_count": self.ready_count,
            "visible_remaining_work": self.visible_remaining_work,
            "observed_mean_burst": self.observed_mean_burst,
            "load_level": self.load_level,
            "load_reference": self.load_reference,
        }
        if self.snapshot is not None:
            payload.update(self.snapshot.as_dict())
        return payload


class DegradationDetector:
    """Reference-and-threshold detector with hysteresis, cooldown and a work gate."""

    def __init__(self, config: HealingConfig) -> None:
        if not isinstance(config, HealingConfig):
            raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
        self._config = config
        self._suspect_streak = 0

    @property
    def config(self) -> HealingConfig:
        """The declared detection configuration."""
        return self._config

    @property
    def suspect_streak(self) -> int:
        """Number of consecutive suspect evaluations seen so far."""
        return self._suspect_streak

    def reset(self) -> None:
        """Clear the hysteresis counter at the start of a trace."""
        self._suspect_streak = 0

    def evaluate(
        self,
        monitor: CausalMonitor,
        *,
        ready_count: int,
        visible_remaining_work: int,
        observed_mean_burst: float,
        load_level: float,
        load_reference: float,
        decisions_since_conclusion: int,
        adaptation_enabled: bool,
        interventions_used: int,
    ) -> DetectionOutcome:
        """Evaluate the current rolling statistic against the stable reference.

        Args:
            monitor: The causal monitor holding the completions observed so far.
            ready_count: Number of ready processes visible at this epoch.
            visible_remaining_work: Total remaining burst of those ready processes.
            observed_mean_burst: Mean burst of all processes that have arrived so far
                (the workload's own job scale, used by the work gate).
            load_level, load_reference: Observed queue length now, and the trace's own
                normal queue length, used by the congestion gate.
            decisions_since_conclusion: Decisions elapsed since the last concluded
                incident (used for the cooldown).
            adaptation_enabled: Whether the safe-fallback state still allows adaptation.
            interventions_used: Configuration changes already applied in this trace.

        Returns:
            A :class:`DetectionOutcome`. ``absorbable`` tells the caller that the window
            looked normal and may be blended into the stable reference.

        Raises:
            ValidationError: If the monitor has no evidence to evaluate.
        """
        if not monitor.has_pending_evidence and monitor.completed_count == 0:
            raise ValidationError("detector requires at least one observed completion")
        if ready_count < 1:
            raise ValidationError("a decision epoch requires at least one ready process")
        if visible_remaining_work < 0:
            raise ValidationError("visible_remaining_work must be >= 0")
        if not observed_mean_burst >= 0.0:
            raise ValidationError("observed_mean_burst must be a non-negative number")
        if not math.isfinite(load_level) or load_level < 0.0:
            raise ValidationError(f"load_level must be finite and >= 0, got {load_level!r}")
        if not math.isfinite(load_reference) or load_reference < 0.0:
            raise ValidationError(
                f"load_reference must be finite and >= 0, got {load_reference!r}"
            )
        if decisions_since_conclusion < 0:
            raise ValidationError("decisions_since_conclusion must be >= 0")

        config = self._config
        snapshot = monitor.snapshot()

        if snapshot.window_size < config.min_window_completions:
            # Not enough telemetry: abstain, and do not blend a thin window into the
            # reference either (a two-sample mean is not evidence about normal behaviour).
            return DetectionOutcome(
                False,
                "insufficient_telemetry",
                self._suspect_streak,
                float("nan"),
                False,
                snapshot,
                ready_count,
                visible_remaining_work,
                observed_mean_burst,
                load_level,
                load_reference,
            )

        threshold = snapshot.reference_wait + config.threshold_band_multiplier * snapshot.band
        suspect = snapshot.window_mean_wait > threshold
        self._suspect_streak = self._suspect_streak + 1 if suspect else 0

        def outcome(triggered: bool, reason: str, absorbable: bool) -> DetectionOutcome:
            return DetectionOutcome(
                triggered,
                reason,
                self._suspect_streak,
                threshold,
                absorbable,
                snapshot,
                ready_count,
                visible_remaining_work,
                observed_mean_burst,
                load_level,
                load_reference,
            )

        if not suspect:
            # First observation establishes the reference; later ones update it slowly.
            return outcome(False, "within_band", True)

        if not adaptation_enabled:
            return outcome(False, "adaptation_disabled", False)
        if interventions_used >= config.max_interventions_per_trace:
            return outcome(False, "intervention_budget_exhausted", False)
        if decisions_since_conclusion < config.cooldown_decisions:
            return outcome(False, "cooldown", False)
        if self._suspect_streak < config.hysteresis_evaluations:
            return outcome(False, "hysteresis_pending", False)
        if ready_count < config.min_ready_processes:
            self._suspect_streak = 0
            return outcome(False, "insufficient_ready_work", False)
        # A zero reference means the trace has not shown any backlog yet, so there is
        # nothing for this gate to compare against and it does not block the incident.
        if load_reference > 0.0 and load_level < config.min_load_factor * load_reference:
            self._suspect_streak = 0
            return outcome(False, "load_below_reference", False)
        if visible_remaining_work < config.min_remaining_work_factor * max(
            1.0, observed_mean_burst
        ):
            self._suspect_streak = 0
            return outcome(False, "insufficient_remaining_work", False)
        return outcome(True, "degradation_detected", False)
