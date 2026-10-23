"""Mid-trace workload drift: phase changes and uncertainty injection.

This module implements the distinctive contribution recommended in Review 2:
workloads that change *during* execution.  In a real system load is not
stationary — a batch-heavy queue can become interactive, or a normal
arrival pattern can turn into a storm.  The simulator's workload generator
produces stationary traces by default; this module wraps any generated
workload and injects a deterministic phase shift at a declared injection
time.

The injection is causal and reproducible: every random draw flows through
``numpy.random.Generator`` seeded with ``derive_seed``, so the same base
workload always yields the same drifted workload.  Pre-injection arrivals,
pids and priorities are untouched, so the prefix is identical to the stable
trace — the controller cannot distinguish the two until the drift arrives.

Four drift kinds are supported, matching the safety-reliability narrative:

* ``burst_shock`` — multiply CPU time of every process arriving after injection;
  models a sudden arrival of compute-intensive jobs.
* ``arrival_storm`` — compress arrival times after injection; models a load spike.
* ``phase_transition`` — replace the second half of the trace with a different
  workload family (e.g. batch→interactive); models a workload regime change.
* ``priority_inversion`` — skew priorities after injection; models contention.

The module is additive: no existing workload family or seed changes
behaviour when drift is not requested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import derive_seed
from errors import ValidationError
from workload.models import Process, Workload

__all__ = [
    "DRIFT_KINDS",
    "DRIFT_KIND_BURST_SHOCK",
    "DRIFT_KIND_ARRIVAL_STORM",
    "DRIFT_KIND_PHASE_TRANSITION",
    "DRIFT_KIND_PRIORITY_INVERSION",
    "DRIFT_INJECTION_FRACTION",
    "DRIFT_MULTIPLIERS",
    "DriftSpec",
    "injection_time_for_workload",
    "apply_drift",
    "generate_drift_workload",
]

DRIFT_KIND_BURST_SHOCK = "burst_shock"
DRIFT_KIND_ARRIVAL_STORM = "arrival_storm"
DRIFT_KIND_PHASE_TRANSITION = "phase_transition"
DRIFT_KIND_PRIORITY_INVERSION = "priority_inversion"

DRIFT_KINDS: Tuple[str, ...] = (
    DRIFT_KIND_BURST_SHOCK,
    DRIFT_KIND_ARRIVAL_STORM,
    DRIFT_KIND_PHASE_TRANSITION,
    DRIFT_KIND_PRIORITY_INVERSION,
)

# Fraction of arrival span at which drift is injected.  Chosen so that
# there is enough prefix to establish a stable reference and enough suffix
# to measure recovery.
DRIFT_INJECTION_FRACTION: float = 0.35

# Declared multipliers for the two simple drift kinds.
DRIFT_MULTIPLIERS: Dict[str, float] = {
    DRIFT_KIND_BURST_SHOCK: 4.0,
    DRIFT_KIND_ARRIVAL_STORM: 0.35,
}


@dataclass(frozen=True)
class DriftSpec:
    """Declarative specification of a mid-trace drift injection.

    Attributes:
        kind: One of DRIFT_KINDS.
        injection_fraction: Fraction of arrival span where drift starts (0, 1).
        burst_multiplier: For burst_shock — factor applied to bursts after injection.
        arrival_compression: For arrival_storm — factor applied to gaps after injection.
        phase_target_family: For phase_transition — family name to draw second half from.
        priority_skew_fraction: For priority_inversion — share of post-injection
            processes whose priority is flipped.
        seed: Seed for any randomness in the drift itself (e.g. target family draw).
    """

    kind: str
    injection_fraction: float = DRIFT_INJECTION_FRACTION
    burst_multiplier: float = 4.0
    arrival_compression: float = 0.35
    phase_target_family: Optional[str] = None
    priority_skew_fraction: float = 0.7
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in DRIFT_KINDS:
            raise ValidationError(f"unknown drift kind {self.kind!r}; known: {list(DRIFT_KINDS)}")
        if not 0.0 < self.injection_fraction < 1.0:
            raise ValidationError(f"injection_fraction must be in (0, 1), got {self.injection_fraction}")
        if not math.isfinite(self.burst_multiplier) or self.burst_multiplier <= 0:
            raise ValidationError(f"burst_multiplier must be finite and > 0, got {self.burst_multiplier}")
        if not 0.0 < self.arrival_compression <= 1.0:
            raise ValidationError(f"arrival_compression must be in (0, 1], got {self.arrival_compression}")
        if not 0.0 <= self.priority_skew_fraction <= 1.0:
            raise ValidationError(f"priority_skew_fraction must be in [0, 1], got {self.priority_skew_fraction}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError(f"seed must be a non-negative int, got {self.seed!r}")
        if self.kind == DRIFT_KIND_PHASE_TRANSITION and not self.phase_target_family:
            raise ValidationError("phase_target_family is required for phase_transition drift")


def injection_time_for_workload(workload: Workload, fraction: float = DRIFT_INJECTION_FRACTION) -> int:
    """Return the simulated time at which drift is injected for a workload."""
    if not isinstance(workload, Workload):
        raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
    arrivals = [p.arrival_time for p in workload]
    if not arrivals:
        raise ValidationError("workload must not be empty for drift injection")
    if not 0.0 < fraction < 1.0:
        raise ValidationError(f"fraction must be in (0, 1), got {fraction}")
    span = max(arrivals) - min(arrivals)
    return int(min(arrivals) + round(fraction * span))


def apply_drift(
    workload: Workload,
    spec: DriftSpec,
    *,
    target_workload: Optional[Workload] = None,
) -> Workload:
    """Apply a drift injection to a workload deterministically.

    Args:
        workload: Base workload (pre-drift).
        spec: Drift specification.
        target_workload: For phase_transition — a pre-generated workload of the
            target family to splice in.  When provided, its process count and
            burst/priority distributions define the second phase.

    Returns:
        A new Workload with the same pre-injection prefix and a drifted suffix.

    Raises:
        ValidationError: If inputs are invalid.
    """
    if not isinstance(workload, Workload):
        raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
    if not isinstance(spec, DriftSpec):
        raise ValidationError(f"spec must be a DriftSpec, got {type(spec).__name__}")

    boundary = injection_time_for_workload(workload, spec.injection_fraction)

    if spec.kind == DRIFT_KIND_BURST_SHOCK:
        return _apply_burst_shock(workload, boundary, spec.burst_multiplier)
    if spec.kind == DRIFT_KIND_ARRIVAL_STORM:
        return _apply_arrival_storm(workload, boundary, spec.arrival_compression)
    if spec.kind == DRIFT_KIND_PRIORITY_INVERSION:
        return _apply_priority_inversion(workload, boundary, spec.priority_skew_fraction, spec.seed)
    if spec.kind == DRIFT_KIND_PHASE_TRANSITION:
        if target_workload is None:
            raise ValidationError("phase_transition drift requires target_workload")
        return _apply_phase_transition(workload, boundary, target_workload, spec.seed)
    raise ValidationError(f"unknown drift kind {spec.kind!r}")


def _apply_burst_shock(workload: Workload, boundary: int, factor: float) -> Workload:
    processes: List[Process] = []
    for p in workload:
        if p.arrival_time < boundary:
            processes.append(p)
        else:
            processes.append(
                Process(
                    pid=p.pid,
                    arrival_time=p.arrival_time,
                    burst_time=max(1, int(round(p.burst_time * factor))),
                    priority=p.priority,
                )
            )
    return Workload(f"{workload.name}+{DRIFT_KIND_BURST_SHOCK}", processes, description=DRIFT_KIND_BURST_SHOCK)


def _apply_arrival_storm(workload: Workload, boundary: int, compression: float) -> Workload:
    processes: List[Process] = []
    for p in workload:
        if p.arrival_time < boundary:
            processes.append(p)
        else:
            compressed = boundary + int(round((p.arrival_time - boundary) * compression))
            processes.append(
                Process(
                    pid=p.pid,
                    arrival_time=compressed,
                    burst_time=p.burst_time,
                    priority=p.priority,
                )
            )
    return Workload(f"{workload.name}+{DRIFT_KIND_ARRIVAL_STORM}", processes, description=DRIFT_KIND_ARRIVAL_STORM)


def _apply_priority_inversion(
    workload: Workload, boundary: int, skew_fraction: float, seed: int
) -> Workload:
    rng = np.random.default_rng(derive_seed(seed, 99))
    processes: List[Process] = []
    post_count = sum(1 for p in workload if p.arrival_time >= boundary)
    # Highest priority number in trace defines inversion range.
    max_prio = max(p.priority for p in workload) if workload.processes else 5
    min_prio = min(p.priority for p in workload) if workload.processes else 1
    for p in workload:
        if p.arrival_time < boundary:
            processes.append(p)
        else:
            if rng.random() < skew_fraction:
                # Invert: low priority becomes high and vice versa.
                inverted = (max_prio + min_prio) - p.priority
                # Clamp to valid range.
                inverted = max(min_prio, min(max_prio, inverted))
                processes.append(
                    Process(pid=p.pid, arrival_time=p.arrival_time, burst_time=p.burst_time, priority=inverted)
                )
            else:
                processes.append(p)
    return Workload(f"{workload.name}+{DRIFT_KIND_PRIORITY_INVERSION}", processes, description=DRIFT_KIND_PRIORITY_INVERSION)


def _apply_phase_transition(
    workload: Workload, boundary: int, target: Workload, seed: int
) -> Workload:
    # Pre-injection stays as-is; post-injection draws from target's burst/priority
    # distributions but re-uses the original arrival timeline (stretched/compressed
    # to post-boundary window) so pid count is preserved.
    pre = [p for p in workload if p.arrival_time < boundary]
    post_original = [p for p in workload if p.arrival_time >= boundary]
    if not post_original:
        return workload
    # Map target processes cyclically onto post-injection slots.
    target_bursts = [p.burst_time for p in target.processes]
    target_priorities = [p.priority for p in target.processes]
    if not target_bursts:
        return workload
    processes: List[Process] = list(pre)
    for idx, orig in enumerate(post_original):
        ti = idx % len(target_bursts)
        processes.append(
            Process(
                pid=orig.pid,
                arrival_time=orig.arrival_time,
                burst_time=target_bursts[ti],
                priority=target_priorities[ti],
            )
        )
    return Workload(f"{workload.name}+{DRIFT_KIND_PHASE_TRANSITION}", processes, description=DRIFT_KIND_PHASE_TRANSITION)


def generate_drift_workload(
    base_workload: Workload,
    kind: str,
    seed: int = 0,
    *,
    injection_fraction: float = DRIFT_INJECTION_FRACTION,
    target_workload: Optional[Workload] = None,
) -> Workload:
    """Convenience: generate a drifted workload from a base trace in one call.

    This is the recommended entry point for experiments that add mid-trace
    drift to synthetic traces.  It wraps ``DriftSpec`` construction so callers
    need only name the drift kind and a seed.
    """
    spec = DriftSpec(
        kind=kind,
        injection_fraction=injection_fraction,
        seed=seed,
        phase_target_family=target_workload.name if target_workload is not None else None,
        # For convenience allow None target for non-transition kinds.
    )
    # For non-transition kinds the phase_target_family must be None; adjust.
    if kind != DRIFT_KIND_PHASE_TRANSITION:
        spec = DriftSpec(kind=kind, injection_fraction=injection_fraction, seed=seed)
    else:
        if target_workload is None:
            raise ValidationError("phase_transition requires target_workload")
        spec = DriftSpec(
            kind=kind,
            injection_fraction=injection_fraction,
            seed=seed,
            phase_target_family=target_workload.name,
        )
    return apply_drift(base_workload, spec, target_workload=target_workload)
