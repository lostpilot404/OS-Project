"""Shared reward calculation for the healing controller and its audit probe.

The controller's learning reward is defined as the phase-adjusted verification
residual minus the declared acceptance boundary, minus an intervention cost
when a configuration actually changed, minus a rollback penalty only on a
confirmed failure. This module exposes that calculation as a single function so
the in-run controller and the off-line action-effect probe cannot drift apart.

It also exposes helpers for the closure statistic and for building the trigger
phase from an :class:`IncidentRecord`, which the probe needs to replay a
verification window faithfully without inspecting future information.
"""

from __future__ import annotations

import math
from typing import Optional

from errors import ValidationError
from healing.config import HealingConfig
from healing.verifier import TriggerPhase, VerificationEvidence, VerificationVerdict, closure_residual, expected_closure

__all__ = [
    "closure_fraction_from_loads",
    "phase_from_incident",
    "cost_in_band_units",
    "controller_reward",
]


def closure_fraction_from_loads(
    trigger_load: float,
    reference_load: float,
    post_load: Optional[float],
    load_band: float,
) -> Optional[float]:
    """Return the normalised load closure for one verification window.

    This mirrors ``SelfHealingController._load_closure``: the fraction of the
    gap between the queue level at the trigger and the trace's normal level
    that the window closed. The denominator is floored by the load dispersion
    band so a tiny gap cannot produce an enormous ratio.

    Args:
        trigger_load: Queue level observed at the trigger epoch.
        reference_load: Trace's normal queue level at the trigger.
        post_load: Mean queue level observed inside the verification window,
            or ``None`` when no interval was observed.
        load_band: Dispersion band of the load statistic at the trigger.

    Returns:
        The closure fraction, or ``None`` when no post-window load exists or
        the gap is degenerate.
    """
    if post_load is None:
        return None
    trigger = float(trigger_load)
    reference = float(reference_load)
    band = max(float(load_band), 1e-9)
    gap = max(trigger - reference, band)
    if gap <= 0.0:
        return None
    return float((trigger - float(post_load)) / gap)


def phase_from_incident(incident) -> TriggerPhase:
    """Build the declared trigger phase from an :class:`IncidentRecord`.

    All quantities are taken from the incident's recorded trigger epoch, so the
    reconstruction stays causal and cannot see the future.

    Args:
        incident: An :class:`healing.controller.IncidentRecord`.

    Returns:
        A :class:`TriggerPhase` suitable for :func:`expected_closure`.
    """
    return TriggerPhase(
        excess_band_units=float(incident.excess_band_units),
        relative_band=float(
            float(incident.band_at_trigger)
            / max(float(incident.reference_at_trigger), 1e-9)
        ),
        load_band=float(incident.load_band_at_trigger),
        trigger_load=float(incident.trigger_load_level),
        reference_load=float(incident.pre_load_level),
        ready_count=int(incident.ready_count_at_trigger),
        visible_work=float(incident.visible_remaining_work_at_trigger),
        decision_index=int(incident.decision_index),
        elapsed_time=int(incident.simulated_time),
    )


def cost_in_band_units(
    reference_wait: float,
    band: float,
    unit_cost: float,
) -> float:
    """Convert a waiting-time cost into dispersion-band units.

    Mirrors ``SelfHealingController._band_scaled``.

    Args:
        reference_wait: Stable reference waiting time at the trigger.
        band: Dispersion band at the trigger.
        unit_cost: Cost as a fraction of the reference waiting time
            (e.g. ``healing_config.intervention_cost``).

    Returns:
        Cost expressed in band units.
    """
    if unit_cost == 0.0:
        return 0.0
    return float(unit_cost * max(float(reference_wait), 1e-9) / max(float(band), 1e-9))


def controller_reward(
    *,
    residual: Optional[float],
    config: HealingConfig,
    cost_band_units: float = 0.0,
    rollback_penalty_band: float = 0.0,
    changed: bool = True,
    confirmed_failure: bool = False,
    held: bool = False,
) -> float:
    """Return the controller's learning reward for one closed window.

    This is the authoritative definition used by the live controller and by
    the replay probe. It deliberately mirrors ``SelfHealingController._conclude_pending``
    and ``SelfHealingController.end_episode``:

    * the reward is centred on the phase-adjusted residual (``0`` when no
      closure was observed, i.e. the declared neutral value);
    * the declared accept boundary is subtracted so a change must beat the
      certification threshold by more than its cost;
    * an intervention cost is charged only when the configuration actually
      changed, scaled into band units;
    * the rollback penalty is subtracted only when the verifier confirmed a failure
      (rejected), never for inconclusive evidence.

    Args:
        residual: Phase-adjusted closure residual of the window (``None``
            when the window produced no load evidence; treated as ``0``).
        config: Declared thresholds.
        cost_band_units: Intervention cost in band units (``0`` for a hold).
        rollback_penalty_band: Rollback penalty in the same units.
        changed: Whether the incident changed the live configuration.
        confirmed_failure: Whether the verifier's verdict was ``rejected``.
        held: Whether the incident was a hold (no change to verify). When
            ``True`` the cost is ignored even if ``changed`` is ``True``.

    Returns:
        The scalar reward.

    Raises:
        ValidationError: If ``config`` is not a :class:`HealingConfig`.
    """
    if not isinstance(config, HealingConfig):
        raise ValidationError("controller_reward() requires a HealingConfig")
    observed = 0.0 if residual is None else float(residual)
    baseline = float(config.accept_residual_factor)
    if held or not changed:
        # Held interventions carry no cost and no rollback penalty, but are
        # scored on the same scale so holding competes with acting.
        return float(observed - baseline)
    # Intervention: cost always charged when the configuration changed,
    # rollback penalty only on a confirmed failure.
    penalty = float(rollback_penalty_band) if confirmed_failure else 0.0
    return float(observed - baseline - float(cost_band_units) - penalty)
