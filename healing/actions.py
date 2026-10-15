"""Recovery action space of the self-healing scheduler.

A recovery action either leaves the configuration alone (``HOLD``) or changes the
scheduler configuration that the live event engine will use for its next dispatch. The
configuration is a pair ``(policy, quantum)``: the policy is one of FCFS, SJF,
Round Robin or Priority, and the quantum is the Round-Robin time slice (it only affects
Round-Robin service, matching :class:`scheduler.runtime.RuntimeActionDecision.quantum`).

The action space is deliberately small (8 actions) so a tabular controller can cover it
during episodic training and so every decision stays auditable. ``REVERT_TO_LAST_ACCEPTED``
lets a learned controller undo its own last accepted change.

Two kinds of requested action have no effect on the schedule and are therefore *collapsed
to HOLD* by :func:`apply_healing_action`, never counted and never charged as
interventions:

* requesting a policy or a reverted configuration that is already in force;
* changing only the Round-Robin quantum while a non-Round-Robin policy is in force, since
  the quantum is used by Round-Robin service alone.

Collapsing these is what keeps the recovery ledger and the false-recovery statistics
honest: an audit of "interventions" must not be diluted by actions that cannot change a
single scheduling decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from config import ACTION_NAMES, ACTION_ROUND_ROBIN
from errors import ValidationError

__all__ = [
    "HOLD",
    "SET_FCFS",
    "SET_SJF",
    "SET_ROUND_ROBIN",
    "SET_PRIORITY",
    "SHRINK_QUANTUM",
    "GROW_QUANTUM",
    "REVERT_TO_LAST_ACCEPTED",
    "HEALING_ACTION_NAMES",
    "CONFIGURATION_CHANGING_ACTIONS",
    "HealingActionOutcome",
    "SchedulerSettings",
    "apply_healing_action",
]

HOLD: int = 0
SET_FCFS: int = 1
SET_SJF: int = 2
SET_ROUND_ROBIN: int = 3
SET_PRIORITY: int = 4
SHRINK_QUANTUM: int = 5
GROW_QUANTUM: int = 6
REVERT_TO_LAST_ACCEPTED: int = 7

HEALING_ACTION_NAMES: Tuple[str, ...] = (
    "hold",
    "set_fcfs",
    "set_sjf",
    "set_round_robin",
    "set_priority",
    "shrink_quantum",
    "grow_quantum",
    "revert_to_last_accepted",
)

#: Actions that can change the live configuration. ``HOLD`` cannot; the others can, but
#: may still collapse to a hold when the requested configuration is already in force.
CONFIGURATION_CHANGING_ACTIONS = frozenset(range(1, len(HEALING_ACTION_NAMES)))

_POLICY_OF_ACTION = {
    SET_FCFS: 0,
    SET_SJF: 1,
    SET_ROUND_ROBIN: 2,
    SET_PRIORITY: 3,
}


@dataclass(frozen=True)
class SchedulerSettings:
    """The configuration the live event engine is currently using."""

    policy: int
    quantum: int

    def __post_init__(self) -> None:
        if isinstance(self.policy, bool) or self.policy not in range(len(ACTION_NAMES)):
            raise ValidationError(f"policy must be an index in 0..3, got {self.policy!r}")
        if isinstance(self.quantum, bool) or not isinstance(self.quantum, int) or self.quantum < 1:
            raise ValidationError(f"quantum must be an integer >= 1, got {self.quantum!r}")

    @property
    def policy_name(self) -> str:
        """Human-readable policy label."""
        return ACTION_NAMES[self.policy]

    def as_dict(self) -> dict:
        """Return ``{"policy", "policy_name", "quantum"}`` for artifacts."""
        return {
            "policy": self.policy,
            "policy_name": self.policy_name,
            "quantum": self.quantum,
        }


@dataclass(frozen=True)
class HealingActionOutcome:
    """Result of applying one recovery action to a configuration."""

    action: int
    settings: SchedulerSettings
    changed: bool
    note: str

    @property
    def action_name(self) -> str:
        """Human-readable action label."""
        return HEALING_ACTION_NAMES[self.action]


def apply_healing_action(
    action: int,
    *,
    current: SchedulerSettings,
    last_accepted: Optional[SchedulerSettings],
    min_quantum: int,
    max_quantum: int,
) -> HealingActionOutcome:
    """Apply one recovery action, returning the resulting configuration.

    Args:
        action: Index into :data:`HEALING_ACTION_NAMES`.
        current: Configuration in force before the action.
        last_accepted: Most recent configuration that passed verification, or ``None``
            when only the episode's initial configuration is known.
        min_quantum, max_quantum: Bounds for the quantum adaptation actions.

    Returns:
        A :class:`HealingActionOutcome` whose ``changed`` flag is ``True`` only when the
        configuration actually differs from ``current``. A requested but noneffective
        change is reported with a note and collapses to a hold in the caller, so it is
        never counted or charged as an intervention.

    Raises:
        ValidationError: If the action index or the quantum bounds are invalid.
    """
    if isinstance(action, bool) or not isinstance(action, int) or action not in range(
        len(HEALING_ACTION_NAMES)
    ):
        raise ValidationError(
            f"healing action must be an integer in 0..{len(HEALING_ACTION_NAMES) - 1}, got {action!r}"
        )
    if min_quantum < 1 or max_quantum < min_quantum:
        raise ValidationError(
            f"invalid quantum bounds: min={min_quantum}, max={max_quantum}"
        )
    if not isinstance(current, SchedulerSettings):
        raise ValidationError("current must be a SchedulerSettings")
    if last_accepted is not None and not isinstance(last_accepted, SchedulerSettings):
        raise ValidationError("last_accepted must be a SchedulerSettings or None")

    if action == HOLD:
        return HealingActionOutcome(HOLD, current, False, "hold: configuration unchanged")

    if action in _POLICY_OF_ACTION:
        target = SchedulerSettings(policy=_POLICY_OF_ACTION[action], quantum=current.quantum)
        if target == current:
            return HealingActionOutcome(
                action, current, False, "requested policy already in force"
            )
        return HealingActionOutcome(action, target, True, "policy changed")

    if action == SHRINK_QUANTUM:
        if current.policy != ACTION_ROUND_ROBIN:
            return HealingActionOutcome(
                action, current, False,
                "quantum only affects Round-Robin service; current policy is "
                f"{current.policy_name}",
            )
        quantum = max(min_quantum, current.quantum // 2)
        if quantum == current.quantum:
            return HealingActionOutcome(
                action, current, False, "quantum already at its minimum"
            )
        return HealingActionOutcome(
            action, SchedulerSettings(policy=current.policy, quantum=quantum), True,
            "quantum halved under Round Robin",
        )

    if action == GROW_QUANTUM:
        if current.policy != ACTION_ROUND_ROBIN:
            return HealingActionOutcome(
                action, current, False,
                "quantum only affects Round-Robin service; current policy is "
                f"{current.policy_name}",
            )
        quantum = min(max_quantum, current.quantum * 2)
        if quantum == current.quantum:
            return HealingActionOutcome(
                action, current, False, "quantum already at its maximum"
            )
        return HealingActionOutcome(
            action, SchedulerSettings(policy=current.policy, quantum=quantum), True,
            "quantum doubled under Round Robin",
        )

    # REVERT_TO_LAST_ACCEPTED
    if last_accepted is None:
        return HealingActionOutcome(
            action, current, False, "no previously accepted configuration to revert to"
        )
    if last_accepted == current:
        return HealingActionOutcome(
            action, current, False, "current configuration is already the last accepted one"
        )
    return HealingActionOutcome(action, last_accepted, True, "reverted to last accepted")
