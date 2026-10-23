"""Loadable demonstration scenarios for the live dashboard.

Each entry describes what the scenario is *for*, which engine path executes it and which
parameters the interface may change.  The catalogue is static metadata only: no numbers
here are results, and nothing in this module runs a simulation.  Expectations are stated
as intent ("verified accepted recovery") and are re-checked against the engine by
``tests/test_dashboard_api.py`` so a stale description cannot silently drift.

Scenario kinds
--------------
``teaching``
    The three-process worked example.  Executed by the real scheduler classes for all four
    policies and checked against the hand calculation in ``docs/TEACHER_LIVE_DEMO.md``.
``healing``
    One of the two documented self-healing fixtures, executed through
    :func:`experiments.demo.run_fixture` (controller + runtime simulator + metrics).
``custom``
    A client-supplied workload, executed by the same scheduler classes the simulator
    uses; self-healing can be switched on for the same workload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["SCENARIOS", "SCENARIO_BY_ID", "catalogue", "get_scenario", "normalise_options"]


#: Default editable workload for the custom scenario (five processes, arbitrary units).
DEFAULT_PROCESSES: Tuple[Dict[str, int], ...] = (
    {"pid": 1, "arrival_time": 0, "burst_time": 8, "priority": 2},
    {"pid": 2, "arrival_time": 1, "burst_time": 4, "priority": 1},
    {"pid": 3, "arrival_time": 2, "burst_time": 9, "priority": 3},
    {"pid": 4, "arrival_time": 3, "burst_time": 2, "priority": 4},
    {"pid": 5, "arrival_time": 4, "burst_time": 6, "priority": 2},
)

SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "teaching",
        "kind": "teaching",
        "name": "Small teaching workload",
        "tagline": "Three processes, four policies, arithmetic you can do on the board",
        "description": (
            "The documented three-process example (P1 arrival 0 / burst 5 / priority 2, "
            "P2 0/3/1, P3 0/1/3) executed by the real FCFS, SJF, Priority and Round Robin "
            "classes with quantum 2 and no switch cost. Every waiting and completion time "
            "is checked against the hand calculation in the teacher guide."
        ),
        "expectation": (
            "Hand-verified means: FCFS 4.33, SJF 1.67, Priority 3.67, Round Robin 4.33 "
            "simulated time units per process."
        ),
        "fixed_parameters": {"processes": 3, "quantum": 2, "switching_cost": 0},
        "options": ["policy_view"],
    },
    {
        "id": "recovery-accepted",
        "kind": "healing",
        "name": "Successful recovery (seed 13)",
        "tagline": "Degradation detected, SJF selected, observation window accepts it",
        "description": (
            "The 'mixed' generator workload at seed 13 under the waiting-time recovery "
            "rule. The controller detects degradation, switches FCFS to SJF and keeps SJF "
            "when the observation window closes with an accepted residual."
        ),
        "expectation": "Verification closes accepted; SJF stays in force to the end of the trace.",
        "fixed_parameters": {"family": "mixed", "seed": 13, "rule": "waiting-time rule"},
        "options": [],
    },
    {
        "id": "recovery-rollback",
        "kind": "healing",
        "name": "Inconclusive verification and rollback (seed 3)",
        "tagline": "The intervention is not proven, so the previous policy is restored",
        "description": (
            "The 'mixed' generator workload at seed 3 under the same waiting-time rule. "
            "Degradation is detected and SJF is selected, but the residual stays inside the "
            "declared null range, so the verdict is inconclusive and the controller rolls "
            "back to FCFS. Inconclusive is not the same as rejected."
        ),
        "expectation": (
            "Verification closes inconclusive; the trace rolls back to FCFS with no "
            "confirmed failure."
        ),
        "fixed_parameters": {"family": "mixed", "seed": 3, "rule": "waiting-time rule"},
        "options": [],
    },
    {
        "id": "custom",
        "kind": "custom",
        "name": "Custom workload",
        "tagline": "Your own processes, your own policy, executed by the real scheduler",
        "description": (
            "Edit the process table (pid, arrival, burst, priority), choose FCFS, SJF, "
            "Priority or Round Robin, set the round-robin quantum and the per-switch cost, "
            "and optionally run the same workload through the self-healing control loop."
        ),
        "expectation": "No fixed expectation: the result is whatever the engine computes for your input.",
        "fixed_parameters": {},
        "options": ["processes", "policy", "quantum", "switching_cost", "healing"],
        "defaults": {
            "processes": [dict(row) for row in DEFAULT_PROCESSES],
            "policy": "FCFS",
            "quantum": 4,
            "switching_cost": 0,
            "healing": False,
        },
        "limits": {"processes": [1, 40], "quantum": [1, 64], "switching_cost": [0, 10]},
    },
)

SCENARIO_BY_ID: Dict[str, Dict[str, Any]] = {scenario["id"]: scenario for scenario in SCENARIOS}


def catalogue() -> Tuple[Dict[str, Any], ...]:
    """Return the scenario list exactly as the API serves it (deep-copied, JSON-safe)."""
    import copy

    return tuple(copy.deepcopy(scenario) for scenario in SCENARIOS)


def get_scenario(scenario_id: str) -> Dict[str, Any]:
    """Return one scenario definition.

    Raises:
        KeyError: If ``scenario_id`` is not in the catalogue.
    """
    try:
        import copy

        return copy.deepcopy(SCENARIO_BY_ID[scenario_id])
    except KeyError:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; available: {', '.join(SCENARIO_BY_ID)}"
        ) from None


def normalise_options(scenario: Dict[str, Any], raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge client options with a scenario's defaults and refuse unsupported keys.

    Only options listed in ``scenario['options']`` are accepted, so a request cannot smuggle
    arbitrary keys into the engine call.

    Raises:
        ValueError: If an unsupported option is supplied.
    """
    options = raw or {}
    allowed = set(scenario["options"])
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"scenario {scenario['id']!r} does not accept option(s): {', '.join(unknown)}"
        )
    merged = dict(scenario.get("defaults", {}))
    merged.update(options)
    return merged
