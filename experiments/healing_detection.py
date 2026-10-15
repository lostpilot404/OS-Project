"""Controlled degradation scenarios for the detector: recall, delay and false triggers.

The detection thresholds are derived from the *no-action control* in
:mod:`experiments.healing_thresholds`. That tells us how often the loop declares a
degradation event when nothing was done and no workload change was injected -- an upper
bound on the false-trigger rate. It does **not** tell us whether the detector reacts to a
degradation that is actually there.

This module supplies the missing half: it takes real workloads from a declared split,
injects a step change into a deterministic fraction of them, and runs the *hold-only*
control (the full monitor/detect/diagnose loop with a policy that never intervenes) on both
versions of every trace. Because nothing is ever changed by that arm, every detection it
declares is attributable to the workload itself, never to a configuration change.

Declared scenarios
------------------
``stable``
    The workload as generated. Any detection here is a false trigger by construction.
``burst_shock``
    From ``injection_time`` every newly arriving process needs ``factor`` times its original
    CPU time. The queue therefore grows and waiting genuinely degrades.
``arrival_storm``
    From ``injection_time`` all later arrivals are compressed towards the injection instant,
    so the same work arrives in a much shorter window and the ready queue backs up.

Reported quantities
-------------------
* **recall** -- share of injected traces where an incident is declared at or after the
  injection instant (and before the trace ends), per scenario and family.
* **detection delay** -- simulated time and decisions between the injection instant and the
  declaration.
* **false-trigger rate** -- incidents per stable trace, and the share of stable traces that
  trigger at least once.
* **precision proxy** -- detections on injected traces divided by all detections. It is a
  proxy, not a real precision: a stable trace *can* legitimately degrade after the injection
  point in the injected variant only, and the injection is a workload property, not a
  ground-truth label of the detector's target event.

The measurement is a characterisation, not a competition: it is run on the validation split
(never on the final-test split), and it is reported with its limitations in
``docs/EVALUATION_RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ACTION_FCFS
from errors import ValidationError
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from healing.policies import HoldOnlyPolicy
from scheduler import RuntimeSimulator
from workload.models import Process, Workload

__all__ = [
    "SCENARIOS",
    "INJECTION_MULTIPLIERS",
    "DetectionScenario",
    "scenario_rows",
    "run_detection_scenarios",
]

#: Scenario names, in report order.
SCENARIOS: Tuple[str, ...] = ("stable", "burst_shock", "arrival_storm")

#: Declared multipliers of the two injections.
INJECTION_MULTIPLIERS: Dict[str, float] = {"burst_shock": 4.0, "arrival_storm": 0.35}

#: Fraction of a trace's arrival span at which the step change is injected. Later injections
#: leave too little remaining trace for a detector to react, earlier ones make the "before"
#: period too short to establish a reference.
INJECTION_FRACTION = 0.35


@dataclass(frozen=True)
class DetectionScenario:
    """One (workload, scenario) pair and what the hold-only control observed on it."""

    family: str
    repetition: int
    scenario: str
    workload_fingerprint: str
    num_processes: int
    injection_time: int
    last_arrival_time: int
    decisions: int
    incidents: int
    interventions: int
    first_detection_time: Optional[int]
    first_detection_decision: Optional[int]
    detection_delay_time: Optional[int]
    detection_delay_decisions: Optional[int]
    detected: bool
    total_waiting_time: float
    detector_reason: str

    def as_row(self) -> Dict[str, Any]:
        """Flat dictionary for the scenario artifact."""
        return {
            "family": self.family,
            "repetition": self.repetition,
            "scenario": self.scenario,
            "workload_fingerprint": self.workload_fingerprint,
            "num_processes": self.num_processes,
            "injection_time": self.injection_time,
            "last_arrival_time": self.last_arrival_time,
            "decisions": self.decisions,
            "incidents": self.incidents,
            "interventions": self.interventions,
            "detected": self.detected,
            "first_detection_time": self.first_detection_time,
            "first_detection_decision": self.first_detection_decision,
            "detection_delay_time": self.detection_delay_time,
            "detection_delay_decisions": self.detection_delay_decisions,
            "total_waiting_time": self.total_waiting_time,
            "detector_reason": self.detector_reason,
        }


def _injection_time(workload: Workload) -> int:
    """Return the declared injection instant of one workload."""
    arrivals = [process.arrival_time for process in workload]
    if not arrivals:
        raise ValidationError("scenario workloads must not be empty")
    span = max(arrivals) - min(arrivals)
    return int(min(arrivals) + round(INJECTION_FRACTION * span))


def inject_scenario(workload: Workload, scenario: str) -> Workload:
    """Return the injected variant of one workload, deterministically.

    Args:
        workload: The base workload (``stable`` scenario returns it unchanged).
        scenario: One of :data:`SCENARIOS`.

    Returns:
        A new :class:`workload.models.Workload` sharing pids and priorities. Processes that
        arrive before the injection instant are untouched, so the pre-injection behaviour of
        every scenario is identical to the stable trace.

    Raises:
        ValidationError: If ``scenario`` is unknown.
    """
    if scenario not in SCENARIOS:
        raise ValidationError(f"unknown detection scenario {scenario!r}")
    if scenario == "stable":
        return workload
    boundary = _injection_time(workload)
    factor = INJECTION_MULTIPLIERS[scenario]
    processes: List[Process] = []
    if scenario == "burst_shock":
        for process in workload:
            if process.arrival_time < boundary:
                processes.append(process)
            else:
                processes.append(
                    Process(
                        pid=process.pid,
                        arrival_time=process.arrival_time,
                        burst_time=max(1, int(round(process.burst_time * factor))),
                        priority=process.priority,
                    )
                )
    else:  # arrival_storm
        for process in workload:
            if process.arrival_time < boundary:
                processes.append(process)
            else:
                compressed = boundary + int(
                    round((process.arrival_time - boundary) * factor)
                )
                processes.append(
                    Process(
                        pid=process.pid,
                        arrival_time=compressed,
                        burst_time=process.burst_time,
                        priority=process.priority,
                    )
                )
    return Workload(f"{workload.name}+{scenario}", processes, description=scenario)


def run_detection_scenarios(
    workloads: Sequence[Any],
    *,
    config,
    healing: Optional[HealingConfig] = None,
    scenarios: Sequence[str] = SCENARIOS,
) -> List[DetectionScenario]:
    """Run the hold-only control on every (workload, scenario) pair.

    Args:
        workloads: The split's workloads (``experiments.healing_experiment.HealingWorkload``).
        config: The study configuration (scheduler plus declared healing thresholds).
        healing: Optional threshold override (defaults to the declared configuration).
        scenarios: Scenarios to run, in order.

    Returns:
        One :class:`DetectionScenario` per (workload, scenario) pair.
    """
    thresholds = healing or config.healing
    simulator = RuntimeSimulator(config.scheduler)
    rows: List[DetectionScenario] = []
    for item in workloads:
        boundary = _injection_time(item.workload)
        last_arrival = max(process.arrival_time for process in item.workload)
        for scenario in scenarios:
            workload = inject_scenario(item.workload, scenario)
            controller = SelfHealingController(
                thresholds,
                config.scheduler,
                HoldOnlyPolicy(),
                initial_policy=ACTION_FCFS,
                label=f"detection-{scenario}",
            )
            controller.begin_episode(0)
            run = simulator.run(workload, controller, training=False, policy_name="detection")
            episode = controller.end_episode()
            relevant = [
                incident
                for incident in episode.incidents
                if scenario == "stable" or incident.simulated_time >= boundary
            ]
            first = relevant[0] if relevant else None
            detected = first is not None
            injection_decision = next(
                (
                    record.decision_index
                    for record in run.decisions
                    if record.end_time >= boundary
                ),
                len(run.decisions),
            )
            rows.append(
                DetectionScenario(
                    family=item.family,
                    repetition=item.repetition,
                    scenario=scenario,
                    workload_fingerprint=workload.fingerprint,
                    num_processes=len(workload),
                    injection_time=boundary,
                    last_arrival_time=last_arrival,
                    decisions=episode.decisions,
                    incidents=len(relevant),
                    interventions=episode.interventions,
                    first_detection_time=(first.simulated_time if first else None),
                    first_detection_decision=(first.decision_index if first else None),
                    detection_delay_time=(
                        (first.simulated_time - boundary) if first else None
                    ),
                    detection_delay_decisions=(
                        (first.decision_index - injection_decision) if first else None
                    ),
                    detected=detected,
                    total_waiting_time=float(
                        sum(outcome.waiting_time for outcome in run.result.outcomes)
                    ),
                    detector_reason=(first.detector_reason if first else "none"),
                )
            )
    return rows


def scenario_rows(scenarios: pd.DataFrame) -> List[Dict[str, Any]]:
    """Summarise the scenario artifact into the study's reported detection statistics."""
    if scenarios.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for scenario, subset in scenarios.groupby("scenario", sort=False):
        stable = scenario == "stable"
        detections = subset["detected"].astype(bool)
        delays = subset["detection_delay_time"].dropna().astype(float)
        rows.append(
            {
                "scenario": scenario,
                "traces": int(len(subset)),
                "traces_with_detection": int(detections.sum()),
                "detection_share": float(detections.mean()),
                "incidents_per_trace": float(subset["incidents"].astype(float).mean()),
                "median_delay_time_units": float(np.median(delays)) if len(delays) else float("nan"),
                "median_delay_time_units_normalised": (
                    float(
                        np.median(delays)
                        / max(1.0, subset["last_arrival_time"].astype(float).mean())
                    )
                    if len(delays)
                    else float("nan")
                ),
                "interventions": int(subset["interventions"].astype(float).sum()),
                "mean_total_waiting_time": float(
                    subset["total_waiting_time"].astype(float).mean()
                ),
                "kind": "false-trigger probe" if stable else "injected degradation",
            }
        )
    injected = scenarios[scenarios["scenario"] != "stable"]
    stable_frame = scenarios[scenarios["scenario"] == "stable"]
    total_detections = int(scenarios["detected"].astype(bool).sum())
    injected_detections = int(injected["detected"].astype(bool).sum())
    delayed = injected[injected["detected"].astype(bool)]
    rows.append(
        {
            "scenario": "all",
            "traces": int(len(scenarios)),
            "traces_with_detection": total_detections,
            "detection_share": float(scenarios["detected"].astype(bool).mean()),
            "incidents_per_trace": float(scenarios["incidents"].astype(float).mean()),
            # Share of all declared detections that happened on an injected trace: a proxy
            # for precision, because the injection is a workload property and not a label of
            # the event the detector is looking for.
            "precision_proxy": (
                float(injected_detections / total_detections) if total_detections else float("nan")
            ),
            "recall_on_injected": (
                float(injected["detected"].astype(bool).mean()) if len(injected) else float("nan")
            ),
            "false_trigger_rate_per_stable_trace": (
                float(stable_frame["incidents"].astype(float).mean())
                if len(stable_frame)
                else float("nan")
            ),
            "median_detection_delay_decisions": (
                float(np.median(delayed["detection_delay_decisions"].dropna().astype(float)))
                if len(delayed)
                else float("nan")
            ),
            "median_detection_delay_time_units": (
                float(np.median(delayed["detection_delay_time"].dropna().astype(float)))
                if len(delayed)
                else float("nan")
            ),
            "median_delay_time_units": (
                float(np.median(delayed["detection_delay_time"].dropna().astype(float)))
                if len(delayed)
                else float("nan")
            ),
            "median_delay_time_units_normalised": (
                float(
                    np.median(delayed["detection_delay_time"].dropna().astype(float))
                    / max(1.0, scenarios["last_arrival_time"].astype(float).mean())
                )
                if len(delayed)
                else float("nan")
            ),
            "interventions": int(scenarios["interventions"].astype(float).sum()),
            "mean_total_waiting_time": float(
                scenarios["total_waiting_time"].astype(float).mean()
            ),
            "kind": "summary",
        }
    )
    return rows
