"""Engine-facing service layer behind the live dashboard API.

The dashboard is a *consumer*: every number in a response is produced here by calling the
project's own scheduler, runtime simulator, healing controller and metric functions.  This
module adds no scheduling, detection, recovery or verification logic -- it validates a
request, calls the engine, and shapes the recorded output for the browser.

What is reused
--------------
* :func:`experiments.demo.run_fixture` / :func:`experiments.demo.run_healing_workload`
  for the two documented fixtures and for custom workloads with self-healing enabled.
* ``scheduler.POLICY_CLASSES`` for FCFS/SJF/Priority/Round Robin, including the policy
  comparison endpoint.
* :func:`evaluation.metrics.compute_metrics` for every metric, with the project's own
  definitions (waiting = completion - arrival - burst, and so on).
* :mod:`dashboard.model` for schema validation of engine output, :mod:`dashboard.charts`
  for the SVG charts, :mod:`dashboard.render` for the lifecycle stages and
  :mod:`dashboard.research` for the committed study tables.

Isolation
---------
Nothing here writes to the repository.  Runtime artifacts are held by the API's run
registry in a per-run temporary directory; research artifacts are opened read-only.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import SchedulerConfig
from evaluation.metrics import compute_metrics
from experiments.demo import run_fixture, run_healing_workload, with_scheduler
from experiments.healing_config import HealingExperimentConfig
from healing.policies import HealingContextEncoder
from scheduler import POLICY_CLASSES, SchedulingPolicy
from workload.models import Process, Workload

from dashboard import scenarios as scenario_catalogue
from dashboard.charts import (
    BarItem,
    GanttRow,
    Marker,
    gantt_chart,
    horizontal_bars,
    lifecycle_flow,
    research_bars,
    verification_band,
)
from dashboard.model import (
    DashboardError,
    Fixture,
    Incident,
    Interval,
    Metrics,
    Outcome,
    Process as RecordedProcess,
    parse_fixture,
)
from dashboard.render import lifecycle_stages
from dashboard.research import load_research
from dashboard.teaching import build_teaching_panel

__all__ = [
    "SimulationError",
    "REPOSITORY_ROOT",
    "RESEARCH_SUMMARY_PATH",
    "engine_info",
    "execute",
    "compare_policies",
    "research_panel",
    "describe_workload",
    "POLICY_IDS",
]

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_SUMMARY_PATH = REPOSITORY_ROOT / "results" / "healing" / "healing_summary.json"

#: Scheduler policy identifiers accepted by the API, mapped to the engine's names.
POLICY_IDS: Dict[str, str] = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "priority": "Priority",
    "round_robin": "Round Robin",
}

#: Hard bounds on a custom request, so one browser click cannot hang the service.
MAX_PROCESSES = 40
MAX_QUANTUM = 64
MAX_SWITCH_COST = 10
MAX_BURST = 10_000
MAX_ARRIVAL = 100_000

#: Number of cells in the healing controller's tabular state space. Derived from the
#: encoder's own bin definition, so the interface cannot print a stale "/648".
STATE_SPACE_SIZE: int = math.prod(HealingContextEncoder.RADICES)


class SimulationError(ValueError):
    """A request that the engine cannot execute, with a message meant for a human."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


# --------------------------------------------------------------------------------------
# Small serialisation helpers
# --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _duration_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _metrics_view(metrics: Any) -> Dict[str, Any]:
    """Return one canonical metrics mapping for either an engine or a dashboard object."""
    if isinstance(metrics, Metrics):
        source: Mapping[str, Any] = asdict(metrics)
    elif hasattr(metrics, "as_dict"):
        source = metrics.as_dict()
    else:
        source = dict(metrics)
    keys = (
        "policy_name",
        "workload_name",
        "workload_fingerprint",
        "num_processes",
        "total_waiting_time",
        "total_turnaround_time",
        "total_response_time",
        "avg_waiting_time",
        "avg_turnaround_time",
        "avg_response_time",
        "cpu_utilization",
        "throughput",
        "context_switches",
        "context_switches_per_process",
        "cpu_busy_time",
        "cpu_idle_time",
        "total_elapsed_time",
    )
    view = {key: source.get(key) for key in keys}
    for key in ("avg_waiting_time", "avg_turnaround_time", "avg_response_time",
                "cpu_utilization", "throughput", "context_switches_per_process"):
        value = view.get(key)
        if isinstance(value, float):
            view[key] = round(value, 6)
    return view


def _interval_view(interval: Interval) -> Dict[str, Any]:
    return {
        "index": interval.index,
        "pid": interval.pid,
        "policy": interval.policy,
        "start": interval.start,
        "end": interval.end,
        "duration": interval.duration,
        "switch_before": interval.switch_before,
        "idle_before": interval.idle_before,
        "waiting_increment": interval.waiting_increment,
        "completed_pids": list(interval.completed_pids),
    }


def _outcome_rows(fixture: Fixture) -> List[Dict[str, Any]]:
    return [
        {
            "pid": outcome.pid,
            "arrival_time": outcome.arrival_time,
            "burst_time": outcome.burst_time,
            "priority": outcome.priority,
            "start_time": outcome.start_time,
            "completion_time": outcome.completion_time,
            "waiting_time": outcome.waiting_time,
            "turnaround_time": outcome.turnaround_time,
            "response_time": outcome.response_time,
        }
        for outcome in sorted(fixture.outcomes, key=lambda row: row.pid)
    ]


def _gantt_rows(outcomes: Sequence[Any]) -> List[GanttRow]:
    return [
        GanttRow(
            pid=row.pid,
            arrival_time=row.arrival_time,
            completion_time=row.completion_time,
            burst_time=row.burst_time,
            priority=row.priority,
            waiting_time=row.waiting_time,
        )
        for row in sorted(outcomes, key=lambda row: row.pid)
    ]


def _segments(intervals: Sequence[Interval]) -> List[Dict[str, Any]]:
    """Merge consecutive intervals that ran under the same policy."""
    merged: List[Dict[str, Any]] = []
    for interval in intervals:
        if merged and merged[-1]["policy"] == interval.policy:
            merged[-1]["end"] = interval.end
            merged[-1]["decisions"] += 1
        else:
            merged.append(
                {
                    "policy": interval.policy,
                    "start": interval.start,
                    "end": interval.end,
                    "decisions": 1,
                }
            )
    return merged


def _schedule_view(intervals: Sequence[Interval], domain_end: int) -> Dict[str, Any]:
    return {
        "intervals": [_interval_view(interval) for interval in intervals],
        "segments": _segments(intervals),
        "domain_end": domain_end,
        "initial_policy": intervals[0].policy if intervals else "n/a",
        "final_policy": intervals[-1].policy if intervals else "n/a",
        "idle_units": sum(interval.idle_before for interval in intervals),
        "switch_units": sum(interval.switch_before for interval in intervals),
    }


def _gantt_svg(
    prefix: str,
    fixture: Fixture,
    *,
    seed: int,
    show_policy_row: bool = True,
    width: int = 1080,
) -> str:
    """Draw the recorded intervals of one fixture with the shared Gantt renderer."""
    incident = fixture.primary_incident
    markers: List[Marker] = []
    if incident is not None:
        markers.append(
            Marker(
                incident.simulated_time,
                f"degradation detected at t={incident.simulated_time}",
            )
        )
        if incident.window_end_time is not None:
            label = (
                f"rollback at t={incident.window_end_time}"
                if incident.rollback_target_policy
                else f"verification closes at t={incident.window_end_time}"
            )
            markers.append(Marker(incident.window_end_time, label, "#64748b"))
    return gantt_chart(
        name=f"{prefix}-gantt",
        title=f"CPU execution timeline ({fixture.family}, seed {seed})",
        description=(
            "Recorded dispatch intervals: the CPU lane shows the exact service sequence, the "
            "lower lanes show each process's own slices, and the colour identifies the policy "
            "in force."
        ),
        rows=_gantt_rows(fixture.outcomes),
        intervals=fixture.intervals,
        segments=fixture.policy_segments,
        domain_end=fixture.domain_end,
        markers=tuple(markers),
        show_policy_row=show_policy_row,
        width=width,
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# Scenario execution
# --------------------------------------------------------------------------------------


def _scenario_view(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": scenario["id"],
        "kind": scenario["kind"],
        "name": scenario["name"],
        "tagline": scenario.get("tagline", ""),
        "description": scenario.get("description", ""),
        "expectation": scenario.get("expectation", ""),
    }


def _workload_view(
    fixture: Fixture, family: str, seed: Optional[int], processes: Sequence[Process]
) -> Dict[str, Any]:
    return {
        "name": fixture.metrics.workload_name,
        "family": family,
        "seed": seed,
        "fingerprint": fixture.fingerprint,
        "process_count": len(processes),
        "processes": [
            {
                "pid": process.pid,
                "arrival_time": process.arrival_time,
                "burst_time": process.burst_time,
                "priority": process.priority,
            }
            for process in processes
        ],
    }


def _healing_view(fixture: Fixture) -> Optional[Dict[str, Any]]:
    """Lifecycle, incident and verdict material for a fixture that ran the control loop."""
    incident = fixture.primary_incident
    if incident is None:
        return {
            "triggered": False,
            "summary": (
                "The controller ran for this workload but declared no incident: the monitored "
                "waiting time never crossed its threshold, so no policy change was attempted."
            ),
            "stages": [],
            "final_text": "No intervention; the initial policy ran to the end of the trace.",
            "incident": None,
        }
    stages, final_text = lifecycle_stages(fixture)
    return {
        "triggered": True,
        "summary": _verdict_summary(incident),
        "stages": [
            {
                "title": stage.title,
                "lines": list(stage.lines),
                "tone": stage.tone,
                # Semantic tone of the stage ("ok", "warn", "bad", "muted", "accent"). The
                # interface colours each stage from this key rather than by re-reading a hex
                # value or matching the title, so an inconclusive rollback can never be
                # painted as a success.
                "tone_key": stage.tone_key,
                "detail": stage.detail,
            }
            for stage in stages
        ],
        "final_text": final_text,
        "incident": _incident_view(incident),
    }


def _verdict_summary(incident: Incident) -> str:
    """One sentence built only from the recorded verdict, reason and residual."""
    residual = (
        f"{incident.closure_residual:+.4f}" if incident.closure_residual is not None else "n/a"
    )
    if incident.verdict == "held":
        # The detector fired but the selected action left the configuration unchanged, so
        # no window existed to judge. Calling this "inconclusive" would claim evidence that
        # was never collected.
        return (
            f"No configuration change was made ({incident.verifier_reason}): the selected "
            f"action {incident.action_name} left {incident.settings_before_policy} in force, "
            "so there was nothing to verify and the trace continued unchanged."
        )
    if incident.verdict == "accepted":
        return (
            f"Verification closed accepted ({incident.verifier_reason}): the residual {residual} "
            f"reached the accept boundary {incident.accept_residual_factor:+.2f}, so "
            f"{incident.settings_after_policy} stayed in force."
        )
    if incident.verdict == "rejected":
        return (
            f"Verification closed rejected ({incident.verifier_reason}): the residual {residual} "
            f"fell to or below the reject boundary {incident.reject_residual_factor:+.2f}, the "
            f"change failed and the configuration was rolled back."
        )
    return (
        f"Verification closed inconclusive ({incident.verifier_reason}): the residual {residual} "
        f"stayed inside the null range between {incident.reject_residual_factor:+.2f} and "
        f"{incident.accept_residual_factor:+.2f}. The change is neither proven nor rejected, so "
        + (
            f"the controller rolled back to {incident.rollback_target_policy}."
            if incident.rollback_target_policy
            else "the configuration was kept."
        )
    )


def _incident_view(incident: Incident) -> Dict[str, Any]:
    """Recorded incident fields the interface displays, with no derived claims."""
    return {
        "decision_index": incident.decision_index,
        "simulated_time": incident.simulated_time,
        "window_end_time": incident.window_end_time,
        "detector_reason": incident.detector_reason,
        "suspicion_evaluations": incident.suspicion_evaluations,
        "trigger_threshold": incident.trigger_threshold,
        "reference_at_trigger": incident.reference_at_trigger,
        "band_at_trigger": incident.band_at_trigger,
        "window_mean_at_trigger": incident.window_mean_at_trigger,
        "window_size_at_trigger": incident.window_size_at_trigger,
        "window_median_at_trigger": incident.window_median_at_trigger,
        "window_max_at_trigger": incident.window_max_at_trigger,
        "excess_band_units": incident.excess_band_units,
        "ready_count_at_trigger": incident.ready_count_at_trigger,
        "visible_remaining_work_at_trigger": incident.visible_remaining_work_at_trigger,
        "diagnosis": incident.diagnosis,
        "diagnosis_statement": incident.diagnosis_statement,
        "diagnosis_matched": incident.diagnosis_matched,
        "context_state_index": incident.context_state_index,
        # Size of the tabular context space, read from the encoder that produced the index
        # rather than typed into the interface.
        "state_space_size": STATE_SPACE_SIZE,
        "feature_bins": list(incident.feature_bins),
        "offered_actions": incident.offered_actions,
        "action": incident.action,
        "action_name": incident.action_name,
        "action_changed": incident.action_changed,
        "action_note": incident.action_note,
        "request_note": incident.request_note,
        "settings_before_policy": incident.settings_before_policy,
        "settings_before_quantum": incident.settings_before_quantum,
        "settings_after_policy": incident.settings_after_policy,
        "settings_after_quantum": incident.settings_after_quantum,
        "verdict": incident.verdict,
        "verifier_reason": incident.verifier_reason,
        "confirmed_failure": incident.confirmed_failure,
        "consecutive_failures": incident.consecutive_failures,
        "window_truncated": incident.window_truncated,
        "verification_decisions": incident.verification_decisions,
        "verification_completions": incident.verification_completions,
        "closure_fraction": incident.closure_fraction,
        "expected_closure": incident.expected_closure,
        "closure_residual": incident.closure_residual,
        "accept_residual_factor": incident.accept_residual_factor,
        "reject_residual_factor": incident.reject_residual_factor,
        "post_window_mean": incident.post_window_mean,
        "improvement": incident.improvement,
        "reward": incident.reward,
        "rollback_target_policy": incident.rollback_target_policy,
        "rollback_target_quantum": incident.rollback_target_quantum,
        "switch_overhead_units": incident.switch_overhead_units,
        "fallback_triggered": incident.fallback_triggered,
    }


def _healing_charts(prefix: str, fixture: Fixture, seed: int) -> Dict[str, str]:
    incident = fixture.primary_incident
    if incident is None:
        return {}
    stages, _ = lifecycle_stages(fixture)
    return {
        "lifecycle": lifecycle_flow(
            name=f"{prefix}-lifecycle",
            title=f"Self-healing lifecycle (seed {seed})",
            description=(
                "The recorded control-loop stages: normal scheduling, monitoring, degradation "
                "detection, diagnosis, selected intervention, observation, verification and the "
                "final decision."
            ),
            stages=stages,
        ),
        "verification": verification_band(
            name=f"{prefix}-band",
            title=f"Verification residual against the declared boundaries (seed {seed})",
            description=(
                "The accept and reject boundaries declared by the healing configuration; the "
                "marker is the residual recorded for this incident."
            ),
            incident=incident,
        ),
    }


def _waiting_comparison_chart(
    prefix: str,
    label_a: str,
    value_a: float,
    label_b: str,
    value_b: float,
    colour_a: str,
    colour_b: str,
) -> str:
    return horizontal_bars(
        name=f"{prefix}-waiting",
        title="Mean waiting time: self-healing run versus FCFS on the same workload",
        description=(
            "Mean waiting time per process of the self-healing trace and of a plain FCFS run on "
            "the identical workload."
        ),
        items=(
            BarItem(label=label_a, value=value_a, colour=colour_a, strong=True),
            BarItem(label=label_b, value=value_b, colour=colour_b),
        ),
        unit="Mean waiting time (simulated time units per process)",
        width=720,
    )


def _run_healing_scenario(scenario: Mapping[str, Any], options: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    seed = int(scenario["fixed_parameters"]["seed"])
    family = str(scenario["fixed_parameters"]["family"])
    config = HealingExperimentConfig()
    run = run_fixture(seed, config, family)
    fixture = parse_fixture(run.payload, source=f"live engine (seed {seed})")
    domain = fixture.domain_end
    charts = {"gantt": _gantt_svg("healing", fixture, seed=seed)}
    charts.update(_healing_charts("healing", fixture, seed))
    charts["waiting"] = _waiting_comparison_chart(
        "healing",
        "Self-healing trace",
        fixture.metrics.avg_waiting_time,
        "FCFS (same workload)",
        fixture.fcfs_metrics.avg_waiting_time,
        "#6b4a86",
        "#3d5a80",
    )
    return {
        "kind": "healing",
        "scenario": _scenario_view(scenario),
        "executed_at": _now(),
        "duration_ms": _duration_ms(started),
        "settings": {
            "family": family,
            "seed": seed,
            "quantum": config.scheduler.round_robin_quantum,
            "switching_cost": config.scheduler.switching_cost,
            "healing": True,
            "recovery_rule": "waiting-time rule",
        },
        "workload": _workload_view(fixture, family, seed, [p for p in _fixture_processes(fixture)]),
        "schedule": _schedule_view(fixture.intervals, domain),
        "metrics": _metrics_view(fixture.metrics),
        "baseline": _metrics_view(fixture.fcfs_metrics),
        "outcomes": _outcome_rows(fixture),
        "healing": _healing_view(fixture),
        "charts": charts,
        "notes": [
            "Executed now by experiments.demo.run_fixture (controller, runtime simulator and "
            "metrics); the payload is validated against the dashboard schema before display.",
            "Times are simulated time units, not seconds.",
            "Illustrative fixture, not a held-out study result.",
        ],
    }


def _fixture_processes(fixture: Fixture) -> Tuple[Any, ...]:
    """Process inputs of a fixture, in workload order."""
    return tuple(fixture.processes)


def _run_teaching_scenario(scenario: Mapping[str, Any], options: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    panel = build_teaching_panel()
    names = {policy.name: policy for policy in panel.policies}
    selected = options.get("policy_view") or "FCFS"
    if selected not in names:
        raise SimulationError(
            f"unknown policy {selected!r} for the teaching example; choose one of "
            f"{', '.join(names)}",
            field="policy_view",
        )
    chosen = names[selected]
    fixtures = _teaching_fixtures(panel)
    fixture = fixtures[selected]
    charts = {
        "gantt": _gantt_svg("teaching", fixture, seed=0, show_policy_row=False, width=900),
        "waiting": horizontal_bars(
            name="teaching-waiting",
            title="Mean waiting time of the four policies on the same three processes",
            description="Each bar is one real execution of the documented example.",
            items=tuple(
                BarItem(
                    label=policy.name,
                    value=policy.mean_waiting_time,
                    colour={
                        "FCFS": "#3d5a80",
                        "SJF": "#2f6f4f",
                        "Round Robin": "#9a6b1f",
                        "Priority": "#6b4a86",
                    }[policy.name],
                    note=f"mean turnaround {policy.mean_turnaround_time:.2f}",
                    strong=policy.name == panel.best_by_waiting,
                )
                for policy in panel.policies
            ),
            unit="Mean waiting time (simulated time units per process)",
            width=760,
        ),
    }
    variants = {
        policy.name: _gantt_svg(
            f"teaching-{index}", fixtures[policy.name], seed=0, show_policy_row=False, width=900
        )
        for index, policy in enumerate(panel.policies)
    }
    return {
        "kind": "teaching",
        "scenario": _scenario_view(scenario),
        "executed_at": _now(),
        "duration_ms": _duration_ms(started),
        "settings": {
            "quantum": panel.quantum,
            "switching_cost": panel.switching_cost,
            "policy": selected,
            "healing": False,
        },
        "workload": {
            "name": "teacher-hand-example",
            "family": "teaching",
            "seed": None,
            "fingerprint": _workload_fingerprint(panel),
            "process_count": len(panel.processes),
            "processes": [
                {
                    "pid": process.pid,
                    "arrival_time": process.arrival_time,
                    "burst_time": process.burst_time,
                    "priority": process.priority,
                }
                for process in panel.processes
            ],
        },
        "schedule": _schedule_view(chosen.intervals, chosen.domain_end),
        "metrics": _metrics_view(_teaching_metrics(chosen)),
        "baseline": _metrics_view(_teaching_metrics(names["FCFS"])),
        "outcomes": [
            {
                "pid": outcome.pid,
                "arrival_time": outcome.arrival_time,
                "burst_time": outcome.burst_time,
                "priority": outcome.priority,
                "start_time": outcome.start_time,
                "completion_time": outcome.completion_time,
                "waiting_time": outcome.waiting_time,
                "turnaround_time": outcome.turnaround_time,
                "response_time": outcome.response_time,
            }
            for outcome in sorted(names[selected].outcomes, key=lambda row: row.pid)
        ],
        "healing": None,
        "teaching": {
            "quantum": panel.quantum,
            "switching_cost": panel.switching_cost,
            "hand_verified": panel.hand_verified,
            "best_by_waiting": panel.best_by_waiting,
            "selected": selected,
            "policies": [
                {
                    "name": policy.name,
                    "service_order": policy.service_order,
                    "metrics": _metrics_view(_teaching_metrics(policy)),
                    "schedule": _schedule_view(policy.intervals, policy.domain_end),
                    "outcomes": [
                        {
                            "pid": outcome.pid,
                            "arrival_time": outcome.arrival_time,
                            "burst_time": outcome.burst_time,
                            "priority": outcome.priority,
                            "start_time": outcome.start_time,
                            "completion_time": outcome.completion_time,
                            "waiting_time": outcome.waiting_time,
                            "turnaround_time": outcome.turnaround_time,
                            "response_time": outcome.response_time,
                        }
                        for outcome in sorted(policy.outcomes, key=lambda row: row.pid)
                    ],
                    "chart_key": policy.name,
                }
                for policy in panel.policies
            ],
        },
        "charts": {**charts, "gantt_variants": variants},
        "notes": [
            "Executed now: all four policies ran on the same three processes through "
            "scheduler.POLICY_CLASSES with quantum 2 and switching cost 0.",
            "Waiting and completion times are checked against the hand calculation in "
            "docs/TEACHER_LIVE_DEMO.md; the run fails loudly if they no longer agree.",
        ],
    }


def _teaching_metrics(policy: Any) -> Dict[str, Any]:
    """Turn a TeachingPolicy record into the canonical metrics mapping.

    Every aggregate the other scenarios serialise is filled here too (turnaround and
    response totals, context switches per process), so a payload from this scenario carries
    the same keys with the same meaning as a custom or a healing run.
    """
    count = len(policy.outcomes)
    total_turnaround = sum(row.turnaround_time for row in policy.outcomes)
    total_response = sum(row.response_time for row in policy.outcomes)
    return {
        "policy_name": policy.name,
        "workload_name": "teacher-hand-example",
        "workload_fingerprint": "",
        "num_processes": count,
        "total_waiting_time": sum(row.waiting_time for row in policy.outcomes),
        "total_turnaround_time": total_turnaround,
        "total_response_time": total_response,
        "avg_waiting_time": policy.mean_waiting_time,
        "avg_turnaround_time": policy.mean_turnaround_time,
        "avg_response_time": policy.mean_response_time,
        "cpu_utilization": policy.cpu_utilization,
        "throughput": policy.throughput,
        "cpu_busy_time": policy.elapsed_time,
        "cpu_idle_time": 0,
        "context_switches": policy.context_switches,
        "context_switches_per_process": policy.context_switches / count if count else 0.0,
        "total_elapsed_time": policy.elapsed_time,
    }


def _workload_fingerprint(panel: Any) -> str:
    return Workload(
        "teacher-hand-example",
        [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in panel.processes],
    ).fingerprint


def _teaching_fixtures(panel: Any) -> Dict[str, Fixture]:
    """Wrap each executed teaching policy in the dashboard's validated fixture type.

    ``fcfs_metrics`` is the *executed* FCFS result on the same three processes, not a
    placeholder: the type is shared with the healing fixtures, where the field really is a
    baseline, so filling it with zeros would plant wrong data in every teaching payload.
    """
    fcfs_policy = next(policy for policy in panel.policies if policy.name == "FCFS")
    fcfs_metrics = Metrics(
        policy_name="FCFS",
        workload_name="teacher-hand-example",
        workload_fingerprint="",
        num_processes=len(fcfs_policy.outcomes),
        total_waiting_time=sum(row.waiting_time for row in fcfs_policy.outcomes),
        avg_waiting_time=fcfs_policy.mean_waiting_time,
        avg_turnaround_time=fcfs_policy.mean_turnaround_time,
        avg_response_time=fcfs_policy.mean_response_time,
        cpu_utilization=fcfs_policy.cpu_utilization,
        context_switches=fcfs_policy.context_switches,
        total_elapsed_time=fcfs_policy.elapsed_time,
        cpu_busy_time=fcfs_policy.elapsed_time,
        cpu_idle_time=0,
        throughput=fcfs_policy.throughput,
    )
    fixtures: Dict[str, Fixture] = {}
    for policy in panel.policies:
        fixture = Fixture(
            family="teaching",
            seed=0,
            fingerprint="",
            processes=tuple(
                RecordedProcess(
                    outcome.pid, outcome.arrival_time, outcome.burst_time, outcome.priority
                )
                for outcome in policy.outcomes
            ),
            decisions=(),
            intervals=policy.intervals,
            policy_segments=(),
            incidents=(),
            outcomes=policy.outcomes,
            metrics=Metrics(
                policy_name=policy.name,
                workload_name="teacher-hand-example",
                workload_fingerprint="",
                num_processes=len(policy.outcomes),
                total_waiting_time=sum(row.waiting_time for row in policy.outcomes),
                avg_waiting_time=policy.mean_waiting_time,
                avg_turnaround_time=policy.mean_turnaround_time,
                avg_response_time=policy.mean_response_time,
                cpu_utilization=policy.cpu_utilization,
                context_switches=policy.context_switches,
                total_elapsed_time=policy.elapsed_time,
                cpu_busy_time=policy.elapsed_time,
                cpu_idle_time=0,
                throughput=policy.throughput,
            ),
            fcfs_metrics=fcfs_metrics,
        )
        fixtures[policy.name] = fixture
    return fixtures


def _processes_from_options(options: Mapping[str, Any]) -> List[Process]:
    raw = options.get("processes")
    if raw is None:
        raw = scenario_catalogue.get_scenario("custom")["defaults"]["processes"]
    if not isinstance(raw, (list, tuple)):
        raise SimulationError("processes must be a list of objects", field="processes")
    if not raw:
        raise SimulationError("a workload needs at least one process", field="processes")
    if len(raw) > MAX_PROCESSES:
        raise SimulationError(
            f"at most {MAX_PROCESSES} processes per run, got {len(raw)}", field="processes"
        )
    processes: List[Process] = []
    seen: set[int] = set()
    for index, row in enumerate(raw):
        where = f"processes[{index}]"
        if not isinstance(row, Mapping):
            raise SimulationError(f"{where} must be an object", field="processes")
        missing = [key for key in ("pid", "arrival_time", "burst_time", "priority") if key not in row]
        if missing:
            raise SimulationError(
                f"{where} is missing {', '.join(missing)}", field="processes"
            )
        values = {}
        for key in ("pid", "arrival_time", "burst_time", "priority"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise SimulationError(
                    f"{where}.{key} must be a whole number, got {value!r}", field="processes"
                )
            values[key] = value
        if values["burst_time"] > MAX_BURST:
            raise SimulationError(
                f"{where}.burst_time must be <= {MAX_BURST} time units", field="processes"
            )
        if values["arrival_time"] > MAX_ARRIVAL:
            raise SimulationError(
                f"{where}.arrival_time must be <= {MAX_ARRIVAL} time units", field="processes"
            )
        if values["pid"] in seen:
            raise SimulationError(
                f"duplicate pid {values['pid']} in the workload", field="processes"
            )
        seen.add(values["pid"])
        try:
            processes.append(
                Process(
                    pid=values["pid"],
                    arrival_time=values["arrival_time"],
                    burst_time=values["burst_time"],
                    priority=values["priority"],
                )
            )
        except Exception as error:  # engine validation messages are already specific
            raise SimulationError(f"{where}: {error}", field="processes") from error
    return processes


def _int_option(options: Mapping[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimulationError(f"{key} must be a whole number, got {value!r}", field=key)
    if not low <= value <= high:
        raise SimulationError(f"{key} must be between {low} and {high}, got {value}", field=key)
    return value


def _policy_from_options(options: Mapping[str, Any]) -> Tuple[str, type[SchedulingPolicy]]:
    raw = options.get("policy", "FCFS")
    if not isinstance(raw, str):
        raise SimulationError("policy must be a string identifier", field="policy")
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {name.lower().replace(" ", "_"): name for name in POLICY_IDS.values()}
    aliases.update(POLICY_IDS)
    if key not in aliases:
        raise SimulationError(
            f"unknown policy {raw!r}; choose FCFS, SJF, Priority or Round Robin", field="policy"
        )
    name = aliases[key]
    for policy_class in POLICY_CLASSES:
        if policy_class.name == name:
            return name, policy_class
    raise SimulationError(f"policy {name!r} is not implemented by the engine", field="policy")


def _run_custom_scenario(scenario: Mapping[str, Any], options: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    processes = _processes_from_options(options)
    quantum = _int_option(options, "quantum", 4, 1, MAX_QUANTUM)
    switching_cost = _int_option(options, "switching_cost", 0, 0, MAX_SWITCH_COST)
    healing = options.get("healing", False)
    if not isinstance(healing, bool):
        raise SimulationError("healing must be true or false", field="healing")
    policy_name, policy_class = _policy_from_options(options)
    if healing and policy_name != "FCFS":
        # The control loop always starts from FCFS and chooses the policy itself. Silently
        # ignoring a requested policy made the Studio's selector look effective when it was
        # not, so the mismatch is refused instead.
        raise SimulationError(
            f"the self-healing loop always starts from FCFS and chooses the policy itself; "
            f"remove 'policy' (or set it to FCFS) to run this workload through the loop, or "
            f"turn healing off to execute {policy_name} directly",
            field="policy",
        )
    model_config = SchedulerConfig(round_robin_quantum=quantum, switching_cost=switching_cost)
    workload = Workload("custom", processes)

    if healing:
        experiment = with_scheduler(HealingExperimentConfig(), quantum, switching_cost)
        payload = run_healing_workload(workload, experiment, family="custom", seed=0)
        fixture = parse_fixture(payload, source="live engine (custom workload)")
        charts = {"gantt": _gantt_svg("custom", fixture, seed=0)}
        charts.update(_healing_charts("custom", fixture, 0))
        charts["waiting"] = _waiting_comparison_chart(
            "custom",
            "Self-healing trace",
            fixture.metrics.avg_waiting_time,
            "FCFS (same workload)",
            fixture.fcfs_metrics.avg_waiting_time,
            "#6b4a86",
            "#3d5a80",
        )
        return {
            "kind": "custom",
            "scenario": _scenario_view(scenario),
            "executed_at": _now(),
            "duration_ms": _duration_ms(started),
            "settings": {
                "quantum": quantum,
                "switching_cost": switching_cost,
                "policy": "FCFS -> recovery rule",
                "healing": True,
            },
            "workload": _workload_view(fixture, "custom", None, list(fixture.processes)),
            "schedule": _schedule_view(fixture.intervals, fixture.domain_end),
            "metrics": _metrics_view(fixture.metrics),
            "baseline": _metrics_view(fixture.fcfs_metrics),
            "outcomes": _outcome_rows(fixture),
            "healing": _healing_view(fixture),
            "charts": charts,
            "notes": [
                "Executed now on the submitted workload by the self-healing controller and the "
                "runtime simulator, then validated against the dashboard schema.",
                "What the controller does depends on the recorded trace; a short workload may "
                "legitimately produce no incident.",
            ],
        }

    result = policy_class(model_config).run(workload)
    metrics = compute_metrics(result)
    intervals = _intervals_from_trace(result.trace, policy_name, switching_cost)
    fixture = Fixture(
        family="custom",
        seed=0,
        fingerprint=workload.fingerprint,
        processes=tuple(processes),
        decisions=(),
        intervals=tuple(intervals),
        policy_segments=(),
        incidents=(),
        outcomes=tuple(_outcome_from_engine(outcome) for outcome in result.outcomes),
        metrics=fixture_metrics(metrics, policy_name, workload, intervals),
        fcfs_metrics=fixture_metrics(metrics, policy_name, workload, intervals),
    )
    charts = {"gantt": _gantt_svg("custom", fixture, seed=0)}
    return {
        "kind": "custom",
        "scenario": _scenario_view(scenario),
        "executed_at": _now(),
        "duration_ms": _duration_ms(started),
        "settings": {
            "quantum": quantum,
            "switching_cost": switching_cost,
            "policy": policy_name,
            "healing": False,
        },
        "workload": {
            "name": workload.name,
            "family": "custom",
            "seed": None,
            "fingerprint": workload.fingerprint,
            "process_count": len(processes),
            "processes": [
                {
                    "pid": process.pid,
                    "arrival_time": process.arrival_time,
                    "burst_time": process.burst_time,
                    "priority": process.priority,
                }
                for process in processes
            ],
        },
        "schedule": _schedule_view(intervals, metrics.total_elapsed_time),
        "metrics": _metrics_view(metrics),
        "baseline": None,
        "outcomes": [
            {
                "pid": outcome.pid,
                "arrival_time": outcome.arrival_time,
                "burst_time": outcome.burst_time,
                "priority": outcome.priority,
                "start_time": outcome.start_time,
                "completion_time": outcome.completion_time,
                "waiting_time": outcome.waiting_time,
                "turnaround_time": outcome.turnaround_time,
                "response_time": outcome.response_time,
            }
            for outcome in sorted(result.outcomes, key=lambda row: row.pid)
        ],
        "healing": None,
        "charts": charts,
        "notes": [
            f"Executed now by scheduler.{policy_class.__name__} on the submitted workload; "
            "metrics come from evaluation.metrics.compute_metrics.",
            "Switch and idle gaps are derived from the recorded slice boundaries and the "
            "configured switching cost.",
        ],
    }


def _outcome_from_engine(outcome: Any) -> Outcome:
    """Convert an engine ProcessOutcome into the dashboard's validated Outcome."""
    return Outcome(
        pid=outcome.pid,
        arrival_time=outcome.arrival_time,
        burst_time=outcome.burst_time,
        priority=outcome.priority,
        start_time=outcome.start_time,
        completion_time=outcome.completion_time,
    )


def fixture_metrics(
    metrics: Any, policy_name: str, workload: Workload, intervals: Sequence[Interval]
) -> Metrics:
    """Shrink engine metrics into the dashboard's validated ``Metrics`` record."""
    return Metrics(
        policy_name=policy_name,
        workload_name=workload.name,
        workload_fingerprint=workload.fingerprint,
        num_processes=metrics.num_processes,
        total_waiting_time=metrics.total_waiting_time,
        avg_waiting_time=metrics.avg_waiting_time,
        avg_turnaround_time=metrics.avg_turnaround_time,
        avg_response_time=metrics.avg_response_time,
        cpu_utilization=metrics.cpu_utilization,
        context_switches=metrics.context_switches,
        total_elapsed_time=metrics.total_elapsed_time,
        total_turnaround_time=metrics.total_turnaround_time,
        total_response_time=metrics.total_response_time,
        cpu_busy_time=metrics.cpu_busy_time,
        cpu_idle_time=metrics.cpu_idle_time,
        throughput=metrics.throughput,
    )


def _intervals_from_trace(
    trace: Iterable[Any], policy_name: str, switching_cost: int
) -> List[Interval]:
    """Build recorded intervals from an engine trace.

    The engine records only service slices, so the gap before a slice is split into the
    configured switch cost (when a different process ran before) and genuine idle time.
    Nothing else is inferred.
    """
    intervals: List[Interval] = []
    previous_end: Optional[int] = None
    previous_pid: Optional[int] = None
    for index, slice_ in enumerate(trace):
        gap = slice_.start_time - previous_end if previous_end is not None else slice_.start_time
        switch = min(gap, switching_cost) if (previous_pid is not None and previous_pid != slice_.pid) else 0
        idle = max(gap - switch, 0)
        intervals.append(
            Interval(
                index=index,
                pid=slice_.pid,
                policy=policy_name,
                start=slice_.start_time,
                end=slice_.end_time,
                switch_before=switch,
                idle_before=idle,
                waiting_increment=None,
                completed_pids=(),
            )
        )
        previous_end = slice_.end_time
        previous_pid = slice_.pid
    return intervals


# --------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------


def execute(scenario_id: str, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run one scenario through the real engine and return the dashboard payload.

    Args:
        scenario_id: Catalogue identifier (``teaching``, ``recovery-accepted``,
            ``recovery-rollback`` or ``custom``).
        options: Scenario-specific options; see :mod:`dashboard.scenarios`.

    Returns:
        A JSON-ready payload with workload, schedule, metrics, healing record and charts.

    Raises:
        SimulationError: If the scenario or its options are not supported.
        DashboardError: If the engine output fails the dashboard schema validation.
    """
    try:
        scenario = scenario_catalogue.get_scenario(scenario_id)
    except KeyError as error:
        # ``str(KeyError(...))`` is the repr of the message, so it arrives wrapped in
        # double quotes; the declared text is ``error.args[0]``.
        raise SimulationError(str(error.args[0] if error.args else error), field="scenario") from None
    try:
        merged = scenario_catalogue.normalise_options(scenario, dict(options or {}))
    except ValueError as error:
        raise SimulationError(str(error), field="options") from None
    kind = scenario["kind"]
    if kind == "teaching":
        return _run_teaching_scenario(scenario, merged)
    if kind == "healing":
        return _run_healing_scenario(scenario, merged)
    if kind == "custom":
        return _run_custom_scenario(scenario, merged)
    raise SimulationError(f"scenario {scenario_id!r} has an unsupported kind {kind!r}")


def describe_workload(processes: Sequence[Process]) -> Dict[str, Any]:
    """Return the validated description of a workload (fingerprint, totals, order)."""
    workload = Workload("custom", list(processes))
    return {
        "name": workload.name,
        "fingerprint": workload.fingerprint,
        "process_count": workload.size,
        "total_burst_time": workload.total_burst_time,
        "processes": [
            {
                "pid": process.pid,
                "arrival_time": process.arrival_time,
                "burst_time": process.burst_time,
                "priority": process.priority,
            }
            for process in workload.processes
        ],
    }


def compare_policies(
    processes: Sequence[Mapping[str, Any]],
    quantum: int = 4,
    switching_cost: int = 0,
    source_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all four conventional policies on one workload and return their metrics.

    Every bar is a separate real execution of the engine on the *same* process list, so the
    comparison is like-for-like.

    Raises:
        SimulationError: If the workload or the scheduler configuration is invalid.
    """
    started = time.perf_counter()
    options: Dict[str, Any] = {"processes": list(processes)}
    checked_processes = _processes_from_options(options)
    quantum = _int_option({"quantum": quantum}, "quantum", 4, 1, MAX_QUANTUM)
    switching_cost = _int_option(
        {"switching_cost": switching_cost}, "switching_cost", 0, 0, MAX_SWITCH_COST
    )
    config = SchedulerConfig(round_robin_quantum=quantum, switching_cost=switching_cost)
    workload = Workload("custom", checked_processes)

    policies: List[Dict[str, Any]] = []
    for policy_class in POLICY_CLASSES:
        result = policy_class(config).run(workload)
        metrics = compute_metrics(result)
        policies.append(
            {
                "name": policy_class.name,
                "policy_id": next(
                    (key for key, name in POLICY_IDS.items() if name == policy_class.name), ""
                ),
                "metrics": _metrics_view(metrics),
            }
        )

    best = min(policies, key=lambda row: row["metrics"]["avg_waiting_time"])
    colours = {
        "FCFS": "#3d5a80",
        "SJF": "#2f6f4f",
        "Round Robin": "#9a6b1f",
        "Priority": "#6b4a86",
    }
    charts = {
        "waiting": horizontal_bars(
            name="compare-waiting",
            title="Mean waiting time by policy on this workload",
            description="Each bar is a separate execution of the named policy on the same processes.",
            items=tuple(
                BarItem(
                    label=row["name"],
                    value=row["metrics"]["avg_waiting_time"],
                    colour=colours.get(row["name"], "#64748b"),
                    strong=row["name"] == best["name"],
                )
                for row in policies
            ),
            unit="Mean waiting time (simulated time units per process)",
            width=760,
        ),
        "response": horizontal_bars(
            name="compare-response",
            title="Mean response time by policy on this workload",
            description="Each bar is a separate execution of the named policy on the same processes.",
            items=tuple(
                BarItem(
                    label=row["name"],
                    value=row["metrics"]["avg_response_time"],
                    colour=colours.get(row["name"], "#64748b"),
                    strong=row["name"] == best["name"],
                )
                for row in policies
            ),
            unit="Mean response time (simulated time units per process)",
            width=760,
        ),
    }
    return {
        "kind": "comparison",
        "source_run_id": source_run_id,
        "executed_at": _now(),
        "duration_ms": _duration_ms(started),
        "settings": {"quantum": quantum, "switching_cost": switching_cost},
        "workload": describe_workload(checked_processes),
        "policies": policies,
        "best_by_waiting": best["name"],
        "charts": charts,
        "notes": [
            "Each policy executed separately on the identical workload; all metrics come from "
            "evaluation.metrics.compute_metrics.",
            "Lower waiting and response times are better. Times are simulated time units.",
        ],
    }


def _measured_limits() -> Optional[Dict[str, Any]]:
    """Read the study's own verification/detection measurements for the Research page.

    The page states what the verifier and the detector actually did on the held-out split.
    Those figures live in the committed ``healing_summary.json`` but are not part of the
    arm table ``load_research`` validates, so they are read here, read-only, and omitted
    (``None``) rather than guessed when the keys are absent.
    """
    try:
        document = json.loads(RESEARCH_SUMMARY_PATH.read_text(encoding="utf-8"))
        discrimination = document["verification_discrimination"]
        derivation = document["threshold_derivation"]
        return {
            "split": discrimination.get("split"),
            "interventions": discrimination.get("n_interventions"),
            "accepted": discrimination.get("accepted_n"),
            "accepted_helped_share": discrimination.get("accepted_helped_share"),
            "certifiable_share": discrimination.get(
                "certifiable_share_of_helpful_at_accept_residual_factor"
            ),
            "base_helped_share": discrimination.get("base_helped_share"),
            "rejected": discrimination.get("rejected_n"),
            "rejected_harmful_share": discrimination.get("rejected_harmful_share"),
            "control_incidents_per_trace": derivation.get("control_incidents_per_trace"),
            "declared_false_accept_rate": derivation.get(
                "declared_false_accept_rate_on_control"
            ),
        }
    except (OSError, KeyError, TypeError, ValueError):
        return None


def research_panel() -> Dict[str, Any]:
    """Read the committed study artifacts (read-only) and shape them for the browser."""
    summary = load_research(RESEARCH_SUMMARY_PATH)
    arms = [
        {
            "arm": arm.arm,
            "label": arm.label,
            "short_label": arm.short_label,
            "kind": arm.kind,
            "mean_waiting_time": round(arm.mean_waiting_time, 6),
            "p95_waiting_time": arm.p95_waiting_time,
            "mean_response_time": arm.mean_response_time,
            "context_switches": arm.context_switches,
        }
        for arm in summary.arms
    ]
    return {
        "source": summary.source_name,
        "title": summary.title,
        "split": summary.split,
        "workloads": summary.workloads,
        "final_test_seed": summary.final_test_seed,
        "training_seeds": list(summary.training_seeds),
        "arms": arms,
        "comparisons": [
            {
                "target": row.target,
                "reference": row.reference,
                "label": row.label,
                "metric": row.metric,
                "difference": round(row.difference, 6),
                "ci_low": round(row.ci_low, 6),
                "ci_high": round(row.ci_high, 6),
                "excludes_zero": row.excludes_zero,
            }
            for row in summary.comparisons
        ],
        "charts": {
            "waiting": research_bars(
                name="research-waiting",
                title="Held-out study: mean waiting time by arm",
                description=(
                    "Committed final-test means from healing_summary.json. These are historical "
                    "aggregates over the whole held-out split, not results of this session."
                ),
                arms=summary.arms,
            )
        },
        "measured_limits": _measured_limits(),
        "caveats": [
            "These are committed final-test aggregate means, not the output of the run shown "
            "above.",
            "The frozen comparison did not establish Q-learning recovery as better than the "
            "waiting-time rule.",
            "SJF is best on mean waiting time in this evaluated design; that is not a claim of "
            "general superiority.",
        ],
    }


def engine_info() -> Dict[str, Any]:
    """Describe the engine the API will call, for the health endpoint and the UI footer."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "policies": [policy.name for policy in POLICY_CLASSES],
        "repo_root": str(REPOSITORY_ROOT),
        "research_available": RESEARCH_SUMMARY_PATH.exists(),
        "research_source": RESEARCH_SUMMARY_PATH.relative_to(REPOSITORY_ROOT).as_posix()
        if RESEARCH_SUMMARY_PATH.exists()
        else None,
    }
