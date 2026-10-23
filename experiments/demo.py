"""Small presentation fixtures using the unmodified self-healing scheduling loop.

These are illustrative generator seeds, not the validation or final-test experiment.
No fitting or threshold changes are performed. The waiting-time rule makes this demo
independent of a saved model or the expensive five-seed training run.

Three entry points share one execution path:

* :func:`run_healing_workload` runs *any* workload through the controller + runtime
  simulator and returns the recorded fixture record.  The live dashboard uses it for the
  documented seeds and for custom workloads with self-healing switched on.
* :func:`run_fixture` generates the documented ``mixed`` workload for one seed and adds
  the CLI transcript.
* :func:`run_demo` prints both fixed examples and writes ``demo.json``; its console output
  and payload are unchanged by the existence of the dashboard API.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ACTION_NAMES
from evaluation.metrics import compute_metrics
from experiments.healing_config import HealingExperimentConfig
from healing.controller import SelfHealingController
from healing.policies import HealingContext, RuleWaitingTimePolicy
from scheduler import FCFS, RuntimeSimulator
from workload.generator import WorkloadGenerator
from workload.models import Workload

#: Workload family used by both illustrative fixtures.
DEMO_FAMILY = "mixed"
#: Seeds of the two illustrative fixtures (the order the CLI prints them in).
DEMO_SEEDS = (3, 13)


@dataclass(frozen=True)
class FixtureRun:
    """One executed demo fixture: the recorded payload and the CLI transcript."""

    seed: int
    payload: Dict[str, object]
    transcript: List[str]


def run_healing_workload(
    workload: Workload,
    config: HealingExperimentConfig,
    family: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one workload through the real self-healing loop and record the fixture.

    This is the single adapter between the engine and the dashboard schema: the returned
    mapping is exactly what ``demo.json`` stores for one fixture, so the API and the CLI
    cannot drift apart.

    Args:
        workload: The workload to schedule (already validated by :class:`Workload`).
        config: Experiment configuration holding the scheduler and healing settings.
        family: Workload condition name recorded in the payload.
        seed: Generator seed recorded in the payload (``None`` for custom workloads).

    Returns:
        A JSON-ready fixture record with processes, decisions, incidents, outcomes and
        metrics.
    """
    controller = SelfHealingController(
        config.healing, config.scheduler, RuleWaitingTimePolicy(config.healing),
        label="demo_rule_wait",
    )
    controller.begin_episode(0)
    run = RuntimeSimulator(config.scheduler).run(
        workload, controller, training=False, policy_name="demo_rule_wait"
    )
    episode = controller.end_episode()
    metrics = compute_metrics(run.result)
    baseline = compute_metrics(FCFS(config.scheduler).run(workload))

    decisions = []
    for record in run.decisions:
        observation = record.observation
        decisions.append(
            {
                "index": record.decision_index,
                "observation": asdict(observation),
                "policy": ACTION_NAMES[record.decision.action],
                "state": record.decision.state_index,
                "reason": record.decision.fallback_reason,
                "pid": record.chosen_pid,
                "start": record.start_time,
                "end": record.end_time,
                "switch_overhead": record.switch_overhead,
                "waiting_increment": record.waiting_cost_increment,
                "completed_pids": list(record.completed_pids),
                "completed_waiting_times": list(record.completed_waiting_times),
            }
        )

    incidents = []
    for incident in episode.incidents:
        context = HealingContext(
            diagnosis_index=incident.diagnosis_index,
            excess_band_units=incident.excess_band_units,
            ready_count=incident.ready_count_at_trigger,
            current_policy=incident.settings_before.policy,
            relative_band=incident.band_at_trigger / max(incident.reference_at_trigger, 1e-9),
            remaining_work_ratio=incident.visible_remaining_work_at_trigger / max(incident.reference_at_trigger, 1e-9),
        )
        row = incident.as_row()
        row["context"] = context.as_dict()
        row["feature_bins"] = list(controller.encoder.feature_bins(context))
        incidents.append(row)

    return {
        "family": family if family is not None else workload.name,
        "seed": seed,
        "fingerprint": workload.fingerprint,
        "processes": [asdict(p) for p in workload.processes],
        "decisions": decisions,
        "incidents": incidents,
        "outcomes": [asdict(o) for o in run.result.outcomes],
        "metrics": metrics.as_dict(),
        "fcfs_metrics": baseline.as_dict(),
    }


def _transcript(fixture: Dict[str, Any], family: str, seed: int) -> List[str]:
    """Build the exact CLI lines printed for one fixture."""
    lines: List[str] = [
        f"\nINPUT {family} seed={seed}; fingerprint={fixture['fingerprint']}",
        "Processes (arrival, burst, priority):",
    ]
    for process in fixture["processes"]:
        lines.append(
            f"  P{process['pid']}: "
            f"({process['arrival_time']}, {process['burst_time']}, {process['priority']})"
        )
    lines.append("Execution decisions (ready IDs are arrived, unfinished work):")
    for decision in fixture["decisions"]:
        observation = decision["observation"]
        ready = ",".join(str(p["pid"]) for p in observation["ready_processes"])
        lines.append(
            f"  d{decision['index']:02} t={observation['current_time']} ready=[{ready}] "
            f"{decision['policy']} -> P{decision['pid']} "
            f"[{decision['start']},{decision['end']}) "
            f"switch={decision['switch_overhead']} wait+={decision['waiting_increment']}"
        )
    lines.append("Concluded incident evidence (verdict issued after the observation window):")
    for incident in fixture["incidents"]:
        lines.append(
            f"  Trigger d{incident['decision_index']} t={incident['simulated_time']}: "
            f"completed-window wait={incident['window_mean_at_trigger']:.3f} "
            f"> threshold={incident['trigger_threshold']:.3f}"
        )
        lines.append(f"  Diagnosis: {incident['diagnosis_statement']}")
        lines.append(
            f"  Context bins={incident['feature_bins']} -> "
            f"state={incident['context_state_index']}/648"
        )
        lines.append(
            f"  Offered: {incident['offered_actions']}; "
            f"rule selected: {incident['action_name']}"
        )
        lines.append(
            f"  Verdict={incident['verdict']}; reason={incident['verifier_reason']}; "
            f"residual={incident['closure_residual']}; reward={incident['reward']}"
        )
        rollback = incident["rollback_target_policy"]
        posture = "rollback to " + rollback if rollback else "keep configuration"
        lines.append(
            f"  At window end t={incident['window_end_time']}: {posture}; "
            f"confirmed_failure={incident['confirmed_failure']}"
        )
    metrics = fixture["metrics"]
    baseline = fixture["fcfs_metrics"]
    lines.append(
        f"SUMMARY mean wait={metrics['avg_waiting_time']:.3f}; "
        f"FCFS same input={baseline['avg_waiting_time']:.3f}; "
        f"difference={metrics['avg_waiting_time'] - baseline['avg_waiting_time']:+.3f}"
    )
    lines.append(
        f"  mean response={metrics['avg_response_time']:.3f}; "
        f"switches={metrics['context_switches']}; elapsed={metrics['total_elapsed_time']}; "
        f"utilization={metrics['cpu_utilization']:.3f}%"
    )
    lines.append("  Waiting per process = completion - arrival - burst; mean = sum / process count.")
    return lines


def run_fixture(
    seed: int,
    config: HealingExperimentConfig | None = None,
    family: str = DEMO_FAMILY,
) -> FixtureRun:
    """Run one illustrative fixture through the real engine.

    Args:
        seed: Generator seed of the workload (3 and 13 are the documented fixtures).
        config: Experiment configuration; the frozen default is used when omitted.
        family: Workload family name; ``"mixed"`` is the documented demo condition.

    Returns:
        The fixture payload and the exact lines the CLI prints for it.

    Raises:
        ConfigurationError: If ``family`` is not one of the configured workload families.
    """
    config = config or HealingExperimentConfig()
    workload = WorkloadGenerator(config.families).generate(family, seed)
    payload = run_healing_workload(workload, config, family=family, seed=seed)
    return FixtureRun(seed=seed, payload=payload, transcript=_transcript(payload, family, seed))


def with_scheduler(
    config: HealingExperimentConfig, quantum: int, switching_cost: int
) -> HealingExperimentConfig:
    """Return the same experiment design with a different scheduler configuration.

    Only the round-robin quantum and the switch cost change; thresholds, windows, reward
    and rule stay exactly as frozen in :class:`HealingExperimentConfig`.
    """
    return replace(
        config,
        scheduler=replace(config.scheduler, round_robin_quantum=quantum, switching_cost=switching_cost),
    )


def run_demo(output_dir: Path) -> dict:
    """Run two fixed examples, print measured evidence, and save a portable JSON trace."""
    config = HealingExperimentConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "illustrative self-healing rule fixtures; no Q training or final-test evaluation",
        "scheduler": asdict(config.scheduler),
        "healing": config.healing.as_dict(),
        "fixtures": [],
    }
    print("Self-healing scheduler: deterministic quick demonstration")
    print("Real engine + default thresholds + waiting-time recovery rule; no Q training.")
    print("Times below are simulated units, not seconds. Fixtures are not study results.")
    for seed in DEMO_SEEDS:
        fixture = run_fixture(seed, config)
        for line in fixture.transcript:
            print(line)
        payload["fixtures"].append(fixture.payload)
    path = output_dir / "demo.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved real inputs, observations, intervals, incidents and metrics: {path}")
    return payload
