"""Small presentation fixtures using the unmodified self-healing scheduling loop.

These are illustrative generator seeds, not the validation or final-test experiment.
No fitting or threshold changes are performed. The waiting-time rule makes this demo
independent of a saved model or the expensive five-seed training run.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from config import ACTION_NAMES
from evaluation.metrics import compute_metrics
from experiments.healing_config import HealingExperimentConfig
from healing.controller import SelfHealingController
from healing.policies import HealingContext, RuleWaitingTimePolicy
from scheduler import FCFS, RuntimeSimulator
from workload.generator import WorkloadGenerator


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
    for seed in (3, 13):
        workload = WorkloadGenerator(config.families).generate("mixed", seed)
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
        print(f"\nINPUT mixed seed={seed}; fingerprint={workload.fingerprint}")
        print("Processes (arrival, burst, priority):")
        for process in workload.processes:
            print(f"  P{process.pid}: ({process.arrival_time}, {process.burst_time}, {process.priority})")
        decisions = []
        print("Execution decisions (ready IDs are arrived, unfinished work):")
        for record in run.decisions:
            observation = record.observation
            decision = {
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
            decisions.append(decision)
            ready = ",".join(str(p.pid) for p in observation.ready_processes)
            print(
                f"  d{record.decision_index:02} t={observation.current_time} ready=[{ready}] "
                f"{decision['policy']} -> P{record.chosen_pid} "
                f"[{record.start_time},{record.end_time}) "
                f"switch={record.switch_overhead} wait+={record.waiting_cost_increment}"
            )
        print("Concluded incident evidence (verdict issued after the observation window):")
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
            print(
                f"  Trigger d{incident.decision_index} t={incident.simulated_time}: "
                f"completed-window wait={incident.window_mean_at_trigger:.3f} "
                f"> threshold={incident.trigger_threshold:.3f}"
            )
            print(f"  Diagnosis: {incident.diagnosis_statement}")
            print(f"  Context bins={row['feature_bins']} -> state={incident.context_state_index}/648")
            print(f"  Offered: {incident.offered_actions}; rule selected: {incident.action_name}")
            print(
                f"  Verdict={incident.verdict}; reason={incident.verifier_reason}; "
                f"residual={incident.closure_residual}; reward={incident.reward}"
            )
            posture = "rollback to " + incident.rollback_target.policy_name if incident.rollback_target else "keep configuration"
            print(f"  At window end t={incident.window_end_time}: {posture}; confirmed_failure={incident.confirmed_failure}")
        print(
            f"SUMMARY mean wait={metrics.avg_waiting_time:.3f}; "
            f"FCFS same input={baseline.avg_waiting_time:.3f}; "
            f"difference={metrics.avg_waiting_time - baseline.avg_waiting_time:+.3f}"
        )
        print(
            f"  mean response={metrics.avg_response_time:.3f}; "
            f"switches={metrics.context_switches}; elapsed={metrics.total_elapsed_time}; "
            f"utilization={metrics.cpu_utilization:.3f}%"
        )
        print("  Waiting per process = completion - arrival - burst; mean = sum / process count.")
        payload["fixtures"].append({
            "family": "mixed", "seed": seed, "fingerprint": workload.fingerprint,
            "processes": [asdict(p) for p in workload.processes],
            "decisions": decisions, "incidents": incidents,
            "outcomes": [asdict(o) for o in run.result.outcomes],
            "metrics": metrics.as_dict(), "fcfs_metrics": baseline.as_dict(),
        })
    path = output_dir / "demo.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved real inputs, observations, intervals, incidents and metrics: {path}")
    return payload
