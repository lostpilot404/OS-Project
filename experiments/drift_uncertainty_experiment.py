"""Reproducible evaluation under workload drift and burst uncertainty (Task 6)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import SchedulerConfig
from evaluation.metrics import compute_metrics
from evaluation.reliability import compute_reliability_metrics, reliability_summary_for_arms
from experiments.healing_config import HealingExperimentConfig, default_arms
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from scheduler import POLICY_CLASSES
from scheduler.runtime import RuntimeSimulator
from workload.drift import DRIFT_KINDS, DriftSpec, apply_drift
from workload.generator import WorkloadGenerator
from workload.models import Workload

from experiments.healing_experiment import (
    FIXED_POLICY_CLASSES,
    HealingWorkload,
    build_healing_workloads,
    train_healing_models,
    _make_policy,
)


ARMS_TO_TEST = ["fixed_fcfs", "fixed_sjf", "detector_only", "rule_wait", "q_recovery", "rule_fairness"]


def run_drift_evaluation(
    *,
    final_test_seed: int = 23819,
    repetitions: int = 5,
    drift_kind: str = "burst_shock",
    estimator_enabled: bool = True,
    estimator_error_cv: float = 0.30,
    training_episodes_per_seed: int = 200,
    output_dir: Path = Path("results/drift"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = HealingExperimentConfig(
        final_test_seed=final_test_seed,
        final_test_repetitions=repetitions,
        training_episodes_per_seed=training_episodes_per_seed,
        healing=HealingConfig(
            burst_estimator_enabled=estimator_enabled,
            burst_estimator_error_cv=estimator_error_cv,
            drift_enabled=False,
        ),
        bootstrap_replicates=100,
        bootstrap_seed=104729,
    )
    print(f"Drift experiment: seed={final_test_seed}, reps={repetitions}, drift={drift_kind}, estimator={estimator_enabled} cv={estimator_error_cv}")
    print(f"Training Q with {training_episodes_per_seed} episodes/seed on stationary workloads")

    generator = WorkloadGenerator(config.families)
    models_tuple, _ = train_healing_models(config, generator)
    models = {m.training_seed: m for m in models_tuple}
    print(f"Trained {len(models)} Q models")

    base_workloads = build_healing_workloads("final_test", config, generator)

    # Apply drift to copies (deterministic, same seed → same injection time)
    drifted_workloads: List[HealingWorkload] = []
    for hw in base_workloads:
        spec = DriftSpec(kind=drift_kind, seed=hw.seed)
        drifted = apply_drift(hw.workload, spec)
        drifted_workloads.append(HealingWorkload(split=hw.split, family=hw.family, repetition=hw.repetition, seed=hw.seed, workload=drifted))

    test_workloads = drifted_workloads
    simulator = RuntimeSimulator(config.scheduler)
    from healing.policies import HealingContextEncoder
    encoder = HealingContextEncoder(config.healing)

    all_arms = default_arms(config)
    arm_by_name = {}
    for a in all_arms:
        # For q_recovery there are 5 entries with same name but different seed; keep list
        arm_by_name.setdefault(a.name, []).append(a)

    per_arm_traces: Dict[str, List[Dict[str, Any]]] = {}

    for arm_name in ARMS_TO_TEST:
        if arm_name not in arm_by_name:
            continue
        # For fixed/healing single-instance arms, one entry; for q_recovery 5 seeds
        arm_entries = arm_by_name[arm_name]
        print(f"\nEvaluating arm {arm_name} on {len(test_workloads)} drifted workloads...")
        all_traces = []
        for arm in arm_entries:
            # Policy
            if arm.kind == "fixed":
                # Fixed arms: run standalone scheduler per workload
                policy_class_name = FIXED_POLICY_CLASSES[arm.name]
                scheduler_cls = next(cls for cls in POLICY_CLASSES if cls.__name__ == policy_class_name)
                for hw in test_workloads:
                    scheduler = scheduler_cls(config.scheduler)
                    result = scheduler.run(hw.workload)
                    rel = compute_reliability_metrics(result, incidents=[], audit_gains=[])
                    all_traces.append({"result": result, "episode": None, "reliability": rel, "workload": hw, "arm_entry": arm})
            else:
                # Healing arms
                if arm.policy_kind == "q_learning":
                    model = models.get(arm.training_seed)
                    if model is None:
                        continue
                    policy = model.policy
                else:
                    policy = _make_policy(arm, config, encoder)
                for idx, hw in enumerate(test_workloads):
                    controller = SelfHealingController(
                        config.healing, config.scheduler, policy,
                        initial_policy=arm.initial_policy, label=arm.name,
                    )
                    controller.begin_episode(idx)
                    run = simulator.run(hw.workload, controller, training=False, policy_name=arm.name)
                    episode = controller.end_episode()
                    rel = compute_reliability_metrics(run.result, incidents=episode.incidents, audit_gains=[])
                    all_traces.append({"result": run.result, "episode": episode, "reliability": rel, "workload": hw, "arm_entry": arm})
        per_arm_traces[arm_name] = all_traces
        means = float(np.mean([t["reliability"].mean_waiting for t in all_traces])) if all_traces else 0
        p95s = float(np.mean([t["reliability"].p95_waiting for t in all_traces])) if all_traces else 0
        print(f"  {arm_name}: mean_wait={means:.2f} p95={p95s:.2f} n={len(all_traces)}")

    # Reliability summary
    per_arm_metrics = {arm: [t["reliability"] for t in traces] for arm, traces in per_arm_traces.items()}
    summary_rows = reliability_summary_for_arms(per_arm_metrics, baseline_arm="fixed_fcfs")

    # Write summary CSV
    summary_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    # Per-trace CSV
    detail_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_per_trace.csv"
    with open(detail_path, "w", newline="") as f:
        fieldnames = ["arm", "family", "repetition", "mean_waiting", "p95_waiting", "max_waiting", "starved", "fairness", "throughput", "cpu_util", "switches", "incidents", "interventions", "rollback_rate", "recovery_delay_decisions", "fallback"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm, traces in per_arm_traces.items():
            for t in traces:
                rel = t["reliability"]
                writer.writerow({
                    "arm": arm,
                    "family": t["workload"].family,
                    "repetition": t["workload"].repetition,
                    "mean_waiting": rel.mean_waiting,
                    "p95_waiting": rel.p95_waiting,
                    "max_waiting": rel.max_waiting,
                    "starved": rel.starved_processes,
                    "fairness": rel.fairness_index,
                    "throughput": rel.throughput,
                    "cpu_util": rel.cpu_utilization,
                    "switches": rel.context_switches,
                    "incidents": rel.incidents,
                    "interventions": rel.interventions,
                    "rollback_rate": rel.rollback_rate,
                    "recovery_delay_decisions": rel.recovery_delay_decisions,
                    "fallback": rel.fallback_triggered,
                })

    # Markdown report
    lines = []
    lines.append(f"# Drift & Uncertainty Evaluation — {drift_kind} drift, estimator={estimator_enabled} cv={estimator_error_cv}")
    lines.append("")
    lines.append(f"**Workloads:** {len(test_workloads)} traces ({repetitions} reps × {len(config.families)} families), seed {final_test_seed}, drift `{drift_kind}` via workload/drift.py at fraction 0.35 (deterministic, pre-injection prefix identical).")
    lines.append(f"**Estimator:** {'enabled' if estimator_enabled else 'disabled (exact burst baseline)'}; when enabled EWMA per-bucket over completed totals (observable service sums), log-normal noise cv={estimator_error_cv} bias=0.0 alpha={config.healing.burst_estimator_alpha}; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation.")
    lines.append(f"**Training:** {len(models)} Q tables, {training_episodes_per_seed} episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training.")
    lines.append(f"**Q seeds:** {list(config.training_seeds)}")
    lines.append("")
    lines.append("| Arm | Mean wait | P95 | Max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay | Fallback |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        rr = row.get("rollback_rate")
        rr_s = f"{rr:.2f}" if rr is not None else "n/a"
        rd = row.get("mean_recovery_delay_decisions")
        rd_s = f"{rd:.1f}" if rd is not None else "n/a"
        lines.append(f"| {row['arm']} | {row['mean_waiting']:.2f} | {row['p95_waiting']:.2f} | {row['max_waiting']:.2f} | {row['starved_processes']:.2f} | {row['fairness_index']:.3f} | {row['throughput']:.4f} | {row['cpu_utilization']:.1f} | {row['context_switches']:.1f} | {row['incidents']:.2f} | {row['interventions']:.2f} | {rr_s} | {rd_s} | {row['fallback_rate']:.3f} |")
    lines.append("")
    lines.append("**Interpretation notes:**")
    lines.append("- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified.")
    lines.append("- False-intervention / regression from audit require matched replay (not run here for lightweight script); rollback/recovery/fallback from incident ledger.")
    lines.append("- Estimator documented as 'error modelled as log-normal', not 'unbiased' or 'realistic'; no claim about real OS burst prediction.")
    lines.append("- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, Q coverage ~3-7% of 648 states (flat after ~900 episodes), intervals conditional on declared design.")

    report_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_report.md"
    report_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nArtifacts: {summary_path}, {detail_path}, {report_path}")

    return {"summary_rows": summary_rows, "per_arm_traces": per_arm_traces, "output_dir": str(output_dir), "report_path": str(report_path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23819)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--drift", type=str, default="burst_shock", choices=list(DRIFT_KINDS))
    parser.add_argument("--estimator", action="store_true", default=True)
    parser.add_argument("--no-estimator", dest="estimator", action="store_false")
    parser.add_argument("--cv", type=float, default=0.30)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="results/drift")
    args = parser.parse_args()
    run_drift_evaluation(
        final_test_seed=args.seed,
        repetitions=args.reps,
        drift_kind=args.drift,
        estimator_enabled=args.estimator,
        estimator_error_cv=args.cv,
        training_episodes_per_seed=args.train_episodes,
        output_dir=Path(args.out),
    )
