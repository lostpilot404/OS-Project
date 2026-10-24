"""Reproducible evaluation under workload drift and burst uncertainty.

This script implements the causal, paired drift experiment required by the
project brief:

- Training uses only stationary workloads (6 families, poisson_arrivals held out,
  200 episodes/seed, 5 Q seeds 7101-7105, never sees drift).
- Testing uses drifted workloads (seed 23819, 10 reps × 7 families = 70 traces,
  drift injected at 0.35 of arrival span via workload/drift.py, pre-injection
  prefix identical so each arm sees the same workload instance).
- Burst uncertainty is via healing/burst_estimator.py EWMA per-bucket + log-normal
  noise cv=0.30 when enabled; when disabled exact bursts are visible (baseline).
- All arms are evaluated on identical workload instances (paired by
  family/repetition/fingerprint), stratified bootstrap CIs are computed.

Formulas and denominators are documented inline for auditability.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import SchedulerConfig
from evaluation.metrics import compute_metrics
from evaluation.paired import paired_bootstrap_difference
from evaluation.reliability import _p95 as _compute_p95
from evaluation.reliability import compute_reliability_metrics, reliability_summary_for_arms
from experiments.healing_config import HealingExperimentConfig, default_arms
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from scheduler import POLICY_CLASSES
from scheduler.runtime import RuntimeSimulator
from workload.drift import DRIFT_INJECTION_FRACTION, DRIFT_KINDS, DriftSpec, apply_drift
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

# Declared workload families (config.py) — 7 families, their documented meaning:
# short_jobs: uniform arrivals, short uniform bursts 2-10
# long_jobs: uniform arrivals, long uniform bursts 15-30
# mixed: mix of short/long
# cpu_bursty: bursty arrivals, bimodal bursts (batch-like)
# priority_skewed: priority correlates with burst
# staggered_interactive: staggered arrivals, interactive bursts
# poisson_arrivals: Poisson arrivals (held out of training, tests distribution shift)
# Training uses 6 families (all but poisson_arrivals); testing uses all 7.


def _comparison_interpretations(comparisons: List[Dict[str, Any]]) -> List[str]:
    """Interpret measured intervals without drift-specific overrides or rounding tests."""
    lines = []
    for row in comparisons:
        if "mean_difference" not in row:
            continue
        low, high = float(row["ci95_low"]), float(row["ci95_high"])
        status = "excludes zero" if low > 0 or high < 0 else "includes zero"
        lines.append(
            f"- {row['target']} minus {row['reference']}: "
            f"{row['mean_difference']:.6g}, 95% CI [{low:.6g}, {high:.6g}] "
            f"({status}, using unrounded bounds). Negative favors target for waiting time; "
            "statistical separation alone does not establish practical importance."
        )
    return lines


def run_drift_evaluation(
    *,
    final_test_seed: int = 23819,
    repetitions: int = 10,
    drift_kind: str = "burst_shock",
    estimator_enabled: bool = True,
    estimator_error_cv: float = 0.30,
    training_episodes_per_seed: int = 200,
    output_dir: Path = Path("results/drift"),
) -> Dict[str, Any]:
    # Validate the drift before training or creating output.
    DriftSpec(kind=drift_kind, phase_target_family="short_jobs" if drift_kind == "phase_transition" else None)
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
        bootstrap_replicates=1000,
        bootstrap_seed=104729,
    )
    print(f"Drift experiment: seed={final_test_seed}, reps={repetitions}, drift={drift_kind}, estimator={estimator_enabled} cv={estimator_error_cv}")
    print(f"Training Q with {training_episodes_per_seed} episodes/seed on stationary workloads (held-out drift at test)")

    generator = WorkloadGenerator(config.families)
    models_tuple, _ = train_healing_models(config, generator)
    models = {m.training_seed: m for m in models_tuple}
    print(f"Trained {len(models)} Q models (seeds {list(models.keys())})")

    base_workloads = build_healing_workloads("final_test", config, generator)

    # Apply drift to copies (deterministic, same seed → same injection time)
    drifted_workloads: List[HealingWorkload] = []
    for hw in base_workloads:
        target = generator.generate("short_jobs", hw.seed) if drift_kind == "phase_transition" else None
        spec = DriftSpec(kind=drift_kind, seed=hw.seed,
                         phase_target_family="short_jobs" if target is not None else None)
        drifted = apply_drift(hw.workload, spec, target_workload=target)
        drifted_workloads.append(HealingWorkload(split=hw.split, family=hw.family, repetition=hw.repetition, seed=hw.seed, workload=drifted))

    test_workloads = drifted_workloads
    simulator = RuntimeSimulator(config.scheduler)
    from healing.policies import HealingContextEncoder
    encoder = HealingContextEncoder(config.healing)

    all_arms = default_arms(config)
    arm_by_name = {}
    for a in all_arms:
        arm_by_name.setdefault(a.name, []).append(a)

    per_arm_traces: Dict[str, List[Dict[str, Any]]] = {}
    # For paired bootstrap we need a tidy frame with one row per (arm, training_seed, family, rep, fingerprint)
    paired_rows: List[Dict[str, Any]] = []

    for arm_name in ARMS_TO_TEST:
        if arm_name not in arm_by_name:
            continue
        arm_entries = arm_by_name[arm_name]
        print(f"\nEvaluating arm {arm_name} on {len(test_workloads)} drifted workloads...")
        all_traces = []
        for arm in arm_entries:
            if arm.kind == "fixed":
                policy_class_name = FIXED_POLICY_CLASSES[arm.name]
                scheduler_cls = next(cls for cls in POLICY_CLASSES if cls.__name__ == policy_class_name)
                for hw in test_workloads:
                    scheduler = scheduler_cls(config.scheduler)
                    result = scheduler.run(hw.workload)
                    rel = compute_reliability_metrics(result, incidents=[], audit_gains=[])
                    all_traces.append({"result": result, "episode": None, "reliability": rel, "workload": hw, "arm_entry": arm})
                    # Paired frame row for this trace
                    paired_rows.append({
                        "method": arm_name,
                        "family": hw.family,
                        "repetition": hw.repetition,
                        "workload_fingerprint": hw.fingerprint,
                        "training_seed": arm.training_seed if hasattr(arm, 'training_seed') else None,
                        "avg_waiting_time": rel.mean_waiting,
                        "p95_waiting_time": rel.p95_waiting,
                        "max_waiting_time": rel.max_waiting,
                        "mean_waiting": rel.mean_waiting,
                    })
            else:
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
                    paired_rows.append({
                        "method": arm_name,
                        "family": hw.family,
                        "repetition": hw.repetition,
                        "workload_fingerprint": hw.fingerprint,
                        "training_seed": arm.training_seed,
                        "avg_waiting_time": rel.mean_waiting,
                        "p95_waiting_time": rel.p95_waiting,
                        "max_waiting_time": rel.max_waiting,
                        "mean_waiting": rel.mean_waiting,
                    })
        per_arm_traces[arm_name] = all_traces
        means = float(np.mean([t["reliability"].mean_waiting for t in all_traces])) if all_traces else 0
        p95s = float(np.mean([t["reliability"].p95_waiting for t in all_traces])) if all_traces else 0
        print(f"  {arm_name}: mean_wait={means:.2f} p95={p95s:.2f} n={len(all_traces)}")

    # Per-trace aggregate metrics; regression below is separately paired by
    # workload key after averaging training seeds, regardless of repetition count.
    per_arm_metrics = {arm: [t["reliability"] for t in traces] for arm, traces in per_arm_traces.items()}
    summary_rows = reliability_summary_for_arms(per_arm_metrics, baseline_arm="fixed_fcfs")
    # Pooled P95 (tail over all process waits, not per-trace mean of maxes).
    # _p95 uses ceil(0.95*n)-1, so per-trace P95 with n=15 is definitionally max.
    # Pooled pools n≈1050 (70×15) or 5250 (350×15) values so P95 < max and interpretable.
    for row in summary_rows:
        arm = row["arm"]
        traces = per_arm_traces.get(arm, [])
        pooled_waits: List[float] = []
        for t in traces:
            # result.outcomes holds per-process ProcessMetrics.waiting_time
            for pm in t["result"].outcomes:
                pooled_waits.append(float(pm.waiting_time))
        if pooled_waits:
            row["pooled_p95_waiting"] = float(_compute_p95(pooled_waits))
            row["pooled_max_waiting"] = float(max(pooled_waits))
            row["pooled_n"] = len(pooled_waits)
        else:
            row["pooled_p95_waiting"] = None
            row["pooled_max_waiting"] = None
            row["pooled_n"] = 0

    baseline_traces = per_arm_traces.get("fixed_fcfs", [])
    if baseline_traces:
        baseline_by_key = {
            (t["workload"].family, t["workload"].repetition, t["workload"].fingerprint):
            t["reliability"].mean_waiting for t in baseline_traces
        }
        for row in summary_rows:
            if row["arm"] == "fixed_fcfs":
                continue
            grouped = {}
            for trace in per_arm_traces[row["arm"]]:
                hw = trace["workload"]
                key = (hw.family, hw.repetition, hw.fingerprint)
                grouped.setdefault(key, []).append(trace["reliability"].mean_waiting)
            if set(grouped) != set(baseline_by_key):
                raise ValueError("Regression comparison requires identical workload keys")
            row["regression_rate"] = sum(
                sum(values) / len(values) > baseline_by_key[key]
                for key, values in grouped.items()
            ) / len(grouped)

    # Write summary CSV
    summary_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    # Per-trace CSV with full metric set (denominators explicit)
    detail_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_per_trace.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["arm", "family", "repetition", "fingerprint", "training_seed", "mean_waiting", "p95_waiting", "max_waiting", "starved", "fairness", "throughput", "cpu_util", "switches", "switching_overhead", "incidents", "interventions", "accepted", "rejected", "inconclusive", "rollback_rate", "recovery_delay_decisions", "recovery_delay_time", "false_intervention_rate", "regression_vs_fcfs", "fallback"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm, traces in per_arm_traces.items():
            for t in traces:
                rel = t["reliability"]
                writer.writerow({
                    "arm": arm,
                    "family": t["workload"].family,
                    "repetition": t["workload"].repetition,
                    "fingerprint": t["workload"].fingerprint,
                    "training_seed": getattr(t["arm_entry"], 'training_seed', ''),
                    "mean_waiting": rel.mean_waiting,
                    "p95_waiting": rel.p95_waiting,
                    "max_waiting": rel.max_waiting,
                    "starved": rel.starved_processes,
                    "fairness": rel.fairness_index,
                    "throughput": rel.throughput,
                    "cpu_util": rel.cpu_utilization,
                    "switches": rel.context_switches,
                    "switching_overhead": rel.switching_overhead,
                    "incidents": rel.incidents,
                    "interventions": rel.interventions,
                    "accepted": rel.accepted,
                    "rejected": rel.rejected,
                    "inconclusive": rel.inconclusive,
                    "rollback_rate": rel.rollback_rate,
                    "recovery_delay_decisions": rel.recovery_delay_decisions,
                    "recovery_delay_time": rel.recovery_delay_time,
                    "false_intervention_rate": rel.false_intervention_rate,
                    "regression_vs_fcfs": None,  # per-trace flag computed via paired workloads; summary gives rate
                    "fallback": rel.fallback_triggered,
                })

    # Paired bootstrap comparisons for mean waiting (key comparisons)
    # Build tidy DataFrame for paired_bootstrap_difference: needs method, family, repetition, fingerprint, training_seed, metric
    paired_df = pd.DataFrame(paired_rows)
    comparisons: List[Dict[str, Any]] = []
    if not paired_df.empty:
        # Key comparisons from healing_experiment (drift-adapted)
        targets = [("rule_wait", "fixed_fcfs"), ("q_recovery", "fixed_fcfs"), ("q_recovery", "rule_wait"), ("rule_wait", "detector_only"), ("q_recovery", "fixed_sjf")]
        for target, ref in targets:
            if target in paired_df["method"].unique() and ref in paired_df["method"].unique():
                try:
                    res = paired_bootstrap_difference(paired_df, target, ref, "mean_waiting", replicates=config.bootstrap_replicates, seed=config.bootstrap_seed, method_column="method", seed_column="training_seed")
                    comparisons.append({"target": target, "reference": ref, "metric": "mean_waiting", **res})
                except Exception as e:
                    comparisons.append({"target": target, "reference": ref, "metric": "mean_waiting", "error": str(e)})

    comp_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_comparisons.csv"
    if comparisons:
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
            writer.writeheader()
            for row in comparisons:
                writer.writerow(row)

    # Markdown report with full methodology and honest interpretation
    lines = []
    lines.append(f"# Drift & Uncertainty Evaluation — {drift_kind} drift, estimator={estimator_enabled} cv={estimator_error_cv}")
    lines.append("")
    lines.append(f"**Workloads:** {len(test_workloads)} traces ({repetitions} reps × {len(config.families)} families: {', '.join(f.name for f in config.families)}), seed {final_test_seed}, drift `{drift_kind}` via workload/drift.py at fraction {DRIFT_INJECTION_FRACTION} (deterministic per-workload seed, pre-injection prefix identical).")
    lines.append(f"**Workload families:** 7 synthetic families (arrival pattern × burst distribution × priority). Training uses 6 families cycling (all but poisson_arrivals, which is held out to test distribution shift); testing uses all 7. Each trace has 15 processes by default; burst_shock multiplies post-injection bursts by 4.0, arrival_storm compresses post-injection gaps by 0.35, etc. See config.py and workload/drift.py for exact generators.")
    est_note = f"""**Estimator:** {'enabled' if estimator_enabled else 'disabled (exact burst baseline)'}; when enabled EWMA per-bucket over completed totals (observable service sums, alpha={config.healing.burst_estimator_alpha}), log-normal multiplicative noise `exp(N(bias, sigma²))` where `sigma = error_cv = {estimator_error_cv}` is the log-space standard deviation (parameter name `burst_estimator_error_cv` retained for backward compatibility; not mathematically identical to the multiplier's CV `sqrt(exp(sigma²)-1)` ≈ 0.307 for sigma=0.30) bias={config.healing.burst_estimator_bias} via derive_seed; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation via EstimationError (not used for decisions). When enabled=False, controller sees exact remaining_burst (simulator assumption, same as frozen study). SJF baseline `fixed_sjf` always uses exact burst_time (oracle, not equal-information) even when estimator is enabled for adaptive arms; no estimated-burst SJF variant is included in this run. Sampling behavior unchanged (parameter name and value retained)."""
    lines.append(est_note)
    lines.append(f"**Training:** {len(models)} Q tables, {training_episodes_per_seed} episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training. Seeds: {list(config.training_seeds)}.")
    lines.append(f"**Evaluation:** Each workload instance presented to every arm (paired by fingerprint); {len(test_workloads)} traces for single-policy arms, {len(models) * len(test_workloads)} for q_recovery ({len(models)} seeds). Metrics via evaluation/reliability.py and evaluation/metrics.py. Bootstrap: family-stratified crossed bootstrap {config.bootstrap_replicates} replicates seed {config.bootstrap_seed}, same as frozen study.")
    lines.append("")
    lines.append("**Formulas & denominators (explicit):**")
    lines.append("- mean_waiting = sum(waiting_time)/n, waiting_time = completion−arrival−burst per process (includes preemptions); per-trace p95 = ceil(0.95n)th ordered waiting — with n=15 per trace, ceil(0.95*15)=15 so per-trace p95 is definitionally the max (not evidence about tail shape); max = max(waiting); pooled_p95_waiting = ceil(0.95*N)th over all process waits pooled across traces (N≈1050 for 70-trace arms, 5250 for q_recovery), interpretable tail; starved = count(waiting > 3×median); fairness = (sum w)²/(n sum w²) (Jain, 1=perfect); throughput = n/total_time; cpu_util = 100×busy/total_time; switches = count(pid changes); switching_overhead = switches×switching_cost.")
    lines.append("- incidents = detector events; interventions = action_changed (HOLD excluded); rollback_rate = (rejected+inconclusive)/interventions (per trace, mean over traces); recovery_delay_decisions = mean verification_decisions per incident (where incidents>0); fallback = any incident with fallback_triggered (safe fallback after 2 consecutive confirmed failures).")
    lines.append("- false_intervention_rate = share of interventions with matched_gain≤0 from counterfactual audit (requires audit; None in lightweight drift script — see frozen study's healing_audit.csv for full audit). regression_rate = share of traces where mean_waiting > baseline trace at same (family,rep) (for q_recovery: mean over 5 seeds per workload vs baseline).")
    lines.append("- Aggregation: per-arm means over traces (for q_recovery over all seed/workload records, not independent workload counts); comparisons are paired per-workload differences, family-stratified.")
    lines.append("")
    lines.append("| Arm | Mean wait | P95 (per-trace mean of max) | Pooled P95 (all waits) | Pooled max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay (dec) | Fallback | Regression vs FCFS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        rr = row.get("rollback_rate")
        rr_s = f"{rr:.2f}" if rr is not None else "n/a"
        rd = row.get("mean_recovery_delay_decisions")
        rd_s = f"{rd:.1f}" if rd is not None else "n/a"
        reg = row.get("regression_rate")
        reg_s = f"{reg:.2f}" if reg is not None else "n/a"
        pp95 = row.get("pooled_p95_waiting")
        pp95_s = f"{pp95:.2f}" if pp95 is not None else "n/a"
        pmax = row.get("pooled_max_waiting")
        pmax_s = f"{pmax:.2f}" if pmax is not None else "n/a"
        lines.append(f"| {row['arm']} | {row['mean_waiting']:.2f} | {row['p95_waiting']:.2f} (=max, n=15) | {pp95_s} (N={row.get('pooled_n',0)}) | {pmax_s} | {row['starved_processes']:.2f} | {row['fairness_index']:.3f} | {row['throughput']:.4f} | {row['cpu_utilization']:.1f} | {row['context_switches']:.1f} | {row['incidents']:.2f} | {row['interventions']:.2f} | {rr_s} | {rd_s} | {row['fallback_rate']:.3f} | {reg_s} |")
    lines.append("")
    if comparisons:
        lines.append("**Paired bootstrap mean-waiting differences (target − reference, 95% CI, family-stratified crossed over seeds×workloads; Q records share workload instances across training seeds and are not independent workloads; crossed bootstrap in evaluation/paired.py::paired_bootstrap_difference resamples seeds and workloads independently retaining pairing and stratification):**")
        lines.append("| Target | Reference | Mean diff | 95% CI |")
        lines.append("|---|---|---:|---|")
        for c in comparisons:
            if "mean_difference" in c:
                lines.append(f"| {c['target']} | {c['reference']} | {c['mean_difference']:.2f} | [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}] |")
            else:
                lines.append(f"| {c['target']} | {c['reference']} | error | {c.get('error','')} |")
        lines.append("")
    lines.append("**Interpretation (evidence-limited, no overclaim):**")
    lines.append("- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified; q_recovery reports mean over 5 seeds and per-seed traces in CSV.")
    lines.extend(_comparison_interpretations(comparisons))
    lines.append("- Fixed SJF receives exact bursts; when the estimator is enabled this is an oracle, not an equal-information comparison.")
    lines.append("- Recovery delays measure verification-window closure, not independently proven recovery. Audit-dependent false-intervention rates are unavailable in this lightweight run.")
    if drift_kind == "phase_transition":
        lines.append("- Phase transition uses short_jobs suffix bursts/priorities drawn with the base workload seed; original arrivals and process IDs are retained.")
    lines.append("- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, declared sample/episode budget, intervals conditional on declared design. Conclusions apply to simulator, not production kernel.")
    lines.append("")
    lines.append("**Traceability:** Every reported mean is sum(col)/n over per_trace CSV rows (fingerprint links to workload); comparisons are paired per-workload differences. Raw per-trace CSV is named in Artifacts below. Metric means use `reliability_summary_for_arms`; this script separately computes fingerprint-paired seed-averaged regression rates and pooled tails.")
    lines.append("**Reproduce:** `PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30` (+ `--no-estimator`, `--drift arrival_storm`).")
    report_path = output_dir / f"drift_{drift_kind}_est{estimator_enabled}_cv{estimator_error_cv}_report.md"
    lines.append(f"**Artifacts:** {summary_path.name}, {detail_path.name}, {comp_path.name if comparisons else 'no comparisons'}, {report_path.name}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nArtifacts: {summary_path}, {detail_path}, {comp_path if comparisons else ''}, {report_path}")

    return {"summary_rows": summary_rows, "per_arm_traces": per_arm_traces, "comparisons": comparisons, "output_dir": str(output_dir), "report_path": str(report_path)}


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
