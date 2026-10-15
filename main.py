"""Command-line entry point for the self-healing CPU scheduling study.

Usage::

    python main.py healing-study       # primary system (default): self-healing study
    python main.py study               # preserved offline policy-selection study
    python main.py experiment          # preserved legacy offline pipeline
    python main.py train               # preserved legacy offline training only
    python main.py runtime-experiment  # preserved non-healing runtime appendix

The project system is the **self-healing scheduler**: a discrete-event simulator that
monitors realized performance, detects degradation against a declared reference with
thresholds, hysteresis and cooldown, diagnoses the cause from observable evidence, asks a
recovery policy (a predeclared rule or a tabular Q-learning controller) for an action,
verifies the action over a declared observation window, and accepts, rolls back or falls
back safely while recording every decision.  The offline whole-workload selector and the
non-healing runtime simulator are preserved as compatibility/historical workflows; they
are not the project system.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from config import ACTION_NAMES, build_default_config
from experiments.healing_experiment import run_healing_experiment
from experiments.run_experiment import run_experiment
from experiments.runtime_experiment import run_runtime_experiment
from experiments.study import run_study
from experiments.train import train

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Adaptive CPU scheduling with a self-healing scheduler: monitor, detect, "
            "diagnose, recover with a rule or a tabular Q-learning controller, verify, "
            "roll back. Other commands expose the preserved historical workflows."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="healing-study",
        choices=(
            "healing-study",
            "study",
            "experiment",
            "train",
            "runtime-experiment",
        ),
        help=(
            "'healing-study' (default) runs the self-healing scheduling study: training, "
            "held-out evaluation, recovery-ledger and counterfactual-recovery audit; "
            "'study' runs the preserved offline policy-selection study; 'experiment' runs "
            "the preserved legacy offline pipeline; 'train' trains the legacy selector "
            "only; 'runtime-experiment' runs the preserved non-healing runtime appendix."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where result files are written (default: configured results directory)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="where figures are written (default: configured figures directory)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip optional Matplotlib figures; result tables and report are still generated",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help=(
            "skip the validation action-effect probe (the reward-versus-ground-truth "
            "analysis); the rest of the study is unaffected"
        ),
    )
    return parser


def _report_training(history) -> None:
    """Print the measured summary for one training seed."""
    summary = history.summary()
    print(f"Training seed {summary['seed']}")
    print("-" * (14 + len(str(summary["seed"]))))
    print(f"  terminal episodes              : {summary['episodes']}")
    print(f"  mean reward, first 100          : {summary['mean_reward_first_100']:+.4f}")
    print(f"  mean reward, last 100           : {summary['mean_reward_last_100']:+.4f}")
    print(f"  mean reward, all episodes       : {summary['mean_reward_all']:+.4f}")
    print(f"  random-action branch rate       : {summary['random_exploration_episode_rate']:.1%}")
    print(f"  final epsilon probability       : {summary['final_epsilon']:.4f}")
    print(f"  states visited                  : {summary['visited_states']} / {summary['n_states']}")
    print("  selected policies:")
    for name in ACTION_NAMES:
        print(f"    {name:<12}: {summary['action_counts'][name]}")


def _report_evaluation(evaluation, summary) -> None:
    """Print unique workload, model decision, and aggregate comparison counts."""
    values = summary["evaluation"]
    print()
    print("Evaluation")
    print("----------")
    print(f"  workload conditions            : {', '.join(evaluation.families)}")
    print(f"  unique workloads               : {values['observed_unique_workloads']}")
    print(f"  repetitions per condition      : {evaluation.repetitions}")
    print(f"  independent training seeds     : {', '.join(map(str, evaluation.training_seeds))}")
    print(f"  model decision rows            : {values['selector_decision_rows']}")
    print(f"  evaluation master seed         : {evaluation.evaluation_seed}")
    print()
    ratios = summary["tables"]["selector_vs_baselines"]
    print("Aggregate mean reductions (positive means selector mean is lower):")
    for policy in ACTION_NAMES:
        row = next(
            record
            for record in ratios
            if record["policy"] == policy and record["metric"] == "avg_waiting_time"
        )
        baseline = float(row["mean_baseline"])
        selector = float(row["mean_selector"])
        reduction = 100.0 * (baseline - selector) / baseline if baseline else float("nan")
        print(f"  vs {policy:<12}: {reduction:+.2f}% waiting-time reduction")
    print()
    print("Learned selections by condition (pooled across independent training seeds):")
    for record in summary["tables"]["policy_selection"]:
        selected = [
            f"{name} {record[f'count_{name}']}"
            for name in ACTION_NAMES
            if record[f"count_{name}"]
        ]
        print(
            f"  {record['family']:<22} {', '.join(selected):<44} "
            f"unseen-state fallback rows {record['unseen_state_decisions']}"
        )
    print("\nDetailed generated result report: results/report.md (or --results-dir/report.md).")


def _report_study(artifacts) -> None:
    """Print the measured study summary."""
    summary = artifacts.summary
    test = summary["test"]
    design = summary["configuration"]["design"]
    print("Offline policy-selection study (risk-aware objective)")
    print("----------------------------------------------------")
    print(f"  primary objective              : {design['primary_objective']}")
    print(f"  training seeds                 : {', '.join(map(str, design['training_seeds']))}")
    print(f"  training episodes / seed       : {design['training_episodes']}")
    print(f"  held-out test workloads        : {test['workloads']}")
    print(f"  best fixed policy (train only) : {design['best_fixed_action']}")
    print()
    print("  Test-split mean declared-objective cost / regret:")
    for arm, values in test["arms"].items():
        print(
            f"    {arm:<22}: cost {values['mean_objective_cost']:.4f}   "
            f"regret {values['mean_regret']:.4f}   "
            f"hindsight-best {100 * values['hindsight_best_rate']:.1f}%"
        )
    print()
    print("Artifacts written")
    print("-----------------")
    for name, path in artifacts.paths.items():
        if name == "figures":
            for figure in path:
                print(f"  figure: {figure}")
        else:
            print(f"  {name:<22}: {path}")


def _share(value) -> str:
    """Format an optional share: undefined shares are printed as ``n/a``, never as a number."""
    return "n/a" if value is None else f"{float(value):.1%}"


def _number(value, width: int = 6) -> str:
    """Format an optional number, printing ``n/a`` when the artifact has no value."""
    return "   n/a" if value is None else f"{float(value):>{width}.2f}"


def _report_healing_study(artifacts) -> None:
    """Print the measured outcome of the self-healing study."""
    summary = artifacts.summary
    print("Self-healing CPU scheduling study")
    print("--------------------------------")
    print(f"  final-test workloads          : {summary['workloads']}")
    print(f"  independent training seeds    : {', '.join(map(str, summary['training_seeds']))}")
    print(f"  held-out final-test seed      : {summary['final_test_seed']}")
    print()
    print("Final-test means")
    print("----------------")
    print(f"  {'arm':<24}{'wait':>9}{'p95':>9}{'resp':>9}{'incid':>8}{'intv':>7}")
    for row in summary["means"]:
        print(
            f"  {row['arm']:<24}{row['avg_waiting_time']:>9.2f}{row['p95_waiting_time']:>9.2f}"
            f"{row['avg_response_time']:>9.2f}{row['incidents']:>8.2f}{row['interventions']:>7.2f}"
        )
    print()
    print("Verification verdicts on interventions")
    print("--------------------------------------")
    for row in summary["detection_summary"]:
        if row["split"] != "final_test" or not row["interventions"]:
            continue
        print(
            f"  {row['arm']:<24} interventions={row['interventions']:<5} "
            f"accepted={_share(row['accepted_share_of_interventions'])}  "
            f"rejected={_share(row['rejected_share_of_interventions'])}  "
            f"inconclusive={_share(row['inconclusive_share_of_interventions'])}  "
            f"episodes with fallback={row['episodes_with_fallback']}"
        )
    if summary["audit_summary"]:
        print()
        print("Counterfactual audit of interventions (held-out split)")
        print("------------------------------------------------------")
        print("  matched replay: same controller, this one incident forced to hold")
        for row in summary["audit_summary"]:
            print(
                f"  {row['arm']:<24} n={row['audited_interventions']:<5} "
                f"matched gain {_number(row['mean_matched_gain'], 8)}  "
                f"never-heal gain {_number(row['episode_level_mean_matched_gain'], 8)}  "
                f"helped {_share(1 - row['false_recovery_rate'])}  "
                f"| not certified {_share(row['unverified_share_of_helpful'])} of the helpful ones"
            )
    if summary.get("detection_scenarios"):
        print()
        print("Detector on controlled degradations (validation split)")
        print("------------------------------------------------------")
        for row in summary["detection_scenarios"]:
            if row["kind"] == "summary":
                print(
                    f"  recall on injected traces {_share(row['recall_on_injected'])}  "
                    f"precision proxy {_share(row['precision_proxy'])}  "
                    f"median delay {_number(row['median_detection_delay_decisions'], 5)} decisions  "
                    f"false triggers/trace {_number(row['false_trigger_rate_per_stable_trace'], 5)}"
                )
                continue
            print(
                f"  {row['scenario']:<14} traces={row['traces']:<4} "
                f"detected={_share(row['detection_share'])}  "
                f"median delay={_number(row['median_delay_time_units'], 7)}  "
                f"incidents/trace={_number(row['incidents_per_trace'], 5)}"
            )
    print()
    print("Artifacts")
    print("---------")
    for name, path in sorted(artifacts.paths.items()):
        print(f"  {name:<20} {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface and return its exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "healing-study":
        artifacts = run_healing_experiment(
            results_dir=args.results_dir or Path("results/healing"),
            figures_dir=args.figures_dir or Path("figures/healing"),
            make_figures=not args.no_figures,
            probe=not args.no_probe,
        )
        _report_healing_study(artifacts)
        return 0

    if args.command == "study":
        artifacts = run_study(
            results_dir=args.results_dir or Path("results/study"),
            figures_dir=args.figures_dir or Path("figures/study"),
            make_figures=not args.no_figures,
        )
        _report_study(artifacts)
        return 0

    if args.command == "runtime-experiment":
        artifacts = run_runtime_experiment(
            results_dir=args.results_dir or Path("results/runtime")
        )
        summary = artifacts.summary
        print("Causal event-driven runtime experiment")
        print("---------------------------------------")
        print(
            f"  training seeds                 : "
            f"{', '.join(map(str, summary['configuration']['training_seeds']))}"
        )
        print(f"  training episodes/model        : {summary['configuration']['training_episodes_per_seed']}")
        print(f"  validation workloads           : {summary['validation_unique_workloads']}")
        print(f"  final held-out test workloads  : {summary['final_test_unique_workloads']}")
        print("  final-test mean waiting time:")
        for method, values in summary["final_test_means"].items():
            if "avg_waiting_time" in values:
                print(f"    {method:<22}: {values['avg_waiting_time']:.3f}")
        demo = summary["demo"]
        print(
            f"  learned within-trace switches : {demo['policy_switch_count']} "
            f"({demo['decision_count']} decisions; {demo['path']})"
        )
        print("  artifacts:")
        for name, path in artifacts.paths.items():
            print(f"    {name:<24}: {path}")
        return 0

    config = build_default_config()
    if args.command == "train":
        for seed in config.training.seeds:
            model_config = replace(
                config,
                training=replace(config.training, seed=seed, replicates=1),
            )
            _report_training(train(model_config).history)
        return 0

    artifacts = run_experiment(
        config=config,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        make_figures=not args.no_figures,
    )
    _report_evaluation(artifacts.evaluation, artifacts.summary)
    print()
    print("Artifacts written")
    print("-----------------")
    for name, path in artifacts.paths.items():
        if name == "figures":
            for figure in path:
                print(f"  figure: {figure}")
        else:
            print(f"  {name:<18}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
