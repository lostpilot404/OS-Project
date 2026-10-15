"""Command-line entry point for the offline policy-selection study.

Usage::

    python main.py study               # full study: train, held-out test, artifacts
    python main.py experiment          # preserved legacy offline pipeline
    python main.py train               # preserved legacy offline training only
    python main.py runtime-experiment  # preserved event-driven simulator appendix

The project system is the **offline policy-selection and evaluation study**: one
policy per complete workload (FCFS / SJF / Round Robin / Priority), chosen before
execution by tabular Q-learning from workload characteristics.  It is not a runtime
OS scheduler.  The legacy offline pipeline and the event-driven simulator remain
runnable as preserved references.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from config import ACTION_NAMES, build_default_config
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
            "Offline CPU scheduling policy selection with tabular Q-learning: one "
            "policy per workload, selected from workload characteristics and "
            "evaluated with counterfactuals, regret and structured explanations."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="study",
        choices=("study", "experiment", "train", "runtime-experiment"),
        help=(
            "'study' (default) runs the risk-aware policy-selection study with "
            "ablations, baselines and held-out testing; 'experiment' runs the "
            "preserved legacy offline pipeline; 'train' trains the legacy selector "
            "only; 'runtime-experiment' runs the preserved event-driven simulator "
            "appendix."
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface and return its exit code."""
    args = _build_parser().parse_args(argv)
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
