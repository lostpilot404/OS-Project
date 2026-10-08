"""Command-line entry point of the project.

Usage::

    python3 main.py experiment        # train, evaluate, tabulate, plot  (default)
    python3 main.py train             # train only and report the training history

Every command uses the frozen configuration in :func:`config.build_default_config`; no
hyperparameter can be changed from the command line, so a reported result can always be
traced back to an exact configuration (which is written to ``results/config.json``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from config import ACTION_NAMES, build_default_config
from experiments.evaluate import ADAPTIVE_CLASSIC_REGIME
from experiments.run_experiment import run_experiment
from experiments.train import train

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Reinforcement-learning-based workload-aware CPU scheduling with dynamic "
            "policy selection (tabular Q-learning over FCFS, SJF, Round Robin and "
            "Priority on a single simulated CPU)."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="experiment",
        choices=("experiment", "train"),
        help="'experiment' trains, evaluates, tabulates and plots; 'train' only trains.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where result files are written (default: the configured results directory)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="where figures are written (default: the configured figures directory)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip figure creation (useful on machines without Matplotlib)",
    )
    return parser


def _report_training(history) -> None:
    """Print the training summary."""
    summary = history.summary()
    print("Training")
    print("--------")
    print(f"  episodes                       : {summary['episodes']}")
    print(f"  master seed                    : {summary['seed']}")
    print(f"  mean reward, first 100 episodes: {summary['mean_reward_first_100']:+.4f}")
    print(f"  mean reward, last 100 episodes : {summary['mean_reward_last_100']:+.4f}")
    print(f"  exploration rate (all episodes): {summary['exploration_rate']:.1%}")
    print(f"  final epsilon                  : {summary['final_epsilon']:.4f}")
    print(f"  visited states                 : {summary['visited_states']} of {summary['n_states']}")
    print("  policy selections (all episodes):")
    for name in ACTION_NAMES:
        print(f"    {name:<12}: {summary['action_counts'][name]}")
    print("  Round-Robin quantum multipliers chosen by the controller:")
    for multiplier, count in summary["controller_multiplier_counts"].items():
        print(f"    x{multiplier:<5}: {count}")


def _report_evaluation(evaluation, summary) -> None:
    """Print the measured evaluation results."""
    print()
    print("Evaluation")
    print("----------")
    print(f"  workload conditions            : {', '.join(evaluation.families)}")
    print(f"  repetitions per condition      : {evaluation.repetitions}")
    print(f"  evaluation master seed         : {evaluation.evaluation_seed}")
    print(f"  distinct workloads evaluated   : {evaluation.metrics['workload_fingerprint'].nunique()}")
    print()

    ratios = summary["tables"]["adaptive_ratio_vs_baselines"]
    print("  Baseline vs adaptive scheduler, workload by workload (mean ratio > 1 means")
    print("  the baseline is worse for a cost metric such as waiting time):")
    for policy in ACTION_NAMES:
        for metric in ("avg_waiting_time", "avg_turnaround_time", "avg_response_time", "throughput"):
            row = next(
                (
                    record
                    for record in ratios
                    if record["policy"] == policy and record["metric"] == metric
                ),
                None,
            )
            if row is None:
                continue
            print(
                f"    {policy:<12} {metric:<20} baseline={row['mean_baseline']:.4f} "
                f"adaptive={row['mean_adaptive']:.4f} ratio={row['mean_ratio']:.4f}"
            )
    print()

    print("  Policy selected by the agent, per workload condition:")
    for record in summary["tables"]["policy_selection"]:
        selected = sorted(
            (
                (name, record[f"count_{name}"])
                for name in ACTION_NAMES
                if record.get(f"count_{name}")
            ),
            key=lambda item: (-item[1], item[0]),
        )
        chosen = ", ".join(f"{name} x{count}" for name, count in selected) or "none"
        print(
            f"    {record['family']:<18} {chosen:<40} mean reward "
            f"{record['mean_reward']:+.4f}"
        )
    print()
    print(f"  Adaptive regime reported above: {ADAPTIVE_CLASSIC_REGIME}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    config = build_default_config()

    if args.command == "train":
        result = train(config)
        _report_training(result.history)
        return 0

    artifacts = run_experiment(
        config=config,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        make_figures=not args.no_figures,
    )
    _report_training(artifacts.training.history)
    _report_evaluation(artifacts.evaluation, artifacts.summary)

    print()
    print("Artefacts written")
    print("-----------------")
    for name, path in artifacts.paths.items():
        if name == "figures":
            for figure in path:  # type: ignore[union-attr]
                print(f"  figure: {figure}")
        else:
            print(f"  {name:<16}: {path}")
    print()
    print("Every number above was measured by this run; see README.md for interpretation.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
