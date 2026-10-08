"""The full experiment: train, evaluate, tabulate, plot and save.

This module is the single entry point used by ``main.py experiment``.  It performs the
complete study in a fixed order and writes every artefact (configuration, training
history, Q-table, per-workload results, summary tables and figures) to disk, so that a
result can always be traced back to the run that produced it:

``results/config.json``              the exact configuration of the run
``results/training_history.json``    per-episode training record
``results/q_table.json``             the learned Q-table with its state bins
``results/workloads.csv``            every evaluated workload and its fingerprint
``results/metrics.csv``              one row per (workload, scheduler, regime)
``results/decisions.csv``            one row per adaptive decision
``results/summary.json``             the tables and headline numbers
``figures/*.png``                    the figures

Nothing is written by hand: every number in every artefact is computed from the
scheduling runs of this execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import ACTION_NAMES, ExperimentConfig, build_default_config
from evaluation.comparison import (
    ALL_METRICS,
    adaptive_ratio_table,
    best_policy_per_family,
    family_summary,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import (
    ADAPTIVE_CLASSIC_REGIME,
    ADAPTIVE_LEARNED_REGIME,
    ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
    EvaluationResult,
    evaluate,
)
from experiments.train import TrainingResult, train
from visualization.plots import create_all_figures
from workload.generator import WorkloadGenerator

__all__ = ["ExperimentArtifacts", "run_experiment"]


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Paths and tables produced by a full experiment run.

    Attributes:
        training: The training result.
        evaluation: The evaluation result.
        summary: The headline numbers and tables, ready for JSON serialisation.
        paths: Written artefact paths keyed by artefact name.
    """

    training: TrainingResult
    evaluation: EvaluationResult
    summary: Dict[str, object]
    paths: Dict[str, Path]


def run_experiment(
    config: Optional[ExperimentConfig] = None,
    results_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    make_figures: bool = True,
) -> ExperimentArtifacts:
    """Run the complete study and write all artefacts.

    Args:
        config: Configuration to use; the frozen default is used when omitted.
        results_dir: Directory for result files; defaults to the configured directory.
        figures_dir: Directory for figures; defaults to the configured directory.
        make_figures: Whether to create the Matplotlib figures.

    Returns:
        The training and evaluation results, the summary and the written paths.
    """
    config = config or build_default_config()
    config.validate()
    results_dir = Path(results_dir or config.results_dir)
    figures_dir = Path(figures_dir or config.figures_dir)

    generator = WorkloadGenerator(config.families)
    training = train(config, generator=generator)
    evaluation = evaluate(config, training, generator=generator)

    summary = _build_summary(config, training, evaluation)
    paths = _write_artifacts(config, training, evaluation, summary, results_dir)
    if make_figures:
        figures = create_all_figures(
            metrics=evaluation.metrics,
            decisions=evaluation.decisions,
            history=training.history,
            q_table_rows=training.q_table_rows(),
            figures_dir=figures_dir,
        )
        paths["figures"] = figures
    return ExperimentArtifacts(
        training=training, evaluation=evaluation, summary=summary, paths=paths
    )


def _build_summary(
    config: ExperimentConfig, training: TrainingResult, evaluation: EvaluationResult
) -> Dict[str, object]:
    """Assemble every reported table and headline number."""
    baseline_regime = "baseline"
    summary: Dict[str, object] = {
        "configuration": config_snapshot(config),
        "training": training.history.summary(),
        "evaluation": evaluation.summary(),
        "tables": {
            "policy_summary_baselines": _frame_records(
                policy_summary(evaluation.metrics, baseline_regime)
            ),
            "policy_summary_adaptive_classic": _frame_records(
                policy_summary(evaluation.metrics, ADAPTIVE_CLASSIC_REGIME)
            ),
            "adaptive_ratio_vs_baselines": _frame_records(
                adaptive_ratio_table(evaluation.metrics)
            ),
            "best_baseline_per_family": _frame_records(best_policy_per_family(evaluation.metrics)),
            "family_policy_means": _frame_records(family_summary(evaluation.metrics)),
            "policy_selection": _frame_records(policy_selection_table(evaluation.decisions)),
            "state_occupancy": _frame_records(state_occupancy_table(evaluation.decisions)),
        },
        "metrics_compared": list(ALL_METRICS),
        "actions": list(ACTION_NAMES),
    }
    regimes = set(evaluation.metrics["regime"])
    if ADAPTIVE_LEARNED_REGIME in regimes:
        summary["tables"]["policy_summary_adaptive_learned_quantum"] = _frame_records(
            policy_summary(evaluation.metrics, ADAPTIVE_LEARNED_REGIME)
        )
    if ROUND_ROBIN_LEARNED_QUANTUM_REGIME in regimes:
        summary["tables"]["policy_summary_round_robin_learned_quantum"] = _frame_records(
            policy_summary(evaluation.metrics, ROUND_ROBIN_LEARNED_QUANTUM_REGIME)
        )
        summary["tables"]["round_robin_classic_vs_learned_quantum"] = _frame_records(
            adaptive_ratio_table(
                evaluation.metrics,
                adaptive_regime=ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
                baselines=["Round Robin"],
            )
        )
        multiplier_counts = (
            evaluation.metrics.loc[
                evaluation.metrics["regime"] == ROUND_ROBIN_LEARNED_QUANTUM_REGIME
            ]
            .groupby("family")
            .size()
        )
        summary["round_robin_learned_quantum_workloads"] = int(multiplier_counts.sum())
    return summary


def config_snapshot(config: ExperimentConfig) -> Dict[str, object]:
    """Return a JSON-serialisable snapshot of the configuration.

    This is what makes a run reproducible: it records every hyperparameter, the quantum,
    the workload-generation rules, both master seeds and the number of episodes.
    """
    return {
        "name": config.name,
        "scheduler": {
            "round_robin_quantum": config.scheduler.round_robin_quantum,
            "switching_cost": config.scheduler.switching_cost,
            "lower_priority_number_is_higher_priority": (
                config.scheduler.lower_priority_number_is_higher_priority
            ),
        },
        "workload_families": [
            {
                "name": family.name,
                "description": family.description,
                "num_processes": family.num_processes,
                "burst_distribution": family.burst_distribution,
                "burst_time_min": family.burst_time_min,
                "burst_time_max": family.burst_time_max,
                "short_burst_max": family.short_burst_max,
                "short_burst_fraction": family.short_burst_fraction,
                "arrival_pattern": family.arrival_pattern,
                "arrival_window": family.arrival_window,
                "arrival_rate": family.arrival_rate,
                "priority_pattern": family.priority_pattern,
                "priority_min": family.priority_min,
                "priority_max": family.priority_max,
                "high_priority_cutoff": family.high_priority_cutoff,
                "high_priority_fraction": family.high_priority_fraction,
            }
            for family in config.families
        ],
        "state": {
            "state_variables": list(config.state.state_variables),
            "bins_per_variable": config.state.bins_per_variable,
            "n_states": config.state.n_states,
            "burst_profile_range": list(config.state.burst_profile_range),
            "burst_dispersion_range": list(config.state.burst_dispersion_range),
            "offered_load_range": list(config.state.offered_load_range),
            "priority_spread_range": list(config.state.priority_spread_range),
        },
        "reward": {
            "weight_waiting_time": config.reward.weight_waiting_time,
            "weight_turnaround_time": config.reward.weight_turnaround_time,
            "weight_response_time": config.reward.weight_response_time,
            "weight_context_switches": config.reward.weight_context_switches,
            "weight_cpu_utilization": config.reward.weight_cpu_utilization,
            "weight_throughput": config.reward.weight_throughput,
            "reference_clip": config.reward.reference_clip,
        },
        "q_learning": {
            "learning_rate": config.q_learning.learning_rate,
            "discount_factor": config.q_learning.discount_factor,
            "epsilon_start": config.q_learning.epsilon_start,
            "epsilon_min": config.q_learning.epsilon_min,
            "epsilon_decay_per_episode": config.q_learning.epsilon_decay_per_episode,
            "initial_value": config.q_learning.initial_value,
        },
        "quantum_controller": {
            "enabled": config.quantum_controller.enabled,
            "use_during_evaluation": config.quantum_controller.use_during_evaluation,
            "multipliers": list(config.quantum_controller.multipliers),
            "learning_rate": config.quantum_controller.learning_rate,
            "discount_factor": config.quantum_controller.discount_factor,
            "epsilon_start": config.quantum_controller.epsilon_start,
            "epsilon_min": config.quantum_controller.epsilon_min,
            "epsilon_decay_per_episode": config.quantum_controller.epsilon_decay_per_episode,
        },
        "training": {
            "episodes": config.training.episodes,
            "seed": config.training.seed,
            "family_cycle": list(config.training.family_cycle),
        },
        "evaluation": {
            "repetitions": config.evaluation.repetitions,
            "seed": config.evaluation.seed,
        },
    }


def _frame_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Convert a table into JSON-serialisable records, keeping the index as columns."""
    reset = frame.reset_index()
    records: List[Dict[str, object]] = json.loads(reset.to_json(orient="records"))
    return records


def _write_artifacts(
    config: ExperimentConfig,
    training: TrainingResult,
    evaluation: EvaluationResult,
    summary: Dict[str, object],
    results_dir: Path,
) -> Dict[str, Path]:
    """Write every result file and return the paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    paths["config"] = _write_json(results_dir / "config.json", config_snapshot(config))
    paths["training_history"] = training.history.to_json(results_dir / "training_history.json")
    paths["q_table"] = _write_json(
        results_dir / "q_table.json",
        {
            "state_variables": list(config.state.state_variables),
            "bins_per_variable": config.state.bins_per_variable,
            "rows": training.q_table_rows(),
        },
    )

    workloads = sorted(
        {
            (row.family, int(row.repetition), row.workload_fingerprint, int(row.num_processes))
            for row in evaluation.metrics.itertuples()
        }
    )
    workloads_frame = pd.DataFrame(
        workloads, columns=["family", "repetition", "workload_fingerprint", "num_processes"]
    )
    paths["workloads"] = _write_frame(workloads_frame, results_dir / "workloads.csv")

    paths["metrics"] = _write_frame(evaluation.metrics, results_dir / "metrics.csv")
    paths["decisions"] = _write_frame(evaluation.decisions, results_dir / "decisions.csv")
    paths["summary"] = _write_json(results_dir / "summary.json", summary)
    return paths


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Write a data frame as CSV and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: Dict[str, object]) -> Path:
    """Write a JSON document and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
