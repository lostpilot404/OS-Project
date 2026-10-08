"""The full experiment: verify the classes, train, evaluate, tabulate, plot and save.

This module is the single entry point used by ``main.py experiment``.  It performs the
complete study in a fixed order and writes every artefact to disk, so that a result can
always be traced back to the run that produced it:

``results/config.json``              the exact configuration of the run
``results/class_verification.json``  the pre-training verification of the workload classes
``results/training_history.json``    per-episode training record
``results/q_table.json``             the learned Q-table with its state bins
``results/workloads.csv``            every evaluated workload and its fingerprint
``results/metrics.csv``              one row per (workload, scheduler, regime)
``results/decisions.csv``            one row per adaptive decision
``results/summary.json``             the tables and headline numbers
``figures/*.png``                    the figures

The order matters and is part of the experimental design: the workload classes are
**verified first** (which conventional policy does each class actually favour, and is
the reward-optimal action a consistent function of the encoded state?), then the agent is
**trained** on the training stream, and finally it is **evaluated** on held-out
workloads from a different master seed.  Nothing is written by hand: every number in
every artefact is computed from the scheduling runs of this execution.
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
    oracle_agreement_table,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import ADAPTIVE_REGIME, BASELINE_REGIME, EvaluationResult, evaluate
from experiments.train import TrainingResult, train
from experiments.verify_classes import VerificationResult, verify_classes
from rl.state import StateEncoder
from visualization.plots import create_all_figures
from workload.generator import WorkloadGenerator

__all__ = ["ExperimentArtifacts", "run_experiment"]


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Paths and tables produced by a full experiment run.

    Attributes:
        verification: The pre-training verification of the workload classes.
        training: The training result.
        evaluation: The evaluation result.
        summary: The headline numbers and tables, ready for JSON serialisation.
        paths: Written artefact paths keyed by artefact name.
    """

    verification: VerificationResult
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
        The verification, training and evaluation results, the summary and the written
        paths.
    """
    config = config or build_default_config()
    config.validate()
    results_dir = Path(results_dir or config.results_dir)
    figures_dir = Path(figures_dir or config.figures_dir)

    generator = WorkloadGenerator(config.families)
    encoder = StateEncoder(config.state)

    # 1. Verify the workload classes before anything is trained: which policy does each
    #    class favour, and is the reward-optimal action a function of the state?
    verification = verify_classes(config, generator=generator, encoder=encoder)
    # 2. Train on the training stream.
    training = train(config, generator=generator, encoder=encoder)
    # 3. Evaluate on held-out workloads from a different master seed.
    evaluation = evaluate(config, training, generator=generator)

    summary = _build_summary(config, verification, training, evaluation)
    paths = _write_artifacts(
        config, verification, training, evaluation, summary, results_dir
    )
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
        verification=verification,
        training=training,
        evaluation=evaluation,
        summary=summary,
        paths=paths,
    )


def _build_summary(
    config: ExperimentConfig,
    verification: VerificationResult,
    training: TrainingResult,
    evaluation: EvaluationResult,
) -> Dict[str, object]:
    """Assemble every reported table and headline number."""
    summary: Dict[str, object] = {
        "configuration": config_snapshot(config),
        "verification": verification.summary(),
        "training": training.history.summary(),
        "evaluation": evaluation.summary(),
        "tables": {
            "policy_summary_baselines": _frame_records(
                policy_summary(evaluation.metrics, BASELINE_REGIME)
            ),
            "policy_summary_adaptive": _frame_records(
                policy_summary(evaluation.metrics, ADAPTIVE_REGIME)
            ),
            "adaptive_ratio_vs_baselines": _frame_records(
                adaptive_ratio_table(evaluation.metrics)
            ),
            "best_baseline_per_family": _frame_records(best_policy_per_family(evaluation.metrics)),
            "family_policy_means": _frame_records(family_summary(evaluation.metrics)),
            "policy_selection": _frame_records(policy_selection_table(evaluation.decisions)),
            "state_occupancy": _frame_records(state_occupancy_table(evaluation.decisions)),
            "oracle_agreement": _frame_records(oracle_agreement_table(evaluation.decisions)),
        },
        "metrics_compared": list(ALL_METRICS),
        "actions": list(ACTION_NAMES),
    }
    return summary


def config_snapshot(config: ExperimentConfig) -> Dict[str, object]:
    """Return a JSON-serialisable snapshot of the configuration.

    This is what makes a run reproducible: it records every hyperparameter, the quantum,
    the workload-generation rules, the state variables and ranges, all three master
    seeds and the number of episodes.
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
                "long_burst_min": family.long_burst_min,
                "arrival_pattern": family.arrival_pattern,
                "arrival_window": family.arrival_window,
                "head_window": family.head_window,
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
            "long_job_share_range": list(config.state.long_job_share_range),
            "arrival_concentration_range": list(config.state.arrival_concentration_range),
            "offered_load_range": list(config.state.offered_load_range),
            "priority_spread_range": list(config.state.priority_spread_range),
            "priority_burst_alignment_range": list(config.state.priority_burst_alignment_range),
            "long_burst_threshold": config.state.long_burst_threshold,
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
        "training": {
            "episodes": config.training.episodes,
            "seed": config.training.seed,
            "family_cycle": list(config.training.family_cycle),
        },
        "evaluation": {
            "repetitions": config.evaluation.repetitions,
            "seed": config.evaluation.seed,
            "verification_repetitions": config.evaluation.verification_repetitions,
            "verification_seed": config.evaluation.verification_seed,
        },
    }


def _frame_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Convert a table into JSON-serialisable records, keeping the index as columns."""
    reset = frame.reset_index()
    records: List[Dict[str, object]] = json.loads(reset.to_json(orient="records"))
    return records


def _write_artifacts(
    config: ExperimentConfig,
    verification: VerificationResult,
    training: TrainingResult,
    evaluation: EvaluationResult,
    summary: Dict[str, object],
    results_dir: Path,
) -> Dict[str, Path]:
    """Write every result file and return the paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    paths["config"] = _write_json(results_dir / "config.json", config_snapshot(config))
    paths["class_verification"] = verification.to_json(
        results_dir / "class_verification.json"
    )
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
