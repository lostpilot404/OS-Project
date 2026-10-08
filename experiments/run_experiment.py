"""Reproducible multi-seed end-to-end experiment runner.

The experiment trains independent agents on fixed seeds, evaluates each agent on the same
held-out workload set, writes paired per-workload results, and produces a report directly
from those artifacts. Plotting imports are deferred until figures are requested so
``--no-figures`` does not require Matplotlib at import time.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import ACTION_NAMES, ExperimentConfig, build_default_config
from errors import ValidationError
from evaluation.comparison import (
    ALL_METRICS,
    selector_ratio_table,
    best_policy_per_family,
    family_summary,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import (
    SELECTOR_REGIME,
    BASELINE_REGIME,
    EvaluationResult,
    evaluate,
)
from experiments.report import render_report, update_readme_results
from experiments.train import TrainingResult, train
from rl.state import StateEncoder
from workload.generator import WorkloadGenerator

__all__ = ["ExperimentArtifacts", "run_experiment", "config_snapshot", "software_snapshot"]


@dataclass(frozen=True)
class ExperimentArtifacts:
    """All trained models, paired evaluation rows, summaries, and written artifact paths."""

    training: TrainingResult
    training_runs: Tuple[TrainingResult, ...]
    evaluation: EvaluationResult
    summary: Dict[str, object]
    paths: Dict[str, object]


def run_experiment(
    config: ExperimentConfig | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    make_figures: bool = True,
) -> ExperimentArtifacts:
    """Run all configured independent training seeds and the common evaluation set."""
    config = config or build_default_config()
    config.validate()
    results_path = Path(results_dir or config.results_dir)
    figures_path = Path(figures_dir or config.figures_dir)
    generator = WorkloadGenerator(config.families)

    training_runs: List[TrainingResult] = []
    evaluation_runs: List[EvaluationResult] = []
    for seed in config.training.seeds:
        model_config = replace(
            config,
            training=replace(config.training, seed=seed, replicates=1),
        )
        training_result = train(model_config, generator=generator)
        evaluation_result = evaluate(model_config, training_result, generator=generator)
        training_runs.append(training_result)
        evaluation_runs.append(evaluation_result)

    metrics = pd.concat([result.metrics for result in evaluation_runs], ignore_index=True)
    decisions = pd.concat([result.decisions for result in evaluation_runs], ignore_index=True)
    evaluation = EvaluationResult(
        metrics=metrics,
        decisions=decisions,
        families=tuple(family.name for family in config.families),
        repetitions=config.evaluation.repetitions,
        evaluation_seed=config.evaluation.seed,
        training_seeds=tuple(config.training.seeds),
    )
    _check_repeated_baselines(metrics, len(config.training.seeds))
    _check_model_workload_coverage(decisions, len(config.training.seeds))

    summary = _build_summary(config, training_runs, evaluation)
    paths = _write_artifacts(config, training_runs, evaluation, summary, results_path)
    if make_figures:
        # Keep matplotlib and visualization imports out of the --no-figures path.
        from visualization.plots import create_all_figures

        figures = create_all_figures(
            metrics=evaluation.metrics,
            decisions=evaluation.decisions,
            histories=[result.history for result in training_runs],
            q_table_rows=[row for result in training_runs for row in result.q_table_rows()],
            figures_dir=figures_path,
        )
        paths["figures"] = figures
    return ExperimentArtifacts(
        training=training_runs[0],
        training_runs=tuple(training_runs),
        evaluation=evaluation,
        summary=summary,
        paths=paths,
    )


def _build_summary(
    config: ExperimentConfig,
    training_runs: List[TrainingResult],
    evaluation: EvaluationResult,
) -> Dict[str, object]:
    """Derive all report values from the trained models and generated metric rows."""
    summaries = [result.history.summary() for result in training_runs]
    final_rewards = [float(item["mean_reward_all"]) for item in summaries]
    visited = [int(item["visited_states"]) for item in summaries]
    exploration = [float(item["random_exploration_episode_rate"]) for item in summaries]
    training_summary: Dict[str, object] = {
        "runs": summaries,
        "seeds": [result.history.seed for result in training_runs],
        "replicates": len(training_runs),
        "episodes_per_run": config.training.episodes,
        "mean_episode_reward_across_runs": _mean(final_rewards),
        "sd_episode_reward_across_runs": _sample_sd(final_rewards),
        "mean_visited_states": _mean(visited),
        "sd_visited_states": _sample_sd(visited),
        "mean_random_exploration_episode_rate": _mean(exploration),
        "sd_random_exploration_episode_rate": _sample_sd(exploration),
    }

    baseline_summary = policy_summary(evaluation.metrics, BASELINE_REGIME)
    selector_summary = policy_summary(evaluation.metrics, SELECTOR_REGIME)
    ratios = selector_ratio_table(evaluation.metrics)
    family_baseline = family_summary(evaluation.metrics, BASELINE_REGIME)
    family_selector = family_summary(evaluation.metrics, SELECTOR_REGIME)
    selections = policy_selection_table(evaluation.decisions)
    occupancy = state_occupancy_table(evaluation.decisions)
    selection_by_seed = _selection_by_seed(evaluation.decisions)

    workloads = evaluation.decisions.drop_duplicates(
        ["family", "repetition", "workload_fingerprint"]
    )
    training_families = tuple(config.training.family_cycle or (f.name for f in config.families))
    held_out_families = tuple(name for name in evaluation.families if name not in training_families)
    evaluation_summary = evaluation.summary()
    evaluation_summary.update(
        {
            "configured_unique_workloads": len(config.families) * config.evaluation.repetitions,
            "observed_unique_workloads": int(len(workloads)),
            "selector_decision_rows": int(len(evaluation.decisions)),
            "baseline_metric_rows": int(
                len(evaluation.metrics[evaluation.metrics["regime"] == BASELINE_REGIME])
            ),
            "selector_metric_rows": int(
                len(evaluation.metrics[evaluation.metrics["regime"] == SELECTOR_REGIME])
            ),
            "training_families": list(training_families),
            "held_out_families": list(held_out_families),
            "all_baselines_share_same_workloads": True,
        }
    )

    return {
        "configuration": config_snapshot(config),
        "software": software_snapshot(),
        "training": training_summary,
        "evaluation": evaluation_summary,
        "tables": {
            "policy_summary_baselines": _frame_records(baseline_summary),
            "policy_summary_selector": _frame_records(selector_summary),
            "baseline_family_summary": _frame_records(family_baseline),
            "selector_family_summary": _frame_records(family_selector),
            "selector_vs_baselines": _frame_records(ratios),
            "best_baseline_per_family": _frame_records(
                best_policy_per_family(evaluation.metrics)
            ),
            "policy_selection": _frame_records(selections),
            "policy_selection_by_seed": selection_by_seed,
            "state_occupancy": _frame_records(occupancy),
        },
        "metrics_compared": list(ALL_METRICS),
        "reward_metrics": [
            "avg_waiting_time",
            "avg_turnaround_time",
            "avg_response_time",
            "context_switches_per_process",
            "cpu_utilization",
            "throughput",
        ],
        "actions": list(ACTION_NAMES),
    }


def config_snapshot(config: ExperimentConfig) -> Dict[str, object]:
    """Return the complete scientific configuration, including exact state-bin edges."""
    encoder = StateEncoder(config.state)
    ranges = config.state.value_ranges()
    scales = {
        "burst_profile": "logarithmic",
        "burst_dispersion": "linear",
        "offered_load": "logarithmic",
        "priority_spread": "linear",
    }
    training_families = tuple(config.training.family_cycle or (f.name for f in config.families))
    held_out = tuple(f.name for f in config.families if f.name not in training_families)
    return {
        "name": config.name,
        "decision_scope": "offline single-workload batch policy selection; no runtime switching",
        "scheduler": {
            "round_robin_quantum": config.scheduler.round_robin_quantum,
            "switching_cost": config.scheduler.switching_cost,
            "lower_priority_number_is_higher_priority": (
                config.scheduler.lower_priority_number_is_higher_priority
            ),
        },
        "policy_actions": list(ACTION_NAMES),
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
                "staggered_gap_min": family.staggered_gap_min,
                "staggered_gap_max": family.staggered_gap_max,
                "priority_pattern": family.priority_pattern,
                "priority_min": family.priority_min,
                "priority_max": family.priority_max,
                "high_priority_cutoff": family.high_priority_cutoff,
                "high_priority_fraction": family.high_priority_fraction,
            }
            for family in config.families
        ],
        "state": {
            "state_variables_in_mixed_radix_order": list(config.state.state_variables),
            "bins_per_variable": config.state.bins_per_variable,
            "q_table_shape": [config.state.n_states, len(ACTION_NAMES)],
            "n_discrete_states": config.state.n_states,
            "n_state_action_values": config.state.n_states * len(ACTION_NAMES),
            "ranges": {name: list(value) for name, value in ranges.items()},
            "binning_scale": {name: scales[name] for name in config.state.state_variables},
            "exact_interior_edges": {
                name: list(encoder.binner(name).interior_edges)
                for name in config.state.state_variables
            },
            "overflow_behavior": "clamp into lowest or highest open-ended bin",
            "index_order": "first listed variable is most significant; mixed-radix flattening",
        },
        "reward": {
            "formula": (
                "sum(w_cost * (1 - clipped(candidate/reference))) + "
                "sum(w_benefit * (clipped(candidate/reference) - 1))"
            ),
            "reference_policies": list(ACTION_NAMES),
            "reference": "arithmetic mean across the four policies on the identical workload",
            "cost_metrics": [
                "avg_waiting_time",
                "avg_turnaround_time",
                "avg_response_time",
                "context_switches_per_process",
            ],
            "benefit_metrics": ["cpu_utilization", "throughput"],
            "weight_waiting_time": config.reward.weight_waiting_time,
            "weight_turnaround_time": config.reward.weight_turnaround_time,
            "weight_response_time": config.reward.weight_response_time,
            "weight_context_switches": config.reward.weight_context_switches,
            "weight_cpu_utilization": config.reward.weight_cpu_utilization,
            "weight_throughput": config.reward.weight_throughput,
            "reference_clip_upper_bound": config.reward.reference_clip,
            "zero_reference_ratio": 1.0,
        },
        "q_learning": {
            "learning_rate_alpha": config.q_learning.learning_rate,
            "discount_factor_gamma": config.q_learning.discount_factor,
            "gamma_effect_in_project": "none: every workload episode is terminal",
            "initial_q_value": config.q_learning.initial_value,
            "initialization_interpretation": "neutral zero; not optimistic",
            "unseen_state_behavior": (
                "all actions tie initially; np.argmax deterministically selects action 0 (FCFS); "
                "reported as an untrained fallback"
            ),
            "action_selection": "epsilon-greedy; training samples uniformly from four actions with probability epsilon",
            "epsilon_start": config.q_learning.epsilon_start,
            "epsilon_min": config.q_learning.epsilon_min,
            "epsilon_decay_per_episode": config.q_learning.epsilon_decay_per_episode,
            "epsilon_schedule": "max(epsilon_min, epsilon_start * decay ** episode)",
            "training_update": "Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))",
            "evaluation": "greedy, deterministic, no Q-table or visit-count updates",
        },
        "training": {
            "episodes_per_independent_model": config.training.episodes,
            "base_seed": config.training.seed,
            "independent_seeds": list(config.training.seeds),
            "replicates": config.training.replicates,
            "family_cycle": list(config.training.family_cycle),
            "training_families": list(training_families),
            "held_out_families": list(held_out),
        },
        "evaluation": {
            "unique_repetitions_per_family": config.evaluation.repetitions,
            "evaluation_seed": config.evaluation.seed,
            "training_evaluation_fingerprint_overlap_allowed": False,
            "metric_rows_per_unique_workload_per_training_seed": len(ACTION_NAMES) + 1,
        },
    }


def software_snapshot() -> Dict[str, object]:
    """Capture runtime versions and a checksum of the exact dependency lock file."""
    packages: Dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "matplotlib", "pytest"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    lock_path = Path(__file__).resolve().parents[1] / "requirements.lock"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "requirements_lock_sha256": lock_hash,
    }


def _selection_by_seed(decisions: pd.DataFrame) -> List[Dict[str, object]]:
    """Return model-specific selection counts, retaining held-out/fallback distinctions."""
    rows: List[Dict[str, object]] = []
    for (seed, family), group in decisions.groupby(["training_seed", "family"], sort=False):
        record: Dict[str, object] = {
            "training_seed": int(seed),
            "family": family,
            "unique_workloads": int(group["workload_fingerprint"].nunique()),
            "unseen_state_decisions": int((~group["state_seen_in_training"]).sum()),
            "mean_reward": float(group["reward"].mean()),
        }
        counts = group["policy_name"].value_counts()
        for action in ACTION_NAMES:
            record[f"count_{action}"] = int(counts.get(action, 0))
        rows.append(record)
    return rows


def _check_repeated_baselines(metrics: pd.DataFrame, model_count: int) -> None:
    """Ensure every model seed has exactly the same deterministic baseline measurements."""
    baseline = metrics[metrics["regime"] == BASELINE_REGIME]
    metric_columns = list(ALL_METRICS)
    counts = baseline.groupby(["family", "repetition", "policy"], sort=False)[
        "training_seed"
    ].nunique()
    if not counts.eq(model_count).all():
        raise ValidationError("baseline rows are missing from one or more training-seed pairs")
    spread = baseline.groupby(["family", "repetition", "policy"], sort=False)[
        metric_columns
    ].nunique(dropna=False)
    if (spread > 1).any().any():
        raise ValidationError("identical baseline workloads produced inconsistent metrics")


def _check_model_workload_coverage(decisions: pd.DataFrame, model_count: int) -> None:
    """Require one decision from every model for every unique workload."""
    counts = decisions.groupby(["family", "repetition"], sort=False)["training_seed"].nunique()
    if not counts.eq(model_count).all():
        raise ValidationError("one or more workloads are missing decisions from a training seed")
    fingerprint_counts = decisions.groupby(["family", "repetition"], sort=False)[
        "workload_fingerprint"
    ].nunique()
    if not fingerprint_counts.eq(1).all():
        raise ValidationError("training models did not evaluate the same workload inputs")


def _frame_records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Convert a table to JSON-compatible records while retaining index labels."""
    reset = frame.reset_index()
    return json.loads(reset.to_json(orient="records"))


def _write_artifacts(
    config: ExperimentConfig,
    training_runs: List[TrainingResult],
    evaluation: EvaluationResult,
    summary: Dict[str, object],
    results_dir: Path,
) -> Dict[str, object]:
    """Write machine-readable experiment artifacts and a generated report."""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {}
    snapshot = config_snapshot(config)
    _write_json(
        results_dir / "config.json",
        {"configuration": snapshot, "software": software_snapshot()},
    )
    paths["config"] = results_dir / "config.json"

    history_payload = {
        "training_seeds": [result.history.seed for result in training_runs],
        "runs": [result.history.as_dict() for result in training_runs],
    }
    _write_json(results_dir / "training_history.json", history_payload)
    paths["training_history"] = results_dir / "training_history.json"

    q_table = {
        "state_variables": list(config.state.state_variables),
        "action_names": list(ACTION_NAMES),
        "shape_per_training_seed": [config.state.n_states, len(ACTION_NAMES)],
        "initial_value": config.q_learning.initial_value,
        "training_seeds": [result.history.seed for result in training_runs],
        "rows": [row for result in training_runs for row in result.q_table_rows()],
    }
    _write_json(results_dir / "q_table.json", q_table)
    paths["q_table"] = results_dir / "q_table.json"

    metrics = evaluation.metrics
    decisions = evaluation.decisions
    unique_workloads = decisions.drop_duplicates(
        ["family", "repetition", "workload_fingerprint"]
    )
    workload_columns = [
        "family",
        "repetition",
        "workload_fingerprint",
        "selector_num_processes",
        "state_index",
        "state_burst_profile",
        "state_burst_dispersion",
        "state_offered_load",
        "state_priority_spread",
    ]
    workload_frame = unique_workloads[workload_columns].rename(
        columns={"selector_num_processes": "num_processes"}
    )
    paths["workloads"] = _write_frame(workload_frame, results_dir / "workloads.csv")
    paths["metrics"] = _write_frame(metrics, results_dir / "metrics.csv")
    paths["decisions"] = _write_frame(decisions, results_dir / "decisions.csv")
    summary_path = _write_json(results_dir / "summary.json", summary)
    paths["summary"] = summary_path
    # Render both human-readable surfaces from the JSON artifact that users inspect.
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = results_dir / "report.md"
    report_path.write_text(render_report(saved_summary), encoding="utf-8")
    paths["report"] = report_path
    repository_root = Path(__file__).resolve().parents[1]
    if results_dir.resolve() == (repository_root / "results").resolve():
        update_readme_results(repository_root / "README.md", saved_summary)
    return paths


def _mean(values: List[float] | List[int]) -> float:
    """Arithmetic mean with a stable empty-list fallback."""
    return float(sum(values) / len(values)) if values else 0.0


def _sample_sd(values: List[float] | List[int]) -> float:
    """Sample standard deviation (zero for a single replicate)."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Write a deterministic CSV without the pandas index."""
    frame.to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: Dict[str, object]) -> Path:
    """Write formatted JSON and return its path."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path
