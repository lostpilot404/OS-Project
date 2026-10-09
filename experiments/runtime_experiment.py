"""Reproducible training/evaluation pipeline for causal runtime adaptation.

This experiment is separate from the legacy offline selector. Models are trained on
sequential traces, validation workloads are distinct, and final-test workloads are not
generated until all agents have been trained and validation has completed. All methods
receive the same final workload instances. The four fixed baselines use the existing
standalone scheduler implementations.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES, derive_seed
from errors import ValidationError
from evaluation.comparison import ALL_METRICS
from evaluation.metrics import compute_metrics
from experiments.runtime_config import RuntimeExperimentConfig
from rl.runtime_controller import RuntimeQController
from rl.runtime_heuristic import CausalHeuristic
from rl.runtime_state import RuntimeStateEncoder
from scheduler import POLICY_CLASSES
from scheduler.runtime import RuntimeDecisionRecord, RuntimeRun, RuntimeSimulator
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = [
    "RuntimeWorkload",
    "TrainedRuntimeModel",
    "RuntimeExperimentArtifacts",
    "build_runtime_workloads",
    "run_runtime_experiment",
]

Q_METHOD = "Runtime Q-learning"
HEURISTIC_METHOD = "Causal heuristic"
DEMO_SELECTION_RULE = (
    "validation only: maximize policy switches among Q traces with zero unseen-state "
    "fallbacks; if none qualify, use all validation Q traces; break ties by ascending "
    "training seed, family, then repetition"
)
_SPLIT_TAGS = {"training": 11, "validation": 23, "final_test": 37}


@dataclass(frozen=True)
class RuntimeWorkload:
    """One reproducible workload and its experimental design coordinates."""

    split: str
    family: str
    repetition: int
    seed: int
    workload: Workload

    @property
    def fingerprint(self) -> str:
        return self.workload.fingerprint


@dataclass(frozen=True)
class TrainedRuntimeModel:
    """Sequentially trained Q controller and its disjoint training workload set."""

    training_seed: int
    controller: RuntimeQController
    training_fingerprints: frozenset[str]
    train_wall_ns: int
    update_ns: int


@dataclass(frozen=True)
class RuntimeExperimentArtifacts:
    """Trained agents, paired tables, summaries, and written paths."""

    models: Tuple[TrainedRuntimeModel, ...]
    training_metrics: pd.DataFrame
    validation_metrics: pd.DataFrame
    final_test_metrics: pd.DataFrame
    validation_decisions: pd.DataFrame
    final_test_decisions: pd.DataFrame
    summary: Dict[str, Any]
    paths: Dict[str, Path]


def build_runtime_workloads(
    split: str,
    config: RuntimeExperimentConfig,
    generator: Optional[WorkloadGenerator] = None,
) -> List[RuntimeWorkload]:
    """Build one seeded workload per family/repetition in a named split."""
    if split not in _SPLIT_TAGS:
        raise ValidationError(f"unknown runtime split {split!r}")
    generator = generator or WorkloadGenerator(config.families)
    if split == "training":
        family_names = config.training_families
        repetitions = config.training_episodes_per_seed
        base_seeds = config.training_seeds
    elif split == "validation":
        family_names = config.validation_families
        repetitions = config.validation_repetitions
        base_seeds = (config.validation_seed,)
    else:
        family_names = config.final_test_families
        repetitions = config.final_test_repetitions
        base_seeds = (config.final_test_seed,)

    # Training visits one family per sequential episode and has an independent stream
    # per model. Evaluation streams are independent from all training seeds.
    items: List[RuntimeWorkload] = []
    for master_seed in base_seeds:
        if split == "training":
            for episode in range(repetitions):
                family_name = family_names[episode % len(family_names)]
                family_position = family_names.index(family_name)
                workload_seed = derive_seed(
                    master_seed,
                    _SPLIT_TAGS[split],
                    family_position,
                    episode,
                )
                items.append(
                    RuntimeWorkload(
                        split=split,
                        family=family_name,
                        repetition=episode,
                        seed=workload_seed,
                        workload=generator.generate(family_name, workload_seed),
                    )
                )
            continue
        for family_index, family_name in enumerate(family_names):
            for repetition in range(repetitions):
                workload_seed = derive_seed(
                    master_seed,
                    _SPLIT_TAGS[split],
                    family_index,
                    repetition,
                )
                items.append(
                    RuntimeWorkload(
                        split=split,
                        family=family_name,
                        repetition=repetition,
                        seed=workload_seed,
                        workload=generator.generate(family_name, workload_seed),
                    )
                )
    return items


def _train_models(
    config: RuntimeExperimentConfig,
    generator: WorkloadGenerator,
) -> tuple[Tuple[TrainedRuntimeModel, ...], pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    encoder = RuntimeStateEncoder(config.scheduler.round_robin_quantum)
    heuristic = CausalHeuristic(config.scheduler.round_robin_quantum)
    simulator = RuntimeSimulator(config.scheduler)
    family_index = {family_name: index for index, family_name in enumerate(config.training_families)}
    models: List[TrainedRuntimeModel] = []
    episode_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for model_seed in config.training_seeds:
        controller = RuntimeQController(
            encoder=encoder,
            config=config.q_learning,
            seed=derive_seed(model_seed, 1),
            fallback_action=heuristic.choose_action,
        )
        fingerprints: set[str] = set()
        train_wall_ns = 0
        update_ns = 0
        started_training = perf_counter_ns()
        for episode in range(config.training_episodes_per_seed):
            family_name = config.training_families[episode % len(config.training_families)]
            workload_seed = derive_seed(
                model_seed,
                _SPLIT_TAGS["training"],
                family_index[family_name],
                episode,
            )
            workload = generator.generate(family_name, workload_seed)
            fingerprints.add(workload.fingerprint)
            run = simulator.run(
                workload,
                controller,
                training=True,
                episode=episode,
                policy_name="Runtime Q-learning (training)",
            )
            train_wall_ns += run.wall_ns
            update_ns += run.update_ns
            metrics = compute_metrics(run.result).as_dict()
            exploratory = sum(record.decision.explored for record in run.decisions)
            episode_rows.append(
                {
                    "training_seed": model_seed,
                    "episode": episode,
                    "family": family_name,
                    "workload_seed": workload_seed,
                    "workload_fingerprint": workload.fingerprint,
                    "episode_reward": run.total_reward,
                    "decision_count": run.decision_count,
                    "policy_switch_count": run.policy_switch_count,
                    "exploratory_decisions": exploratory,
                    "unseen_state_explorations": sum(
                        record.decision.fallback_reason
                        == "forced-unseen-state-exploration"
                        for record in run.decisions
                    ),
                    "observation_ms": run.observation_ns / 1e6,
                    "selection_ms": run.selection_ns / 1e6,
                    "q_update_ms": run.update_ns / 1e6,
                    "simulator_wall_ms": run.wall_ns / 1e6,
                    **metrics,
                }
            )
            manifest_rows.append(
                {
                    "split": "training",
                    "training_seed": model_seed,
                    "episode": episode,
                    "family": family_name,
                    "repetition": episode,
                    "workload_seed": workload_seed,
                    "workload_fingerprint": workload.fingerprint,
                    "process_count": workload.size,
                }
            )
        train_wall_ns = max(train_wall_ns, perf_counter_ns() - started_training)
        models.append(
            TrainedRuntimeModel(
                training_seed=model_seed,
                controller=controller,
                training_fingerprints=frozenset(fingerprints),
                train_wall_ns=train_wall_ns,
                update_ns=update_ns,
            )
        )
        counts = controller.agent.state_action_visit_counts
        coverage_rows.append(
            {
                "split": "training",
                "training_seed": model_seed,
                "episodes": config.training_episodes_per_seed,
                "training_decisions": int(counts.sum()),
                "visited_states": int(np.count_nonzero(counts.sum(axis=1))),
                "total_states": controller.agent.n_states,
                "visited_state_actions": int(np.count_nonzero(counts)),
                "total_state_actions": int(counts.size),
                "unvisited_state_actions": int(counts.size - np.count_nonzero(counts)),
                "state_action_coverage": float(np.count_nonzero(counts) / counts.size),
                "training_wall_ms": train_wall_ns / 1e6,
                "training_q_update_ms": update_ns / 1e6,
            }
        )
    return (
        tuple(models),
        pd.DataFrame(episode_rows),
        pd.DataFrame(manifest_rows),
        coverage_rows,
    )


def _evaluate_split(
    split: str,
    workloads: Sequence[RuntimeWorkload],
    models: Sequence[TrainedRuntimeModel],
    config: RuntimeExperimentConfig,
    coverage_rows: List[Dict[str, Any]],
    keep_demo_candidates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, TrainedRuntimeModel, RuntimeWorkload, RuntimeRun]]]:
    simulator = RuntimeSimulator(config.scheduler)
    heuristic = CausalHeuristic(config.scheduler.round_robin_quantum)
    metrics_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    demos: list[tuple[int, TrainedRuntimeModel, RuntimeWorkload, RuntimeRun]] = []

    # Fixed baselines are the untouched standalone schedulers, not wrappers that make
    # decisions at runtime. Each sees the identical workload object used by adaptive runs.
    for item in workloads:
        for policy_class in POLICY_CLASSES:
            policy = policy_class(config.scheduler)
            metrics = compute_metrics(policy.run(item.workload)).as_dict()
            metrics_rows.append(
                _metric_row(
                    split,
                    policy.name,
                    None,
                    item,
                    metrics,
                    policy_switch_count=0,
                    decision_count=0,
                    observation_ms=np.nan,
                    selection_ms=np.nan,
                    update_ms=np.nan,
                    simulator_wall_ms=np.nan,
                    total_reward=np.nan,
                )
            )

        heuristic_run = simulator.run(
            item.workload,
            heuristic,
            training=False,
            policy_name=HEURISTIC_METHOD,
        )
        metrics_rows.append(
            _adaptive_metric_row(split, HEURISTIC_METHOD, None, item, heuristic_run)
        )
        decision_rows.extend(
            _decision_rows(split, HEURISTIC_METHOD, None, item, heuristic_run.decisions)
        )

        for model in models:
            before = np.array(model.controller.agent.q_table, copy=True)
            q_run = simulator.run(
                item.workload,
                model.controller,
                training=False,
                policy_name=Q_METHOD,
            )
            after = model.controller.agent.q_table
            if not np.array_equal(before, after):
                raise ValidationError("evaluation mutated the runtime Q table")
            metrics_rows.append(
                _adaptive_metric_row(split, Q_METHOD, model.training_seed, item, q_run)
            )
            decision_rows.extend(
                _decision_rows(split, Q_METHOD, model.training_seed, item, q_run.decisions)
            )
            if keep_demo_candidates:
                demos.append((q_run.policy_switch_count, model, item, q_run))

    metrics = pd.DataFrame(metrics_rows)
    decisions = pd.DataFrame(decision_rows)
    if split != "training":
        for model in models:
            model_rows = decisions[
                (decisions["method"] == Q_METHOD)
                & (decisions["training_seed"] == model.training_seed)
            ]
            counts = model.controller.agent.state_action_visit_counts
            no_action_states = model_rows[model_rows["fallback_reason"] == "unseen-state-causal-fallback"]
            all_states = model_rows["state_index"].dropna().astype(int)
            coverage_rows.append(
                {
                    "split": split,
                    "training_seed": model.training_seed,
                    "eval_decisions": int(len(model_rows)),
                    "distinct_eval_states": int(all_states.nunique()),
                    "unseen_state_decisions": int(len(no_action_states)),
                    "distinct_unseen_states": int(
                        no_action_states["state_index"].nunique()
                    ),
                    "unvisited_action_opportunities": int(
                        model_rows["unvisited_action_count"].fillna(0).sum()
                    ),
                    "decisions_with_unvisited_actions": int(
                        (model_rows["unvisited_action_count"].fillna(0) > 0).sum()
                    ),
                    "training_visited_state_actions": int(np.count_nonzero(counts)),
                    "total_state_actions": int(counts.size),
                }
            )
    return metrics, decisions, demos


def _select_demo_candidate(
    candidates: Sequence[tuple[int, TrainedRuntimeModel, RuntimeWorkload, RuntimeRun]],
) -> tuple[int, TrainedRuntimeModel, RuntimeWorkload, RuntimeRun]:
    """Select an illustrative trace from validation data only.

    Prefer no-fallback traces, then maximize the observed number of policy changes. Ties
    are broken by ascending model seed, family name, and repetition. The split check is
    deliberately enforced here so a future caller cannot accidentally feed final-test
    switch counts into demonstration selection.
    """
    if not candidates:
        raise ValidationError("no validation Q-learning traces are available for the demo")
    for switch_count, _model, item, run in candidates:
        if item.split != "validation":
            raise ValidationError(
                "the illustrative runtime demonstration must be selected from validation"
            )
        if switch_count != run.policy_switch_count:
            raise ValidationError("demo candidate switch count does not match its runtime trace")

    no_fallback = [
        candidate
        for candidate in candidates
        if all(
            record.decision.fallback_reason is None
            for record in candidate[3].decisions
        )
    ]
    pool = no_fallback or list(candidates)
    return min(
        pool,
        key=lambda entry: (
            -entry[0],
            entry[1].training_seed,
            entry[2].family,
            entry[2].repetition,
        ),
    )


def _metric_row(
    split: str,
    method: str,
    training_seed: Optional[int],
    item: RuntimeWorkload,
    metrics: Dict[str, Any],
    *,
    policy_switch_count: int,
    decision_count: int,
    observation_ms: float,
    selection_ms: float,
    update_ms: float,
    simulator_wall_ms: float,
    total_reward: float,
) -> Dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "training_seed": training_seed,
        "family": item.family,
        "repetition": item.repetition,
        "workload_seed": item.seed,
        "workload_fingerprint": item.fingerprint,
        **metrics,
        "policy_switch_count": policy_switch_count,
        "decision_count": decision_count,
        "observation_ms": observation_ms,
        "selection_ms": selection_ms,
        "q_update_ms": update_ms,
        "controller_overhead_ms": observation_ms + selection_ms,
        "simulator_wall_ms": simulator_wall_ms,
        "total_reward": total_reward,
    }


def _adaptive_metric_row(
    split: str,
    method: str,
    training_seed: Optional[int],
    item: RuntimeWorkload,
    run: RuntimeRun,
) -> Dict[str, Any]:
    return _metric_row(
        split,
        method,
        training_seed,
        item,
        compute_metrics(run.result).as_dict(),
        policy_switch_count=run.policy_switch_count,
        decision_count=run.decision_count,
        observation_ms=run.observation_ns / 1e6,
        selection_ms=run.selection_ns / 1e6,
        update_ms=run.update_ns / 1e6,
        simulator_wall_ms=run.wall_ns / 1e6,
        total_reward=run.total_reward,
    )


def _decision_rows(
    split: str,
    method: str,
    training_seed: Optional[int],
    item: RuntimeWorkload,
    records: Sequence[RuntimeDecisionRecord],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for record in records:
        observation = record.observation
        rows.append(
            {
                "split": split,
                "method": method,
                "training_seed": training_seed,
                "family": item.family,
                "repetition": item.repetition,
                "workload_seed": item.seed,
                "workload_fingerprint": item.fingerprint,
                "decision_index": record.decision_index,
                "decision_time": observation.current_time,
                "state_index": record.decision.state_index,
                "action": record.decision.action,
                "action_name": record.policy_name,
                "previous_policy": observation.previous_policy,
                "chosen_pid": record.chosen_pid,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "switch_overhead": record.switch_overhead,
                "policy_switch": record.policy_switch,
                "epsilon": record.decision.epsilon,
                "explored": record.decision.explored,
                "fallback_reason": record.decision.fallback_reason,
                "unvisited_action_count": record.decision.unvisited_action_count,
                "ready_count": len(observation.ready_processes),
                "completed_count": observation.completed_count,
                "arrived_count": observation.arrived_count,
                "mean_ready_arrival_age": observation.mean_ready_arrival_age,
                "observed_mean_burst": observation.observed_mean_burst,
                "observed_priority_spread": observation.observed_priority_spread,
                "ready_processes_compact": ";".join(
                    f"{process.pid},{process.arrival_time},{process.burst_time},"
                    f"{process.remaining_burst},{process.priority}"
                    for process in observation.ready_processes
                ),
                "waiting_cost_increment": record.waiting_cost_increment,
                "reward": record.reward,
                "terminal": record.terminal,
                "next_state_index": record.next_state_index,
            }
        )
    return rows


def _manifest_rows(workloads: Iterable[RuntimeWorkload]) -> list[Dict[str, Any]]:
    return [
        {
            "split": item.split,
            "family": item.family,
            "repetition": item.repetition,
            "workload_seed": item.seed,
            "workload_fingerprint": item.fingerprint,
            "process_count": item.workload.size,
        }
        for item in workloads
    ]


def _assert_split_separation(
    models: Sequence[TrainedRuntimeModel],
    validation: Sequence[RuntimeWorkload],
    final_test: Sequence[RuntimeWorkload],
) -> None:
    validation_fingerprints = {item.fingerprint for item in validation}
    test_fingerprints = {item.fingerprint for item in final_test}
    if validation_fingerprints & test_fingerprints:
        raise ValidationError("validation and final-test workload fingerprints overlap")
    for model in models:
        if model.training_fingerprints & (validation_fingerprints | test_fingerprints):
            raise ValidationError(
                f"training seed {model.training_seed} overlaps validation/final-test workloads"
            )


def _paired_matrices(
    frame: pd.DataFrame,
    method: str,
    metric: str,
) -> tuple[list[str], Dict[str, np.ndarray], list[Any]]:
    subset = frame[frame["method"] == method].copy()
    if subset.empty:
        raise ValidationError(f"no rows for method {method!r}")
    seed_values = sorted(subset["training_seed"].dropna().unique().tolist())
    if not seed_values:
        seed_values = [None]
        subset["_seed_key"] = 0
        seed_keys: list[Any] = [None]
    else:
        subset["_seed_key"] = subset["training_seed"]
        seed_keys = seed_values
    family_names = sorted(subset["family"].unique().tolist())
    matrices: Dict[str, np.ndarray] = {}
    for family in family_names:
        family_frame = subset[subset["family"] == family]
        workload_keys = sorted(
            set(zip(family_frame["repetition"], family_frame["workload_fingerprint"]))
        )
        matrix = np.empty((len(seed_keys), len(workload_keys)), dtype=float)
        for seed_index, seed in enumerate(seed_keys):
            seed_key = 0 if seed is None else seed
            seed_frame = family_frame[family_frame["_seed_key"] == seed_key]
            lookup = {
                (row.repetition, row.workload_fingerprint): float(getattr(row, metric))
                for row in seed_frame.itertuples()
            }
            if set(workload_keys) - set(lookup):
                raise ValidationError(
                    f"method {method!r} has incomplete paired workloads in {family!r}"
                )
            matrix[seed_index, :] = [lookup[key] for key in workload_keys]
        matrices[family] = matrix
    return family_names, matrices, seed_keys


def _paired_bootstrap_difference(
    frame: pd.DataFrame,
    target: str,
    reference: str,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    """Family-stratified, crossed workload/model-seed paired bootstrap interval."""
    target_families, target_data, target_seeds = _paired_matrices(frame, target, metric)
    ref_families, ref_data, ref_seeds = _paired_matrices(frame, reference, metric)
    if target_families != ref_families:
        raise ValidationError("comparison methods do not cover the same workload families")
    rng = np.random.default_rng(seed)
    observed_family_means: list[float] = []
    for family in target_families:
        if target_data[family].shape[1] != ref_data[family].shape[1]:
            raise ValidationError("comparison methods do not cover paired workload counts")
        target_mean = float(target_data[family].mean())
        ref_mean = float(ref_data[family].mean())
        observed_family_means.append(target_mean - ref_mean)
    observed = float(np.mean(observed_family_means))

    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        target_seed_indices = rng.integers(0, len(target_seeds), size=len(target_seeds))
        ref_seed_indices = rng.integers(0, len(ref_seeds), size=len(ref_seeds))
        family_draws: list[float] = []
        for family in target_families:
            workload_count = target_data[family].shape[1]
            sampled_workloads = rng.integers(0, workload_count, size=workload_count)
            target_values = target_data[family][target_seed_indices][:, sampled_workloads]
            reference_values = ref_data[family][ref_seed_indices][:, sampled_workloads]
            family_draws.append(float(np.mean(target_values - reference_values)))
        draws[replicate] = float(np.mean(family_draws))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _summary_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for method, subset in metrics.groupby("method", sort=False):
        for metric in (*ALL_METRICS, "policy_switch_count"):
            values = subset[metric].dropna().astype(float)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n_metric_rows": int(len(values)),
                    "n_unique_workloads": int(subset["workload_fingerprint"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _comparison_table(
    metrics: pd.DataFrame,
    config: RuntimeExperimentConfig,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    targets = [method for method in metrics["method"].unique() if method != "SJF"]
    comparisons = [(target, "SJF") for target in targets]
    if Q_METHOD in metrics["method"].unique() and HEURISTIC_METHOD in metrics["method"].unique():
        comparisons.append((Q_METHOD, HEURISTIC_METHOD))
    for target, reference in comparisons:
        for metric_index, metric in enumerate(ALL_METRICS):
            interval = _paired_bootstrap_difference(
                metrics,
                target,
                reference,
                metric,
                replicates=config.bootstrap_replicates,
                seed=derive_seed(config.bootstrap_seed, metric_index, len(rows)),
            )
            rows.append(
                {
                    "target": target,
                    "reference": reference,
                    "metric": metric,
                    **interval,
                    "direction_note": (
                        "negative favors target" if metric in (
                            "avg_waiting_time",
                            "avg_turnaround_time",
                            "avg_response_time",
                            "context_switches",
                        )
                        else "positive favors target"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _q_table_rows(models: Sequence[TrainedRuntimeModel]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for model in models:
        counts = model.controller.agent.state_action_visit_counts
        values = model.controller.agent.q_table
        for state in range(model.controller.agent.n_states):
            for action, name in enumerate(ACTION_NAMES):
                rows.append(
                    {
                        "training_seed": model.training_seed,
                        "state_index": state,
                        "action": action,
                        "action_name": name,
                        "q_value": float(values[state, action]),
                        "visit_count": int(counts[state, action]),
                        "visited": bool(counts[state, action] > 0),
                    }
                )
    return pd.DataFrame(rows)


def _software_snapshot() -> Dict[str, Any]:
    package_versions: Dict[str, Optional[str]] = {}
    for package in ("numpy", "pandas", "pytest", "matplotlib"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
    }


def _config_snapshot(config: RuntimeExperimentConfig) -> Dict[str, Any]:
    gamma = config.q_learning.discount_factor
    if gamma == 1.0:
        learning_objective = (
            "maximize undiscounted episodic return; with gamma=1, sum_t r_t equals "
            "negative total waiting time divided by RR quantum"
        )
    else:
        learning_objective = (
            f"maximize per-decision discounted return sum_t {gamma:g}^t r_t; this "
            "discounts by decision count, not simulated elapsed time, and is not the "
            "undiscounted total-waiting objective"
        )
    return {
        "experiment": "causal_event_driven_runtime_policy_adaptation",
        "training_episodes_per_seed": config.training_episodes_per_seed,
        "training_seeds": list(config.training_seeds),
        "training_families": list(config.training_families),
        "validation_seed": config.validation_seed,
        "validation_repetitions": config.validation_repetitions,
        "final_test_seed": config.final_test_seed,
        "final_test_repetitions": config.final_test_repetitions,
        "validation_and_test_families": list(config.validation_families),
        "workload_family_parameters": [asdict(family) for family in config.families],
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_seed": config.bootstrap_seed,
        "scheduler": asdict(config.scheduler),
        "q_learning": asdict(config.q_learning),
        "state_encoder": {
            "features": [
                "ready_count(1,2-3,4+)",
                "completed_count(0,1-3,4+)",
                "median_ready_remaining_burst(<=q,<=4q,>4q)",
                "mean_ready_arrival_age(<=q,<=4q,>4q)",
                "arrived_work_priority_spread(zero,nonzero)",
            ],
            "state_count": 162,
        },
        "reward": "r_t = -incremental_total_waiting_time / max(1, RR_quantum)",
        "learning_objective": learning_objective,
        "ready_queue_contract": (
            "FCFS/RR use live FIFO insertion order; SJF/Priority ties preserve that order; "
            "arrivals at a service endpoint enter before an unfinished RR job is requeued"
        ),
        "demonstration_selection_rule": DEMO_SELECTION_RULE,
        "priority_convention": "lower numeric value is higher priority by default",
        "burst_information_assumption": (
            "exact burst is known on arrival for arrived processes, matching the standalone SJF simulator"
        ),
    }


def _mean_by_method(metrics: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for method, subset in metrics.groupby("method", sort=False):
        row: Dict[str, float] = {}
        for metric in (*ALL_METRICS, "policy_switch_count", "decision_count"):
            values = subset[metric].dropna().astype(float)
            if len(values):
                row[metric] = float(values.mean())
        for metric in ("observation_ms", "selection_ms", "q_update_ms", "controller_overhead_ms", "simulator_wall_ms"):
            values = subset[metric].dropna().astype(float)
            if len(values):
                row[f"mean_{metric}"] = float(values.mean())
        result[str(method)] = row
    return result


def _render_report(summary: Dict[str, Any]) -> str:
    final_means = summary["final_test_means"]
    gamma = float(summary["configuration"]["q_learning"]["discount_factor"])
    if gamma == 1.0:
        learning_lines = (
            "updates omit the bootstrap. This run uses gamma=1: since each finite workload is an",
            "episode and `r = -delta_wait/q`, the undiscounted return is exactly negative total",
            "waiting time divided by the Round-Robin quantum. There is no per-decision or",
            "simulated-time discount. Unvisited actions are masked for greedy evaluation and",
            "bootstrapping; wholly unseen states use the documented causal heuristic fallback.",
            "Evaluation is deterministic and read-only.",
        )
    else:
        learning_lines = (
            f"updates omit the bootstrap. This run uses gamma={gamma:g} per decision, so later",
            "waiting increments are discounted by decision count, not simulated elapsed time.",
            "That objective is not the undiscounted total-waiting objective. Unvisited actions",
            "are masked for greedy evaluation and bootstrapping; wholly unseen states use the",
            "documented causal heuristic fallback. Evaluation is deterministic and read-only.",
        )
    lines = [
        "# Causal runtime-adaptation experiment",
        "",
        "This is the sequential, event-driven runtime experiment. It is separate from the",
        "legacy offline selector: no controller receives a `Workload`, unarrived process,",
        "future arrival/burst, or counterfactual schedule metric. The environment privately",
        "advances its event queue and exposes an immutable arrived-work observation at each",
        "dispatch/quantum decision epoch.",
        "",
        "## Design and split integrity",
        "",
        f"- Independent Q-learning seeds: {', '.join(map(str, summary['configuration']['training_seeds']))}.",
        f"- Training: {summary['configuration']['training_episodes_per_seed']} sequential episodes/model across {len(summary['configuration']['training_families'])} families; {summary['training_workloads']} total model-episodes.",
        f"- Validation: {summary['validation_unique_workloads']} distinct workloads; used for reporting and the predeclared illustrative-demo selection rule, not model/hyperparameter tuning.",
        f"- Untouched final test: {summary['final_test_unique_workloads']} distinct workloads, generated only after the validation-based demo was selected.",
        f"- Final test and validation workload fingerprints overlap: {summary['split_fingerprint_overlap']}.",
        f"- Round Robin quantum / per-PID-change switch cost: {summary['configuration']['scheduler']['round_robin_quantum']} / {summary['configuration']['scheduler']['switching_cost']} time units.",
        "- Exact burst lengths are assumed known when a process arrives (also required by SJF); no unarrived process details are exposed.",
        "",
        "## Sequential learning and causal heuristic",
        "",
        "At each decision epoch, the Q learner encodes five coarse causal features, chooses",
        "FCFS, SJF, Round Robin, or Priority, executes the chosen policy's next dispatch or",
        "quantum segment, and observes the next causal state. It updates with",
        "`Q(s,a) <- Q(s,a) + alpha * (r + gamma max_known_a' Q(s',a') - Q(s,a))`; terminal",
        *learning_lines,
        "",
        "## Ready-queue contract",
        "",
        "The live queue contains arrived, unfinished, non-running processes. Dispatch removes",
        "one process; completion removes it permanently. FCFS and RR select the current FIFO",
        "head. RR requeues an unfinished process at the tail after admitting endpoint arrivals.",
        "SJF and Priority select by their primary key, with ties preserving current queue order.",
        "A policy change never rebuilds the queue or resets remaining bursts. Simultaneous",
        "arrivals are admitted by `(arrival_time, pid)` before RR requeue at a service endpoint.",
        "",
        "The predeclared non-RL heuristic selects RR for at least three ready jobs with mean",
        "arrival age at least one quantum; otherwise Priority if ready priorities differ;",
        "otherwise SJF if the largest visible remaining burst is at least twice the smallest;",
        "otherwise FCFS. Fixed baselines are the four preserved standalone implementations.",
        "",
        "## Final held-out test means",
        "",
        "| Method | Mean wait | Mean turnaround | Mean response | CPU util. % | Throughput | Context switches | Policy switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = (
        "FCFS", "SJF", "Round Robin", "Priority", HEURISTIC_METHOD, Q_METHOD
    )
    for method in display:
        row = final_means.get(method, {})
        values = [
            row.get("avg_waiting_time", float("nan")),
            row.get("avg_turnaround_time", float("nan")),
            row.get("avg_response_time", float("nan")),
            row.get("cpu_utilization", float("nan")),
            row.get("throughput", float("nan")),
            row.get("context_switches", float("nan")),
            row.get("policy_switch_count", float("nan")),
        ]
        lines.append(
            f"| {method} | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "Q-learning rows average across the five independently trained models and the same paired test workloads; fixed and heuristic rows contain one deterministic result per workload.",
        "",
        "## Paired uncertainty",
        "",
        f"Intervals below use a {summary['configuration']['bootstrap_replicates']}-replicate, family-stratified crossed bootstrap: resample test workloads within each family and independently resample training-model seeds. Intervals quantify uncertainty across these specified seeds/workloads; they do not justify claims about all operating systems or workload populations.",
        "A difference is target minus reference; negative is favorable for waiting/turnaround/response/context-switch costs, positive for utilization/throughput. No p-values or blanket significance claims are reported.",
        "",
        "| Target | Reference | Metric | Difference | 95% interval |",
        "|---|---|---|---:|---:|",
    ])
    for row in summary["paired_comparisons"]:
        lines.append(
            f"| {row['target']} | {row['reference']} | {row['metric']} | "
            f"{row['mean_difference']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
        )
    lines.extend([
        "",
        "## Coverage and controller overhead",
        "",
        "Per-model state/action visit counts, unseen-state fallbacks, unvisited-action exposure,",
        "policy decision sequences/times, and all six scheduler metrics are in the CSV outputs.",
        "Observation construction, action selection, Q updates, and total simulator wall time",
        "are recorded separately; timing is host-dependent and is not charged to simulated time.",
        "",
        f"- Illustrative validation-split Q trace (not representative): `{summary['demo']['path']}` ({summary['demo']['policy_switch_count']} policy changes across {summary['demo']['decision_count']} decisions; training seed {summary['demo']['training_seed']}, {summary['demo']['family']} repetition {summary['demo']['repetition']}).",
        f"- Demo selection rule (validation only): {DEMO_SELECTION_RULE}.",
        f"- Mean Q-controller observation/selection overhead per evaluation trace: {summary['mean_runtime_q_observation_ms']:.6f} ms / {summary['mean_runtime_q_selection_ms']:.6f} ms.",
        f"- Mean Q-controller observation/action-selection time per decision: {summary['mean_runtime_q_observation_us_per_decision']:.3f} us / {summary['mean_runtime_q_selection_us_per_decision']:.3f} us.",
        f"- Mean Q training update time: {summary['mean_training_q_update_us_per_transition']:.3f} us per transition.",
        "",
        "## Limitations and legacy reference",
        "",
        "This is a single-CPU synthetic simulator, not a kernel scheduler. It assumes exact",
        "arrived burst lengths, uses a small hand-binned state abstraction, and has no I/O,",
        "multicore contention, deadlines, or hardware latency model. The historical offline",
        "selector in `rl/adaptive.py` remains separately available for reproducibility, but",
        "it sees complete-workload features and counterfactual schedule metrics; it is not a",
        "causal runtime baseline and is not included in these fair paired comparisons.",
        "",
        "## Artifact map",
        "",
        "- `runtime_training_metrics.csv`: per-episode sequential training outcomes.",
        "- `runtime_validation_metrics.csv` / `runtime_final_test_metrics.csv`: paired metrics.",
        "- `runtime_validation_decisions.csv` / `runtime_final_test_decisions.csv`: large causal state/action event logs, regenerated by the command above and intentionally excluded from version control.",
        "- `runtime_workload_manifest.csv`: split seeds and fingerprints for regeneration.",
        "- `runtime_state_action_coverage.csv` / `runtime_q_table.csv`: state/action visits and learned values.",
        "- `runtime_summary.json`: configuration, software, split audit, aggregate results, and intervals.",
        "",
    ])
    return "\n".join(lines)


def run_runtime_experiment(
    config: Optional[RuntimeExperimentConfig] = None,
    results_dir: str | Path = "results/runtime",
) -> RuntimeExperimentArtifacts:
    """Train independent online agents, validate, and evaluate on untouched test traces."""
    config = config or RuntimeExperimentConfig()
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = WorkloadGenerator(config.families)

    models, training_metrics, train_manifest, coverage_rows = _train_models(config, generator)
    training_fingerprints_by_seed = {
        model.training_seed: model.training_fingerprints for model in models
    }
    training_union = set().union(*training_fingerprints_by_seed.values())

    validation_workloads = build_runtime_workloads("validation", config, generator)
    validation_metrics, validation_decisions, validation_demo_candidates = _evaluate_split(
        "validation",
        validation_workloads,
        models,
        config,
        coverage_rows,
        keep_demo_candidates=True,
    )
    validation_fingerprints = {item.fingerprint for item in validation_workloads}
    if training_union & validation_fingerprints:
        raise ValidationError("training and validation workload fingerprints overlap")

    # Choose the illustrative trace using the predeclared validation-only rule before
    # generating the final-test workloads. Test outcomes cannot enter this selection.
    demo_switches, demo_model, demo_item, demo_run = _select_demo_candidate(
        validation_demo_candidates
    )

    # The final split is generated only after training and validation are complete. It is
    # used for aggregate reporting only, never hyperparameter/model/demo selection.
    final_workloads = build_runtime_workloads("final_test", config, generator)
    _assert_split_separation(models, validation_workloads, final_workloads)
    final_metrics, final_decisions, _ = _evaluate_split(
        "final_test",
        final_workloads,
        models,
        config,
        coverage_rows,
        keep_demo_candidates=False,
    )

    manifest = pd.DataFrame(
        [*train_manifest.to_dict("records"), *_manifest_rows(validation_workloads), *_manifest_rows(final_workloads)]
    )
    coverage = pd.DataFrame(coverage_rows)
    training_metrics = training_metrics.sort_values(
        ["training_seed", "episode"], kind="stable"
    ).reset_index(drop=True)
    validation_metrics = validation_metrics.sort_values(
        ["method", "training_seed", "family", "repetition"], kind="stable", na_position="first"
    ).reset_index(drop=True)
    final_metrics = final_metrics.sort_values(
        ["method", "training_seed", "family", "repetition"], kind="stable", na_position="first"
    ).reset_index(drop=True)
    validation_decisions = validation_decisions.sort_values(
        ["method", "training_seed", "family", "repetition", "decision_index"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    final_decisions = final_decisions.sort_values(
        ["method", "training_seed", "family", "repetition", "decision_index"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)

    comparison = _comparison_table(final_metrics, config)
    test_summary = _summary_statistics(final_metrics)
    validation_summary = _summary_statistics(validation_metrics)
    q_summary = _mean_by_method(final_metrics)
    training_coverage = coverage[coverage["split"] == "training"]
    mean_update_ms = float(training_metrics["q_update_ms"].sum() / max(1, training_metrics["decision_count"].sum()))
    q_eval_rows = final_metrics[final_metrics["method"] == Q_METHOD]
    mean_q_observation = float(q_eval_rows["observation_ms"].mean())
    mean_q_selection = float(q_eval_rows["selection_ms"].mean())
    q_eval_decisions = max(1.0, float(q_eval_rows["decision_count"].sum()))
    mean_q_observation_us_per_decision = float(
        q_eval_rows["observation_ms"].sum() / q_eval_decisions * 1000.0
    )
    mean_q_selection_us_per_decision = float(
        q_eval_rows["selection_ms"].sum() / q_eval_decisions * 1000.0
    )

    demo_path = output / "runtime_learned_switch_demo.json"
    demo_payload = _demo_payload(demo_model, demo_item, demo_run)
    demo_path.write_text(json.dumps(demo_payload, indent=2, sort_keys=True), encoding="utf-8")

    workload_split_counts = {
        "training_workloads": int(len(train_manifest)),
        "validation_unique_workloads": int(len(validation_workloads)),
        "final_test_unique_workloads": int(len(final_workloads)),
        "split_fingerprint_overlap": False,
        "training_fingerprint_count_union": int(len(training_union)),
    }
    summary: Dict[str, Any] = {
        "configuration": _config_snapshot(config),
        "software": _software_snapshot(),
        **workload_split_counts,
        "training_models": [
            {
                "training_seed": model.training_seed,
                "episodes": config.training_episodes_per_seed,
                "training_workload_fingerprints": len(model.training_fingerprints),
                "training_wall_ms": model.train_wall_ns / 1e6,
                "q_update_ms": model.update_ns / 1e6,
                "state_action_coverage": float(
                    training_coverage.loc[
                        training_coverage["training_seed"] == model.training_seed,
                        "state_action_coverage",
                    ].iloc[0]
                ),
            }
            for model in models
        ],
        "validation_means": _mean_by_method(validation_metrics),
        "final_test_means": q_summary,
        "final_test_metric_summary": test_summary.to_dict("records"),
        "validation_metric_summary": validation_summary.to_dict("records"),
        "paired_comparisons": comparison.to_dict("records"),
        "mean_runtime_q_observation_ms": mean_q_observation,
        "mean_runtime_q_selection_ms": mean_q_selection,
        "mean_runtime_q_observation_us_per_decision": mean_q_observation_us_per_decision,
        "mean_runtime_q_selection_us_per_decision": mean_q_selection_us_per_decision,
        "mean_training_q_update_ms": mean_update_ms,
        "mean_training_q_update_us_per_transition": mean_update_ms * 1000.0,
        "demo": {
            "path": demo_path.name,
            "source_split": demo_item.split,
            "illustrative_not_representative": True,
            "selection_rule": DEMO_SELECTION_RULE,
            "training_seed": demo_model.training_seed,
            "family": demo_item.family,
            "repetition": demo_item.repetition,
            "workload_seed": demo_item.seed,
            "workload_fingerprint": demo_item.fingerprint,
            "policy_switch_count": demo_switches,
            "decision_count": demo_run.decision_count,
            "fallback_decisions": sum(
                record.decision.fallback_reason is not None
                for record in demo_run.decisions
            ),
            "actions": [record.policy_name for record in demo_run.decisions],
            "policy_switch_times": [
                record.observation.current_time
                for record in demo_run.decisions
                if record.policy_switch
            ],
        },
        "split_fingerprints": {
            "training_union_sha256": hashlib.sha256(
                "\n".join(sorted(training_union)).encode("ascii")
            ).hexdigest(),
            "validation_sha256": hashlib.sha256(
                "\n".join(sorted(validation_fingerprints)).encode("ascii")
            ).hexdigest(),
            "final_test_sha256": hashlib.sha256(
                "\n".join(sorted(item.fingerprint for item in final_workloads)).encode("ascii")
            ).hexdigest(),
        },
    }

    paths: Dict[str, Path] = {}
    frames = {
        "training_metrics": (training_metrics, "runtime_training_metrics.csv"),
        "validation_metrics": (validation_metrics, "runtime_validation_metrics.csv"),
        "final_test_metrics": (final_metrics, "runtime_final_test_metrics.csv"),
        "validation_decisions": (validation_decisions, "runtime_validation_decisions.csv"),
        "final_test_decisions": (final_decisions, "runtime_final_test_decisions.csv"),
        "workload_manifest": (manifest, "runtime_workload_manifest.csv"),
        "state_action_coverage": (coverage, "runtime_state_action_coverage.csv"),
        "q_table": (_q_table_rows(models), "runtime_q_table.csv"),
        "paired_comparisons": (comparison, "runtime_paired_comparisons.csv"),
        "final_test_summary": (test_summary, "runtime_final_test_summary.csv"),
        "validation_summary": (validation_summary, "runtime_validation_summary.csv"),
    }
    for name, (frame, filename) in frames.items():
        path = output / filename
        frame.to_csv(path, index=False)
        paths[name] = path
    summary_path = output / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    paths["learned_switch_demo"] = demo_path
    report_path = output / "runtime_report.md"
    report_path.write_text(_render_report(summary), encoding="utf-8")
    paths["report"] = report_path

    return RuntimeExperimentArtifacts(
        models=models,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        final_test_metrics=final_metrics,
        validation_decisions=validation_decisions,
        final_test_decisions=final_decisions,
        summary=summary,
        paths=paths,
    )


def _demo_payload(
    model: TrainedRuntimeModel,
    item: RuntimeWorkload,
    run: RuntimeRun,
) -> Dict[str, Any]:
    """Serialize a real learned trace with observations and actual action transitions."""
    return {
        "status": "verified_learned_runtime_policy_switch",
        "source_split": item.split,
        "illustrative_not_representative": True,
        "selection_rule": DEMO_SELECTION_RULE,
        "training_seed": model.training_seed,
        "family": item.family,
        "repetition": item.repetition,
        "workload_seed": item.seed,
        "workload_fingerprint": item.fingerprint,
        "workload_processes": [
            {
                "pid": process.pid,
                "arrival_time": process.arrival_time,
                "burst_time": process.burst_time,
                "priority": process.priority,
            }
            for process in item.workload.processes
        ],
        "policy_switch_count": run.policy_switch_count,
        "decision_count": run.decision_count,
        "policy_switch_times": [
            record.observation.current_time
            for record in run.decisions
            if record.policy_switch
        ],
        "decisions": [
            {
                "decision_index": record.decision_index,
                "time": record.observation.current_time,
                "previous_policy": record.observation.previous_policy,
                "mean_ready_arrival_age": record.observation.mean_ready_arrival_age,
                "state_index": record.decision.state_index,
                "action": record.decision.action,
                "policy": record.policy_name,
                "chosen_pid": record.chosen_pid,
                "fallback_reason": record.decision.fallback_reason,
                "unvisited_action_count": record.decision.unvisited_action_count,
                "ready": [
                    {
                        "pid": p.pid,
                        "arrival_time": p.arrival_time,
                        "remaining_burst": p.remaining_burst,
                        "priority": p.priority,
                    }
                    for p in record.observation.ready_processes
                ],
                "policy_switch": record.policy_switch,
            }
            for record in run.decisions
        ],
        "trace": [
            {"pid": slice_.pid, "start": slice_.start_time, "end": slice_.end_time}
            for slice_ in run.result.trace
        ],
        "metrics": compute_metrics(run.result).as_dict(),
        "controller_training_updates": int(model.controller.agent.state_action_visit_counts.sum()),
        "evaluation_updates": 0,
        "fallback_decisions": sum(
            record.decision.fallback_reason is not None for record in run.decisions
        ),
        "all_choices_from_trained_state": all(
            record.decision.fallback_reason is None for record in run.decisions
        ),
    }
