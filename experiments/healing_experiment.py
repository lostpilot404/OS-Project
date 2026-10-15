"""Self-healing scheduling study: training, held-out evaluation, and recovery audit.

The experiment produced by this module is the project's primary evidence. For every
predeclared arm (see :mod:`experiments.healing_config`) and every workload of a split it
records the six established scheduling metrics plus the tail statistic, and for the healing
arms it also records the whole recovery ledger: trigger evidence, diagnosis, offered
actions, chosen action, verification window, verdict, reward and realized switch overhead.

Three analyses make the recovery loop checkable rather than asserted:

``detection summary``
    Counts of every detector outcome per arm (detected, held, accepted, rolled back,
    censored, budget-exhausted, work-gated, ...), so "the loop recovers" cannot hide
    behind an aggregate.
``counterfactual replay audit``
    For each intervention the trace is replayed with its exact pre-incident configuration
    prefix and then *held* from the incident onward. The difference between that
    counterfactual's total waiting time and the realized run's is the measured marginal
    value of adapting at that point; a non-positive value is the study's definition of a
    false recovery. An episode-level version of the same audit answers whether adapting at
    all helped a trace.
``action-effect probe``
    On the validation split, every candidate action is applied at every incident of a
    detection-powered trace and run to completion. The probe reports, per diagnosis, the
    true effect of each action next to the reward the controller would have received, which
    is what turns "the reward encourages useful recovery" into a measurement.

Everything is seeded: two runs of the same configuration produce byte-identical tables.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ACTION_NAMES, derive_seed
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from evaluation.paired import paired_bootstrap_difference, percentile
from experiments.healing_config import (
    ARM_FIXED,
    ARM_HEALING,
    POLICY_KIND_HOLD,
    POLICY_KIND_Q,
    POLICY_KIND_RULE_FAIRNESS,
    POLICY_KIND_RULE_WAITING,
    SPLIT_TAGS,
    HealingArm,
    HealingExperimentConfig,
    default_arms,
)
from healing.actions import HEALING_ACTION_NAMES
from healing.controller import (
    EpisodeRecord,
    SelfHealingController,
    available_actions,
    replay_action_at_incident,
    replay_hold_counterfactual,
)
from healing.diagnosis import DIAGNOSIS_CATEGORIES
from healing.policies import (
    HealingContextEncoder,
    HoldOnlyPolicy,
    QLearningHealingPolicy,
    RuleHealingPolicy,
    RuleWaitingTimePolicy,
)
from scheduler import POLICY_CLASSES
from scheduler.runtime import RuntimeSimulator
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = [
    "HealingWorkload",
    "TrainedHealingModel",
    "HealingExperimentArtifacts",
    "build_healing_workloads",
    "train_healing_models",
    "run_healing_experiment",
]

METRICS = (
    "avg_waiting_time",
    "p95_waiting_time",
    "max_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "throughput",
    "context_switches",
)
#: Fixed arms run through the untouched standalone scheduler implementations.
FIXED_POLICY_CLASSES = {
    "fixed_fcfs": "FCFS",
    "fixed_sjf": "SJF",
    "fixed_round_robin": "RoundRobin",
    "fixed_priority": "Priority",
}

COMPARISONS = (
    ("rule_wait", "fixed_fcfs"),
    ("rule_wait", "fixed_sjf"),
    ("rule_wait", "detector_only"),
    ("rule_fairness", "fixed_fcfs"),
    ("q_recovery", "fixed_fcfs"),
    ("q_recovery", "fixed_sjf"),
    ("q_recovery", "rule_wait"),
    ("q_recovery", "detector_only"),
    ("q_recovery_from_sjf", "fixed_sjf"),
    ("rule_wait_from_sjf", "fixed_sjf"),
)


@dataclass(frozen=True)
class HealingWorkload:
    """One reproducible workload and its experimental-design coordinates."""

    split: str
    family: str
    repetition: int
    seed: int
    workload: Workload

    @property
    def fingerprint(self) -> str:
        """Stable digest of the process table, used to pair every comparison."""
        return self.workload.fingerprint


@dataclass(frozen=True)
class TrainedHealingModel:
    """One independently trained recovery policy and its disjoint training evidence."""

    training_seed: int
    policy: QLearningHealingPolicy
    controller: SelfHealingController
    training_fingerprints: frozenset
    episodes: Tuple[EpisodeRecord, ...]
    train_wall_ns: int


@dataclass(frozen=True)
class HealingExperimentArtifacts:
    """Every table, summary and path produced by one run of the study."""

    models: Tuple[TrainedHealingModel, ...]
    training_episodes: pd.DataFrame
    metrics: Dict[str, pd.DataFrame]
    incidents: Dict[str, pd.DataFrame]
    audit: pd.DataFrame
    episode_audit: pd.DataFrame
    comparisons: pd.DataFrame
    detection_summary: pd.DataFrame
    action_probe: pd.DataFrame
    coverage: pd.DataFrame
    summary: Dict[str, Any]
    paths: Dict[str, Path]


def build_healing_workloads(
    split: str,
    config: HealingExperimentConfig,
    generator: Optional[WorkloadGenerator] = None,
) -> List[HealingWorkload]:
    """Build the seeded workloads of one split.

    Training episodes cycle through the training families so that each training seed sees
    the same sequence of conditions; validation and final test cover every declared family
    with independent repetitions.

    Raises:
        ValidationError: If the split name is unknown.
    """
    if split not in SPLIT_TAGS:
        raise ValidationError(f"unknown split {split!r}")
    generator = generator or WorkloadGenerator(config.families)
    items: List[HealingWorkload] = []
    if split == "training":
        family_names = config.training_families
        for master_seed in config.training_seeds:
            for episode in range(config.training_episodes_per_seed):
                family = family_names[episode % len(family_names)]
                position = family_names.index(family)
                seed = derive_seed(master_seed, SPLIT_TAGS[split], position, episode)
                items.append(
                    HealingWorkload(split, family, episode, seed, generator.generate(family, seed))
                )
        return items
    family_names = config.validation_families
    repetitions = (
        config.validation_repetitions if split == "validation" else config.final_test_repetitions
    )
    master_seed = config.validation_seed if split == "validation" else config.final_test_seed
    for family_index, family in enumerate(family_names):
        for repetition in range(repetitions):
            seed = derive_seed(master_seed, SPLIT_TAGS[split], family_index, repetition)
            items.append(
                HealingWorkload(split, family, repetition, seed, generator.generate(family, seed))
            )
    return items


def _make_policy(arm: HealingArm, config: HealingExperimentConfig, encoder: HealingContextEncoder):
    """Build the recovery policy of one arm."""
    if arm.policy_kind == POLICY_KIND_HOLD:
        return HoldOnlyPolicy()
    if arm.policy_kind == POLICY_KIND_RULE_FAIRNESS:
        return RuleHealingPolicy(config.healing)
    if arm.policy_kind == POLICY_KIND_RULE_WAITING:
        return RuleWaitingTimePolicy(config.healing)
    if arm.policy_kind == POLICY_KIND_Q:
        return QLearningHealingPolicy(
            encoder=encoder,
            q_config=config.q_learning,
            seed=derive_seed(arm.training_seed, 1),
        )
    raise ValidationError(f"arm {arm.name!r} has an unsupported policy kind {arm.policy_kind!r}")


def train_healing_models(
    config: HealingExperimentConfig,
    generator: WorkloadGenerator,
) -> Tuple[Tuple[TrainedHealingModel, ...], pd.DataFrame]:
    """Train one independent Q-learning recovery policy per declared training seed.

    Training is sequential: one complete trace per episode, in the declared family order,
    with epsilon decayed by episode. Only the training split is used, and the resulting
    fingerprints are kept so the study can prove the splits are disjoint.

    Raises:
        ValidationError: If the training split would be empty.
    """
    encoder = HealingContextEncoder(config.healing)
    simulator = RuntimeSimulator(config.scheduler)
    workloads = build_healing_workloads("training", config, generator)
    if not workloads:
        raise ValidationError("the training split is empty")
    by_seed: Dict[int, List[HealingWorkload]] = {seed: [] for seed in config.training_seeds}
    for item in workloads:
        # The workload seed encodes its training seed; recompute it the same way.
        for seed in config.training_seeds:
            if item.seed == derive_seed(
                seed,
                SPLIT_TAGS["training"],
                config.training_families.index(item.family),
                item.repetition,
            ):
                by_seed[seed].append(item)
                break

    models: List[TrainedHealingModel] = []
    episode_rows: List[Dict[str, Any]] = []
    for seed in config.training_seeds:
        policy = QLearningHealingPolicy(encoder=encoder, q_config=config.q_learning, seed=derive_seed(seed, 1))
        controller = SelfHealingController(
            config.healing,
            config.scheduler,
            policy,
            initial_policy=0,
            label=f"q_recovery_seed{seed}",
        )
        fingerprints: set[str] = set()
        started = perf_counter_ns()
        episodes: List[EpisodeRecord] = []
        total_reward = 0.0
        for episode_index, item in enumerate(by_seed[seed]):
            fingerprints.add(item.fingerprint)
            controller.begin_episode(episode_index)
            run = simulator.run(
                item.workload, controller, training=True, episode=episode_index
            )
            record = controller.end_episode()
            episodes.append(record)
            rewards = [
                incident.reward
                for incident in record.incidents
                if incident.reward is not None
            ]
            total_reward += float(sum(rewards))
            episode_rows.append(
                {
                    "training_seed": seed,
                    "episode_index": episode_index,
                    "family": item.family,
                    "repetition": item.repetition,
                    "workload_fingerprint": item.fingerprint,
                    "decisions": record.decisions,
                    "incidents": len(record.incidents),
                    "interventions": record.interventions,
                    "accepted": record.accepted,
                    "rolled_back": record.rolled_back,
                    "held": record.holds,
                    "censored": record.censored,
                    "fallback_triggered": record.fallback_triggered,
                    "episode_reward": float(sum(rewards)),
                    "simulated_waiting_time": float(
                        sum(outcome.waiting_time for outcome in run.result.outcomes)
                    ),
                    "epsilon": float(policy.agent.epsilon_for_episode(episode_index)),
                }
            )
        models.append(
            TrainedHealingModel(
                training_seed=seed,
                policy=policy,
                controller=controller,
                training_fingerprints=frozenset(fingerprints),
                episodes=tuple(episodes),
                train_wall_ns=perf_counter_ns() - started,
            )
        )
    return tuple(models), pd.DataFrame(episode_rows)


def _metric_row(
    split: str,
    arm: HealingArm,
    item: HealingWorkload,
    metrics: WorkloadMetrics,
    waits: Sequence[int],
    *,
    policy_switch_count: int,
    decision_count: int,
    episode: Optional[EpisodeRecord],
) -> Dict[str, Any]:
    """One tidy metric row, including the healing bookkeeping when available."""
    row: Dict[str, Any] = {
        "split": split,
        "arm": arm.name,
        "kind": arm.kind,
        "family": item.family,
        "repetition": item.repetition,
        "workload_fingerprint": item.fingerprint,
        "training_seed": arm.training_seed if arm.kind == ARM_HEALING else np.nan,
        "initial_policy": ACTION_NAMES[arm.initial_policy] if arm.kind == ARM_HEALING else "",
        "policy_name": metrics.policy_name,
        "num_processes": metrics.num_processes,
        "p95_waiting_time": percentile(list(waits), 0.95),
        "max_waiting_time": float(max(waits)) if waits else 0.0,
        "median_waiting_time": float(np.median(waits)) if waits else 0.0,
        "policy_switch_count": policy_switch_count,
        "decision_count": decision_count,
        "incidents": 0,
        "interventions": 0,
        "accepted": 0,
        "rolled_back": 0,
        "held": 0,
        "censored": 0,
        "fallback_triggered": False,
        "final_policy": "",
    }
    for metric in (
        "avg_waiting_time",
        "avg_turnaround_time",
        "avg_response_time",
        "cpu_utilization",
        "throughput",
        "context_switches",
    ):
        row[metric] = float(getattr(metrics, metric))
    if episode is not None:
        row.update(
            {
                "incidents": len(episode.incidents),
                "interventions": episode.interventions,
                "accepted": episode.accepted,
                "rolled_back": episode.rolled_back,
                "held": episode.holds,
                "censored": episode.censored,
                "fallback_triggered": episode.fallback_triggered,
                "final_policy": episode.settings_final.policy_name,
                "final_quantum": episode.settings_final.quantum,
            }
        )
    return row


def _incident_rows(
    split: str,
    arm: HealingArm,
    item: HealingWorkload,
    episodes: Sequence[EpisodeRecord],
) -> List[Dict[str, Any]]:
    """Flatten every incident of one workload/arm run into ledger rows."""
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        for incident in episode.incidents:
            row = incident.as_row()
            row.update(
                {
                    "split": split,
                    "arm": arm.name,
                    "family": item.family,
                    "repetition": item.repetition,
                    "workload_fingerprint": item.fingerprint,
                    "training_seed": arm.training_seed,
                    "initial_policy": ACTION_NAMES[arm.initial_policy],
                }
            )
            rows.append(row)
    return rows


def _evaluate_arm_on_split(
    arm: HealingArm,
    split: str,
    workloads: Sequence[HealingWorkload],
    config: HealingExperimentConfig,
    models: Dict[int, TrainedHealingModel],
    *,
    audit: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run one arm over one split; return metric rows, incident rows and audit rows."""
    simulator = RuntimeSimulator(config.scheduler)
    encoder = HealingContextEncoder(config.healing)
    metric_rows: List[Dict[str, Any]] = []
    incident_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    for index, item in enumerate(workloads):
        if arm.kind == ARM_FIXED:
            policy_class = FIXED_POLICY_CLASSES[arm.name]
            scheduler = next(
                cls(config.scheduler) for cls in POLICY_CLASSES if cls.__name__ == policy_class
            )
            result = scheduler.run(item.workload)
            metrics = compute_metrics(result)
            metric_rows.append(
                _metric_row(
                    split,
                    arm,
                    item,
                    metrics,
                    [outcome.waiting_time for outcome in result.outcomes],
                    policy_switch_count=0,
                    decision_count=0,
                    episode=None,
                )
            )
            continue

        if arm.policy_kind == POLICY_KIND_Q:
            model = models[arm.training_seed]
            policy = model.policy
        else:
            policy = _make_policy(arm, config, encoder)
        controller = SelfHealingController(
            config.healing,
            config.scheduler,
            policy,
            initial_policy=arm.initial_policy,
            label=arm.name,
        )
        controller.begin_episode(index)
        run = simulator.run(item.workload, controller, training=False, policy_name=arm.name)
        episode = controller.end_episode()
        metrics = compute_metrics(run.result)
        waits = [outcome.waiting_time for outcome in run.result.outcomes]
        metric_rows.append(
            _metric_row(
                split,
                arm,
                item,
                metrics,
                waits,
                policy_switch_count=run.policy_switch_count,
                decision_count=run.decision_count,
                episode=episode,
            )
        )
        incident_rows.extend(_incident_rows(split, arm, item, [episode]))
        episode_rows.append(
            {
                "split": split,
                "arm": arm.name,
                "family": item.family,
                "repetition": item.repetition,
                "workload_fingerprint": item.fingerprint,
                "training_seed": arm.training_seed,
                "episode_index": index,
                "decisions": episode.decisions,
                "incidents": len(episode.incidents),
                "interventions": episode.interventions,
                "accepted": episode.accepted,
                "rolled_back": episode.rolled_back,
                "held": episode.holds,
                "censored": episode.censored,
                "fallback_triggered": episode.fallback_triggered,
                "total_waiting_time": float(sum(waits)),
                "detector_reasons": ";".join(
                    f"{reason}={count}" for reason, count in episode.detector_reasons
                ),
            }
        )
        if audit:
            audit_rows.extend(
                _audit_episode(simulator, split, arm, item, run, episode, index)
            )
    return metric_rows, incident_rows, audit_rows, episode_rows


def _audit_episode(
    simulator: RuntimeSimulator,
    split: str,
    arm: HealingArm,
    item: HealingWorkload,
    run,
    episode: EpisodeRecord,
    episode_index: int,
) -> List[Dict[str, Any]]:
    """Counterfactual replay audit of every intervention plus the episode-level test."""
    rows: List[Dict[str, Any]] = []
    actual_total = float(sum(outcome.waiting_time for outcome in run.result.outcomes))
    interventions = [incident for incident in episode.incidents if incident.action_changed]
    for incident in interventions:
        counterfactual = replay_hold_counterfactual(simulator, item.workload, run.decisions, incident)
        hold_total = float(sum(outcome.waiting_time for outcome in counterfactual.result.outcomes))
        rows.append(
            {
                "split": split,
                "arm": arm.name,
                "family": item.family,
                "repetition": item.repetition,
                "workload_fingerprint": item.fingerprint,
                "training_seed": arm.training_seed,
                "episode_index": episode_index,
                "decision_index": incident.decision_index,
                "diagnosis": incident.diagnosis,
                "action_name": incident.action_name,
                "verdict": incident.verdict,
                "hold_total_waiting": hold_total,
                "actual_total_waiting": actual_total,
                "intervention_gain": hold_total - actual_total,
                "helped": bool(hold_total - actual_total > 0),
            }
        )
    if interventions:
        first = interventions[0]
        counterfactual = replay_hold_counterfactual(simulator, item.workload, run.decisions, first)
        hold_total = float(sum(outcome.waiting_time for outcome in counterfactual.result.outcomes))
        rows.append(
            {
                "split": split,
                "arm": arm.name,
                "family": item.family,
                "repetition": item.repetition,
                "workload_fingerprint": item.fingerprint,
                "training_seed": arm.training_seed,
                "episode_index": episode_index,
                "decision_index": first.decision_index,
                "diagnosis": first.diagnosis,
                "action_name": first.action_name,
                "verdict": "episode_level",
                "hold_total_waiting": hold_total,
                "actual_total_waiting": actual_total,
                "intervention_gain": hold_total - actual_total,
                "helped": bool(hold_total - actual_total > 0),
            }
        )
    return rows


def _detection_summary(incidents: pd.DataFrame, metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per-arm counts of detections, interventions and verdicts within one split."""
    rows: List[Dict[str, Any]] = []
    for arm_name, metric_subset in metrics.groupby("arm", sort=False):
        subset = incidents[incidents["arm"] == arm_name] if not incidents.empty else incidents
        episodes = len(metric_subset)
        interventions = int(subset["action_changed"].sum()) if len(subset) else 0
        rollbacks = int((subset["verdict"] == "rolled_back").sum()) if len(subset) else 0
        accepted = int((subset["verdict"] == "accepted").sum()) if len(subset) else 0
        holds = int((subset["verdict"] == "held").sum()) if len(subset) else 0
        censored = int((subset["verdict"] == "censored").sum()) if len(subset) else 0
        rows.append(
            {
                "split": split,
                "arm": arm_name,
                "incidents": int(len(subset)),
                "incidents_per_episode": (len(subset) / episodes) if episodes else 0.0,
                "interventions": interventions,
                "held": holds,
                "accepted": accepted,
                "rolled_back": rollbacks,
                "censored": censored,
                "interventions_per_episode": (interventions / episodes) if episodes else 0.0,
                "accepted_share_of_interventions": (
                    accepted / interventions if interventions else 0.0
                ),
                "in_run_false_recovery_rate": (
                    rollbacks / interventions if interventions else 0.0
                ),
                "episodes": episodes,
                "episodes_with_fallback": int(metric_subset["fallback_triggered"].sum()),
                "diagnoses": ";".join(
                    f"{category}={int((subset['diagnosis'] == category).sum())}"
                    for category in DIAGNOSIS_CATEGORIES
                )
                if len(subset)
                else "",
            }
        )
    return pd.DataFrame(rows)


def _comparison_table(metrics: pd.DataFrame, config: HealingExperimentConfig) -> pd.DataFrame:
    """Paired bootstrap intervals for the predeclared arm comparisons."""
    rows: List[Dict[str, Any]] = []
    available = set(metrics["arm"].unique())
    for target, reference in COMPARISONS:
        if target not in available or reference not in available:
            continue
        for metric_index, metric in enumerate(METRICS):
            interval = paired_bootstrap_difference(
                metrics,
                target,
                reference,
                metric,
                replicates=config.bootstrap_replicates,
                seed=derive_seed(config.bootstrap_seed, metric_index, len(rows)),
                method_column="arm",
            )
            rows.append(
                {
                    "target": target,
                    "reference": reference,
                    "metric": metric,
                    **interval,
                    "favours": "lower" if metric != "throughput" else "higher",
                }
            )
    return pd.DataFrame(rows)


def action_effect_probe(
    config: HealingExperimentConfig,
    workloads: Sequence[HealingWorkload],
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Measure the true effect and the received reward of every action at every incident.

    The probe runs a detection-powered but never-intervening controller over the supplied
    traces, then replays each incident once per candidate action: the recorded
    configuration prefix, the candidate action applied at the incident, and holding from
    then on. Both the realized total waiting time (ground truth) and the reward the
    controller's own verification window would have produced are recorded, which is what
    makes it possible to check whether the reward ranks actions the way the workload does.

    Args:
        config: Study design (thresholds, scheduler, seeds).
        workloads: Traces to probe (validation split in the declared study).
        progress: Print a progress line every ten traces.

    Returns:
        One row per ``(incident, action)`` with ``reward_band_units``,
        ``true_effect_per_process`` and the component evidence.
    """
    simulator = RuntimeSimulator(config.scheduler)
    generator = WorkloadGenerator(config.families)
    episodes_by_key = {}
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(workloads):
        if progress and index % 10 == 0:
            print(f"  action probe: trace {index + 1}/{len(workloads)}", flush=True)
        controller = SelfHealingController(
            config.healing, config.scheduler, HoldOnlyPolicy(), label="probe"
        )
        controller.begin_episode(index)
        run = simulator.run(item.workload, controller, training=False, policy_name="probe")
        episode = controller.end_episode()
        episodes_by_key[item.fingerprint] = (item, run, episode)
        actual_total = float(sum(outcome.waiting_time for outcome in run.result.outcomes))
        process_count = max(1, len(run.result.outcomes))
        for incident in episode.incidents:
            band = max(incident.load_band_at_trigger, 1e-9)
            reference = max(incident.pre_load_level, 1e-9)
            allowed = available_actions(
                incident.settings_before,
                incident.settings_before,
                config.healing,
            )
            for action in range(len(HEALING_ACTION_NAMES)):
                probe_run = replay_action_at_incident(
                    simulator, item.workload, run.decisions, incident, action
                )
                probe_total = float(
                    sum(outcome.waiting_time for outcome in probe_run.result.outcomes)
                )
                window = probe_run.decisions[
                    incident.decision_index : incident.decision_index
                    + config.healing.verify_decisions
                ]
                accrued = sum(record.waiting_cost_increment for record in window)
                duration = sum(
                    record.end_time - record.observation.current_time for record in window
                )
                post_load = accrued / duration if duration else reference
                changed = action in allowed and action != 0
                cost = config.healing.intervention_cost * reference / band if changed else 0.0
                rows.append(
                    {
                        "family": item.family,
                        "repetition": item.repetition,
                        "workload_fingerprint": item.fingerprint,
                        "decision_index": incident.decision_index,
                        "diagnosis": incident.diagnosis,
                        "current_policy": incident.settings_before.policy_name,
                        "action": action,
                        "action_name": HEALING_ACTION_NAMES[action],
                        "offered": action in allowed,
                        "reference_wait": incident.reference_at_trigger,
                        "load_band_at_trigger": band,
                        "pre_load_level": reference,
                        "post_load_level": post_load,
                        "reward_band_units": (reference - post_load) / band - cost,
                        "true_total_waiting": probe_total,
                        "hold_total_waiting": actual_total,
                        "true_effect_per_process": (actual_total - probe_total) / process_count,
                    }
                )
    del generator
    return pd.DataFrame(rows)


def _coverage_rows(
    models: Sequence[TrainedHealingModel],
    encoder: HealingContextEncoder,
) -> pd.DataFrame:
    """State/action coverage of every learned model."""
    rows: List[Dict[str, Any]] = []
    for model in models:
        agent = model.policy.agent
        visits = agent.state_action_visit_counts
        rows.append(
            {
                "training_seed": model.training_seed,
                "n_states": int(agent.n_states),
                "n_actions": int(agent.n_actions),
                "visited_states": int((agent.visit_counts > 0).sum()),
                "visited_state_action_pairs": int((visits > 0).sum()),
                "state_action_coverage": float((visits > 0).sum() / (agent.n_states * agent.n_actions)),
                "training_episodes": len(model.episodes),
                "training_incidents": int(sum(len(episode.incidents) for episode in model.episodes)),
                "training_interventions": int(sum(episode.interventions for episode in model.episodes)),
                "encoder_states": encoder.n_states,
            }
        )
    return pd.DataFrame(rows)


def _select_demo(
    incidents: pd.DataFrame,
    metrics: pd.DataFrame,
    episodes: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Choose the validation trace used as the worked demonstration.

    Deterministic rule, declared before the run: prefer the episode with the most accepted
    interventions from the first learned arm (``q_recovery``), then the most rolled-back
    interventions, then the most incidents; ties break on family and repetition. If no
    learned episode qualifies, fall back to the rule arms with the same ordering.
    """
    if incidents.empty or episodes.empty:
        return None
    candidates = episodes[episodes["split"] == "validation"].copy()
    if candidates.empty:
        return None
    candidates = candidates[candidates["interventions"] > 0]
    if candidates.empty:
        return None

    def rank(row: pd.Series) -> Tuple[int, int, int, int, int, int, str, int]:
        arm_priority = 0 if row["arm"] == "q_recovery" else (1 if row["arm"].startswith("rule") else 2)
        shows_accept = 0 if int(row["accepted"]) > 0 else 1
        shows_rollback = 0 if int(row["rolled_back"]) > 0 else 1
        return (
            arm_priority,
            shows_accept,
            shows_rollback,
            -int(row["accepted"]),
            -int(row["interventions"]),
            -int(row["incidents"]),
            str(row["family"]),
            int(row["repetition"]),
        )

    ordered = candidates.copy()
    ordered["_rank"] = ordered.apply(rank, axis=1)
    chosen = ordered.sort_values("_rank").iloc[0]
    # A learned arm appears once per training seed, so the demonstration must also be
    # restricted to the seed of the selected episode; otherwise the artifact would mix
    # five different controllers and could not be reproduced by re-running one of them.
    seed = int(chosen["training_seed"]) if not pd.isna(chosen["training_seed"]) else None
    match = incidents[
        (incidents["arm"] == chosen["arm"])
        & (incidents["family"] == chosen["family"])
        & (incidents["repetition"] == chosen["repetition"])
        & (incidents["split"] == "validation")
    ]
    metric_match = metrics[
        (metrics["arm"] == chosen["arm"])
        & (metrics["family"] == chosen["family"])
        & (metrics["repetition"] == chosen["repetition"])
        & (metrics["split"] == "validation")
    ]
    if seed is not None:
        match = match[match["training_seed"] == seed]
        metric_match = metric_match[metric_match["training_seed"] == seed]
    return {
        "arm": chosen["arm"],
        "family": chosen["family"],
        "repetition": int(chosen["repetition"]),
        "training_seed": seed,
        "workload_fingerprint": chosen["workload_fingerprint"],
        "incidents": match.to_dict(orient="records"),
        "metrics": metric_match.to_dict(orient="records"),
        "episode": chosen.drop(labels=["_rank"]).to_dict(),
    }


def _demo_payload(
    demo: Dict[str, Any],
    workloads: Sequence[HealingWorkload],
    config: HealingExperimentConfig,
    models: Sequence[TrainedHealingModel],
) -> Dict[str, Any]:
    """Expand the chosen demonstration into a self-contained, traceable artifact.

    The trace is re-run deterministically with exactly the controller and trained model
    that produced the selected episode, so the artifact is the same trace the tables
    summarize rather than a fresh sample.
    """
    item = next(
        workload
        for workload in workloads
        if workload.family == demo["family"] and workload.repetition == demo["repetition"]
    )
    arm = next(
        candidate
        for candidate in default_arms(config)
        if candidate.name == demo["arm"]
        and candidate.training_seed == (demo.get("training_seed") or 0)
    )
    simulator = RuntimeSimulator(config.scheduler)
    encoder = HealingContextEncoder(config.healing)
    if arm.policy_kind == POLICY_KIND_Q:
        trained_model = next(
            model for model in models if model.training_seed == arm.training_seed
        )
        policy = trained_model.policy
    else:
        policy = _make_policy(arm, config, encoder)
    controller = SelfHealingController(
        config.healing, config.scheduler, policy, initial_policy=arm.initial_policy, label=arm.name
    )
    controller.begin_episode(0)
    run = simulator.run(item.workload, controller, training=False, policy_name=arm.name)
    episode = controller.end_episode()
    expected = len(demo.get("incidents", ()))
    if expected and len(episode.incidents) != expected:
        # The artifact must show the trace the tables summarize, not a fresh sample.
        raise ValidationError(
            "demonstration trace does not reproduce the selected episode: "
            f"{len(episode.incidents)} incidents replayed, {expected} recorded"
        )
    decisions = [
        {
            "decision_index": record.decision_index,
            "simulated_time": record.observation.current_time,
            "selected_pid": record.chosen_pid,
            "executed_from": record.start_time,
            "executed_to": record.end_time,
            "policy": ACTION_NAMES[record.decision.action],
            "quantum": record.decision.quantum,
            "switch_overhead": record.switch_overhead,
            "policy_changed": record.policy_switch,
            "completed_pids": list(record.completed_pids),
            "completed_waiting_times": list(record.completed_waiting_times),
            "forced_hold_reason": record.decision.fallback_reason,
        }
        for record in run.decisions
    ]
    return {
        "selection_rule": (
            "validation split only, deterministic: prefer the learned arm, then an episode that "
            "shows both an accepted and a rolled-back verdict, then more accepted verdicts, then "
            "more interventions, then more incidents; ties break on family and repetition. The "
            "episode is identified by arm *and* training seed, so the trace below is the same "
            "trace the tables summarize."
        ),
        "arm": {
            "name": arm.name,
            "description": arm.description,
            "initial_policy": ACTION_NAMES[arm.initial_policy],
            "policy_kind": arm.policy_kind,
            "training_seed": arm.training_seed,
        },
        "workload": {
            "family": item.family,
            "repetition": item.repetition,
            "seed": item.seed,
            "fingerprint": item.fingerprint,
            "processes": [
                {
                    "pid": process.pid,
                    "arrival_time": process.arrival_time,
                    "burst_time": process.burst_time,
                    "priority": process.priority,
                }
                for process in item.workload.processes
            ],
        },
        "incidents": [incident.as_row() for incident in episode.incidents],
        "decisions": decisions,
        "result": {
            "policy_name": run.result.policy_name,
            "total_elapsed_time": run.result.total_elapsed_time,
            "context_switches": run.result.context_switches,
            "total_waiting_time": float(sum(o.waiting_time for o in run.result.outcomes)),
            "mean_waiting_time": compute_metrics(run.result).avg_waiting_time,
            "mean_response_time": compute_metrics(run.result).avg_response_time,
            "outcomes": [
                {
                    "pid": outcome.pid,
                    "arrival_time": outcome.arrival_time,
                    "burst_time": outcome.burst_time,
                    "start_time": outcome.start_time,
                    "completion_time": outcome.completion_time,
                    "waiting_time": outcome.waiting_time,
                    "response_time": outcome.response_time,
                }
                for outcome in run.result.outcomes
            ],
        },
        "healing_config": config.healing.as_dict(),
    }


def _render_report(
    summary: Dict[str, Any],
    comparisons: pd.DataFrame,
    detection: pd.DataFrame,
    audit: pd.DataFrame,
    thresholds: Any = None,
) -> str:
    """Render the human-readable report from the machine-readable tables."""
    lines = [
        "# Self-healing CPU scheduling: generated report",
        "",
        "This file is generated from the machine-readable artifacts in this directory "
        "(`healing_metrics_*.csv`, `healing_incidents_*.csv`, `healing_audit.csv`, "
        "`healing_comparisons.csv`, `healing_summary.json`). Every number below appears in "
        "those files.",
        "",
        f"* split: **{summary['split']}**",
        f"* workloads per arm: {summary['workloads']}",
        f"* training seeds: {summary['training_seeds']}",
        f"* final-test seed: {summary['final_test_seed']}",
        f"* bootstrap replicates: {summary['bootstrap_replicates']}",
        "",
        "## Mean metrics by arm",
        "",
        "| Arm | Mean waiting | P95 waiting | Mean response | Context switches | Incidents | Interventions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["means"]:
        lines.append(
            f"| {row['arm']} | {row['avg_waiting_time']:.3f} | {row['p95_waiting_time']:.3f} | "
            f"{row['avg_response_time']:.3f} | {row['context_switches']:.3f} | "
            f"{row['incidents']:.3f} | {row['interventions']:.3f} |"
        )
    lines += ["", "## Predeclared paired comparisons (family-stratified bootstrap)", "",
              "| Target | Reference | Metric | Difference | 95% CI |", "|---|---|---|---:|---|"]
    for row in comparisons.itertuples():
        lines.append(
            f"| {row.target} | {row.reference} | {row.metric} | {row.mean_difference:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines += ["", "## Detection and recovery summary", "",
              "| Arm | Episodes | Incidents | Interventions | Accepted | Rolled back | Censored | In-run false-recovery rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in detection.itertuples():
        lines.append(
            f"| {row.arm} | {row.episodes} | {row.incidents} | {row.interventions} | {row.accepted} | "
            f"{row.rolled_back} | {row.censored} | {row.in_run_false_recovery_rate:.3f} |"
        )
    if thresholds is not None:
        lines += [
            "",
            "## Declared thresholds and what they were derived from",
            "",
            "Both constants are derived from the *no-action control* on the training split "
            "(`healing_threshold_derivation.csv`, `healing_control_incidents.csv`); the "
            "final-test split never informed them.",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| control traces ({thresholds.split}) | {thresholds.traces} |",
            f"| degradation events declared with no intervention | {thresholds.incidents} |",
            f"| events per trace (false-trigger rate of the detector) | "
            f"{thresholds.incidents_per_trace:.3f} |",
            f"| null load closure p50 | {thresholds.null_p50:.3f} |",
            f"| null load closure p95 | {thresholds.null_p95:.3f} |",
            f"| derived `accept_closure_factor` | {thresholds.derived_accept_closure_factor} |",
            f"| declared `accept_closure_factor` | {thresholds.declared_accept_closure_factor} |",
            f"| false-accept rate of the declared rule on the control | "
            f"{thresholds.declared_false_accept_rate:.3f} |",
            "",
            "Congestion-gate sweep (what each `min_load_factor` removes from the control):",
            "",
            "| min_load_factor | incidents per trace |",
            "|---|---:|",
        ]
        for row in thresholds.gate_sweep:
            lines.append(f"| {row['min_load_factor']:.2f} | {row['incidents_per_trace']:.3f} |")
    if not audit.empty:
        lines += ["", "## Counterfactual replay audit of interventions", "",
                  "An intervention's gain is `hold_total_waiting - actual_total_waiting`: the total waiting "
                  "time the same trace would have accumulated if the controller had held its pre-incident "
                  "configuration from that point on, minus what the realized run accumulated.", "",
                  "| Arm | Audited interventions | Mean gain | Helped | False recoveries | "
                  "Verified and helped | Rejected but helpful |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm_name, subset in audit.groupby("arm", sort=False):
            gains = subset["intervention_gain"].astype(float)
            verdicts = subset["verdict"].astype(str)
            helped = gains > 0
            rolled = verdicts == "rolled_back"
            accepted = verdicts == "accepted"
            lines.append(
                f"| {arm_name} | {len(gains)} | {gains.mean():+.3f} | {int(helped.sum())} | "
                f"{int((~helped).sum())} | {int((accepted & helped).sum())} | "
                f"{int((rolled & helped).sum())} |"
            )
    lines += ["", "Generated by `python main.py healing-study`.", ""]
    return "\n".join(lines)


def run_healing_experiment(
    config: Optional[HealingExperimentConfig] = None,
    *,
    results_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    make_figures: bool = True,
    audit: bool = True,
    probe: bool = True,
    progress: bool = True,
) -> HealingExperimentArtifacts:
    """Run the complete study and write every artifact.

    Args:
        config: Study design; the declared default is used when omitted.
        results_dir: Output directory for tables (defaults to ``results/healing``).
        figures_dir: Output directory for figures (defaults to ``figures/healing``).
        make_figures: Whether to render optional Matplotlib figures.
        audit: Whether to run the counterfactual replay audit.
        probe: Whether to run the validation action-effect probe.
        progress: Print progress lines.

    Returns:
        The :class:`HealingExperimentArtifacts` of the run.
    """
    config = config or HealingExperimentConfig()
    results_dir = Path(results_dir or Path("results") / "healing")
    figures_dir = Path(figures_dir or Path("figures") / "healing")
    results_dir.mkdir(parents=True, exist_ok=True)
    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    generator = WorkloadGenerator(config.families)
    arms = default_arms(config)
    if progress:
        print("Training learned recovery policies on the training split ...", flush=True)
    models, training_episodes = train_healing_models(config, generator)
    trained = {model.training_seed: model for model in models}

    # Imported here because the derivation module reuses this module's workload builder.
    from experiments.healing_thresholds import derivation_rows, derive_thresholds

    if progress:
        print("Deriving the declared thresholds from the training control ...", flush=True)
    threshold_derivation, control_incidents = derive_thresholds(config, "training")
    if progress:
        print(
            "  control incidents per trace %.3f, null closure p95 %.3f, "
            "declared factor %.2f (false-accept on control %.3f)"
            % (
                threshold_derivation.incidents_per_trace,
                threshold_derivation.null_p95,
                threshold_derivation.declared_accept_closure_factor,
                threshold_derivation.declared_false_accept_rate,
            ),
            flush=True,
        )

    metrics_by_split: Dict[str, pd.DataFrame] = {}
    incidents_by_split: Dict[str, pd.DataFrame] = {}
    episodes_by_split: Dict[str, pd.DataFrame] = {}
    audit_frames: List[pd.DataFrame] = []

    for split in ("validation", "final_test"):
        if progress:
            print(f"Evaluating all arms on the {split} split ...", flush=True)
        workloads = build_healing_workloads(split, config, generator)
        metric_rows: List[Dict[str, Any]] = []
        incident_rows: List[Dict[str, Any]] = []
        audit_rows: List[Dict[str, Any]] = []
        episode_rows: List[Dict[str, Any]] = []
        for arm in arms:
            rows, incidents, audits, episodes = _evaluate_arm_on_split(
                arm, split, workloads, config, trained, audit=audit
            )
            metric_rows.extend(rows)
            incident_rows.extend(incidents)
            audit_rows.extend(audits)
            episode_rows.extend(episodes)
            if progress:
                print(f"  {arm.name}: {len(rows)} traces", flush=True)
        metrics_frame = pd.DataFrame(metric_rows)
        incidents_frame = pd.DataFrame(incident_rows)
        episodes_frame = pd.DataFrame(episode_rows)
        metrics_by_split[split] = metrics_frame
        incidents_by_split[split] = incidents_frame
        episodes_by_split[split] = episodes_frame
        if audit_rows:
            audit_frames.append(pd.DataFrame(audit_rows))

    audit_frame = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    evaluation_metrics = metrics_by_split["final_test"]
    comparisons = _comparison_table(evaluation_metrics, config)
    detection_frames = [
        _detection_summary(
            incidents_by_split.get(split, pd.DataFrame()),
            metrics_by_split[split],
            split,
        )
        for split in ("validation", "final_test")
    ]
    detection = pd.concat(detection_frames, ignore_index=True)
    coverage = _coverage_rows(models, HealingContextEncoder(config.healing))

    probe_frame = pd.DataFrame()
    if probe:
        if progress:
            print("Running the validation action-effect probe ...", flush=True)
        probe_families = config.action_probe_families or config.validation_families
        probe_workloads: List[HealingWorkload] = []
        for family_index, family in enumerate(probe_families):
            for repetition in range(config.action_probe_repetitions):
                seed = derive_seed(config.validation_seed, SPLIT_TAGS["validation"], family_index, repetition)
                probe_workloads.append(
                    HealingWorkload(
                        "validation", family, repetition, seed, generator.generate(family, seed)
                    )
                )
        probe_frame = action_effect_probe(config, probe_workloads, progress=progress)

    means: List[Dict[str, Any]] = []
    for arm_name, subset in evaluation_metrics.groupby("arm", sort=False):
        entry: Dict[str, Any] = {"arm": arm_name, "traces": int(len(subset))}
        for metric in METRICS:
            entry[metric] = float(subset[metric].astype(float).mean())
        for column in ("incidents", "interventions", "accepted", "rolled_back", "held", "censored"):
            entry[column] = float(subset[column].astype(float).mean())
        entry["fallback_rate"] = float(subset["fallback_triggered"].astype(bool).mean())
        means.append(entry)

    audit_summary: List[Dict[str, Any]] = []
    if not audit_frame.empty:
        for arm_name, subset in audit_frame[
            audit_frame["split"] == "final_test"
        ].groupby("arm", sort=False):
            gains = subset["intervention_gain"].astype(float)
            verdicts = subset["verdict"].astype(str)
            rolled = verdicts == "rolled_back"
            accepted = verdicts == "accepted"
            helped = gains > 0
            audit_summary.append(
                {
                    "arm": arm_name,
                    "audited_interventions": int(len(gains)),
                    "mean_gain": float(gains.mean()),
                    "median_gain": float(gains.median()),
                    "helped": int(helped.sum()),
                    # A false recovery is an intervention that did not help (the honest,
                    # counterfactual definition; the in-run verdict is not evidence).
                    "false_recoveries": int((~helped).sum()),
                    "false_recovery_rate": float((~helped).mean()),
                    # What the in-run verification did with the interventions, measured
                    # against the counterfactual ground truth.
                    "verified_and_helped": int((accepted & helped).sum()),
                    "verified_but_harmful": int((accepted & ~helped).sum()),
                    "rejected_but_helpful": int((rolled & helped).sum()),
                    "rejected_and_harmful": int((rolled & ~helped).sum()),
                    "in_run_rejection_rate": float(rolled.mean()),
                    "false_reject_rate": (
                        float((rolled & helped).sum() / rolled.sum()) if rolled.any() else float("nan")
                    ),
                }
            )

    summary: Dict[str, Any] = {
        "project_title": (
            "Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads"
        ),
        "system": "self-healing scheduler (monitor -> detect -> diagnose -> act -> verify -> rollback)",
        "split": "final_test",
        "workloads": int(evaluation_metrics["workload_fingerprint"].nunique()),
        "training_seeds": list(config.training_seeds),
        "training_episodes_per_seed": config.training_episodes_per_seed,
        "validation_seed": config.validation_seed,
        "final_test_seed": config.final_test_seed,
        "final_test_repetitions": config.final_test_repetitions,
        "bootstrap_replicates": config.bootstrap_replicates,
        "healing_config": config.healing.as_dict(),
        "scheduler_config": {
            "round_robin_quantum": config.scheduler.round_robin_quantum,
            "switching_cost": config.scheduler.switching_cost,
            "lower_priority_number_is_higher_priority": (
                config.scheduler.lower_priority_number_is_higher_priority
            ),
        },
        "q_learning": {
            "learning_rate": config.q_learning.learning_rate,
            "discount_factor": config.q_learning.discount_factor,
            "epsilon_start": config.q_learning.epsilon_start,
            "epsilon_min": config.q_learning.epsilon_min,
            "epsilon_decay_per_episode": config.q_learning.epsilon_decay_per_episode,
        },
        "threshold_derivation": threshold_derivation.as_dict(),
        "means": means,
        "detection_summary": detection.to_dict(orient="records"),
        "audit_summary": audit_summary,
        "coverage": coverage.to_dict(orient="records"),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    demo = _select_demo(
        incidents_by_split.get("validation", pd.DataFrame()),
        metrics_by_split.get("validation", pd.DataFrame()),
        episodes_by_split.get("validation", pd.DataFrame()),
    )

    paths: Dict[str, Path] = {}
    for split, frame in metrics_by_split.items():
        path = results_dir / f"healing_metrics_{split}.csv"
        frame.to_csv(path, index=False)
        paths[f"metrics_{split}"] = path
    for split, frame in incidents_by_split.items():
        path = results_dir / f"healing_incidents_{split}.csv"
        frame.to_csv(path, index=False)
        paths[f"incidents_{split}"] = path
    for split, frame in episodes_by_split.items():
        path = results_dir / f"healing_episodes_{split}.csv"
        frame.to_csv(path, index=False)
        paths[f"episodes_{split}"] = path
    for name, frame in (
        ("audit", audit_frame),
        ("comparisons", comparisons),
        ("detection_summary", detection),
        ("coverage", coverage),
        ("action_probe", probe_frame),
        ("training_episodes", training_episodes),
        ("threshold_derivation", pd.DataFrame(derivation_rows(threshold_derivation))),
        ("control_incidents", control_incidents),
    ):
        path = results_dir / f"healing_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    summary_path = results_dir / "healing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary"] = summary_path
    report_path = results_dir / "healing_report.md"
    report_path.write_text(
        _render_report(
            summary,
            comparisons,
            detection[detection["split"] == "final_test"],
            audit_frame[audit_frame["split"] == "final_test"] if not audit_frame.empty else audit_frame,
            threshold_derivation,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    if demo is not None:
        demo_path = results_dir / "healing_demo.json"
        demo_path.write_text(
            json.dumps(
                _demo_payload(
                    demo,
                    build_healing_workloads("validation", config, generator),
                    config,
                    models,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        paths["demo"] = demo_path
    if make_figures:
        from visualization.healing_plots import plot_healing_figures

        paths.update(
            plot_healing_figures(
                figures_dir, evaluation_metrics, comparisons, detection, audit_frame, probe_frame
            )
        )

    return HealingExperimentArtifacts(
        models=models,
        training_episodes=training_episodes,
        metrics=metrics_by_split,
        incidents=incidents_by_split,
        audit=audit_frame,
        episode_audit=pd.concat(
            [frame[frame["verdict"] == "episode_level"] for frame in audit_frames],
            ignore_index=True,
        )
        if audit_frames
        else pd.DataFrame(),
        comparisons=comparisons,
        detection_summary=detection,
        action_probe=probe_frame,
        coverage=coverage,
        summary=summary,
        paths=paths,
    )
