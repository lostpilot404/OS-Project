"""Self-healing scheduling study: training, held-out evaluation, and recovery audit.

The experiment produced by this module is the project's primary evidence. For every
predeclared arm (see :mod:`experiments.healing_config`) and every workload of a split it
records the six established scheduling metrics plus the tail statistic, and for the healing
arms it also records the whole recovery ledger: trigger evidence, diagnosis, offered
actions, chosen action, verification window, verdict, reward and realized switch overhead.

Three analyses make the recovery loop checkable rather than asserted:

``detection summary``
    Counts of every detector and verification outcome per arm (detected, held, accepted,
    rejected, inconclusive, window truncated, budget exhausted, work gated, ...), so "the
    loop recovers" cannot hide behind an aggregate.
``counterfactual replay audit``
    *Matched replay*: the same controller, the same workload and the same recorded prefix
    are re-run with exactly one incident forced to hold, every later decision still taken by
    the controller as usual. The difference between that run's total waiting time and the
    realized run's is the marginal value of *that* decision, and a non-positive value is the
    study's definition of a false recovery. A second, clearly separated counterfactual
    (``hold_forever_*``) replays the configuration prefix and then never heals again, which
    answers the different question of what shutting the loop down at that point was worth.
    An episode-level row answers whether adapting at all helped a trace.
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
from healing.config import HealingConfig
from healing.controller import (
    EpisodeRecord,
    SelfHealingController,
    available_actions,
    replay_action_at_incident,
    replay_hold_counterfactual,
)
from healing.diagnosis import DIAGNOSIS_CATEGORIES
from healing.policies import (
    HealingContext,
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

#: Names of the five declared encoder features, in :class:`HealingContextEncoder` order.
_ENCODER_FEATURE_NAMES: Tuple[str, ...] = (
    "diagnosis",
    "excess",
    "ready_count",
    "current_policy",
    "remaining_work",
)

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
    #: Coverage checkpoints ``(episodes, states, state_action_pairs, incidents)`` recorded
    #: during training, so the study can show whether extra episodes still buy new states.
    coverage_curve: Tuple[Dict[str, Any], ...] = ()


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
    detection_scenarios: pd.DataFrame
    action_probe: pd.DataFrame
    coverage: pd.DataFrame
    coverage_curve: pd.DataFrame
    q_seed_variability: pd.DataFrame
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
        curve: List[Dict[str, Any]] = []
        checkpoints = {max(1, config.training_episodes_per_seed // 4) * step for step in range(1, 5)}
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
            if (episode_index + 1) in checkpoints:
                curve.append(
                    {
                        "episodes": episode_index + 1,
                        "visited_states": int((policy.agent.visit_counts > 0).sum()),
                        "visited_state_action_pairs": int(
                            (policy.agent.state_action_visit_counts > 0).sum()
                        ),
                        "incidents": int(
                            sum(len(record.incidents) for record in episodes)
                        ),
                    }
                )
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
                    "rejected": record.rejected,
                    "inconclusive": record.inconclusive,
                    "unverified": record.unverified,
                    "held": record.holds,
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
                coverage_curve=tuple(curve),
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
        "rejected": 0,
        "inconclusive": 0,
        "unverified": 0,
        "held": 0,
        "confirmed_failures": 0,
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
                "rejected": episode.rejected,
                "inconclusive": episode.inconclusive,
                "unverified": episode.unverified,
                "held": episode.holds,
                "confirmed_failures": episode.confirmed_failures,
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
                "rejected": episode.rejected,
                "inconclusive": episode.inconclusive,
                "unverified": episode.unverified,
                "held": episode.holds,
                "confirmed_failures": episode.confirmed_failures,
                "fallback_triggered": episode.fallback_triggered,
                "total_waiting_time": float(sum(waits)),
                "detector_reasons": ";".join(
                    f"{reason}={count}" for reason, count in episode.detector_reasons
                ),
            }
        )
        if audit:
            audit_rows.extend(
                _audit_episode(
                    simulator, split, arm, item, run, episode, index, policy, config
                )
            )
    return metric_rows, incident_rows, audit_rows, episode_rows


def _divergence_index(realized, counterfactual) -> int:
    """Return the first decision index at which two runs differ (``-1`` when identical).

    The audit compares a realized run with a replay of the same controller on the same
    workload; the replay must be byte-identical up to the audited incident and diverge
    exactly there. This helper makes that claim checkable instead of assumed.
    """

    def projection(records):
        return [
            (
                record.decision.action,
                record.decision.quantum,
                record.chosen_pid,
                record.start_time,
                record.end_time,
                record.completed_pids,
            )
            for record in records
        ]

    left, right = projection(realized), projection(counterfactual)
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return -1


def _audit_episode(
    simulator: RuntimeSimulator,
    split: str,
    arm: HealingArm,
    item: HealingWorkload,
    run,
    episode: EpisodeRecord,
    episode_index: int,
    policy,
    config: HealingExperimentConfig,
) -> List[Dict[str, Any]]:
    """Counterfactual audit of every intervention: matched replay plus hold-forever replay.

    * **Matched replay** (``matched_gain``): the same controller, the same workload, the same
      recorded prefix, with this one incident forced to hold and every later decision taken
      by the controller as usual. The difference against the realized run is the marginal
      value of that single decision.
    * **Hold-forever replay** (``hold_forever_gain``): the recorded configuration prefix is
      replayed and then held for good, with no further recovery at all. This answers a
      different question -- what shutting the loop down at this incident would have been
      worth -- and is kept because earlier reports used it.

    Both replays reuse the *same policy object* in evaluation mode (learned policies do not
    update when ``training=False``), so they never contaminate what the controller knows.
    """
    rows: List[Dict[str, Any]] = []
    actual_total = float(sum(outcome.waiting_time for outcome in run.result.outcomes))
    interventions = [incident for incident in episode.incidents if incident.action_changed]
    for incident in interventions:
        controller = SelfHealingController(
            config.healing,
            config.scheduler,
            policy,
            initial_policy=arm.initial_policy,
            label=f"{arm.name}-matched-hold",
            forced_holds=(incident.incident_index,),
        )
        controller.begin_episode(episode_index)
        matched_run = simulator.run(
            item.workload, controller, training=False, policy_name="matched-hold"
        )
        matched_episode = controller.end_episode()
        matched_total = float(
            sum(outcome.waiting_time for outcome in matched_run.result.outcomes)
        )
        divergence = _divergence_index(run.decisions, matched_run.decisions)
        forced = next(
            (
                candidate
                for candidate in matched_episode.incidents
                if candidate.incident_index == incident.incident_index
            ),
            None,
        )

        counterfactual = replay_hold_counterfactual(
            simulator, item.workload, run.decisions, incident
        )
        hold_total = float(
            sum(outcome.waiting_time for outcome in counterfactual.result.outcomes)
        )
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
                "accepted": incident.verdict == "accepted",
                "confirmed_failure": incident.confirmed_failure,
                "closure_fraction": incident.closure_fraction,
                "closure_residual": incident.closure_residual,
                "hold_total_waiting": matched_total,
                "actual_total_waiting": actual_total,
                "matched_gain": matched_total - actual_total,
                "hold_forever_total_waiting": hold_total,
                "hold_forever_gain": hold_total - actual_total,
                "helped": bool(matched_total - actual_total > 0),
                "harmful": bool(matched_total - actual_total < 0),
                "divergence_decision_index": divergence,
                "forced_incident_found": forced is not None,
                "forced_incident_held": bool(forced is not None and not forced.action_changed),
                "counterfactual_incidents": len(matched_episode.incidents),
                "counterfactual_interventions": matched_episode.interventions,
            }
        )
    if interventions:
        first = interventions[0]
        counterfactual = replay_hold_counterfactual(
            simulator, item.workload, run.decisions, first
        )
        hold_total = float(
            sum(outcome.waiting_time for outcome in counterfactual.result.outcomes)
        )
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
                "accepted": False,
                "confirmed_failure": False,
                "closure_fraction": first.closure_fraction,
                "closure_residual": first.closure_residual,
                "hold_total_waiting": hold_total,
                "actual_total_waiting": actual_total,
                "matched_gain": hold_total - actual_total,
                "hold_forever_total_waiting": hold_total,
                "hold_forever_gain": hold_total - actual_total,
                "helped": bool(hold_total - actual_total > 0),
                "harmful": bool(hold_total - actual_total < 0),
                "divergence_decision_index": -1,
                "forced_incident_found": False,
                "forced_incident_held": False,
                "counterfactual_incidents": 0,
                "counterfactual_interventions": 0,
            }
        )
    return rows


def _rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """Rank-based AUC without SciPy: ``P(score of a positive > score of a negative)``.

    Used to report how well the verifier's statistic separates helpful from unhelpful
    interventions. Ties count a half, which is what ranking and the Mann-Whitney form
    below implement; an undefined AUC (one class only) is returned as ``nan``.
    """
    positives = labels.astype(bool)
    if positives.nunique() < 2:
        return float("nan")
    ranks = scores.astype(float).rank()
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    return float((ranks[positives.values].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _json_safe(value: Any) -> Any:
    """Return ``value`` with non-finite floats replaced by ``None``.

    ``NaN`` is not valid JSON and is not equal to itself, so a summary that carries it cannot
    be round-tripped or compared for reproducibility. Every undefined statistic is therefore
    written as ``null``: the artifact says "not measured" instead of "not a number".
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _detection_summary(incidents: pd.DataFrame, metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per-arm counts of detections, interventions and verdicts within one split."""
    rows: List[Dict[str, Any]] = []
    for arm_name, metric_subset in metrics.groupby("arm", sort=False):
        subset = incidents[incidents["arm"] == arm_name] if not incidents.empty else incidents
        episodes = len(metric_subset)
        interventions = int(subset["action_changed"].sum()) if len(subset) else 0
        rejected = int((subset["verdict"] == "rejected").sum()) if len(subset) else 0
        accepted = int((subset["verdict"] == "accepted").sum()) if len(subset) else 0
        inconclusive = int((subset["verdict"] == "inconclusive").sum()) if len(subset) else 0
        holds = int((subset["verdict"] == "held").sum()) if len(subset) else 0
        rows.append(
            {
                "split": split,
                "arm": arm_name,
                "incidents": int(len(subset)),
                "incidents_per_episode": (len(subset) / episodes) if episodes else 0.0,
                "interventions": interventions,
                "held": holds,
                "accepted": accepted,
                "rejected": rejected,
                "inconclusive": inconclusive,
                "interventions_per_episode": (interventions / episodes) if episodes else 0.0,
                "accepted_share_of_interventions": (
                    accepted / interventions if interventions else 0.0
                ),
                "rejected_share_of_interventions": (
                    rejected / interventions if interventions else 0.0
                ),
                "inconclusive_share_of_interventions": (
                    inconclusive / interventions if interventions else 0.0
                ),
                "unverified_share_of_interventions": (
                    (rejected + inconclusive) / interventions if interventions else 0.0
                ),
                "confirmed_failure_share_of_interventions": (
                    rejected / interventions if interventions else 0.0
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


def _probe_window_metrics(
    probe_run,
    incident,
    config: HealingExperimentConfig,
) -> tuple:
    """Replay the verification window of one probe action exactly as the controller would.

    Returns ``(post_load, closure_fraction, residual, decisions, completions, duration, truncated)``.
    """
    from healing.reward import closure_fraction_from_loads, phase_from_incident
    from healing.verifier import expected_closure

    phase = phase_from_incident(incident)
    decisions = probe_run.decisions
    start = int(incident.decision_index)
    accrued = 0
    duration = 0
    completions = 0
    window_decisions = 0
    for offset in range(config.healing.verify_max_decisions):
        idx = start + offset
        if idx >= len(decisions):
            break
        record = decisions[idx]
        accrued += int(record.waiting_cost_increment)
        duration += int(record.end_time - record.observation.current_time)
        completions += len(record.completed_waiting_times)
        window_decisions += 1
        if window_decisions >= config.healing.verify_decisions and completions >= config.healing.verify_completions:
            break
        if window_decisions >= config.healing.verify_max_decisions:
            break
    if duration <= 0:
        post_load = None
        closure = None
    else:
        post_load = accrued / duration
        closure = closure_fraction_from_loads(
            float(incident.trigger_load_level),
            float(incident.pre_load_level),
            float(post_load),
            float(incident.load_band_at_trigger),
        )
    if closure is None:
        residual = None
    else:
        exp = expected_closure(phase, config.healing)
        residual = float(closure) - float(exp)
    truncated = (start + window_decisions) >= len(decisions) and window_decisions < config.healing.verify_decisions
    return post_load, closure, residual, window_decisions, completions, duration, bool(truncated)


def action_effect_probe(
    config: HealingExperimentConfig,
    workloads: Sequence[HealingWorkload],
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Measure the true effect and the actual controller reward of every action at every incident.

    The probe runs a detection-powered but never-intervening controller over the supplied
    traces, then replays each incident once per candidate action: the recorded
    configuration prefix, the candidate action applied at the incident, and holding from
    then on. Both the realized total waiting time (ground truth) and the reward the
    controller's own verification window would have produced are recorded, which is what
    makes it possible to check whether the reward ranks actions the way the workload does.

    The reward is computed from an equivalent replayed verification window, including
    the same phase-adjusted residual, acceptance boundary, intervention cost and rollback
    penalty that the live controller uses. The raw load-change proxy is preserved under
    an explicit column name for exploratory analysis.

    Args:
        config: Study design (thresholds, scheduler, seeds).
        workloads: Traces to probe (validation split in the declared study).
        progress: Print a progress line every ten traces.

    Returns:
        One row per ``(incident, action)`` with ``controller_reward`` (the actual
        reward), ``raw_load_improvement_band_units`` (the exploratory proxy),
        ``true_effect_per_process`` and the full verification evidence.
    """
    from healing.reward import controller_reward, cost_in_band_units, phase_from_incident
    from healing.verifier import VerificationEvidence, verify

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
            wait_band = max(float(incident.band_at_trigger), 1e-9)
            wait_ref = max(float(incident.reference_at_trigger), 1e-9)
            load_band = max(float(incident.load_band_at_trigger), 1e-9)
            load_ref = max(float(incident.pre_load_level), 1e-9)
            allowed = available_actions(
                incident.settings_before,
                incident.settings_before,
                config.healing,
            )
            per_action_rows: List[Dict[str, Any]] = []
            for action in range(len(HEALING_ACTION_NAMES)):
                probe_run = replay_action_at_incident(
                    simulator, item.workload, run.decisions, incident, action
                )
                probe_total = float(
                    sum(outcome.waiting_time for outcome in probe_run.result.outcomes)
                )
                post_load, closure, residual, window_decisions, window_completions, window_duration, truncated = _probe_window_metrics(
                    probe_run, incident, config
                )
                changed = action in allowed and action != 0
                cost_band = cost_in_band_units(wait_ref, wait_band, config.healing.intervention_cost) if changed else 0.0
                rollback_band = cost_in_band_units(wait_ref, wait_band, config.healing.rollback_penalty)
                phase = phase_from_incident(incident)
                evidence = VerificationEvidence(
                    changed=changed,
                    closure_fraction=closure,
                    phase=phase,
                    window_decisions=int(window_decisions),
                    window_completions=int(window_completions),
                    window_duration=int(window_duration),
                    truncated=bool(truncated and changed),
                )
                outcome = verify(evidence, config.healing)
                reward = controller_reward(
                    residual=residual,
                    config=config.healing,
                    cost_band_units=cost_band,
                    rollback_penalty_band=rollback_band,
                    changed=changed,
                    confirmed_failure=outcome.confirmed_failure,
                    held=(outcome.verdict == "held"),
                )
                raw_proxy = None
                if post_load is not None:
                    raw_proxy = (load_ref - float(post_load)) / load_band - (config.healing.intervention_cost * load_ref / load_band if changed else 0.0)
                per_action_rows.append(
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
                        "reference_wait": wait_ref,
                        "load_band_at_trigger": load_band,
                        "pre_load_level": load_ref,
                        "post_load_level": post_load,
                        "closure_fraction": closure,
                        "closure_residual": residual,
                        "expected_closure": outcome.expected,
                        "verdict": outcome.verdict,
                        "confirmed_failure": outcome.confirmed_failure,
                        "window_decisions": window_decisions,
                        "window_completions": window_completions,
                        "window_duration": window_duration,
                        "window_truncated": truncated,
                        "controller_reward": reward,
                        "raw_load_improvement_band_units": raw_proxy,
                        "reward_band_units": reward,
                        "true_total_waiting": probe_total,
                        "hold_total_waiting": actual_total,
                        "true_effect_per_process": (actual_total - probe_total) / process_count,
                        "cost_band_units": cost_band,
                    }
                )
            hold_row = next(
                (row for row in per_action_rows if row["action_name"] == "hold"), None
            )
            hold_residual = hold_row["closure_residual"] if hold_row is not None else 0.0
            hold_raw = hold_row["raw_load_improvement_band_units"] if hold_row is not None else 0.0
            for row in per_action_rows:
                # Legacy ``paired_closure_change`` preserves the raw window-statistic diff used in
                # the published tables (see EVALUATION_RESULTS §4.1, +0.152 for set_sjf).
                # It is computed without the intervention cost so the test
                # ``test_action_probe_reports_the_paired_change_of_the_window_statistic`` keeps
                # its historical definition. The cost-including proxy is carried as
                # ``paired_raw_change`` and the phase-adjusted residual diff as
                # ``paired_residual_change``.
                if row["pre_load_level"] is None or row["post_load_level"] is None or row["load_band_at_trigger"] in (None, 0):
                    row["paired_closure_change"] = 0.0
                else:
                    raw_stat = (float(row["pre_load_level"]) - float(row["post_load_level"])) / max(float(row["load_band_at_trigger"]), 1e-9)
                    hold_stat = (float(hold_row["pre_load_level"]) - float(hold_row["post_load_level"])) / max(float(hold_row["load_band_at_trigger"]), 1e-9) if hold_row is not None and hold_row["post_load_level"] is not None else 0.0
                    row["paired_closure_change"] = float(raw_stat - hold_stat)
                if row["raw_load_improvement_band_units"] is None or hold_raw is None:
                    row["paired_raw_change"] = 0.0
                else:
                    row["paired_raw_change"] = float(row["raw_load_improvement_band_units"]) - float(hold_raw)
                if row["closure_residual"] is None or hold_residual is None:
                    row["paired_residual_change"] = 0.0
                else:
                    row["paired_residual_change"] = float(row["closure_residual"]) - float(hold_residual)
            rows.extend(per_action_rows)
    del generator
    return pd.DataFrame(rows)

def _coverage_rows(
    models: Sequence[TrainedHealingModel],
    encoder: HealingContextEncoder,
) -> pd.DataFrame:
    """State/action coverage of every learned model, with the distribution behind it.

    The project reports coverage as a distribution, not as a single percentage: how many
    states were reachable at all, how the visits are spread over them, how many decisions
    could only fall back to the initial value because every action of their state was still
    unvisited (``actions_effectively_learned_share`` is the complement), and how many of the
    encoder's declared features were actually exercised.
    """
    rows: List[Dict[str, Any]] = []
    for model in models:
        agent = model.policy.agent
        visits = agent.state_action_visit_counts
        state_visits = np.asarray(agent.visit_counts).reshape(-1)
        incidents = [incident for episode in model.episodes for incident in episode.incidents]
        interventions = [incident for incident in incidents if incident.action_changed]
        bins = [_feature_bins(incident) for incident in incidents]
        exercised = {
            name: len({value[position] for value in bins if value is not None})
            for position, name in enumerate(_ENCODER_FEATURE_NAMES)
        }
        # Replay the decisions in training order and record how many *distinct* actions of
        # that state had already been sampled when the decision was taken. With zero known
        # actions the greedy choice can only reproduce the declared default; with one action
        # there is nothing to compare against. Only the two-or-more case is a real comparison.
        known_actions: Dict[int, set] = {}
        decisions_first_visit = 0
        decisions_single_known_action = 0
        decisions_with_comparison = 0
        for incident in incidents:
            known = known_actions.setdefault(int(incident.context_state_index), set())
            if not known:
                decisions_first_visit += 1
            elif len(known) == 1:
                decisions_single_known_action += 1
            else:
                decisions_with_comparison += 1
            known.add(int(incident.action))
        decided_with_table = len(incidents) - decisions_first_visit
        offered_actions = {
            name.strip()
            for incident in incidents
            for name in str(incident.offered_actions).split(",")
            if name.strip()
        }
        chosen_actions = {
            HEALING_ACTION_NAMES[int(incident.action)] for incident in incidents
        }
        observed_bins = [
            len({value[position] for value in bins if value is not None})
            for position in range(len(_ENCODER_FEATURE_NAMES))
        ]
        reachable_states = int(np.prod(observed_bins)) if observed_bins else 0
        rows.append(
            {
                "training_seed": model.training_seed,
                "n_states": int(agent.n_states),
                "n_actions": int(agent.n_actions),
                # States that the *observed* contexts can reach: the product of the distinct
                # bins that the training incidents actually exercised in each dimension. It
                # is a data-driven denominator, reported next to the declared one, because a
                # dimension that never varies makes part of the declared table unreachable.
                "observed_reachable_states": reachable_states,
                "observed_reachable_state_action_pairs": reachable_states * int(agent.n_actions),
                "offered_actions_exercised": len(offered_actions),
                "chosen_actions_exercised": len(chosen_actions),
                "visited_states": int((state_visits > 0).sum()),
                "visited_state_action_pairs": int((visits > 0).sum()),
                "state_action_coverage": float((visits > 0).sum() / (agent.n_states * agent.n_actions)),
                "state_coverage": float((state_visits > 0).sum() / agent.n_states),
                "state_action_coverage_of_observed_reachable": (
                    float((visits > 0).sum() / (reachable_states * int(agent.n_actions)))
                    if reachable_states
                    else float("nan")
                ),
                "state_coverage_of_observed_reachable": (
                    float((state_visits > 0).sum() / reachable_states) if reachable_states else float("nan")
                ),
                "training_episodes": len(model.episodes),
                "training_incidents": len(incidents),
                "training_interventions": len(interventions),
                "encoder_states": encoder.n_states,
                "decisions": len(incidents),
                "decisions_in_sampled_states": decided_with_table,
                "decisions_with_no_prior_evidence": decisions_first_visit,
                "decisions_with_single_known_action": decisions_single_known_action,
                "decisions_with_action_comparison": decisions_with_comparison,
                # Share of decisions answered by comparing at least two sampled actions of
                # the state; the rest could only reuse the declared default (or a single
                # known action), so they are counted as defaulted, not as learned.
                "actions_effectively_learned_share": (
                    decisions_with_comparison / len(incidents) if incidents else float("nan")
                ),
                "actions_defaulted_share": (
                    (decisions_first_visit + decisions_single_known_action) / len(incidents)
                    if incidents
                    else float("nan")
                ),
                "median_visits_per_visited_state": float(np.median(state_visits[state_visits > 0]))
                if (state_visits > 0).any()
                else 0.0,
                "max_state_action_visits": int(visits.max()) if visits.size else 0,
                "states_with_single_sampled_action": int(
                    ((visits > 0).sum(axis=1) == 1).sum()
                ),
                "exercised_bins": ";".join(f"{name}={count}" for name, count in exercised.items()),
                "declared_bins": 6 * 3 * 3 * 4 * 3,
            }
        )
    return pd.DataFrame(rows)


def _feature_bins(incident) -> Optional[Tuple[int, int, int, int, int]]:
    """Feature bins of a recorded incident context, rebuilt from the ledger fields."""
    try:
        context = HealingContext(
            diagnosis_index=incident.diagnosis_index,
            excess_band_units=incident.excess_band_units,
            ready_count=incident.ready_count_at_trigger,
            current_policy=int(incident.settings_before.policy),
            relative_band=(
                incident.band_at_trigger / incident.reference_at_trigger
                if incident.reference_at_trigger > 0.0
                else 0.0
            ),
            remaining_work_ratio=(
                incident.visible_remaining_work_at_trigger
                / max(incident.reference_at_trigger, 1e-9)
            ),
        )
        return HealingContextEncoder(HealingConfig()).feature_bins(context)
    except (KeyError, ValidationError):
        return None


def _q_seed_variability(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-seed performance of the learned arms on the evaluated split.

    Training variability is a property of the learning method, so it is reported separately
    from the arm means (which pool the seeds): same workloads, five independent policies.
    """
    learned = metrics[metrics["arm"].isin(("q_recovery", "q_recovery_from_sjf"))]
    if learned.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (arm_name, seed), subset in learned.groupby(["arm", "training_seed"], sort=True):
        rows.append(
            {
                "arm": arm_name,
                "split": str(subset["split"].iloc[0]),
                "training_seed": int(seed),
                "traces": int(len(subset)),
                "avg_waiting_time": float(subset["avg_waiting_time"].astype(float).mean()),
                "p95_waiting_time": float(subset["p95_waiting_time"].astype(float).mean()),
                "interventions": float(subset["interventions"].astype(float).mean()),
                "accepted": float(subset["accepted"].astype(float).mean()),
                "rejected": float(subset["rejected"].astype(float).mean()),
                "inconclusive": float(subset["inconclusive"].astype(float).mean()),
                "fallback_rate": float(subset["fallback_triggered"].astype(bool).mean()),
            }
        )
    frame = pd.DataFrame(rows)
    for arm_name, subset in frame.groupby("arm", sort=True):
        spread = float(subset["avg_waiting_time"].std(ddof=1)) if len(subset) > 1 else float("nan")
        frame.loc[subset.index, "seed_spread_sd"] = spread
    return frame


def _coverage_curve_rows(models: Sequence[TrainedHealingModel]) -> pd.DataFrame:
    """Coverage as a function of training episodes, for the "more episodes" question.

    The learned agents record their visit counts at each checkpoint below during training;
    the curve shows whether extra episodes still buy *useful* coverage (new states and
    state-action pairs) or only re-visit contexts that were already represented.
    """
    rows: List[Dict[str, Any]] = []
    for model in models:
        for checkpoint in model.coverage_curve:
            rows.append({"training_seed": model.training_seed, **checkpoint})
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
        shows_rollback = 0 if (int(row["rejected"]) + int(row["inconclusive"])) > 0 else 1
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
    *,
    detection_scenarios: Optional[List[Dict[str, Any]]] = None,
    coverage: Optional[pd.DataFrame] = None,
    coverage_curve: Optional[pd.DataFrame] = None,
    q_seed_variability: Optional[pd.DataFrame] = None,
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
    lines += [
        "",
        "## Detection and recovery summary",
        "",
        "Verdicts apply to interventions only: *accepted* is a verified change, *rejected* is a",
        "change whose window closed no part of the trigger gap (a confirmed failure),",
        "*inconclusive* is a change the window could neither verify nor condemn, and *held*",
        "counts incidents that changed nothing.",
        "",
        "| Arm | Episodes | Incidents | Interventions | Accepted | Rejected | Inconclusive | Unverified share | Inconclusive share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detection.itertuples():
        lines.append(
            f"| {row.arm} | {row.episodes} | {row.incidents} | {row.interventions} | {row.accepted} | "
            f"{row.rejected} | {row.inconclusive} | {row.unverified_share_of_interventions:.3f} | "
            f"{row.inconclusive_share_of_interventions:.3f} |"
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
            f"| raw load closure p50 / p95 | {thresholds.null_p50:.3f} / {thresholds.null_p95:.3f} |",
            f"| unconditional `accept_closure_factor` (declared) | "
            f"{thresholds.declared_accept_closure_factor} |",
            f"| false-accept rate of the unconditional rule on the control | "
            f"{thresholds.declared_false_accept_rate:.3f} |",
            f"| phase baseline R^2 / residual sd | {thresholds.baseline_r2:.3f} / "
            f"{thresholds.residual_sd:.3f} |",
            f"| residual p05 / p95 | {thresholds.residual_p05:+.3f} / "
            f"{thresholds.residual_p95:+.3f} |",
            f"| derived accept / reject residual factor | "
            f"{thresholds.derived_accept_residual_factor:+.2f} / "
            f"{thresholds.derived_reject_residual_factor:+.2f} |",
            f"| declared accept / reject residual factor | "
            f"{thresholds.declared_accept_residual_factor:+.2f} / "
            f"{thresholds.declared_reject_residual_factor:+.2f} |",
            f"| false-accept / false-reject rate of the declared rule on the control | "
            f"{thresholds.false_accept_residual_rate:.3f} / "
            f"{thresholds.false_reject_residual_rate:.3f} |",
            f"| out-of-sample (validation control) false-accept / false-reject | "
            f"{thresholds.validation_false_accept_rate:.3f} / "
            f"{thresholds.validation_false_reject_rate:.3f} |",
            "",
            "Congestion-gate sweep (what each `min_load_factor` removes from the control):",
            "",
            "| min_load_factor | incidents per trace |",
            "|---|---:|",
        ]
        for row in thresholds.gate_sweep:
            lines.append(f"| {row['min_load_factor']:.2f} | {row['incidents_per_trace']:.3f} |")
    if not audit.empty:
        lines += ["", "## Counterfactual audit of interventions (matched replay)", "",
                  "Every audited intervention is re-run with the *same controller and the same workload*,",
                  "with that one incident forced to hold (`healing.controller.SelfHealingController`,",
                  "``forced_holds``). `matched_gain = hold_total_waiting - actual_total_waiting`: the total",
                  "waiting time the matched counterfactual accumulated minus the realized run's. Because the",
                  "counterfactual keeps healing after the forced hold, the difference is the marginal value of",
                  "that single decision, not of shutting the loop down (the older `hold_forever_gain` column",
                  "answers that second question and is kept for continuity).", "",
                  "| Arm | Audited interventions | Mean matched gain | Helped | Harmful | False recoveries | "
                  "Verified and helped | Rejected but helpful | Verified but harmful |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm_name, subset in audit.groupby("arm", sort=False):
            # Counted per *intervention*; the per-episode never-heal counterfactual rows
            # (``verdict == "episode_level"``) are reported in their own table below so that
            # no rate in this table has an episode denominator.
            subset = subset[subset["verdict"].astype(str) != "episode_level"]
            gains = subset["matched_gain"].astype(float)
            verdicts = subset["verdict"].astype(str)
            helped = gains > 0
            harmful = gains < 0
            accepted = verdicts == "accepted"
            rejected = verdicts == "rejected"
            lines.append(
                f"| {arm_name} | {len(gains)} | {gains.mean():+.3f} | {int(helped.sum())} | "
                f"{int(harmful.sum())} | {int((~helped).sum())} | {int((accepted & helped).sum())} | "
                f"{int((rejected & helped).sum())} | {int((accepted & harmful).sum())} |"
            )
        episode_view = audit[audit["verdict"].astype(str) == "episode_level"]
        if not episode_view.empty:
            lines += [
                "",
                "The same comparison for the second counterfactual, one row per *episode*: the recorded",
                "configuration prefix with no further healing after that episode's first incident",
                "(`verdict = episode_level`, `divergence_decision_index = -1`). Its denominator is episodes,",
                "not interventions, which is why it is kept out of the table above.",
                "",
                "| Arm | Episodes | Mean matched gain | Helped | Harmful |",
                "|---|---:|---:|---:|---:|",
            ]
            for arm_name, subset in episode_view.groupby("arm", sort=False):
                gains = subset["matched_gain"].astype(float)
                lines.append(
                    f"| {arm_name} | {len(gains)} | {gains.mean():+.3f} | {int((gains > 0).sum())} | "
                    f"{int((gains < 0).sum())} |"
                )
    if detection_scenarios:
        lines += [
            "",
            "## Detector behaviour on controlled degradation scenarios (validation split)",
            "",
            "Injected scenarios change the workload, never the configuration, and the arm is the",
            "hold-only control, so every detection below is caused by the workload itself. `stable`",
            "traces are the false-trigger probe; `burst_shock` multiplies the CPU time of every",
            "process arriving after the injection instant, `arrival_storm` compresses the remaining",
            "arrivals towards it. `detection_share` is the recall against the injected step change,",
            "`incidents_per_trace` on stable traces is the false-trigger frequency, and the delay is",
            "measured from the injection instant. A detection is only counted when it happens at or",
            "after that instant.",
            "",
            "| Scenario / family | Traces | Detected | Detection share | Incidents per stable trace | Median delay (time units) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in detection_scenarios:
            lines.append(
                f"| {row['scenario']} | {row['traces']} | {row['traces_with_detection']} | "
                f"{row['detection_share']:.3f} | {row['incidents_per_trace']:.3f} | "
                f"{row['median_delay_time_units']:.1f} |"
            )
    if coverage is not None and not coverage.empty:
        total_states = int(coverage["n_states"].iloc[0])
        pairs = int(coverage["n_actions"].iloc[0]) * total_states
        lines += [
            "",
            "## Learned-controller coverage",
            "",
            f"Encoder: {total_states} states x {int(coverage['n_actions'].iloc[0])} actions = {pairs} state-action pairs.",
            "Visits are concentrated because the event supply is small: recovery events are rare by",
            "construction, so coverage is reported as a distribution rather than a percentage.",
            "",
            "| Seed | Visited states | Visited pairs | Pairs coverage | Incidents | Decisions in already-sampled states | Median visits/state | Max visits/pair |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in coverage.itertuples():
            lines.append(
                f"| {row.training_seed} | {row.visited_states} | {row.visited_state_action_pairs} | "
                f"{row.state_action_coverage:.4f} | {row.training_incidents} | "
                f"{row.actions_effectively_learned_share:.3f} | {row.median_visits_per_visited_state:.1f} | "
                f"{row.max_state_action_visits} |"
            )
        if coverage_curve is not None and not coverage_curve.empty:
            lines += [
                "",
                "Coverage against training length (does more training still buy new states?):",
                "",
                "| Seed | Episodes | Visited states | Visited pairs | Incidents |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in coverage_curve.itertuples():
                lines.append(
                    f"| {row.training_seed} | {row.episodes} | {row.visited_states} | "
                    f"{row.visited_state_action_pairs} | {row.incidents} |"
                )
    if q_seed_variability is not None and not q_seed_variability.empty:
        lines += [
            "",
            "## Training variability of the learned arms",
            "",
            "| Arm | Training seed | Traces | Mean waiting | P95 waiting | Interventions | Accepted | Rejected | Inconclusive | Seed sd (mean waiting) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in q_seed_variability.itertuples():
            lines.append(
                f"| {row.arm} | {row.training_seed} | {row.traces} | {row.avg_waiting_time:.3f} | "
                f"{row.p95_waiting_time:.3f} | {row.interventions:.3f} | {row.accepted:.3f} | "
                f"{row.rejected:.3f} | {row.inconclusive:.3f} | {row.seed_spread_sd:.3f} |"
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
    scenarios: bool = True,
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
        scenarios: Whether to run the controlled degradation scenarios of
            :mod:`experiments.healing_detection` on the validation split.
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
    from experiments.healing_thresholds import (
        derivation_rows,
        derive_thresholds,
        verification_model_comparison,
    )

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
    if progress:
        print("Comparing candidate phase baselines on the control and the audit ...", flush=True)
    verification_models = verification_model_comparison(
        config,
        control=control_incidents,
        audit=audit_frame,
        incidents=incidents_by_split.get("validation", pd.DataFrame()),
    )
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
    coverage_curve = _coverage_curve_rows(models)
    q_seed_variability = _q_seed_variability(evaluation_metrics)

    scenario_frame = pd.DataFrame()
    scenario_summary: List[Dict[str, Any]] = []
    if scenarios:
        if progress:
            print("Running controlled degradation scenarios on the validation split ...", flush=True)
        from experiments.healing_detection import run_detection_scenarios, scenario_rows

        scenario_frame = pd.DataFrame(
            row.as_row()
            for row in run_detection_scenarios(
                build_healing_workloads("validation", config, generator), config=config
            )
        )
        scenario_summary = scenario_rows(scenario_frame)

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
        for column in (
            "incidents",
            "interventions",
            "accepted",
            "rejected",
            "inconclusive",
            "unverified",
            "held",
            "confirmed_failures",
        ):
            entry[column] = float(subset[column].astype(float).mean())
        entry["fallback_rate"] = float(subset["fallback_triggered"].astype(bool).mean())
        means.append(entry)

    audit_summary: List[Dict[str, Any]] = []
    if not audit_frame.empty:
        for arm_name, subset in audit_frame[
            audit_frame["split"] == "final_test"
        ].groupby("arm", sort=False):
            # The audit frame mixes two populations that must never share a denominator:
            # one row per *intervention* (the causal measurement of that decision) and one
            # row per *episode* describing the counterfactual of never healing again from
            # that episode's first incident (``verdict == "episode_level"``). Every rate
            # below is a per-intervention quantity, so it is computed on the intervention
            # rows only; the episode-level counterfactual is reported separately.
            episode_rows = subset[subset["verdict"].astype(str) == "episode_level"]
            subset = subset[subset["verdict"].astype(str) != "episode_level"]
            gains = subset["matched_gain"].astype(float)
            legacy = subset["hold_forever_gain"].astype(float)
            verdicts = subset["verdict"].astype(str)
            rejected = verdicts == "rejected"
            accepted = verdicts == "accepted"
            inconclusive = verdicts == "inconclusive"
            unverified = ~accepted
            helped = gains > 0
            harmful = gains < 0
            tied = gains == 0
            audit_summary.append(
                {
                    "arm": arm_name,
                    "audited_interventions": int(len(gains)),
                    "mean_matched_gain": float(gains.mean()),
                    "median_matched_gain": float(gains.median()),
                    "mean_hold_forever_gain": float(legacy.mean()),
                    "helped": int(helped.sum()),
                    "harmful": int(harmful.sum()),
                    "unchanged": int(tied.sum()),
                    # A false recovery is an intervention that did not help the trace (the
                    # honest, counterfactual definition; the in-run verdict is not evidence).
                    # Interventions with exactly zero effect are counted as not helping.
                    "false_recoveries": int((~helped).sum()),
                    "false_recovery_rate": float((~helped).mean()),
                    # What the in-run verification did with the interventions, measured
                    # against the counterfactual ground truth.
                    "accepted_and_helped": int((accepted & helped).sum()),
                    "accepted_but_harmful": int((accepted & harmful).sum()),
                    "inconclusive_but_helped": int((inconclusive & helped).sum()),
                    "rejected_but_helped": int((rejected & helped).sum()),
                    "rejected_and_harmful": int((rejected & harmful).sum()),
                    "in_run_accept_rate": float(accepted.mean()),
                    "in_run_rejection_rate": float(rejected.mean()),
                    "in_run_inconclusive_rate": float(inconclusive.mean()),
                    # Share of the genuinely helpful interventions the verifier did not
                    # certify. This is the measured size of the verification mismatch.
                    "unverified_share_of_helpful": (
                        float((unverified & helped).sum() / helped.sum())
                        if helped.any()
                        else float("nan")
                    ),
                    # Share of the genuinely harmful interventions that were rolled back.
                    "harmful_rollback_rate": (
                        float((rejected & harmful).sum() / harmful.sum())
                        if harmful.any()
                        else float("nan")
                    ),
                    "confirmed_failure_share": float(rejected.mean()),
                    # The second counterfactual, kept visibly separate: 0/1 row per episode
                    # answering "what if the loop had never healed again after the first
                    # incident of this episode?".
                    "episode_level_n": int(len(episode_rows)),
                    "episode_level_mean_matched_gain": (
                        float(episode_rows["matched_gain"].astype(float).mean())
                        if len(episode_rows)
                        else float("nan")
                    ),
                }
            )

    discrimination: Dict[str, Any] = {}
    if not audit_frame.empty:
        held = audit_frame[
            (audit_frame["split"] == "final_test")
            & (audit_frame["verdict"].astype(str) != "episode_level")
        ]
        if len(held):
            gains = held["matched_gain"].astype(float)
            helped = gains > 0
            harmful = gains < 0
            verdicts = held["verdict"].astype(str)
            accepted = verdicts == "accepted"
            rejected = verdicts == "rejected"
            residual = held["closure_residual"].astype(float)
            closure = held["closure_fraction"].astype(float)
            null_p95 = float(threshold_derivation.null_p95)
            accept_factor = float(threshold_derivation.declared_accept_residual_factor)
            certifiable = helped & (closure >= null_p95)
            discrimination = {
                "split": "final_test",
                "n_interventions": int(len(held)),
                "helped": int(helped.sum()),
                "harmful": int(harmful.sum()),
                "zero_effect": int((gains == 0).sum()),
                # How well a window statistic can tell a helpful intervention from an
                # unhelpful one: 0.5 is chance, 1.0 is perfect separation.
                "rank_auc_closure_residual_to_helped": _rank_auc(residual, helped),
                "rank_auc_closure_fraction_to_helped": _rank_auc(closure, helped),
                # Enrichment of the verdicts, with the denominators spelled out: each rate
                # is a share *within* one verdict, against the pooled base rate.
                "base_helped_share": float(helped.mean()),
                "accepted_n": int(accepted.sum()),
                "accepted_helped_share": (
                    float((accepted & helped).sum() / accepted.sum())
                    if accepted.any()
                    else float("nan")
                ),
                "rejected_n": int(rejected.sum()),
                "rejected_harmful_share": (
                    float((rejected & harmful).sum() / rejected.sum())
                    if rejected.any()
                    else float("nan")
                ),
                "base_harmful_share": float(harmful.mean()),
                # The certification ceiling: the share of genuinely helpful interventions
                # whose window would clear the declared null boundary (and the declared
                # residual boundary) at all.
                "certifiable_share_of_helpful_at_null_p95": (
                    float(certifiable.sum() / helped.sum()) if helped.any() else float("nan")
                ),
                "certifiable_helpful_at_null_p95": int(certifiable.sum()),
                "certifiable_mean_gain_at_null_p95": (
                    float(gains[certifiable].mean()) if certifiable.any() else float("nan")
                ),
                "certifiable_share_of_helpful_at_accept_residual_factor": (
                    float((helped & (residual >= accept_factor)).sum() / helped.sum())
                    if helped.any()
                    else float("nan")
                ),
                "declared_null_p95": null_p95,
                "declared_accept_residual_factor": accept_factor,
            }

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
        "verification_discrimination": discrimination,
        "coverage": coverage.to_dict(orient="records"),
        "coverage_curve": coverage_curve.to_dict(orient="records"),
        "q_seed_variability": q_seed_variability.to_dict(orient="records"),
        "detection_scenarios": scenario_summary,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    summary = _json_safe(summary)
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
        ("coverage_curve", coverage_curve),
        ("q_seed_variability", q_seed_variability),
        ("detection_scenarios", scenario_frame),
        ("action_probe", probe_frame),
        ("training_episodes", training_episodes),
        ("threshold_derivation", pd.DataFrame(derivation_rows(threshold_derivation))),
        ("verification_models", verification_models),
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
            detection_scenarios=scenario_summary,
            coverage=coverage,
            coverage_curve=coverage_curve,
            q_seed_variability=q_seed_variability,
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
        detection_scenarios=scenario_frame,
        action_probe=probe_frame,
        coverage=coverage,
        coverage_curve=coverage_curve,
        q_seed_variability=q_seed_variability,
        summary=summary,
        paths=paths,
    )