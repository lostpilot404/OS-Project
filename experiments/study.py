"""End-to-end study: training, held-out evaluation, robustness, ablations, artifacts.

The study compares four Q-learning arms (original selector, risk-aware selector and the
two controlled ablations) with three pre-learning-free baselines on frozen splits.  For
every selection it records the decision, computes the four-policy counterfactuals
strictly afterwards, scores both objectives, computes hindsight regret and writes a
structured explanation.  All reported numbers are generated from these records; the
report renderer never invents values.

Determinism: every workload, agent and baseline draw is seeded through
:func:`config.derive_seed`, so a rerun with the same configuration reproduces the study
byte-for-byte (verified by tests).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    ACTION_NAMES,
    ExperimentConfig,
    QLearningConfig,
    RewardConfig,
    RiskObjectiveConfig,
    derive_seed,
)
from errors import ValidationError
from evaluation.counterfactual import CounterfactualReport, run_counterfactuals
from experiments.baselines import best_fixed_action, fingerprint_rule_action, random_action
from experiments.study_config import (
    ARM_LABELS,
    BASELINE_ARMS,
    Q_ARMS,
    STUDY_NAME,
    StudyDesign,
    build_study_config,
    build_study_design,
)
from rl.explain import build_explanation, explain_evaluation_record
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import OBJECTIVE_NAMES, OBJECTIVE_RISK_AWARE
from rl.q_learning import QLearningAgent
from rl.selector import WorkloadPolicySelector
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = ["StudyArtifacts", "run_study"]

_WORKLOAD_STREAM_TAG = 0  # same derivation as experiments.train
_SPLIT_TAGS = {"validation": 11, "test": 12}
_RANDOM_STREAM_TAG = 21
_BOOTSTRAP_SEED = 424242
_BOOTSTRAP_SAMPLES = 2000

#: Metrics copied into aggregate tables.
_SUMMARY_METRICS = (
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "throughput",
    "context_switches",
)


@dataclass(frozen=True)
class StudyArtifacts:
    """Summary and written artifact paths of one study run."""

    summary: Dict[str, object]
    paths: Dict[str, object]


@dataclass(frozen=True)
class _SplitWorkload:
    family: str
    repetition: int
    workload: Workload

    @property
    def fingerprint(self) -> str:
        return self.workload.fingerprint


@dataclass
class _TrainingBundle:
    selectors: Dict[str, Dict[int, WorkloadPolicySelector]]
    rows: List[Dict[str, object]]
    fingerprints: set
    mean_training_costs: Dict[int, float]


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def run_study(
    config: ExperimentConfig | None = None,
    design: StudyDesign | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    make_figures: bool = True,
    explain_training: bool = True,
) -> StudyArtifacts:
    """Run the full study and write every artifact under ``results_dir``."""
    config = config or build_study_config()
    design = design or build_study_design()
    config.validate()
    results_path = Path(results_dir or "results/study")
    figures_path = Path(figures_dir or "figures/study")
    generator = WorkloadGenerator(config.families)
    policies = [cls(config.scheduler) for cls in POLICY_CLASSES]

    training = _train_all_arms(config, design, generator, policies, results_path, explain_training)
    fixed_action = best_fixed_action(training.mean_training_costs)

    splits = {
        "validation": _build_split(config, generator, design.validation_seed, design.validation_repetitions),
        "test": _build_split(config, generator, design.test_seed, design.test_repetitions),
    }
    _check_split_separation(training.fingerprints, splits)

    decision_rows, counterfactual_rows = _evaluate_all_arms(
        config, design, generator, policies, training.selectors, fixed_action, splits, results_path
    )

    decisions = pd.DataFrame(decision_rows)
    counterfactuals = pd.DataFrame(counterfactual_rows)
    summary = _build_summary(config, design, training, decisions, counterfactuals, fixed_action)
    paths = _write_artifacts(
        config, design, summary, training.rows, decisions, counterfactuals, results_path
    )
    if make_figures:
        from visualization.study_plots import create_study_figures

        paths["figures"] = create_study_figures(
            decisions=decisions, training_rows=pd.DataFrame(training.rows), figures_dir=figures_path
        )
    return StudyArtifacts(summary=summary, paths=paths)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


def _make_selector(arm: str, config: ExperimentConfig, seed: int) -> WorkloadPolicySelector:
    encoder_kind, objective = Q_ARMS[arm]
    if encoder_kind == "legacy":
        encoder = StateEncoder(config.state)
    else:
        from config import FingerprintStateConfig

        encoder = FingerprintStateEncoder(FingerprintStateConfig())
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        QLearningConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_per_episode=0.995),
        seed=seed,
    )
    return WorkloadPolicySelector(
        encoder,
        agent,
        objective=objective,
        reward_config=RewardConfig(),
        risk_config=RiskObjectiveConfig(),
    )


def _training_workload(
    generator: WorkloadGenerator,
    config: ExperimentConfig,
    seed: int,
    episode: int,
) -> Workload:
    cycle = config.training.family_cycle
    family = cycle[episode % len(cycle)]
    workload_seed = derive_seed(seed, _WORKLOAD_STREAM_TAG, episode)
    return generator.generate(family, workload_seed)


def _train_all_arms(
    config: ExperimentConfig,
    design: StudyDesign,
    generator: WorkloadGenerator,
    policies: Sequence,
    results_path: Path,
    explain_training: bool,
) -> _TrainingBundle:
    selectors: Dict[str, Dict[int, WorkloadPolicySelector]] = {}
    rows: List[Dict[str, object]] = []
    fingerprints: set = set()
    # Accumulate primary-objective training costs per action (all arms share the
    # workload stream; one arm's episodes are the complete training distribution).
    cost_sums = {action: 0.0 for action in range(len(ACTION_NAMES))}
    cost_count = 0

    explanation_dir = results_path / "explanations"
    explanation_dir.mkdir(parents=True, exist_ok=True)

    for arm in Q_ARMS:
        selectors[arm] = {}
        for seed in design.training_seeds:
            selector = _make_selector(arm, config, seed)
            selectors[arm][seed] = selector
            stream = (
                explanation_dir / f"train_{arm}_seed{seed}.jsonl.gz"
                if explain_training
                else None
            )
            with _open_jsonl(stream) as sink:
                for episode in range(design.training_episodes):
                    workload = _training_workload(generator, config, seed, episode)
                    record = selector.train_episode(workload, episode, policies)
                    fingerprints.add(workload.fingerprint)
                    row = record.as_row()
                    row.update({"arm": arm, "seed": seed, "episode": episode})
                    rows.append(row)
                    if sink is not None:
                        sink.write(
                            json.dumps(
                                explain_evaluation_record(record).as_dict(), sort_keys=True
                            )
                            + "\n"
                        )
                    if arm == "ql_risk":
                        cost_count += 1
                        for action in range(len(ACTION_NAMES)):
                            cost_sums[action] += record.counterfactuals.cost(
                                design.primary_objective, action
                            )
    mean_costs = {action: cost_sums[action] / max(1, cost_count) for action in cost_sums}
    return _TrainingBundle(
        selectors=selectors,
        rows=rows,
        fingerprints=fingerprints,
        mean_training_costs=mean_costs,
    )


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


def _build_split(
    config: ExperimentConfig,
    generator: WorkloadGenerator,
    split_seed: int,
    repetitions: int,
) -> List[_SplitWorkload]:
    items: List[_SplitWorkload] = []
    for family_index, family in enumerate(config.families):
        for repetition in range(repetitions):
            seed = derive_seed(split_seed, _WORKLOAD_STREAM_TAG, family_index, repetition)
            items.append(
                _SplitWorkload(
                    family=family.name,
                    repetition=repetition,
                    workload=generator.generate(family.name, seed),
                )
            )
    return items


def _check_split_separation(training_fingerprints: set, splits: Mapping[str, List[_SplitWorkload]]) -> None:
    seen = set(training_fingerprints)
    for name, items in splits.items():
        fingerprints = {item.fingerprint for item in items}
        if len(fingerprints) != len(items):
            raise ValidationError(f"the {name} split contains duplicate workloads")
        overlap = fingerprints & seen
        if overlap:
            raise ValidationError(
                f"{name} workloads overlap earlier splits/training: {sorted(overlap)[:3]}"
            )
        seen |= fingerprints


def _evaluate_all_arms(
    config: ExperimentConfig,
    design: StudyDesign,
    generator: WorkloadGenerator,
    policies: Sequence,
    selectors: Mapping[str, Mapping[int, WorkloadPolicySelector]],
    fixed_action: int,
    splits: Mapping[str, List[_SplitWorkload]],
    results_path: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    decision_rows: List[Dict[str, object]] = []
    counterfactual_rows: List[Dict[str, object]] = []
    explanation_dir = results_path / "explanations"
    explanation_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        reports: Dict[str, CounterfactualReport] = {}
        for item in items:
            report = run_counterfactuals(item.workload, policies)
            reports[item.fingerprint] = report
            for row in report.as_rows():
                counterfactual_rows.append(
                    {
                        "split": split_name,
                        "family": item.family,
                        "repetition": item.repetition,
                        **row,
                    }
                )

        # Q-learning arms
        for arm, per_seed in selectors.items():
            for seed, selector in per_seed.items():
                stream = explanation_dir / f"{split_name}_{arm}_seed{seed}.jsonl.gz"
                with _open_jsonl(stream) as sink:
                    for item in items:
                        record = selector.evaluate(item.workload, policies)
                        row = _decision_row(arm, seed, split_name, item, record.decision, record.counterfactuals, design.primary_objective)
                        row.update(record.regret.as_dict())
                        for objective, regret in record.regret_by_objective.items():
                            if objective != design.primary_objective:
                                row[f"alt_{objective}_regret"] = regret.regret
                                row[f"alt_{objective}_is_hindsight_best"] = regret.is_hindsight_best
                        decision_rows.append(row)
                        sink.write(
                            json.dumps(
                                explain_evaluation_record(record).as_dict(), sort_keys=True
                            )
                            + "\n"
                        )

        # Non-learning baselines
        for item in items:
            report = reports[item.fingerprint]
            fingerprint = fingerprint_workload_of(item.workload)
            best_fixed_selection = {
                "policy_name": ACTION_NAMES[fixed_action],
                "action": fixed_action,
                "selection_rule": "best_fixed_training_mean",
                "selection_kind": "baseline",
                "selection_evidence": {"source": "lowest mean training cost"},
            }
            rule_action, branch, rule_evidence = fingerprint_rule_action(fingerprint)
            rule_selection = {
                "policy_name": ACTION_NAMES[rule_action],
                "action": rule_action,
                "selection_rule": "fingerprint_rule",
                "selection_kind": "baseline",
                "selection_evidence": {"branch": branch, **rule_evidence},
            }
            deterministic = (
                ("best_fixed", -1, best_fixed_selection),
                ("fingerprint_rule", -1, rule_selection),
            )
            for arm, seed, selection in deterministic:
                decision_rows.append(
                    _baseline_row(arm, seed, split_name, item, selection, report, design.primary_objective)
                )
                _write_explanation(
                    explanation_dir / f"{split_name}_{arm}_seed-1.jsonl.gz",
                    item,
                    selection,
                    report,
                    design.primary_objective,
                )
            for seed in design.random_seeds:
                rng = np.random.default_rng(
                    derive_seed(seed, _RANDOM_STREAM_TAG, _SPLIT_TAGS[split_name], item.repetition)
                )
                action, evidence = random_action(rng)
                selection = {
                    "policy_name": ACTION_NAMES[action],
                    "action": action,
                    "selection_rule": "random_uniform",
                    "selection_kind": "baseline",
                    "selection_evidence": evidence,
                }
                decision_rows.append(
                    _baseline_row("random", seed, split_name, item, selection, report, design.primary_objective)
                )
                _write_explanation(
                    explanation_dir / f"{split_name}_random_seed{seed}.jsonl.gz",
                    item,
                    selection,
                    report,
                    design.primary_objective,
                )
    return decision_rows, counterfactual_rows


def fingerprint_workload_of(workload: Workload):
    from workload.fingerprint import fingerprint_workload

    return fingerprint_workload(workload)


def _decision_row(
    arm: str,
    seed: int,
    split_name: str,
    item: _SplitWorkload,
    decision,
    report: CounterfactualReport,
    primary_objective: str,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "arm": arm,
        "arm_label": ARM_LABELS[arm],
        "model_seed": seed,
        "split": split_name,
        "family": item.family,
        "repetition": item.repetition,
    }
    row.update(decision.as_row())
    row.update(_objective_columns(report, decision.action))
    return row


def _baseline_row(
    arm: str,
    seed: int,
    split_name: str,
    item: _SplitWorkload,
    selection: Mapping[str, object],
    report: CounterfactualReport,
    primary_objective: str,
) -> Dict[str, object]:
    action = int(selection["action"])
    fingerprint = fingerprint_workload_of(item.workload)
    outcome = report.outcomes[action]
    row: Dict[str, object] = {
        "arm": arm,
        "arm_label": ARM_LABELS[arm],
        "model_seed": seed,
        "split": split_name,
        "family": item.family,
        "repetition": item.repetition,
        "workload_name": item.workload.name,
        "workload_fingerprint": item.fingerprint,
        "state_index": -1,
        "state_encoder": "none",
        "state_seen_in_training": False,
        "action": action,
        "policy_name": selection["policy_name"],
        "greedy_action": action,
        "greedy_policy_name": selection["policy_name"],
        "explored": False,
        "epsilon": 0.0,
        "objective": primary_objective,
        "selection_rule": selection["selection_rule"],
        "selection_evidence": json.dumps(selection.get("selection_evidence", {}), sort_keys=True),
    }
    row.update({f"feature_{name}": value for name, value in fingerprint.as_dict().items()})
    row.update({f"selected_{key}": value for key, value in outcome.as_dict().items()})
    row.update(_objective_columns(report, action))
    regret = report.regret(primary_objective, action)
    row.update(regret.as_dict())
    for objective in OBJECTIVE_NAMES:
        if objective != primary_objective:
            alt = report.regret(objective, action)
            row[f"alt_{objective}_regret"] = alt.regret
            row[f"alt_{objective}_is_hindsight_best"] = alt.is_hindsight_best
    return row


def _objective_columns(report: CounterfactualReport, action: int) -> Dict[str, object]:
    columns: Dict[str, object] = {}
    for objective in OBJECTIVE_NAMES:
        breakdown = report.breakdowns[objective][action]
        prefix = "risk_" if objective == OBJECTIVE_RISK_AWARE else "orig_"
        columns[f"{prefix}objective_cost"] = breakdown.cost
        columns[f"{prefix}objective_reward"] = breakdown.reward
        for name, term in zip(breakdown.metric_names, breakdown.terms):
            columns[f"{prefix}term_{name}"] = term
    return columns


@contextlib.contextmanager
def _gzip_text(path: Path, append: bool):
    """Yield a gzip text writer with a fixed mtime and no embedded filename.

    Byte-level reproducibility of the explanation logs requires deterministic gzip
    headers; ``gzip.open`` records the current time by default.
    """
    import io

    raw = path.open("ab" if append else "wb")
    try:
        stream = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        writer = io.TextIOWrapper(stream, encoding="utf-8")
        try:
            yield writer
        finally:
            writer.close()
    finally:
        raw.close()


class _NullSink:
    """Write target that discards everything (used when explanations are disabled)."""

    def write(self, text: str) -> None:
        return None


@contextlib.contextmanager
def _open_jsonl(path: Optional[Path]):
    if path is None:
        yield _NullSink()
        return
    with _gzip_text(path, append=False) as handle:
        yield handle


def _write_explanation(path: Path, item, selection, report, objective: str) -> None:
    explanation = build_explanation(
        workload_name=item.workload.name,
        workload_fingerprint=item.fingerprint,
        fingerprint=fingerprint_workload_of(item.workload),
        selection=selection,
        counterfactuals=report,
        objective=objective,
        state_bins={},
        state_features=(),
    )
    with _gzip_text(path, append=True) as handle:
        handle.write(json.dumps(explanation.as_dict(), sort_keys=True) + "\n")


# --------------------------------------------------------------------------------------
# aggregation and reporting
# --------------------------------------------------------------------------------------


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_bootstrap_ci(differences: Sequence[float]) -> Tuple[float, float, float]:
    """Mean and percentile 95% bootstrap CI of paired per-workload differences."""
    if not differences:
        return 0.0, 0.0, 0.0
    data = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(data), size=(_BOOTSTRAP_SAMPLES, len(data)))
    means = data[indices].mean(axis=1)
    return (
        float(data.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _build_summary(
    config: ExperimentConfig,
    design: StudyDesign,
    training: _TrainingBundle,
    decisions: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    fixed_action: int,
) -> Dict[str, object]:
    primary = design.primary_objective
    cost_col = f"{_prefix(primary)}objective_cost"

    arm_summary = {}
    for arm, frame in decisions.groupby("arm"):
        per_seed = []
        for seed, seed_frame in frame.groupby("model_seed"):
            per_seed.append(float(seed_frame[cost_col].mean()))
        regret = frame["regret"]
        arm_summary[arm] = {
            "label": ARM_LABELS[arm],
            "rows": int(len(frame)),
            "mean_objective_cost": float(frame[cost_col].mean()),
            "sd_objective_cost": float(frame[cost_col].std(ddof=1)) if len(frame) > 1 else 0.0,
            "mean_original_reward": float(frame["orig_objective_reward"].mean()),
            "mean_regret": float(regret.mean()),
            "sd_regret": float(regret.std(ddof=1)) if len(frame) > 1 else 0.0,
            "hindsight_best_rate": float(frame["is_hindsight_best"].mean()),
            "poor_selection_rate": float((regret > 0).mean()),
            "severe_poor_rate": float((regret > design.poor_regret_threshold).mean()),
            "mean_cost_across_model_seeds": _mean(per_seed),
            "sd_cost_across_model_seeds": _sample_sd(per_seed),
            "per_model_seed_mean_cost": per_seed,
            "metrics": {
                metric: {
                    "mean": float(frame[f"selected_{metric}"].mean()),
                    "sd": float(frame[f"selected_{metric}"].std(ddof=1)) if len(frame) > 1 else 0.0,
                }
                for metric in _SUMMARY_METRICS
            },
            "selection_counts": {
                name: int((frame["policy_name"] == name).sum()) for name in ACTION_NAMES
            },
        }

    family_tables = {}
    for (arm, family), frame in decisions.groupby(["arm", "family"]):
        family_tables.setdefault(arm, {})[family] = {
            "rows": int(len(frame)),
            "mean_objective_cost": float(frame[cost_col].mean()),
            "mean_regret": float(frame["regret"].mean()),
            "hindsight_best_rate": float(frame["is_hindsight_best"].mean()),
            "poor_selection_rate": float((frame["regret"] > 0).mean()),
            "mean_original_reward": float(frame["orig_objective_reward"].mean()),
            "selection_counts": {
                name: int((frame["policy_name"] == name).sum()) for name in ACTION_NAMES
            },
        }

    id_frames = {
        arm: frame[frame["family"].isin(design.training_families)]
        for arm, frame in decisions.groupby("arm")
    }
    shift_tables = {}
    for arm in sorted(decisions["arm"].unique()):
        frame = decisions[decisions["arm"] == arm]
        id_frame = id_frames[arm]
        ood_frame = frame[~frame["family"].isin(design.training_families)]
        shift_tables[arm] = {
            "in_distribution_mean_cost": float(id_frame[cost_col].mean()),
            "in_distribution_mean_regret": float(id_frame["regret"].mean()),
            "out_of_distribution_mean_cost": float(ood_frame[cost_col].mean()),
            "out_of_distribution_mean_regret": float(ood_frame["regret"].mean()),
            "out_of_distribution_poor_rate": float((ood_frame["regret"] > 0).mean()),
        }

    paired = {}
    test = decisions[decisions["split"] == "test"]
    # Average over model seeds per workload so every pair is one difference per
    # unique workload (multi-seed arms never duplicate a pairing).
    per_workload = (
        test.groupby(["arm", "family", "repetition", "workload_fingerprint"], sort=True)[
            cost_col
        ]
        .mean()
        .reset_index()
    )
    reference = per_workload[per_workload["arm"] == "ql_risk"]
    for arm in sorted(per_workload["arm"].unique()):
        if arm == "ql_risk":
            continue
        other = per_workload[per_workload["arm"] == arm]
        merged = reference.merge(
            other,
            on=["family", "repetition", "workload_fingerprint"],
            suffixes=("_ref", "_other"),
        )
        if merged.empty:
            continue
        diffs = (merged[f"{cost_col}_ref"] - merged[f"{cost_col}_other"]).tolist()
        mean_diff, lo, hi = _paired_bootstrap_ci(diffs)
        paired[arm] = {
            "mean_cost_difference_ql_risk_minus_arm": mean_diff,
            "ci95_low": lo,
            "ci95_high": hi,
            "paired_workloads": len(diffs),
        }

    training_summary = {}
    for arm in Q_ARMS:
        arm_rows = [row for row in training.rows if row["arm"] == arm]
        by_seed = {}
        for seed in design.training_seeds:
            rewards = [
                float(row["objective_reward"])
                for row in arm_rows
                if row["seed"] == seed
            ]
            by_seed[seed] = {
                "episodes": len(rewards),
                "mean_reward_first_100": _mean(rewards[:100]),
                "mean_reward_last_100": _mean(rewards[-100:]),
                "mean_reward_all": _mean(rewards),
            }
        training_summary[arm] = {
            "per_seed": by_seed,
            "mean_reward_last_100_across_seeds": _mean(
                [by_seed[seed]["mean_reward_last_100"] for seed in design.training_seeds]
            ),
        }

    return {
        "study": design.primary_objective,
        "configuration": {
            "name": STUDY_NAME,
            "design": {
                "training_episodes": design.training_episodes,
                "training_seeds": list(design.training_seeds),
                "validation_seed": design.validation_seed,
                "validation_repetitions": design.validation_repetitions,
                "test_seed": design.test_seed,
                "test_repetitions": design.test_repetitions,
                "random_seeds": list(design.random_seeds),
                "primary_objective": design.primary_objective,
                "poor_regret_threshold": design.poor_regret_threshold,
                "training_families": list(design.training_families),
                "held_out_families": list(design.held_out_families),
                "q_arms": {arm: {"encoder": v[0], "objective": v[1]} for arm, v in Q_ARMS.items()},
                "baseline_arms": list(BASELINE_ARMS),
                "best_fixed_action": ACTION_NAMES[fixed_action],
                "risk_objective": {
                    "weights": {
                        "waiting": RiskObjectiveConfig().weight_waiting_time,
                        "response": RiskObjectiveConfig().weight_response_time,
                        "context_switches": RiskObjectiveConfig().weight_context_switches,
                        "tail_waiting": RiskObjectiveConfig().weight_tail_waiting,
                    },
                    "tail_quantile": RiskObjectiveConfig().tail_quantile,
                    "reference_clip": RiskObjectiveConfig().reference_clip,
                },
            },
            "software": software_snapshot(),
        },
        "training": training_summary,
        "test": {
            "workloads": int(
                decisions[decisions["split"] == "test"]["workload_fingerprint"].nunique()
            ),
            "arms": arm_summary,
            "by_family": family_tables,
            "distribution_shift": shift_tables,
            "paired_vs_ql_risk": paired,
        },
        "validation": {
            "workloads": int(
                decisions[decisions["split"] == "validation"]["workload_fingerprint"].nunique()
            ),
        },
    }


def _prefix(objective: str) -> str:
    return "risk_" if objective == OBJECTIVE_RISK_AWARE else "orig_"


def software_snapshot() -> Dict[str, str]:
    """Versions of the interpreter and the pinned packages actually used."""
    packages = {}
    for name in ("numpy", "pandas", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            packages[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **packages,
    }


def config_digest(config: ExperimentConfig) -> str:
    payload = json.dumps(
        {
            "families": [f.name for f in config.families],
            "scheduler": config.scheduler.__dict__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write_artifacts(
    config: ExperimentConfig,
    design: StudyDesign,
    summary: Dict[str, object],
    training_rows: Sequence[Mapping[str, object]],
    decisions: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    results_path: Path,
) -> Dict[str, object]:
    results_path.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {}

    def write_json(name: str, payload: object) -> None:
        path = results_path / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        paths[name] = path

    decisions_path = results_path / "decisions.csv"
    decisions.to_csv(decisions_path, index=False)
    paths["decisions"] = decisions_path

    counterfactuals_path = results_path / "counterfactuals.csv"
    counterfactuals.to_csv(counterfactuals_path, index=False)
    paths["counterfactuals"] = counterfactuals_path

    training_frame = pd.DataFrame(training_rows)
    training_path = results_path / "training_history.csv"
    training_frame.to_csv(training_path, index=False)
    paths["training_history"] = training_path

    robustness = _robustness_table(decisions, design)
    robustness_path = results_path / "robustness.csv"
    robustness.to_csv(robustness_path, index=False)
    paths["robustness"] = robustness_path

    write_json("summary.json", summary)
    write_json("design.json", summary["configuration"])

    report_path = results_path / "report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")
    paths["report"] = report_path

    examples_path = results_path / "explanations_examples.json"
    examples_path.write_text(
        json.dumps(_example_explanations(results_path), indent=2, sort_keys=True) + "\n"
    )
    paths["explanations_examples"] = examples_path
    paths["explanations_dir"] = results_path / "explanations"

    manifest = {
        "artifacts": {name: str(path) for name, path in paths.items()},
        "config_digest": config_digest(config),
        "decision_rows": int(len(decisions)),
        "counterfactual_rows": int(len(counterfactuals)),
        "training_rows": len(training_rows),
    }
    write_json("manifest.json", manifest)
    return paths


def _robustness_table(decisions: pd.DataFrame, design: StudyDesign) -> pd.DataFrame:
    cost_col = f"{_prefix(design.primary_objective)}objective_cost"
    rows = []
    grouped = decisions.groupby(["split", "arm", "model_seed", "family"], sort=True)
    for (split, arm, seed, family), frame in grouped:
        rows.append(
            {
                "split": split,
                "arm": arm,
                "model_seed": seed,
                "family": family,
                "workloads": int(len(frame)),
                "mean_objective_cost": float(frame[cost_col].mean()),
                "sd_objective_cost": float(frame[cost_col].std(ddof=1)) if len(frame) > 1 else 0.0,
                "mean_regret": float(frame["regret"].mean()),
                "sd_regret": float(frame["regret"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "poor_selection_rate": float((frame["regret"] > 0).mean()),
                "severe_poor_rate": float(
                    (frame["regret"] > design.poor_regret_threshold).mean()
                ),
                "hindsight_best_rate": float(frame["is_hindsight_best"].mean()),
                "mean_original_reward": float(frame["orig_objective_reward"].mean()),
                **{
                    f"mean_{metric}": float(frame[f"selected_{metric}"].mean())
                    for metric in _SUMMARY_METRICS
                },
            }
        )
    return pd.DataFrame(rows)


def _example_explanations(results_path: Path) -> List[Dict[str, object]]:
    """Pick three fully worked explanations by a fixed, documented rule.

    From the test-split explanations of ``ql_risk`` (lowest model seed): the first
    record in file order that (1) is hindsight best, (2) has positive regret, (3) comes
    from an out-of-distribution family with positive regret.  Only file order is used,
    so the selection never depends on metric values beyond these criteria.
    """
    path = results_path / "explanations" / "test_ql_risk_seed42.jsonl.gz"
    if not path.exists():
        return []
    wanted = {"hindsight_best": None, "regret": None, "ood_regret": None}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            regret = float(record["regret"]["regret"])
            best = bool(record["regret"]["is_hindsight_best"])
            family = record["workload_name"]
            if wanted["hindsight_best"] is None and best:
                wanted["hindsight_best"] = record
            if wanted["regret"] is None and regret > 0:
                wanted["regret"] = record
            if (
                wanted["ood_regret"] is None
                and regret > 0
                and family in ("poisson_arrivals", "heavy_tail_stress")
            ):
                wanted["ood_regret"] = record
            if all(value is not None for value in wanted.values()):
                break
    return [
        {"criterion": name, "explanation": value}
        for name, value in wanted.items()
        if value is not None
    ]


def render_report(summary: Mapping[str, object]) -> str:
    """Render the generated study report from ``summary.json`` values only."""
    config = summary["configuration"]["design"]
    test = summary["test"]
    lines = [
        "# Generated study results",
        "",
        "> Generated from `summary.json` by `experiments.study.render_report`; "
        "do not edit result values by hand.",
        "",
        "## Scope",
        "",
        "This is an **offline policy-selection and evaluation system**: one policy per "
        "complete workload, selected before execution from the declared process table. "
        "It is not a runtime OS scheduler and never switches policies during a schedule.",
        "",
        f"- Primary declared objective: `{config['primary_objective']}` "
        f"(weights {config['risk_objective']['weights']}, "
        f"tail quantile {config['risk_objective']['tail_quantile']}).",
        f"- Training: {len(config['training_seeds'])} seeds "
        f"{config['training_seeds']} x {config['training_episodes']} episodes.",
        f"- Held-out test split: {test['workloads']} workloads "
        f"({config['test_repetitions']} per family x 8 families), master seed "
        f"{config['test_seed']}; includes out-of-distribution families "
        f"{config['held_out_families']} that never appear in training.",
        f"- Best-fixed baseline policy (training workloads only): "
        f"{config['best_fixed_action']}.",
        "",
        "## Test-split results by arm",
        "",
        "Cost is the declared objective (lower is better); regret is the hindsight gap "
        "to the best of the four fixed policies on the same workload. "
        "`poor` = regret > 0, `severe` = regret > "
        f"{config['poor_regret_threshold']}.",
        "",
        "| Arm | Mean cost | Mean regret | Hindsight-best % | Poor % | Severe % | "
        "Mean wait | Mean response | Switches/proc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in test["arms"].items():
        metrics = values["metrics"]
        lines.append(
            f"| {values['label']} | {values['mean_objective_cost']:.4f} | "
            f"{values['mean_regret']:.4f} | {100 * values['hindsight_best_rate']:.1f} | "
            f"{100 * values['poor_selection_rate']:.1f} | "
            f"{100 * values['severe_poor_rate']:.1f} | "
            f"{metrics['avg_waiting_time']['mean']:.3f} | "
            f"{metrics['avg_response_time']['mean']:.3f} | "
            f"{metrics['context_switches']['mean']:.2f} |"
        )
    lines += [
        "",
        "## Per-family mean declared-objective cost (test split)",
        "",
    ]
    families = sorted({f for arm in test["by_family"].values() for f in arm})
    header = "| Arm | " + " | ".join(families) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(families))
    for arm, table in test["by_family"].items():
        values = [
            f"{table[family]['mean_objective_cost']:.4f}" if family in table else "-"
            for family in families
        ]
        lines.append(f"| {ARM_LABELS[arm]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Distribution shift (test split)",
        "",
        "| Arm | In-dist mean cost | OOD mean cost | In-dist regret | OOD regret | "
        "OOD poor % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, values in test["distribution_shift"].items():
        lines.append(
            f"| {ARM_LABELS[arm]} | {values['in_distribution_mean_cost']:.4f} | "
            f"{values['out_of_distribution_mean_cost']:.4f} | "
            f"{values['in_distribution_mean_regret']:.4f} | "
            f"{values['out_of_distribution_mean_regret']:.4f} | "
            f"{100 * values['out_of_distribution_poor_rate']:.1f} |"
        )
    lines += [
        "",
        "## Paired comparison against the risk-aware selector (test split)",
        "",
        "Positive differences mean the risk-aware selector has the higher (worse) "
        "declared-objective cost; intervals are workload-level percentile bootstrap "
        f"95% CIs ({_BOOTSTRAP_SAMPLES} resamples, seed {_BOOTSTRAP_SEED}).",
        "",
        "| Compared arm | Mean cost difference (ql_risk - arm) | 95% CI | Workloads |",
        "|---|---:|---:|---:|",
    ]
    for arm, values in test["paired_vs_ql_risk"].items():
        lines.append(
            f"| {ARM_LABELS[arm]} | "
            f"{values['mean_cost_difference_ql_risk_minus_arm']:+.4f} | "
            f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}] | "
            f"{values['paired_workloads']} |"
        )
    lines += [
        "",
        "## Training (mean terminal reward per arm, last-100 window)",
        "",
        "| Arm | Mean reward, last 100 (across seeds) |",
        "|---|---:|",
    ]
    for arm, values in summary["training"].items():
        lines.append(
            f"| {ARM_LABELS[arm]} | "
            f"{values['mean_reward_last_100_across_seeds']:.4f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `decisions.csv` - one row per selection (all arms, validation and test).",
        "- `counterfactuals.csv` - four-policy outcomes per workload (both objectives).",
        "- `robustness.csv` - per arm x seed x family variability, regret and "
        "poor-selection rates.",
        "- `training_history.csv` - per-episode training records with objective values.",
        "- `explanations/` - structured explanations for every selection "
        "(JSONL, gzip).",
        "- `explanations_examples.json` - three fully worked examples chosen by the "
        "documented fixed rule.",
        "",
        "All numbers in this report are regenerated from the recorded seeds; "
        "explanation logs are reproducible and intentionally not tracked in git.",
        "",
    ]
    return "\n".join(lines)
