"""Small deterministic end-to-end checks for the runtime experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from config import ACTION_NAMES, QLearningConfig
from errors import ValidationError
from evaluation.comparison import ALL_METRICS
from experiments.runtime_config import RuntimeExperimentConfig
from experiments.runtime_experiment import (
    _config_snapshot,
    _select_demo_candidate,
    run_runtime_experiment,
)


def _small_runtime_config() -> RuntimeExperimentConfig:
    defaults = RuntimeExperimentConfig()
    families = defaults.families[:2]
    return RuntimeExperimentConfig(
        families=families,
        training_families=tuple(family.name for family in families),
        training_episodes_per_seed=30,
        training_seeds=(1001,),
        validation_seed=2001,
        final_test_seed=3001,
        validation_repetitions=2,
        final_test_repetitions=3,
        bootstrap_replicates=100,
    )


def test_pipeline_separates_splits_pairs_all_methods_and_records_validation_demo(tmp_path) -> None:
    artifacts = run_runtime_experiment(_small_runtime_config(), tmp_path / "first")
    frame = artifacts.final_test_metrics
    assert set(frame["method"]) == {
        *ACTION_NAMES,
        "Causal heuristic",
        "Runtime Q-learning",
    }
    assert set(ALL_METRICS).issubset(frame.columns)
    assert {"policy_switch_count", "decision_count", "observation_ms", "selection_ms", "q_update_ms"}.issubset(
        frame.columns
    )
    assert frame["workload_fingerprint"].nunique() == 6
    for _, workload_rows in frame.groupby("workload_fingerprint"):
        assert set(workload_rows["method"]) == {
            *ACTION_NAMES,
            "Causal heuristic",
            "Runtime Q-learning",
        }

    summary = artifacts.summary
    assert summary["split_fingerprint_overlap"] is False
    assert summary["final_test_unique_workloads"] == 6
    assert summary["validation_unique_workloads"] == 4
    assert set(summary["paired_comparisons"][0]) >= {
        "target", "reference", "metric", "mean_difference", "ci95_low", "ci95_high"
    }

    demo = json.loads(artifacts.paths["learned_switch_demo"].read_text())
    assert demo["status"] == "verified_learned_runtime_policy_switch"
    assert demo["source_split"] == "validation"
    assert demo["illustrative_not_representative"] is True
    assert demo["workload_fingerprint"]
    assert demo["workload_processes"]
    actions = [entry["action"] for entry in demo["decisions"]]
    assert sum(entry["policy_switch"] for entry in demo["decisions"]) == demo[
        "policy_switch_count"
    ]
    assert set(actions).issubset(range(len(ACTION_NAMES)))
    assert artifacts.summary["demo"]["source_split"] == "validation"
    assert artifacts.summary["demo"]["selection_rule"]

    coverage = artifacts.paths["state_action_coverage"].read_text()
    assert "unvisited_state_actions" in coverage
    assert "unvisited_action_opportunities" in coverage
    report = artifacts.paths["report"].read_text()
    assert "untouched final test" in report.lower()
    assert "gamma" in report


def test_runtime_default_uses_undiscounted_waiting_objective_and_explicit_feature_metadata() -> None:
    snapshot = RuntimeExperimentConfig()
    metadata = _config_snapshot(snapshot)
    assert snapshot.q_learning.discount_factor == 1.0
    assert metadata["state_encoder"]["features"][3] == (
        "mean_ready_arrival_age(<=q,<=4q,>4q)"
    )
    assert "live FIFO insertion order" in metadata["ready_queue_contract"]
    assert "undiscounted episodic return" in metadata["learning_objective"]

    discounted_config = replace(snapshot, q_learning=QLearningConfig(discount_factor=0.5))
    discounted_objective = _config_snapshot(discounted_config)["learning_objective"]
    assert "per-decision discounted return" in discounted_objective
    assert "not the undiscounted total-waiting objective" in discounted_objective


def test_demo_selector_rejects_final_test_candidate() -> None:
    candidate = (
        999,
        None,
        SimpleNamespace(split="final_test"),
        SimpleNamespace(policy_switch_count=999),
    )
    with pytest.raises(ValidationError, match="selected from validation"):
        _select_demo_candidate([candidate])


def test_seeded_metrics_and_demo_are_reproducible_excluding_host_timings(tmp_path) -> None:
    config = _small_runtime_config()
    first = run_runtime_experiment(config, tmp_path / "a")
    second = run_runtime_experiment(config, tmp_path / "b")
    stable_columns = [
        "split",
        "method",
        "training_seed",
        "family",
        "repetition",
        "workload_seed",
        "workload_fingerprint",
        *ALL_METRICS,
        "policy_switch_count",
        "decision_count",
        "total_reward",
    ]
    assert first.final_test_metrics[stable_columns].equals(
        second.final_test_metrics[stable_columns]
    )
    assert first.summary["demo"] == second.summary["demo"]
    assert first.summary["paired_comparisons"] == second.summary["paired_comparisons"]
