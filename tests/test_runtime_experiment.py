"""Small deterministic end-to-end checks for the runtime experiment artifacts."""

from __future__ import annotations

import json

from config import ACTION_NAMES
from evaluation.comparison import ALL_METRICS
from experiments.runtime_config import RuntimeExperimentConfig
from experiments.runtime_experiment import run_runtime_experiment


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


def test_pipeline_separates_splits_pairs_all_methods_and_records_learned_switch(tmp_path) -> None:
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
    assert demo["policy_switch_count"] > 0
    assert demo["fallback_decisions"] == 0
    assert demo["all_choices_from_trained_state"] is True
    actions = [entry["action"] for entry in demo["decisions"]]
    assert len(set(actions)) > 1
    assert sum(entry["policy_switch"] for entry in demo["decisions"]) == demo[
        "policy_switch_count"
    ]
    assert set(actions).issubset(range(len(ACTION_NAMES)))

    coverage = artifacts.paths["state_action_coverage"].read_text()
    assert "unvisited_state_actions" in coverage
    assert "unvisited_action_opportunities" in coverage
    report = artifacts.paths["report"].read_text()
    assert "untouched final test" in report.lower()
    assert "gamma" in report


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
