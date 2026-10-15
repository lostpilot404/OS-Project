"""Tests of the self-healing study harness: splits, artifacts, audit and reproducibility."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from config import SchedulerConfig, build_default_config
from errors import ValidationError
from experiments.healing_config import (
    SPLIT_TAGS,
    HealingArm,
    HealingExperimentConfig,
    default_arms,
)
from experiments.healing_experiment import (
    _detection_summary,
    build_healing_workloads,
    run_healing_experiment,
)
from workload.generator import WorkloadGenerator


def _tiny(**overrides) -> HealingExperimentConfig:
    values = {
        "training_seeds": (7101,),
        "training_episodes_per_seed": 25,
        "validation_seed": 8201,
        "final_test_seed": 20719,
        "validation_repetitions": 2,
        "final_test_repetitions": 2,
        "bootstrap_replicates": 120,
        "action_probe_repetitions": 1,
    }
    values.update(overrides)
    return HealingExperimentConfig(**values)


def test_declared_splits_are_disjoint_and_reproducible() -> None:
    """The three splits must never share a workload instance, and rebuilding a split must
    return byte-identical workloads so every arm sees the same traces."""
    config = _tiny()
    generator = WorkloadGenerator(config.families)
    first = build_healing_workloads("training", config, generator)
    second = build_healing_workloads("training", config, generator)
    assert [(w.family, w.repetition, w.workload.fingerprint) for w in first] == [
        (w.family, w.repetition, w.workload.fingerprint) for w in second
    ]
    fingerprints = {}
    for split in ("training", "validation", "final_test"):
        items = build_healing_workloads(split, config, generator)
        assert items, f"{split} must not be empty"
        assert len({w.workload.fingerprint for w in items}) == len(items), "duplicate traces"
        fingerprints[split] = {w.workload.fingerprint for w in items}
    assert not fingerprints["training"] & fingerprints["validation"]
    assert not fingerprints["training"] & fingerprints["final_test"]
    assert not fingerprints["validation"] & fingerprints["final_test"]
    assert SPLIT_TAGS == {"training": 51, "validation": 67, "final_test": 83}
    with pytest.raises(ValidationError, match="split"):
        build_healing_workloads("held_out", config, generator)


def test_experiment_config_validates_its_declared_seeds() -> None:
    """The frozen design is enforced: fresh, disjoint seeds and no empty arm set."""
    config = _tiny()
    assert config.bootstrap_seed == 104729
    assert config.scheduler == SchedulerConfig(round_robin_quantum=4, switching_cost=1)
    with pytest.raises(ValidationError, match="training and evaluation"):
        _tiny(validation_seed=7101)
    with pytest.raises(ValidationError, match="must differ"):
        _tiny(final_test_seed=8201)
    with pytest.raises(ValidationError):
        _tiny(training_seeds=())
    with pytest.raises(ValidationError):
        _tiny(bootstrap_replicates=10)
    with pytest.raises(ValidationError):
        HealingArm(name="x", kind="healing", policy_kind="mystery")
    with pytest.raises(ValidationError):
        HealingArm(name="x", kind="mystery", policy_kind="")
    arms = default_arms(config)
    names = [arm.name for arm in arms]
    assert names[:5] == [
        "fixed_fcfs",
        "fixed_sjf",
        "fixed_round_robin",
        "fixed_priority",
        "detector_only",
    ]
    assert len([name for name in names if name == "q_recovery"]) == len(config.training_seeds)
    assert len(set(names)) == len(names), "arm labels must be unique"


def test_detection_summary_is_split_aware() -> None:
    """Counts are reported per split, and a split with no incidents reports zeros rather
    than disappearing from the table."""
    metrics = pd.DataFrame(
        {
            "split": ["final_test"] * 2,
            "arm": ["a", "a"],
            "fallback_triggered": [False, True],
        }
    )
    incidents = pd.DataFrame(
        {
            "split": ["final_test"] * 4,
            "arm": ["a"] * 4,
            "verdict": ["accepted", "rolled_back", "held", "censored"],
            "action_changed": [True, True, False, True],
            "diagnosis": ["fairness_starvation", "fairness_starvation", "unexplained", "tail_dispersion"],
        }
    )
    summary = _detection_summary(incidents, metrics, "final_test")
    row = summary.set_index("arm").loc["a"]
    assert row["episodes"] == 2
    assert row["incidents"] == 4
    assert row["interventions"] == 3
    assert row["accepted"] == 1 and row["rolled_back"] == 1
    assert row["held"] == 1 and row["censored"] == 1
    assert row["episodes_with_fallback"] == 1
    assert row["in_run_false_recovery_rate"] == pytest.approx(1 / 3)
    assert "fairness_starvation=2" in row["diagnoses"]
    # A split with no incidents produces a zero row for every arm, not an absent one.
    empty = _detection_summary(incidents.iloc[0:0], metrics, "validation")
    assert set(empty["arm"]) == {"a"}
    assert (empty["incidents"] == 0).all()
    assert set(empty["split"]) == {"validation"}


def _run_tiny(tmp_path, **overrides):
    config = _tiny(**overrides)
    return run_healing_experiment(
        config,
        results_dir=tmp_path / "results",
        figures_dir=None,
        make_figures=False,
        probe=False,
    )


def test_experiment_writes_machine_readable_artifacts(tmp_path) -> None:
    """Every table the study reports is written to disk and round-trips exactly."""
    artifacts = _run_tiny(tmp_path)
    for name in (
        "metrics_validation",
        "metrics_final_test",
        "incidents_final_test",
        "episodes_final_test",
        "audit",
        "comparisons",
        "detection_summary",
        "coverage",
        "summary",
        "demo",
        "report",
        "threshold_derivation",
        "control_incidents",
    ):
        assert name in artifacts.paths and artifacts.paths[name].exists(), name
    assert artifacts.paths["report"] == tmp_path / "results" / "healing_report.md"

    metrics = pd.read_csv(artifacts.paths["metrics_final_test"])
    assert set(metrics["arm"]) == {row["arm"] for row in artifacts.detection_summary.to_dict("records")}
    assert len(metrics) == len(artifacts.metrics["final_test"])
    summary = json.loads(artifacts.paths["summary"].read_text(encoding="utf-8"))
    assert summary["final_test_seed"] == 8201 or summary["final_test_seed"] == 20719
    report = (tmp_path / "results" / "healing_report.md").read_text(encoding="utf-8")
    derivation = pd.read_csv(artifacts.paths["threshold_derivation"])
    assert set(derivation["quantity"]) >= {
        "control_incidents_per_trace",
        "null_closure_p95",
        "derived_accept_closure_factor",
        "declared_accept_closure_factor",
        "declared_false_accept_rate_on_control",
    }
    declared = derivation.set_index("quantity").loc["declared_accept_closure_factor", "value"]
    assert float(declared) == pytest.approx(summary["healing_config"]["accept_closure_factor"])
    # The derivation is part of the study's evidence trail: the report must quote it.
    assert "Declared thresholds and what they were derived from" in report
    comparisons = pd.read_csv(artifacts.paths["comparisons"])
    pd.testing.assert_frame_equal(
        comparisons.reset_index(drop=True),
        artifacts.comparisons.reset_index(drop=True),
        check_dtype=False,
    )
    assert "Self-healing" in report
    assert "generated from the machine-readable artifacts" in report
    assert "Counterfactual replay audit" in report
    for arm in {row["arm"] for row in artifacts.detection_summary.to_dict("records")}:
        assert arm in report


def test_hold_only_control_matches_fixed_fcfs_and_incidents_are_consistent(tmp_path) -> None:
    """The detector-only arm must be identical to plain FCFS (detection alone cannot change
    the schedule), and every logged incident must agree with the metrics and the log."""
    artifacts = _run_tiny(tmp_path)
    metrics = artifacts.metrics["final_test"]
    hold = metrics[metrics["arm"] == "detector_only"].set_index(["family", "repetition"])
    fcfs = metrics[metrics["arm"] == "fixed_fcfs"].set_index(["family", "repetition"])
    pd.testing.assert_series_equal(hold["avg_waiting_time"], fcfs["avg_waiting_time"])
    pd.testing.assert_series_equal(hold["context_switches"], fcfs["context_switches"])

    incidents = artifacts.incidents["final_test"]
    if incidents.empty:
        pytest.skip("the tiny configuration detected no incident")
    assert set(incidents["verdict"]) <= {"accepted", "rolled_back", "held", "censored"}
    assert set(incidents["split"]) <= {"validation", "final_test"}
    for row in incidents.to_dict("records"):
        assert row["action_name"] in row["offered_actions"].split(",")
        assert row["window_mean_at_trigger"] > row["trigger_threshold"]
        assert row["diagnosis"] == row["diagnosis_statement"].split(":")[0].strip() or row[
            "diagnosis"
        ] in row["diagnosis_statement"]
    audit = pd.read_csv(artifacts.paths["audit"])
    assert set(audit["verdict"]) <= {"accepted", "rolled_back", "held", "censored", "episode_level"}
    assert "intervention_gain" in audit.columns


def test_identical_seeds_reproduce_identical_results(tmp_path) -> None:
    """A second run of the same frozen configuration must reproduce every number."""
    first = _run_tiny(tmp_path / "a")
    second = _run_tiny(tmp_path / "b")
    pd.testing.assert_frame_equal(
        first.metrics["final_test"].reset_index(drop=True),
        second.metrics["final_test"].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        first.comparisons.sort_values(["target", "reference", "metric"]).reset_index(drop=True),
        second.comparisons.sort_values(["target", "reference", "metric"]).reset_index(drop=True),
        check_dtype=False,
    )
    assert first.summary["audit_summary"] == second.summary["audit_summary"]
    demo_one = json.loads(first.paths["demo"].read_text(encoding="utf-8"))
    demo_two = json.loads(second.paths["demo"].read_text(encoding="utf-8"))
    assert demo_one == demo_two
    assert demo_one["incidents"], "the demonstration trace must show the healing loop"


def test_demo_trace_shows_the_whole_loop(tmp_path) -> None:
    """The demonstration artifact is what the documentation quotes: it must contain the
    detection, the diagnosis, the decision, the verification and the verdict."""
    artifacts = _run_tiny(tmp_path)
    demo = json.loads(artifacts.paths["demo"].read_text(encoding="utf-8"))
    assert demo["workload"]["fingerprint"]
    assert demo["arm"]["name"]
    assert demo["decisions"], "the trace must contain the dispatch decisions"
    assert demo["decisions"][0]["policy"]
    incident = demo["incidents"][0]
    for field in (
        "detector_reason",
        "reference_at_trigger",
        "trigger_threshold",
        "diagnosis",
        "diagnosis_statement",
        "action_name",
        "offered_actions",
        "settings_before_policy",
        "settings_before_quantum",
        "settings_after_policy",
        "settings_after_quantum",
        "verdict",
        "reward",
        "verification_decisions",
        "window_end_time",
    ):
        assert field in incident, field
    assert incident["action_changed"] == (
        (incident["settings_before_policy"], incident["settings_before_quantum"])
        != (incident["settings_after_policy"], incident["settings_after_quantum"])
    )
    assert incident["detector_reason"] == "degradation_detected"
    assert demo["result"]["policy_name"] == demo["arm"]["name"]
    assert demo["result"]["total_waiting_time"] >= 0
    assert demo["result"]["total_elapsed_time"] > 0


def test_thresholds_are_derived_from_the_control_and_not_from_the_final_test(tmp_path) -> None:
    """The derivation runs on training/validation only and reports the declared constants
    against the measured null distribution of the verification statistic."""
    from experiments.healing_thresholds import collect_control_incidents, derive_thresholds

    config = _tiny()
    incidents = collect_control_incidents(config, "training")
    assert incidents.attrs["traces"] > 0
    if not incidents.empty:
        assert (incidents["verdict"] == "held").all(), "the control must never intervene"
        assert incidents["closure_fraction"].notna().all()
        assert (incidents["relative_band"] >= 0.0).all()
    derivation, _ = derive_thresholds(config, "training")
    assert derivation.traces == incidents.attrs["traces"]
    assert derivation.incidents == len(incidents)
    assert derivation.declared_accept_closure_factor == config.healing.accept_closure_factor
    assert 0.0 <= derivation.declared_false_accept_rate <= 1.0
    if len(incidents) >= 20:
        # The declared constant must be at least the measured 95th percentile of the
        # no-action statistic, otherwise the rule would certify noise.
        assert derivation.declared_accept_closure_factor >= derivation.null_p95 - 1e-9
    assert {row["min_load_factor"] for row in derivation.gate_sweep} >= {0.0, 1.0}
    with pytest.raises(ValidationError, match="training/validation"):
        collect_control_incidents(config, "final_test")
