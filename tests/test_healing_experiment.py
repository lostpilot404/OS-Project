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
    action_effect_probe,
    build_healing_workloads,
    run_healing_experiment,
)
from workload.generator import WorkloadGenerator


def _tiny(**overrides) -> HealingExperimentConfig:
    values = {
        "training_seeds": (7101,),
        "training_episodes_per_seed": 25,
        "validation_seed": 8201,
        # The declared held-out stream of the frozen design; the config rejects anything
        # that was already consumed by development.
        "final_test_seed": HealingExperimentConfig().final_test_seed,
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
    # A stream that has already been used for development decisions cannot come back as a
    # held-out split, and can never be reused as a training seed either.
    for retired in (20501, 20719, 21617, 72601):
        with pytest.raises(ValidationError, match="already used"):
            _tiny(final_test_seed=retired)
    assert {20719, 21617, 72601} <= set(HealingExperimentConfig().retired_final_test_seeds), (
        "streams consumed by development must stay retired"
    )
    assert HealingExperimentConfig().final_test_seed == 21977, (
        "the declared held-out stream of the frozen study"
    )
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
            "verdict": ["accepted", "rejected", "held", "inconclusive"],
            "action_changed": [True, True, False, True],
            "diagnosis": ["fairness_starvation", "fairness_starvation", "unexplained", "tail_dispersion"],
        }
    )
    summary = _detection_summary(incidents, metrics, "final_test")
    row = summary.set_index("arm").loc["a"]
    assert row["episodes"] == 2
    assert row["incidents"] == 4
    assert row["interventions"] == 3
    assert row["accepted"] == 1 and row["rejected"] == 1
    assert row["held"] == 1 and row["inconclusive"] == 1
    assert row["episodes_with_fallback"] == 1
    assert row["accepted_share_of_interventions"] == pytest.approx(1 / 3)
    assert row["rejected_share_of_interventions"] == pytest.approx(1 / 3)
    assert row["inconclusive_share_of_interventions"] == pytest.approx(1 / 3)
    assert row["unverified_share_of_interventions"] == pytest.approx(2 / 3)
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
        "detection_scenarios",
        "coverage",
        "coverage_curve",
        "q_seed_variability",
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
    assert summary["final_test_seed"] == _tiny().final_test_seed
    report = (tmp_path / "results" / "healing_report.md").read_text(encoding="utf-8")
    derivation = pd.read_csv(artifacts.paths["threshold_derivation"])
    assert set(derivation["quantity"]) >= {
        "control_incidents_per_trace",
        "null_closure_p95",
        "derived_accept_closure_factor",
        "declared_accept_closure_factor",
        "declared_false_accept_rate_on_control",
        "derived_accept_residual_factor",
        "declared_accept_residual_factor",
        "declared_reject_residual_factor",
        "validation_false_accept_residual_rate",
        "fitted_closure_baseline[intercept]",
        "declared_closure_baseline[intercept]",
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
    assert "Counterfactual audit of interventions (matched replay)" in report
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
    assert set(incidents["verdict"]) <= {"accepted", "rejected", "inconclusive", "held"}
    assert (incidents["confirmed_failure"] == (incidents["verdict"] == "rejected")).all()
    assert (incidents["rollback_target_policy"] == "").eq(
        incidents["verdict"].isin(["accepted", "held"]) | (incidents["action_changed"] == False)
    ).all()
    assert set(incidents["split"]) <= {"validation", "final_test"}
    for row in incidents.to_dict("records"):
        assert row["action_name"] in row["offered_actions"].split(",")
        assert row["window_mean_at_trigger"] > row["trigger_threshold"]
        assert row["diagnosis"] == row["diagnosis_statement"].split(":")[0].strip() or row[
            "diagnosis"
        ] in row["diagnosis_statement"]
    audit = pd.read_csv(artifacts.paths["audit"])
    assert set(audit["verdict"]) <= {
        "accepted",
        "rejected",
        "inconclusive",
        "held",
        "episode_level",
    }
    assert {"matched_gain", "hold_forever_gain", "divergence_decision_index"} <= set(audit.columns)
    audited = audit[audit["verdict"] != "episode_level"]
    if len(audited):
        # The matched replay must reproduce the realized prefix exactly and diverge at the
        # audited incident, and the forced incident must really have held.
        assert (audited["divergence_decision_index"] == audited["decision_index"]).all()
        assert audited["forced_incident_found"].all()
        assert audited["forced_incident_held"].all()


def test_arm_audit_aggregate_uses_an_intervention_denominator(tmp_path) -> None:
    """The per-arm audit summary is a per-intervention quantity: the one-per-episode
    "never heal again" counterfactual must never enter its denominator, and must be
    reported separately instead."""
    artifacts = _run_tiny(tmp_path)
    audit = pd.read_csv(artifacts.paths["audit"])
    summary = {row["arm"]: row for row in artifacts.summary["audit_summary"]}
    assert summary, "the tiny run must audit at least one arm"
    for arm, row in summary.items():
        # the aggregate is declared on the held-out split, so the denominator check is too
        subset = audit[(audit["arm"] == arm) & (audit["split"] == "final_test")]
        interventions = subset[subset["verdict"] != "episode_level"]
        episode_rows = subset[subset["verdict"] == "episode_level"]
        assert row["audited_interventions"] == len(interventions)
        assert row["episode_level_n"] == len(episode_rows)
        if len(interventions):
            # every rate in the entry is defined on the intervention rows only
            assert row["helped"] + row["harmful"] + row["unchanged"] == len(interventions)
            assert row["false_recoveries"] == len(interventions) - row["helped"]
            assert row["false_recovery_rate"] == pytest.approx(
                row["false_recoveries"] / len(interventions)
            )
            assert row["in_run_accept_rate"] == pytest.approx(
                (interventions["verdict"] == "accepted").mean()
            )
        if len(episode_rows):
            assert row["episode_level_mean_matched_gain"] == pytest.approx(
                episode_rows["matched_gain"].astype(float).mean()
            )
    report = artifacts.paths["report"].read_text(encoding="utf-8")
    assert "one row per *episode*" in report


def test_action_probe_reports_the_paired_change_of_the_window_statistic() -> None:
    """The probe's paired-change column is what the documentation uses to explain why one
    verification window cannot certify an action: it must be zero for the hold row of every
    incident and equal to the hold-referenced difference for every other action.

    The tiny harness runs with ``probe=False`` (the probe is its own experiment, not part of
    the artifact contract), so the probe is exercised directly on two validation traces.
    """
    config = _tiny()
    generator = WorkloadGenerator(config.families)
    workloads = build_healing_workloads("validation", config, generator)[:2]
    probe = action_effect_probe(config, workloads)
    assert not probe.empty, "the probe must measure at least one incident"
    assert "paired_closure_change" in probe.columns
    key = ["family", "repetition", "workload_fingerprint", "decision_index"]
    checked = 0
    for _, incident in probe.groupby(key, sort=False):
        hold = incident[incident["action_name"] == "hold"]
        assert len(hold) == 1
        assert hold["paired_closure_change"].iloc[0] == pytest.approx(0.0)
        statistic = (incident["pre_load_level"] - incident["post_load_level"]) / incident[
            "load_band_at_trigger"
        ]
        assert incident["paired_closure_change"].to_numpy() == pytest.approx(
            (statistic - statistic[hold.index[0]]).to_numpy()
        )
        checked += 1
    assert checked >= 1


def test_verification_discrimination_is_recomputable_from_the_audit(tmp_path) -> None:
    """The reported discrimination of the verification statistic must be recomputable from
    the published audit rows: same denominator (interventions), same AUC, same ceiling."""
    artifacts = _run_tiny(tmp_path)
    block = artifacts.summary["verification_discrimination"]
    audit = pd.read_csv(artifacts.paths["audit"])
    held = audit[(audit["split"] == "final_test") & (audit["verdict"] != "episode_level")]
    assert block["n_interventions"] == len(held)
    if not len(held):
        pytest.skip("the tiny configuration audited no intervention")
    gains = held["matched_gain"].astype(float)
    helped = gains > 0
    assert block["helped"] == int(helped.sum())
    assert block["harmful"] == int((gains < 0).sum())
    assert block["base_helped_share"] == pytest.approx(helped.mean())
    assert 0.0 <= block["rank_auc_closure_residual_to_helped"] <= 1.0
    if helped.any():
        certifiable = helped & (
            held["closure_fraction"].astype(float) >= block["declared_null_p95"]
        )
        assert block["certifiable_share_of_helpful_at_null_p95"] == pytest.approx(
            certifiable.sum() / helped.sum()
        )
        assert block["certifiable_helpful_at_null_p95"] == int(certifiable.sum())


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
    assert derivation.declared_accept_residual_factor == config.healing.accept_residual_factor
    assert derivation.declared_reject_residual_factor == config.healing.reject_residual_factor
    assert 0.0 <= derivation.declared_false_accept_rate <= 1.0
    assert 0.0 <= derivation.validation_false_accept_rate <= 1.0
    # The phase baseline must be the declared one, and conditioning must not inflate the
    # null spread it is supposed to shrink.
    assert derivation.declared_baseline == config.healing.closure_baseline
    if len(incidents) >= 20:
        assert derivation.residual_sd <= float(incidents["closure_fraction"].std()) + 1e-9
    if len(incidents) >= 20:
        # The declared constant must be at least the measured 95th percentile of the
        # no-action statistic, otherwise the rule would certify noise.
        assert derivation.declared_accept_closure_factor >= derivation.null_p95 - 1e-9
    assert {row["min_load_factor"] for row in derivation.gate_sweep} >= {0.0, 1.0}
    with pytest.raises(ValidationError, match="training/validation"):
        collect_control_incidents(config, "final_test")
