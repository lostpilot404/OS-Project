"""Regression coverage for drift options and evidence-derived reporting."""
import pytest

from config import build_default_config
from experiments.drift_uncertainty_experiment import _comparison_interpretations, run_drift_evaluation
from workload.generator import WorkloadGenerator
from workload.drift import injection_time_for_workload


def test_phase_transition_generator_preserves_prefix():
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 42)
    drift = gen.generate_with_drift("mixed", 42, "phase_transition")
    assert drift == gen.generate_with_drift("mixed", 42, "phase_transition")
    boundary = injection_time_for_workload(base)
    assert [p for p in drift if p.arrival_time < boundary] == [p for p in base if p.arrival_time < boundary]
    assert [p.arrival_time for p in drift] == [p.arrival_time for p in base]


@pytest.mark.parametrize("low,high,status", [(0.0036, 0.56, "excludes"), (-0.1, 0.5, "includes"), (-0.5, -0.01, "excludes"), (0, 0.5, "includes")])
def test_ci_interpretation_uses_unrounded_evidence(low, high, status):
    text = _comparison_interpretations([dict(target="q_recovery", reference="rule_wait", mean_difference=0.19, ci95_low=low, ci95_high=high)])
    assert f"{status} zero" in text[0]


def test_invalid_drift_rejected_before_output_or_training(tmp_path):
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        run_drift_evaluation(drift_kind="invalid", output_dir=target)
    assert not target.exists()


def test_phase_transition_experiment_outputs_measured_report(tmp_path):
    import csv
    result = run_drift_evaluation(repetitions=1, training_episodes_per_seed=2,
                                 drift_kind="phase_transition", output_dir=tmp_path)
    report = (tmp_path / "drift_phase_transition_estTrue_cv0.3_report.md").read_text(encoding="utf-8")
    assert "7 traces for single-policy arms, 35 for q_recovery" in report
    assert "short_jobs suffix" in report
    assert "~12%" not in report
    for row in result["summary_rows"]:
        if row["arm"] != "fixed_fcfs":
            assert row["regression_rate"] is not None
    with (tmp_path / "drift_phase_transition_estTrue_cv0.3_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 6
