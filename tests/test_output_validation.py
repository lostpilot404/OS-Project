"""Artifact output validation: schema, consistency and explanation coverage.

Runs one tiny study and validates every generated artifact the way a reader would:
required columns, row accounting, non-negative regret, objective arithmetic
recomputed from the recorded terms, decision/counterfactual consistency, and
explanation coverage for every selection.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from config import ACTION_NAMES, RiskObjectiveConfig
from experiments.study import run_study
from experiments.study_config import build_study_design

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory) -> Path:
    design = build_study_design()
    tiny = type(design)(
        training_episodes=5,
        training_seeds=(42,),
        validation_seed=design.validation_seed,
        validation_repetitions=1,
        test_seed=design.test_seed,
        test_repetitions=2,
        random_seeds=(7701,),
        primary_objective=design.primary_objective,
        poor_regret_threshold=design.poor_regret_threshold,
        training_families=design.training_families,
        held_out_families=design.held_out_families,
        q_arms=dict(design.q_arms),
        baseline_arms=design.baseline_arms,
    )
    base = tmp_path_factory.mktemp("output_validation")
    run_study(
        design=tiny,
        results_dir=base / "results",
        figures_dir=base / "figures",
        make_figures=False,
    )
    return base / "results"


class TestDecisionsSchema:
    def test_required_columns_present(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        required = {
            "arm",
            "model_seed",
            "split",
            "family",
            "repetition",
            "workload_fingerprint",
            "policy_name",
            "action",
            "selection_rule",
            "risk_objective_cost",
            "orig_objective_cost",
            "regret",
            "is_hindsight_best",
            "selected_avg_waiting_time",
            "selected_avg_response_time",
            "selected_context_switches",
            "selected_p95_waiting_time",
        }
        assert required <= set(frame.columns)

    def test_row_accounting(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        # 8 families x (1 validation + 2 test) workloads, 1 seed per arm
        per_q_arm = 8 * 3
        assert set(frame["arm"]) == {
            "ql_orig",
            "ql_risk",
            "ql_risk_origstate",
            "ql_orig_fingerprint",
            "best_fixed",
            "fingerprint_rule",
            "random",
        }
        counts = frame.groupby("arm").size().to_dict()
        for arm in ("ql_orig", "ql_risk", "ql_risk_origstate", "ql_orig_fingerprint", "random"):
            assert counts[arm] == per_q_arm
        assert counts["best_fixed"] == 24
        assert counts["fingerprint_rule"] == 24

    def test_no_missing_values_in_key_columns(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        key = [
            "risk_objective_cost",
            "orig_objective_cost",
            "regret",
            "is_hindsight_best",
            "policy_name",
            "workload_fingerprint",
        ]
        assert not frame[key].isna().any().any()

    def test_fingerprints_are_valid_digests(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        assert all(_FINGERPRINT_PATTERN.match(value) for value in frame["workload_fingerprint"])

    def test_regret_is_nonnegative_and_consistent(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        assert (frame["regret"] >= -1e-9).all()
        best_flag = frame["regret"].abs() <= 1e-9
        assert (frame["is_hindsight_best"] == best_flag).all()

    def test_objective_cost_recomputes_from_recorded_terms(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "decisions.csv")
        weights = RiskObjectiveConfig()
        recomputed = (
            weights.weight_waiting_time * frame["risk_term_avg_waiting_time"]
            + weights.weight_response_time * frame["risk_term_avg_response_time"]
            + weights.weight_context_switches
            * frame["risk_term_context_switches_per_process"]
            + weights.weight_tail_waiting * frame["risk_term_p95_waiting_time"]
        )
        assert (recomputed - frame["risk_objective_cost"]).abs().max() < 1e-9

    def test_selected_policy_metrics_match_counterfactuals(
        self, results_dir: Path
    ) -> None:
        decisions = pd.read_csv(results_dir / "decisions.csv")
        counterfactuals = pd.read_csv(results_dir / "counterfactuals.csv")
        merged = decisions.merge(
            counterfactuals,
            left_on=["split", "family", "repetition", "workload_fingerprint", "action"],
            right_on=["split", "family", "repetition", "workload_fingerprint", "action"],
            suffixes=("_dec", "_cf"),
        )
        assert len(merged) == len(decisions)
        for metric in ("avg_waiting_time", "avg_response_time", "p95_waiting_time"):
            # Every arm must record the selected policy's metrics, not just baselines.
            assert merged[f"selected_{metric}"].notna().all(), metric
            assert (
                merged[f"selected_{metric}"] - merged[f"{metric}"]
            ).abs().max() < 1e-9


class TestCounterfactualsSchema:
    def test_four_rows_per_workload(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "counterfactuals.csv")
        grouped = frame.groupby(["split", "workload_fingerprint"])
        assert (grouped.size() == 4).all()
        for _, group in grouped:
            assert set(group["policy_name"]) == set(ACTION_NAMES)

    def test_costs_are_finite(self, results_dir: Path) -> None:
        frame = pd.read_csv(results_dir / "counterfactuals.csv")
        for column in ("orig_objective_cost", "risk_objective_cost"):
            assert frame[column].notna().all()
            assert (frame[column].abs() < 1e6).all()


class TestExplanations:
    def test_every_selection_has_an_explanation(self, results_dir: Path) -> None:
        decisions = pd.read_csv(results_dir / "decisions.csv")
        total_decisions = len(decisions)
        total_explanations = 0
        for path in (results_dir / "explanations").glob("*.jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                total_explanations += sum(1 for _ in handle)
        # one explanation per evaluation selection plus one per training selection
        training_rows = pd.read_csv(results_dir / "training_history.csv")
        assert total_explanations == total_decisions + len(training_rows)

    def test_explanations_have_the_declared_schema(self, results_dir: Path) -> None:
        path = next((results_dir / "explanations").glob("test_ql_risk*.jsonl.gz"))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.loads(handle.readline())
        assert set(record) == {
            "schema_version",
            "workload_name",
            "workload_fingerprint",
            "information_boundary",
            "workload_characteristics",
            "decision",
            "declared_objective",
            "counterfactual_objectives",
            "regret",
            "reasons",
        }
        assert set(record["counterfactual_objectives"]) == set(ACTION_NAMES)
        assert record["reasons"]

    def test_examples_file_carries_full_explanations(self, results_dir: Path) -> None:
        payload = json.loads((results_dir / "explanations_examples.json").read_text(encoding="utf-8"))
        assert payload
        for item in payload:
            assert set(item) == {"criterion", "explanation"}
            assert item["explanation"]["reasons"]


class TestSummaryAndReport:
    def test_rates_are_probabilities(self, results_dir: Path) -> None:
        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        for arm, values in summary["test"]["arms"].items():
            for key in (
                "hindsight_best_rate",
                "poor_selection_rate",
                "severe_poor_rate",
            ):
                assert 0.0 <= values[key] <= 1.0, (arm, key)

    def test_report_contains_generated_values(self, results_dir: Path) -> None:
        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        report = (results_dir / "report.md").read_text(encoding="utf-8")
        assert "offline policy-selection" in report.lower()
        values = summary["test"]["arms"]["ql_risk"]
        assert f"{values['mean_objective_cost']:.4f}" in report
        assert "not a runtime OS scheduler" in report

    def test_manifest_matches_written_files(self, results_dir: Path) -> None:
        manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["decision_rows"] > 0
        assert manifest["counterfactual_rows"] > 0
        assert (results_dir / "decisions.csv").exists()
        assert (results_dir / "counterfactuals.csv").exists()
        assert (results_dir / "robustness.csv").exists()
