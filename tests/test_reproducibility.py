"""End-to-end reproducibility: identical seeds must reproduce identical artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.study import run_study
from experiments.study_config import build_study_design

_FILES_TO_COMPARE = (
    "decisions.csv",
    "counterfactuals.csv",
    "training_history.csv",
    "robustness.csv",
    "summary.json",
)


def _tiny_design():
    design = build_study_design()
    return type(design)(
        training_episodes=5,
        training_seeds=(42, 43),
        validation_seed=design.validation_seed,
        validation_repetitions=1,
        test_seed=design.test_seed,
        test_repetitions=1,
        random_seeds=(7701, 7702),
        primary_objective=design.primary_objective,
        poor_regret_threshold=design.poor_regret_threshold,
        training_families=design.training_families,
        held_out_families=design.held_out_families,
        q_arms=dict(design.q_arms),
        baseline_arms=design.baseline_arms,
    )


def _run(tmp_path: Path, name: str) -> Path:
    run_study(
        design=_tiny_design(),
        results_dir=tmp_path / name / "results",
        figures_dir=tmp_path / name / "figures",
        make_figures=False,
        explain_training=True,
    )
    return tmp_path / name / "results"


class TestReproducibility:
    def test_two_runs_with_identical_configuration_match(self, tmp_path: Path) -> None:
        first = _run(tmp_path, "run1")
        second = _run(tmp_path, "run2")
        for name in _FILES_TO_COMPARE:
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_explanation_logs_are_byte_identical(self, tmp_path: Path) -> None:
        first = _run(tmp_path, "run1")
        second = _run(tmp_path, "run2")
        names = sorted(
            path.name for path in (first / "explanations").iterdir()
        )
        assert names
        for name in names:
            assert (first / "explanations" / name).read_bytes() == (
                second / "explanations" / name
            ).read_bytes(), name

    def test_decisions_identical_across_runs_as_records(self, tmp_path: Path) -> None:
        first = _run(tmp_path, "run1")
        second = _run(tmp_path, "run2")
        frame1 = pd.read_csv(first / "decisions.csv")
        frame2 = pd.read_csv(second / "decisions.csv")
        pd.testing.assert_frame_equal(frame1, frame2)

    def test_summary_json_round_trips(self, tmp_path: Path) -> None:
        results = _run(tmp_path, "run1")
        payload = json.loads((results / "summary.json").read_text())
        assert payload["configuration"]["design"]["test_seed"] == 9501
        assert set(payload["test"]["arms"]) == set(payload["configuration"]["design"]["q_arms"]) | {
            "best_fixed",
            "fingerprint_rule",
            "random",
        }
