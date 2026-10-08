"""Training, evaluation, end-to-end pipeline and reproducibility."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from config import (
    ACTION_NAMES,
    ADAPTIVE_LABEL,
    EvaluationConfig,
    ExperimentConfig,
    SchedulerConfig,
    StateConfig,
    TrainingConfig,
    WorkloadFamilyConfig,
    build_default_config,
    derive_seed,
)
from errors import ConfigurationError, ValidationError
from evaluation.comparison import (
    ALL_METRICS,
    adaptive_ratio_table,
    best_policy_per_family,
    family_summary,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import (
    ADAPTIVE_CLASSIC_REGIME,
    ADAPTIVE_LEARNED_REGIME,
    BASELINE_REGIME,
    ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
    build_evaluation_workloads,
    evaluate,
)
from experiments.run_experiment import run_experiment
from experiments.train import train
from main import main as cli_main
from workload.generator import WorkloadGenerator


def _small_config() -> ExperimentConfig:
    """A small but complete configuration, used to keep the suite fast."""
    families = (
        WorkloadFamilyConfig(name="few_short", description="short jobs", num_processes=4,
                             burst_distribution="bimodal", short_burst_fraction=0.9),
        WorkloadFamilyConfig(name="few_long", description="long jobs", num_processes=4,
                             burst_distribution="uniform", burst_time_min=20, burst_time_max=40,
                             arrival_window=8),
    )
    config = ExperimentConfig(
        name="test_small",
        scheduler=SchedulerConfig(round_robin_quantum=4),
        families=families,
        state=StateConfig(),
        training=TrainingConfig(episodes=12, seed=5, family_cycle=("few_short", "few_long")),
        evaluation=EvaluationConfig(repetitions=2, seed=99),
    )
    config.validate()
    return config


class TestWorkloadGeneration:
    def test_same_seed_same_workload(self) -> None:
        config = _small_config()
        generator = WorkloadGenerator(config.families)
        first = generator.generate("few_short", 1234)
        second = generator.generate("few_short", 1234)
        assert first == second
        assert first.fingerprint == second.fingerprint

    def test_different_seeds_give_different_workloads(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        assert (
            generator.generate("few_short", 1).fingerprint
            != generator.generate("few_short", 2).fingerprint
        )

    def test_process_count_and_attribute_ranges(self) -> None:
        config = build_default_config()
        generator = WorkloadGenerator(config.families)
        for family in config.families:
            workload = generator.generate(family.name, 0)
            assert len(workload) == family.num_processes
            assert all(family.burst_time_min <= p.burst_time <= family.burst_time_max
                       for p in workload.processes)
            assert all(family.priority_min <= p.priority <= family.priority_max
                       for p in workload.processes)
            assert all(p.arrival_time >= 0 for p in workload.processes)
            if family.arrival_pattern == "uniform":
                assert all(p.arrival_time <= family.arrival_window for p in workload.processes)

    def test_pids_are_one_based_and_unique(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        workload = generator.generate("few_long", 7)
        assert sorted(p.pid for p in workload.processes) == list(range(1, len(workload) + 1))

    def test_short_job_family_has_lower_median_burst(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        short = generator.generate("short_jobs", 3)
        long = generator.generate("long_jobs", 3)
        assert short.median_burst_time < long.median_burst_time

    def test_skewed_family_has_lower_priority_spread(self) -> None:
        from rl.state import observe_workload_state

        generator = WorkloadGenerator(build_default_config().families)
        skewed = observe_workload_state(generator.generate("priority_skewed", 1))
        uniform = observe_workload_state(generator.generate("mixed", 1))
        assert skewed.priority_spread < uniform.priority_spread

    def test_bimodal_short_family_has_higher_burst_dispersion(self) -> None:
        from rl.state import observe_workload_state

        generator = WorkloadGenerator(build_default_config().families)
        short_heavy = observe_workload_state(generator.generate("short_jobs", 3))
        cpu_bursty = observe_workload_state(generator.generate("cpu_bursty", 3))
        assert short_heavy.burst_dispersion > cpu_bursty.burst_dispersion

    def test_poisson_arrivals_are_spread_over_time(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        workload = generator.generate("poisson_arrivals", 4)
        assert all(p.arrival_time >= 0 for p in workload.processes)
        assert workload.arrival_span > 0
        assert len({p.arrival_time for p in workload.processes}) > 1
        # Workloads are stored in arrival order, independently of generation order.
        arrivals = [p.arrival_time for p in workload.processes]
        assert arrivals == sorted(arrivals)

    def test_unknown_family_and_bad_seed_are_rejected(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        with pytest.raises(ValidationError, match="unknown workload family"):
            generator.generate("nope", 0)
        with pytest.raises(ValidationError, match="seed"):
            generator.generate("few_short", -1)

    def test_duplicate_family_names_are_rejected(self) -> None:
        family = WorkloadFamilyConfig(name="dup", description="x", num_processes=3)
        with pytest.raises(ValidationError, match="unique"):
            WorkloadGenerator((family, family))


class TestTraining:
    def test_records_one_episode_per_episode(self) -> None:
        config = _small_config()
        result = train(config)
        assert result.history.episodes == config.training.episodes
        assert len(result.history.rewards) == config.training.episodes
        assert len(result.history.family_names) == config.training.episodes

    def test_families_cycle_in_configuration_order(self) -> None:
        config = _small_config()
        result = train(config)
        assert result.history.family_names[:4] == ("few_short", "few_long", "few_short", "few_long")

    def test_training_is_reproducible(self) -> None:
        config = _small_config()
        first = train(config)
        second = train(config)
        assert first.history.rewards == second.history.rewards
        assert first.history.workload_fingerprints == second.history.workload_fingerprints
        assert first.agent.q_table.tolist() == second.agent.q_table.tolist()

    def test_agent_learns_something(self) -> None:
        # With optimistic initialisation and a reward that is positive for good policies,
        # the table must move away from its initial value.
        config = _small_config()
        result = train(config)
        assert not (result.agent.q_table == config.q_learning.initial_value).all()
        assert result.history.visited_states > 0

    def test_workload_stream_matches_derived_seeds(self) -> None:
        config = _small_config()
        result = train(config)
        generator = WorkloadGenerator(config.families)
        expected = generator.generate("few_short", derive_seed(config.training.seed, 0, 0))
        assert result.history.workload_fingerprints[0] == expected.fingerprint

    def test_controller_is_trained_when_enabled(self) -> None:
        config = _small_config()
        result = train(config)
        assert result.controller is not None
        assert result.controller.agent.visit_counts.sum() == config.training.episodes


class TestEvaluation:
    def test_every_scheduler_sees_the_same_workloads(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        grouped = result.metrics.groupby(["family", "repetition"])["workload_fingerprint"].nunique()
        assert (grouped == 1).all()

    def test_all_policies_and_regimes_are_measured(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        assert set(result.metrics["policy"]) == set(ACTION_NAMES) | {
            ADAPTIVE_LABEL,
            "Round Robin (learned quantum)",
        }
        assert set(result.metrics["regime"]) == {
            BASELINE_REGIME,
            ADAPTIVE_CLASSIC_REGIME,
            ADAPTIVE_LEARNED_REGIME,
            ROUND_ROBIN_LEARNED_QUANTUM_REGIME,
        }

    def test_row_count_matches_the_protocol(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        workloads = len(config.families) * config.evaluation.repetitions
        # four baselines + classic adaptive + learned-quantum adaptive + learned-quantum RR
        assert len(result.metrics) == workloads * 7
        assert len(result.decisions) == workloads * 2

    def test_evaluation_workloads_are_disjoint_from_training(self) -> None:
        config = _small_config()
        training = train(config)
        generator = WorkloadGenerator(config.families)
        workloads = build_evaluation_workloads(config, generator)
        assert not ({w.fingerprint for w in workloads} & training.training_fingerprints)

    def test_evaluation_never_learns(self) -> None:
        config = _small_config()
        training = train(config)
        result = evaluate(config, training)
        assert not result.decisions["learned"].any()
        assert (result.decisions["epsilon"] == 0.0).all()

    def test_decision_actions_are_the_greedy_actions(self) -> None:
        config = _small_config()
        training = train(config)
        result = evaluate(config, training)
        for row in result.decisions.itertuples():
            assert row.action == training.agent.greedy_action(row.state_index)

    def test_adaptive_classic_rows_equal_the_chosen_baseline(self) -> None:
        config = _small_config()
        training = train(config)
        result = evaluate(config, training)
        baselines = result.metrics[result.metrics["regime"] == BASELINE_REGIME]
        adaptive_decisions = result.decisions[
            result.decisions["regime"] == ADAPTIVE_CLASSIC_REGIME
        ]
        assert not adaptive_decisions.empty
        for decision in adaptive_decisions.itertuples():
            chosen = ACTION_NAMES[decision.action]
            baseline_row = baselines[
                (baselines["family"] == decision.family)
                & (baselines["repetition"] == decision.repetition)
                & (baselines["policy"] == chosen)
            ].iloc[0]
            assert decision.adaptive_avg_waiting_time == pytest.approx(baseline_row["avg_waiting_time"])

    def test_zero_repetitions_is_rejected_by_configuration(self) -> None:
        with pytest.raises(ConfigurationError):
            EvaluationConfig(repetitions=0)

    def test_equal_seeds_are_rejected_by_the_configuration(self) -> None:
        config = _small_config()
        broken = replace(config, evaluation=replace(config.evaluation, seed=config.training.seed))
        with pytest.raises(ConfigurationError, match="seeds must differ"):
            broken.validate()

    def test_leaked_training_workloads_are_detected(self) -> None:
        config = _small_config()
        training = train(config)
        generator = WorkloadGenerator(config.families)
        leaked = build_evaluation_workloads(config, generator)[0].fingerprint
        poisoned_history = replace(
            training.history,
            workload_fingerprints=training.history.workload_fingerprints + (leaked,),
        )
        poisoned = replace(training, history=poisoned_history)
        with pytest.raises(ValidationError, match="already used during training"):
            evaluate(config, poisoned)


class TestComparisonTables:
    def test_tables_have_the_expected_shape(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        baselines = policy_summary(result.metrics)
        assert set(baselines.index) == set(ACTION_NAMES)
        assert set(baselines.columns) == set(ALL_METRICS)

        ratios = adaptive_ratio_table(result.metrics)
        assert len(ratios) == len(ACTION_NAMES) * len(ALL_METRICS)
        assert ratios["workloads"].eq(len(config.families) * config.evaluation.repetitions).all()
        assert (ratios["ratio_workloads"] <= ratios["workloads"]).all()

        families = family_summary(result.metrics)
        assert isinstance(families, pd.DataFrame)

        selection = policy_selection_table(result.decisions)
        assert "workloads" in selection.columns
        assert selection[[f"count_{name}" for name in ACTION_NAMES]].sum(axis=1).eq(
            selection["workloads"]
        ).all()

        occupancy = state_occupancy_table(result.decisions)
        assert (occupancy["distinct_states"] >= 1).all()

        best = best_policy_per_family(result.metrics)
        assert set(best["best_policy"]).issubset(set(ACTION_NAMES))

    def test_ratio_table_matches_an_independent_pairwise_computation(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        ratios = adaptive_ratio_table(result.metrics)
        metric = "avg_waiting_time"
        row = ratios.loc[("SJF", metric)]

        baseline = result.metrics[
            (result.metrics["regime"] == BASELINE_REGIME) & (result.metrics["policy"] == "SJF")
        ].set_index(["family", "repetition"])[metric]
        adaptive = result.metrics[
            result.metrics["regime"] == ADAPTIVE_CLASSIC_REGIME
        ].set_index(["family", "repetition"])[metric]
        paired = baseline.to_frame("baseline").join(adaptive.to_frame("adaptive"))
        usable = paired[paired["adaptive"] != 0.0]
        expected_ratio = float((usable["baseline"] / usable["adaptive"]).mean())

        assert row["workloads"] == len(paired)
        assert row["ratio_workloads"] == len(usable)
        assert row["mean_ratio"] == pytest.approx(expected_ratio, rel=1e-9)
        assert row["mean_difference"] == pytest.approx(
            float((paired["baseline"] - paired["adaptive"]).mean()), rel=1e-9
        )

    def test_empty_frames_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            policy_selection_table(pd.DataFrame())
        with pytest.raises(ValidationError):
            policy_summary(pd.DataFrame())


class TestEndToEnd:
    def test_run_experiment_writes_every_artefact(self, tmp_path: Path) -> None:
        config = _small_config()
        artifacts = run_experiment(
            config,
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            make_figures=True,
        )
        for name in ("config", "training_history", "q_table", "workloads", "metrics",
                     "decisions", "summary"):
            assert artifacts.paths[name].exists(), name
        assert len(artifacts.paths["figures"]) == 7
        assert all(path.exists() for path in artifacts.paths["figures"])

        payload = json.loads((tmp_path / "results" / "summary.json").read_text())
        assert payload["configuration"]["training"]["episodes"] == config.training.episodes
        assert payload["evaluation"]["distinct_workloads"] == (
            len(config.families) * config.evaluation.repetitions
        )

    def test_rerun_reproduces_identical_results(self, tmp_path: Path) -> None:
        config = _small_config()
        first = run_experiment(
            config,
            results_dir=tmp_path / "a",
            figures_dir=tmp_path / "fa",
            make_figures=False,
        )
        second = run_experiment(
            config,
            results_dir=tmp_path / "b",
            figures_dir=tmp_path / "fb",
            make_figures=False,
        )
        assert (tmp_path / "a" / "metrics.csv").read_text() == (
            tmp_path / "b" / "metrics.csv"
        ).read_text()
        assert (tmp_path / "a" / "decisions.csv").read_text() == (
            tmp_path / "b" / "decisions.csv"
        ).read_text()
        assert (tmp_path / "a" / "q_table.json").read_text() == (
            tmp_path / "b" / "q_table.json"
        ).read_text()
        assert first.summary["training"] == second.summary["training"]

    def test_every_policy_ran_on_every_evaluated_workload(self, tmp_path: Path) -> None:
        config = _small_config()
        artifacts = run_experiment(
            config,
            results_dir=tmp_path / "r",
            figures_dir=tmp_path / "f",
            make_figures=False,
        )
        metrics = artifacts.evaluation.metrics
        counts = metrics[metrics["regime"] == BASELINE_REGIME].groupby("policy").size()
        assert counts.nunique() == 1
        assert set(counts.index) == set(ACTION_NAMES)

    def test_cli_train_command(self, capsys: pytest.CaptureFixture) -> None:
        assert cli_main(["train"]) == 0
        captured = capsys.readouterr().out
        assert "Training" in captured
        assert "episodes" in captured
        assert "SJF" in captured  # the default run selects SJF at least once

    def test_cli_experiment_command_writes_figures(self, tmp_path: Path, capsys) -> None:
        exit_code = cli_main(
            [
                "experiment",
                "--results-dir",
                str(tmp_path / "results"),
                "--figures-dir",
                str(tmp_path / "figures"),
            ]
        )
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Evaluation" in output
        assert "Policy selected by the agent" in output
        assert (tmp_path / "results" / "metrics.csv").exists()
        assert (tmp_path / "figures" / "metric_comparison.png").exists()
