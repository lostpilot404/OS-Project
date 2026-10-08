"""Workload generation, independent training, paired evaluation, and reports."""

from __future__ import annotations

import builtins
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from config import (
    ACTION_NAMES,
    SELECTOR_LABEL,
    EvaluationConfig,
    ExperimentConfig,
    SchedulerConfig,
    StateConfig,
    TrainingConfig,
    WorkloadFamilyConfig,
    build_default_config,
    derive_seed,
)
from errors import ValidationError
from evaluation.comparison import (
    ALL_METRICS,
    selector_ratio_table,
    best_policy_per_family,
    family_summary,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import SELECTOR_REGIME, BASELINE_REGIME, build_evaluation_workloads, evaluate
from experiments.run_experiment import config_snapshot, run_experiment
from experiments.train import train
from main import main as cli_main
from workload.generator import WorkloadGenerator


def _small_config() -> ExperimentConfig:
    """Fast, fully specified training/evaluation design with one held-out condition."""
    families = (
        WorkloadFamilyConfig(
            name="few_short",
            description="short-job mixture",
            num_processes=4,
            burst_distribution="bimodal",
            burst_time_min=1,
            burst_time_max=50,
            short_burst_max=5,
            short_burst_fraction=0.9,
        ),
        WorkloadFamilyConfig(
            name="interactive",
            description="one long process plus staggered short jobs",
            num_processes=4,
            burst_distribution="staggered_interactive",
            burst_time_min=25,
            burst_time_max=40,
            short_burst_max=3,
            arrival_pattern="staggered",
            staggered_gap_min=3,
            staggered_gap_max=5,
            priority_min=1,
            priority_max=1,
        ),
        WorkloadFamilyConfig(
            name="held_out_poisson",
            description="held-out Poisson condition",
            num_processes=4,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=20,
            arrival_pattern="poisson",
            arrival_rate=0.5,
        ),
    )
    config = ExperimentConfig(
        name="test_small_offline_selector",
        scheduler=SchedulerConfig(round_robin_quantum=4),
        families=families,
        state=StateConfig(),
        training=TrainingConfig(
            episodes=24,
            seed=5,
            replicates=2,
            family_cycle=("few_short", "interactive"),
        ),
        evaluation=EvaluationConfig(repetitions=2, seed=99),
    )
    config.validate()
    return config


class TestWorkloadGeneration:
    def test_same_seed_same_workload(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        first = generator.generate("few_short", 1234)
        second = generator.generate("few_short", 1234)
        assert first == second
        assert first.fingerprint == second.fingerprint

    def test_different_seeds_give_different_workloads(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        assert generator.generate("few_short", 1).fingerprint != generator.generate("few_short", 2).fingerprint

    def test_default_family_sizes_and_burst_ranges(self) -> None:
        config = build_default_config()
        generator = WorkloadGenerator(config.families)
        for family in config.families:
            workload = generator.generate(family.name, 0)
            assert len(workload) == family.num_processes
            assert all(p.arrival_time >= 0 for p in workload.processes)
            assert all(family.priority_min <= p.priority <= family.priority_max for p in workload.processes)
            if family.burst_distribution == "staggered_interactive":
                assert family.burst_time_min <= workload.processes[0].burst_time <= family.burst_time_max
                assert all(1 <= p.burst_time <= family.short_burst_max for p in workload.processes[1:])
            else:
                assert all(family.burst_time_min <= p.burst_time <= family.burst_time_max for p in workload.processes)
            if family.arrival_pattern == "uniform":
                assert all(p.arrival_time <= family.arrival_window for p in workload.processes)
            if family.arrival_pattern == "staggered":
                arrivals = [p.arrival_time for p in workload.processes]
                gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
                assert arrivals[0] == 0
                assert all(family.staggered_gap_min <= gap <= family.staggered_gap_max for gap in gaps)

    def test_interactive_distribution_is_not_a_policy_choice(self) -> None:
        family = next(
            family for family in build_default_config().families
            if family.name == "staggered_interactive"
        )
        workload = WorkloadGenerator((family,)).generate(family.name, 15)
        assert len(workload) == 15
        assert workload.processes[0].arrival_time == 0
        assert workload.processes[0].burst_time >= 25
        assert all(p.burst_time <= 3 for p in workload.processes[1:])
        assert all(4 <= b.arrival_time - a.arrival_time <= 6 for a, b in zip(workload.processes, workload.processes[1:]))

    def test_priority_and_burst_conditions_have_distinct_profiles(self) -> None:
        from rl.state import observe_workload_state

        generator = WorkloadGenerator(build_default_config().families)
        skewed = observe_workload_state(generator.generate("priority_skewed", 1))
        uniform = observe_workload_state(generator.generate("mixed", 1))
        short = observe_workload_state(generator.generate("short_jobs", 3))
        cpu = observe_workload_state(generator.generate("cpu_bursty", 3))
        assert skewed.priority_spread < uniform.priority_spread
        assert short.burst_dispersion > cpu.burst_dispersion

    def test_poisson_arrivals_are_sorted_and_spread_over_time(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        workload = generator.generate("poisson_arrivals", 4)
        arrivals = [process.arrival_time for process in workload.processes]
        assert arrivals == sorted(arrivals)
        assert workload.arrival_span > 0
        assert len(set(arrivals)) > 1

    def test_unknown_family_bad_seed_and_duplicate_family_are_rejected(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        with pytest.raises(ValidationError, match="unknown workload family"):
            generator.generate("nope", 0)
        with pytest.raises(ValidationError, match="seed"):
            generator.generate("few_short", -1)
        family = WorkloadFamilyConfig(name="dup", description="x", num_processes=3)
        with pytest.raises(ValidationError, match="unique"):
            WorkloadGenerator((family, family))


class TestTraining:
    def test_records_one_terminal_episode_and_cycles_only_training_families(self) -> None:
        config = _small_config()
        result = train(config)
        assert result.history.episodes == config.training.episodes
        assert len(result.history.rewards) == config.training.episodes
        assert result.history.family_names[:4] == ("few_short", "interactive") * 2
        assert "held_out_poisson" not in result.history.family_names
        assert result.agent.visit_counts.sum() == config.training.episodes

    def test_training_is_reproducible_with_a_fixed_seed(self) -> None:
        config = _small_config()
        first, second = train(config), train(config)
        assert first.history.rewards == second.history.rewards
        assert first.history.workload_fingerprints == second.history.workload_fingerprints
        assert first.agent.q_table.tolist() == second.agent.q_table.tolist()

    def test_training_updates_q_values_from_neutral_initialization(self) -> None:
        config = _small_config()
        assert config.q_learning.initial_value == 0.0
        result = train(config)
        assert not (result.agent.q_table == 0.0).all()
        assert result.history.visited_states > 0

    def test_training_stream_matches_the_documented_seed_derivation(self) -> None:
        config = _small_config()
        result = train(config)
        generator = WorkloadGenerator(config.families)
        expected = generator.generate("few_short", derive_seed(config.training.seed, 0, 0))
        assert result.history.workload_fingerprints[0] == expected.fingerprint

    def test_training_history_distinguishes_random_branch_from_greedy_match(self) -> None:
        summary = train(_small_config()).history.summary()
        assert 0.0 <= summary["random_exploration_episode_rate"] <= 1.0
        assert 0.0 <= summary["greedy_action_match_rate"] <= 1.0
        assert "exploration_rate" not in summary

    def test_training_seeds_are_multiple_and_independent_by_default(self) -> None:
        config = build_default_config()
        assert len(config.training.seeds) >= 2
        assert len(set(config.training.seeds)) == config.training.replicates
        assert config.evaluation.seed not in config.training.seeds


class TestEvaluation:
    def test_every_policy_receives_the_same_workload(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        for _, group in result.metrics.groupby(["training_seed", "family", "repetition"]):
            assert group["workload_fingerprint"].nunique() == 1
            assert len(group) == 5
            assert set(group["policy"]) == set(ACTION_NAMES) | {SELECTOR_LABEL}

    def test_only_four_baselines_and_one_selector_regime_are_reported(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        assert set(result.metrics["policy"]) == set(ACTION_NAMES) | {SELECTOR_LABEL}
        assert set(result.metrics["regime"]) == {BASELINE_REGIME, SELECTOR_REGIME}
        assert "Round Robin (learned quantum)" not in set(result.metrics["policy"])

    def test_row_counts_distinguish_unique_workloads_from_model_decisions(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        unique = len(config.families) * config.evaluation.repetitions
        assert len(result.metrics) == unique * 5
        assert len(result.decisions) == unique
        assert result.summary()["unique_workloads"] == unique

    def test_evaluation_workloads_are_disjoint_from_training(self) -> None:
        config = _small_config()
        training = train(config)
        workloads = build_evaluation_workloads(config, WorkloadGenerator(config.families))
        assert not ({item.fingerprint for item in workloads} & training.training_fingerprints)

    def test_evaluation_is_greedy_and_does_not_mutate_the_q_table(self) -> None:
        config = _small_config()
        training = train(config)
        before_q = training.agent.q_table.copy()
        before_visits = training.agent.visit_counts.copy()
        result = evaluate(config, training)
        assert not result.decisions["learned"].any()
        assert not result.decisions["explored"].any()
        assert (result.decisions["epsilon"] == 0.0).all()
        assert (result.decisions["action"] == result.decisions["greedy_action"]).all()
        assert (before_q == training.agent.q_table).all()
        assert (before_visits == training.agent.visit_counts).all()

    def test_selector_metrics_equal_the_chosen_baseline_on_each_input(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        baselines = result.metrics[result.metrics["regime"] == BASELINE_REGIME]
        for decision in result.decisions.itertuples():
            baseline = baselines[
                (baselines["family"] == decision.family)
                & (baselines["repetition"] == decision.repetition)
                & (baselines["policy"] == decision.policy_name)
            ].iloc[0]
            for metric in ALL_METRICS:
                assert getattr(decision, f"selector_{metric}") == pytest.approx(baseline[metric])
            assert decision.selector_context_switches_per_process == pytest.approx(
                baseline["context_switches_per_process"]
            )

    def test_leakage_is_detected(self) -> None:
        config = _small_config()
        training = train(config)
        leaked = build_evaluation_workloads(config, WorkloadGenerator(config.families))[0].fingerprint
        poisoned_history = replace(
            training.history,
            workload_fingerprints=training.history.workload_fingerprints + (leaked,),
        )
        with pytest.raises(ValidationError, match="already used during training"):
            evaluate(config, replace(training, history=poisoned_history))

    def test_all_independent_training_seeds_are_evaluated_on_same_unique_inputs(self, tmp_path) -> None:
        config = _small_config()
        artifacts = run_experiment(config, tmp_path / "r", tmp_path / "f", make_figures=False)
        assert tuple(run.history.seed for run in artifacts.training_runs) == config.training.seeds
        assert artifacts.evaluation.training_seeds == config.training.seeds
        assert artifacts.evaluation.metrics["workload_fingerprint"].nunique() == (
            len(config.families) * config.evaluation.repetitions
        )
        assert len(artifacts.evaluation.decisions) == (
            len(config.families) * config.evaluation.repetitions * config.training.replicates
        )

    def test_default_experiment_learns_distinct_policies_across_conditions(self, tmp_path) -> None:
        # Independent verification: every seed should choose SJF on the long CPU-bursty
        # condition and Round Robin on the disjoint staggered-interactive condition.
        config = build_default_config()
        artifacts = run_experiment(config, tmp_path / "results", tmp_path / "figures", make_figures=False)
        assert len(artifacts.training_runs) == config.training.replicates >= 2
        decisions = artifacts.evaluation.decisions
        for seed in config.training.seeds:
            cpu = decisions[
                (decisions["training_seed"] == seed)
                & (decisions["family"] == "cpu_bursty")
            ]
            interactive = decisions[
                (decisions["training_seed"] == seed)
                & (decisions["family"] == "staggered_interactive")
            ]
            assert set(cpu["policy_name"]) == {"SJF"}
            assert set(interactive["policy_name"]) == {"Round Robin"}
            assert cpu["state_seen_in_training"].all()
            assert interactive["state_seen_in_training"].all()
        assert set(decisions[decisions["family"] == "poisson_arrivals"]["policy_name"]).issubset(
            set(ACTION_NAMES)
        )


class TestComparisonTables:
    def test_tables_report_means_variability_and_unambiguous_counts(self, tmp_path: Path) -> None:
        config = _small_config()
        artifacts = run_experiment(
            config, tmp_path / "results", tmp_path / "figures", make_figures=False
        )
        metrics = artifacts.evaluation.metrics
        summary = policy_summary(metrics, BASELINE_REGIME)
        assert set(summary.index) == set(ACTION_NAMES)
        assert {f"{metric}_mean" for metric in ALL_METRICS}.issubset(summary.columns)
        assert {f"{metric}_sd" for metric in ALL_METRICS}.issubset(summary.columns)

        ratios = selector_ratio_table(metrics)
        assert len(ratios) == len(ACTION_NAMES) * len(ALL_METRICS)
        expected_unique = len(config.families) * config.evaluation.repetitions
        assert ratios["unique_workloads"].eq(expected_unique).all()
        assert ratios["model_workload_pairs"].eq(expected_unique * config.training.replicates).all()
        assert (ratios["ratio_pairs"] <= ratios["model_workload_pairs"]).all()

        families = family_summary(metrics, BASELINE_REGIME)
        assert isinstance(families, pd.DataFrame)
        selection = policy_selection_table(artifacts.evaluation.decisions)
        assert (selection["unique_workloads"] == config.evaluation.repetitions).all()
        assert (selection["model_decisions"] == config.evaluation.repetitions * config.training.replicates).all()
        assert selection[[f"count_{name}" for name in ACTION_NAMES]].sum(axis=1).eq(
            selection["model_decisions"]
        ).all()
        occupancy = state_occupancy_table(artifacts.evaluation.decisions)
        assert (occupancy["distinct_states"] >= 1).all()
        best = best_policy_per_family(metrics)
        assert set(best["best_policy"]).issubset(set(ACTION_NAMES))

    def test_paired_ratio_matches_independent_calculation(self, tmp_path: Path) -> None:
        config = _small_config()
        artifacts = run_experiment(
            config, tmp_path / "results", tmp_path / "figures", make_figures=False
        )
        metrics = artifacts.evaluation.metrics
        ratios = selector_ratio_table(metrics)
        row = ratios.loc[("SJF", "avg_waiting_time")]
        baseline = metrics[
            (metrics["regime"] == BASELINE_REGIME) & (metrics["policy"] == "SJF")
        ].set_index(["training_seed", "family", "repetition"])["avg_waiting_time"]
        selector = metrics[metrics["regime"] == SELECTOR_REGIME].set_index(
            ["training_seed", "family", "repetition"]
        )["avg_waiting_time"]
        paired = baseline.to_frame("baseline").join(selector.to_frame("selector"))
        usable = paired[paired["selector"] != 0.0]
        assert row["model_workload_pairs"] == len(paired)
        assert row["unique_workloads"] == len(config.families) * config.evaluation.repetitions
        assert row["ratio_pairs"] == len(usable)
        assert row["mean_ratio"] == pytest.approx(
            float((usable["baseline"] / usable["selector"]).mean()), rel=1e-9
        )
        assert row["mean_difference"] == pytest.approx(
            float((paired["baseline"] - paired["selector"]).mean()), rel=1e-9
        )

    def test_unknown_or_empty_frames_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            policy_selection_table(pd.DataFrame())
        with pytest.raises(ValidationError):
            policy_summary(pd.DataFrame())


class TestEndToEnd:
    def test_writes_generated_artifacts_and_six_core_figures(self, tmp_path: Path) -> None:
        config = _small_config()
        artifacts = run_experiment(
            config,
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            make_figures=True,
        )
        for name in (
            "config",
            "training_history",
            "q_table",
            "workloads",
            "metrics",
            "decisions",
            "summary",
            "report",
            "figures",
        ):
            if name == "figures":
                assert len(artifacts.paths[name]) == 6
                assert all(path.exists() for path in artifacts.paths[name])
            else:
                assert artifacts.paths[name].exists(), name
        assert not (tmp_path / "figures" / "round_robin_quantum.png").exists()

        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        workloads = pd.read_csv(tmp_path / "results" / "workloads.csv")
        metrics = pd.read_csv(tmp_path / "results" / "metrics.csv")
        decisions = pd.read_csv(tmp_path / "results" / "decisions.csv")
        report = (tmp_path / "results" / "report.md").read_text()
        assert summary["evaluation"]["observed_unique_workloads"] == len(workloads)
        assert summary["evaluation"]["selector_decision_rows"] == len(decisions)
        assert len(metrics) == summary["evaluation"]["metric_rows"]
        assert "Unique workloads" in report
        assert str(len(workloads)) in report
        assert summary["software"]["requirements_lock_sha256"]
        assert config_snapshot(config)["state"]["q_table_shape"] == [81, 4]

    def test_repeated_runs_are_byte_deterministic_for_data_artifacts(self, tmp_path: Path) -> None:
        config = _small_config()
        for letter in ("a", "b"):
            run_experiment(
                config,
                results_dir=tmp_path / letter,
                figures_dir=tmp_path / f"f{letter}",
                make_figures=False,
            )
        for filename in (
            "config.json",
            "training_history.json",
            "q_table.json",
            "workloads.csv",
            "metrics.csv",
            "decisions.csv",
            "summary.json",
            "report.md",
        ):
            assert (tmp_path / "a" / filename).read_bytes() == (tmp_path / "b" / filename).read_bytes()

    def test_no_figures_path_does_not_import_plotting_or_matplotlib(self, tmp_path: Path, monkeypatch) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "visualization.plots" or name.startswith("matplotlib"):
                raise AssertionError(f"plotting import occurred on --no-figures path: {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        artifacts = run_experiment(
            _small_config(), tmp_path / "results", tmp_path / "figures", make_figures=False
        )
        assert "figures" not in artifacts.paths

    def test_cli_no_figures_and_training_use_offline_language(self, tmp_path, monkeypatch, capsys) -> None:
        import main as main_module

        monkeypatch.setattr(main_module, "build_default_config", _small_config)
        assert cli_main(
            ["experiment", "--no-figures", "--results-dir", str(tmp_path / "r")]
        ) == 0
        output = capsys.readouterr().out
        assert "unique workloads" in output
        assert (tmp_path / "r" / "report.md").exists()
        assert cli_main(["train"]) == 0
        assert "random-action branch rate" in capsys.readouterr().out

    def test_report_uses_generated_summary_values_not_manual_transcription(self, tmp_path) -> None:
        artifacts = run_experiment(
            _small_config(), tmp_path / "results", tmp_path / "figures", make_figures=False
        )
        report = Path(artifacts.paths["report"]).read_text()
        summary = artifacts.summary
        assert str(summary["evaluation"]["observed_unique_workloads"]) in report
        for row in summary["tables"]["policy_selection"]:
            assert str(row["model_decisions"]) in report
