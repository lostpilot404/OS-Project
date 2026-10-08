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
    oracle_agreement_table,
    policy_selection_table,
    policy_summary,
    state_occupancy_table,
)
from experiments.evaluate import (
    ADAPTIVE_REGIME,
    BASELINE_REGIME,
    build_evaluation_workloads,
    evaluate,
)
from experiments.run_experiment import run_experiment
from experiments.train import train
from experiments.verify_classes import verify_classes
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
        evaluation=EvaluationConfig(repetitions=2, seed=99, verification_repetitions=3,
                                    verification_seed=123),
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
            if family.arrival_pattern in ("uniform", "batch_head"):
                assert all(p.arrival_time <= family.arrival_window for p in workload.processes)

    def test_pids_are_one_based_and_unique(self) -> None:
        generator = WorkloadGenerator(_small_config().families)
        workload = generator.generate("few_long", 7)
        assert sorted(p.pid for p in workload.processes) == list(range(1, len(workload) + 1))

    def test_short_job_family_has_lower_median_burst(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        short = generator.generate("short_batch", 3)
        long = generator.generate("long_batch", 3)
        assert short.median_burst_time < long.median_burst_time

    def test_skewed_family_has_lower_priority_spread(self) -> None:
        from rl.state import observe_workload_state

        generator = WorkloadGenerator(build_default_config().families)
        skewed = observe_workload_state(generator.generate("priority_skewed", 1))
        uniform = observe_workload_state(generator.generate("mixed", 1))
        assert skewed.priority_spread < uniform.priority_spread

    def test_interactive_family_has_higher_burst_dispersion(self) -> None:
        from rl.state import observe_workload_state

        generator = WorkloadGenerator(build_default_config().families)
        interactive = observe_workload_state(generator.generate("interactive", 3))
        homogeneous = observe_workload_state(generator.generate("mixed", 3))
        assert interactive.burst_dispersion > homogeneous.burst_dispersion

    def test_interactive_family_has_a_batch_head_of_long_jobs(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        family = generator.family("interactive")
        workload = generator.generate("interactive", 4)
        long_jobs = [p for p in workload.processes if p.burst_time >= family.long_mode_min()]
        short_jobs = [p for p in workload.processes if p.burst_time < family.long_mode_min()]
        assert long_jobs and short_jobs
        assert all(p.arrival_time <= family.head_window for p in long_jobs)
        assert all(p.arrival_time <= family.arrival_window for p in short_jobs)

    def test_batch_families_release_everything_together(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        for name in ("short_batch", "long_batch"):
            workload = generator.generate(name, 4)
            assert all(p.arrival_time == 0 for p in workload.processes)

    def test_aligned_family_tracks_burst_rank(self) -> None:
        generator = WorkloadGenerator(build_default_config().families)
        workload = generator.generate("priority_aligned", 2)
        by_burst = sorted(workload.processes, key=lambda p: (p.burst_time, p.pid))
        priorities = [p.priority for p in by_burst]
        assert priorities == sorted(priorities)

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

    def test_training_result_exposes_the_q_table_with_state_bins(self) -> None:
        result = train(_small_config())
        rows = result.q_table_rows()
        assert len(rows) == result.agent.n_states
        for row in rows:
            assert set(row["state_bins"]) == set(result.encoder.variables)
            assert row["greedy_action"] in ACTION_NAMES


class TestClassVerification:
    def test_verification_runs_before_training_and_is_deterministic(self) -> None:
        config = _small_config()
        generator = WorkloadGenerator(config.families)
        first = verify_classes(config, generator=generator)
        second = verify_classes(config, generator=generator)
        assert first.summary() == second.summary()
        assert first.workloads_total == len(config.families) * 3

    def test_verification_does_not_train(self) -> None:
        config = _small_config()
        result = verify_classes(config)
        # A verification result carries no agent and no Q-table.
        assert not hasattr(result, "agent")

    def test_verification_reports_a_modal_winner_per_class(self) -> None:
        config = _small_config()
        result = verify_classes(config)
        for entry in result.classes:
            assert entry.modal_winner in ACTION_NAMES
            assert entry.winner_shares[entry.modal_winner] > 0.0


class TestEvaluation:
    def test_every_scheduler_sees_the_same_workloads(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        grouped = result.metrics.groupby(["family", "repetition"])["workload_fingerprint"].nunique()
        assert (grouped == 1).all()

    def test_all_policies_and_regimes_are_measured(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        assert set(result.metrics["policy"]) == set(ACTION_NAMES) | {ADAPTIVE_LABEL}
        assert set(result.metrics["regime"]) == {BASELINE_REGIME, ADAPTIVE_REGIME}

    def test_row_count_matches_the_protocol(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        workloads = len(config.families) * config.evaluation.repetitions
        # four baselines + the adaptive scheduler
        assert len(result.metrics) == workloads * 5
        assert len(result.decisions) == workloads

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

    def test_adaptive_rows_equal_the_chosen_baseline(self) -> None:
        config = _small_config()
        training = train(config)
        result = evaluate(config, training)
        baselines = result.metrics[result.metrics["regime"] == BASELINE_REGIME]
        adaptive_decisions = result.decisions[result.decisions["regime"] == ADAPTIVE_REGIME]
        assert not adaptive_decisions.empty
        for decision in adaptive_decisions.itertuples():
            chosen = ACTION_NAMES[decision.action]
            baseline_row = baselines[
                (baselines["family"] == decision.family)
                & (baselines["repetition"] == decision.repetition)
                & (baselines["policy"] == chosen)
            ].iloc[0]
            assert decision.adaptive_avg_waiting_time == pytest.approx(baseline_row["avg_waiting_time"])

    def test_decisions_carry_the_reward_argmax_comparison(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        for column in ("oracle_action", "oracle_policy_name", "oracle_reward",
                       "chosen_minus_oracle_reward", "matches_oracle"):
            assert column in result.decisions.columns
        assert set(result.decisions["oracle_policy_name"]).issubset(set(ACTION_NAMES))
        # The oracle action is the reward-argmax over the four baselines measured on the
        # same workload, so the chosen action's reward can never exceed the oracle's by
        # more than numerical noise.
        gaps = result.decisions["chosen_minus_oracle_reward"]
        assert (gaps <= 1e-9).all()

    def test_state_coverage_is_reported(self) -> None:
        config = _small_config()
        result = evaluate(config, train(config))
        summary = result.summary()
        assert summary["distinct_states_evaluated"] >= 1
        assert summary["evaluated_states_visited_during_training"] <= (
            summary["distinct_states_evaluated"]
        )
        assert "state_visit_count" in result.decisions.columns

    def test_zero_repetitions_is_rejected_by_configuration(self) -> None:
        with pytest.raises(ConfigurationError):
            EvaluationConfig(repetitions=0)

    def test_equal_seeds_are_rejected_by_the_configuration(self) -> None:
        config = _small_config()
        broken = replace(config, evaluation=replace(config.evaluation, seed=config.training.seed))
        with pytest.raises(ConfigurationError, match="seeds must all differ"):
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
        baselines = policy_summary(result.metrics, BASELINE_REGIME)
        assert set(baselines.index) == set(ACTION_NAMES)
        assert set(baselines.columns) == set(ALL_METRICS)

        adaptive = policy_summary(result.metrics, ADAPTIVE_REGIME)
        assert list(adaptive.index) == [ADAPTIVE_LABEL]

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
        assert (occupancy["states_visited_in_training"] <= occupancy["distinct_states"]).all()

        oracle = oracle_agreement_table(result.decisions)
        assert (oracle["workloads"] == config.evaluation.repetitions).all()
        assert len(oracle) == len(config.families)
        assert ((oracle["matches"] >= 0) & (oracle["matches"] <= oracle["workloads"])).all()
        assert ((oracle["agreement_rate"] >= 0.0) & (oracle["agreement_rate"] <= 1.0)).all()

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
            result.metrics["regime"] == ADAPTIVE_REGIME
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
        for name in ("config", "class_verification", "training_history", "q_table", "workloads",
                     "metrics", "decisions", "summary"):
            assert artifacts.paths[name].exists(), name
        assert len(artifacts.paths["figures"]) == 6
        assert all(path.exists() for path in artifacts.paths["figures"])

        payload = json.loads((tmp_path / "results" / "summary.json").read_text())
        assert payload["configuration"]["training"]["episodes"] == config.training.episodes
        assert payload["evaluation"]["distinct_workloads"] == (
            len(config.families) * config.evaluation.repetitions
        )
        # The verification section is part of the summary and precedes training.
        assert payload["verification"]["repetitions_per_family"] == 3
        assert payload["verification"]["workloads_total"] == len(config.families) * 3

        verification_payload = json.loads(
            (tmp_path / "results" / "class_verification.json").read_text()
        )
        assert len(verification_payload["classes"]) == len(config.families)

    def test_config_snapshot_has_no_quantum_controller(self, tmp_path: Path) -> None:
        config = _small_config()
        run_experiment(
            config, results_dir=tmp_path / "results", figures_dir=tmp_path / "figures",
            make_figures=False,
        )
        payload = json.loads((tmp_path / "results" / "config.json").read_text())
        assert "quantum_controller" not in payload
        assert "quantum_controller" not in payload["q_learning"]
        assert payload["scheduler"]["round_robin_quantum"] == 4
        assert payload["state"]["long_burst_threshold"] == 16
        assert payload["evaluation"]["verification_seed"] == 123

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
        assert (tmp_path / "a" / "class_verification.json").read_text() == (
            tmp_path / "b" / "class_verification.json"
        ).read_text()
        assert first.summary["training"] == second.summary["training"]
        assert first.summary["verification"] == second.summary["verification"]

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
        assert "Pre-training class verification" in output
        assert "Evaluation" in output
        assert "Policy selected by the agent" in output
        assert (tmp_path / "results" / "metrics.csv").exists()
        assert (tmp_path / "results" / "class_verification.json").exists()
        assert (tmp_path / "figures" / "metric_comparison.png").exists()
