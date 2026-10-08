"""The pre-training class verification (experiments.verify_classes).

The verification answers, on held-out probe workloads and *before* any training,
which conventional policy each workload class actually favours under the declared
reward, and whether the reward-optimal action is a consistent function of the encoded
state.  These tests pin the structure of that measurement and the design property it
establishes for the default configuration.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from config import ACTION_NAMES, EvaluationConfig, ExperimentConfig, SchedulerConfig, StateConfig, TrainingConfig, WorkloadFamilyConfig, build_default_config
from errors import ConfigurationError
from experiments.verify_classes import verify_classes
from rl.state import StateEncoder
from workload.generator import WorkloadGenerator


def _small_config() -> ExperimentConfig:
    families = (
        WorkloadFamilyConfig(
            name="short", description="short jobs, batch", num_processes=6,
            burst_distribution="uniform", burst_time_min=1, burst_time_max=4,
            arrival_pattern="uniform", arrival_window=0,
        ),
        WorkloadFamilyConfig(
            name="interactive", description="long head plus short stream", num_processes=10,
            burst_distribution="bimodal", burst_time_min=1, burst_time_max=100,
            short_burst_max=2, short_burst_fraction=0.8, long_burst_min=60,
            arrival_pattern="batch_head", arrival_window=40, head_window=2,
        ),
    )
    config = ExperimentConfig(
        name="test_verify",
        scheduler=SchedulerConfig(),
        families=families,
        state=StateConfig(),
        training=TrainingConfig(episodes=4, seed=3, family_cycle=("short", "interactive")),
        evaluation=EvaluationConfig(
            repetitions=2, seed=10, verification_repetitions=3, verification_seed=99
        ),
    )
    config.validate()
    return config


class TestVerificationStructure:
    def test_one_entry_per_family_in_configuration_order(self) -> None:
        config = _small_config()
        result = verify_classes(config)
        assert [entry.family for entry in result.classes] == ["short", "interactive"]
        assert result.repetitions == 3
        assert result.seed == 99

    def test_winner_counts_and_shares_are_consistent(self) -> None:
        result = verify_classes(_small_config())
        for entry in result.classes:
            assert entry.workloads == 3
            assert sum(entry.winner_counts.values()) == entry.workloads
            for name in ACTION_NAMES:
                expected = entry.winner_counts.get(name, 0) / entry.workloads
                assert entry.winner_shares[name] == pytest.approx(expected)
            modal_count = max(entry.winner_counts.values())
            assert entry.winner_counts[entry.modal_winner] == modal_count

    def test_mean_rewards_are_reported_for_every_action(self) -> None:
        result = verify_classes(_small_config())
        for entry in result.classes:
            assert set(entry.mean_reward_per_action) == set(ACTION_NAMES)
            # The best action's mean reward is at least the mean of any other action
            # on the same class (the reward-argmax maximises the mean reward).
            best = max(entry.mean_reward_per_action.values())
            assert entry.mean_reward_per_action[entry.modal_winner] == pytest.approx(best)

    def test_states_are_reported_per_class(self) -> None:
        result = verify_classes(_small_config())
        for entry in result.classes:
            assert entry.distinct_states == len(entry.states)
            assert entry.states == tuple(sorted(entry.states))

    def test_summary_is_json_serialisable(self, tmp_path) -> None:
        result = verify_classes(_small_config())
        payload = result.summary()
        json.dumps(payload)
        written = result.to_json(tmp_path / "class_verification.json")
        assert written.exists()
        reloaded = json.loads(written.read_text())
        assert reloaded["repetitions_per_family"] == result.repetitions
        assert len(reloaded["classes"]) == len(result.classes)

    def test_verification_is_reproducible(self) -> None:
        config = _small_config()
        first = verify_classes(config)
        second = verify_classes(config)
        assert first.summary() == second.summary()

    def test_verification_never_touches_an_agent(self) -> None:
        # The verification is a pure measurement: it takes no agent and returns none.
        result = verify_classes(_small_config())
        assert not hasattr(result, "agent")


class TestStreamDisjointness:
    def test_probe_workloads_are_disjoint_from_training_and_evaluation(self) -> None:
        config = _small_config()
        generator = WorkloadGenerator(config.families)
        result = verify_classes(config, generator=generator)
        from experiments.evaluate import build_evaluation_workloads
        from experiments.train import train

        training = train(config, generator=generator)
        evaluation = build_evaluation_workloads(config, generator)
        probe = set(result.fingerprints)
        assert not (probe & training.training_fingerprints)
        assert not (probe & {w.fingerprint for w in evaluation})

    def test_verification_seed_must_differ_from_the_other_seeds(self) -> None:
        config = _small_config()
        broken = replace(
            config,
            evaluation=replace(
                config.evaluation, verification_seed=config.training.seed
            ),
        )
        with pytest.raises(ConfigurationError, match="seeds must all differ"):
            broken.validate()

    def test_verification_repetitions_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError):
            EvaluationConfig(verification_repetitions=0)


@pytest.fixture(scope="module")
def verification():
    """The pre-training verification of the default configuration."""
    config = build_default_config()
    generator = WorkloadGenerator(config.families)
    encoder = StateEncoder(config.state)
    return verify_classes(config, generator=generator, encoder=encoder)


class TestDefaultDesignProperty:
    """The design property the redesign is required to establish (task item 11).

    Before training, the classes must actually favour different policies: the default
    configuration's verification finds at least two distinct modal winners, the
    interactive classes favour Round Robin, the batch-like classes favour SJF, and the
    reward-optimal action is (almost always) a function of the encoded state.
    """

    def test_at_least_two_distinct_modal_winners(self, verification) -> None:
        assert len(verification.modal_winners) >= 2

    def test_interactive_classes_favour_round_robin(self, verification) -> None:
        by_family = {entry.family: entry for entry in verification.classes}
        for family in ("interactive", "interactive_sparse"):
            assert by_family[family].modal_winner == "Round Robin"
            assert by_family[family].winner_shares["Round Robin"] > 0.5

    def test_batch_like_classes_favour_sjf(self, verification) -> None:
        by_family = {entry.family: entry for entry in verification.classes}
        for family in (
            "short_batch",
            "short_stream",
            "long_batch",
            "priority_aligned",
            "priority_skewed",
            "quantum_sensitive",
            "mixed",
        ):
            assert by_family[family].modal_winner == "SJF"
            assert by_family[family].winner_shares["SJF"] > 0.9

    def test_state_conditional_consistency_is_high(self, verification) -> None:
        # Almost every probe workload's reward-argmax equals the modal argmax of its
        # encoded state, i.e. the state carries the information the agent needs.
        assert verification.state_conditional_consistency >= 0.90
        assert verification.states_with_conflicting_argmax <= 0.25 * verification.states_observed

    def test_every_class_occupies_at_least_one_state(self, verification) -> None:
        for entry in verification.classes:
            assert entry.distinct_states >= 1
