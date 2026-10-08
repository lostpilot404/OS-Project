"""Offline single-workload policy selection and read-only evaluation behavior."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import (
    ACTION_FCFS,
    ACTION_NAMES,
    ACTION_ROUND_ROBIN,
    EvaluationConfig,
    ExperimentConfig,
    QLearningConfig,
    RewardConfig,
    SchedulerConfig,
    StateConfig,
    TrainingConfig,
    WorkloadFamilyConfig,
)
from errors import ValidationError
from rl.adaptive import OfflinePolicySelector
from rl.q_learning import QLearningAgent
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from tests.helpers import workload_from_rows
from workload.models import Workload


def _config() -> ExperimentConfig:
    """A small, validated test configuration with one model seed."""
    family = WorkloadFamilyConfig(name="tiny", description="test family", num_processes=4)
    config = ExperimentConfig(
        name="test",
        scheduler=SchedulerConfig(),
        families=(family,),
        state=StateConfig(),
        reward=RewardConfig(),
        q_learning=QLearningConfig(),
        training=TrainingConfig(episodes=3, seed=7, replicates=1, family_cycle=("tiny",)),
        evaluation=EvaluationConfig(repetitions=1, seed=11),
    )
    config.validate()
    return config


def _workload(name: str = "tiny", rows=None) -> Workload:
    """Deterministic workload with nontrivial queueing and preemption opportunities."""
    rows = rows or [(1, 0, 5, 2), (2, 0, 3, 1), (3, 1, 7, 3), (4, 4, 2, 2)]
    return workload_from_rows(name, rows)


def _scheduler(
    config: ExperimentConfig | None = None,
    *,
    learning_rate: float = 0.1,
    agent_seed: int = 0,
) -> OfflinePolicySelector:
    """Build the four-policy offline selector."""
    config = config or _config()
    encoder = StateEncoder(config.state)
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        replace(config.q_learning, learning_rate=learning_rate),
        seed=agent_seed,
        initial_value=config.q_learning.initial_value,
    )
    policies = [policy_class(config.scheduler) for policy_class in POLICY_CLASSES]
    return OfflinePolicySelector(config, encoder, agent, policies)


class TestConstruction:
    def test_policies_must_cover_exactly_four_action_indices(self) -> None:
        config = _config()
        encoder = StateEncoder(config.state)
        agent = QLearningAgent(encoder.n_states, 4, config.q_learning, seed=0)
        policies = [POLICY_CLASSES[0](config.scheduler), POLICY_CLASSES[1](config.scheduler)]
        with pytest.raises(ValidationError, match="cover actions"):
            OfflinePolicySelector(config, encoder, agent, policies)

    def test_agent_table_dimensions_must_match_state_and_action_spaces(self) -> None:
        config = _config()
        encoder = StateEncoder(config.state)
        agent = QLearningAgent(encoder.n_states - 1, 4, config.q_learning, seed=0)
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        with pytest.raises(ValidationError, match="table dimensions"):
            OfflinePolicySelector(config, encoder, agent, policies)

    def test_encoder_must_match_the_configured_discretization(self) -> None:
        config = _config()
        encoder = StateEncoder(StateConfig(state_variables=("burst_profile",)))
        agent = QLearningAgent(encoder.n_states, 4, config.q_learning, seed=0)
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        with pytest.raises(ValidationError, match="encoder configuration"):
            OfflinePolicySelector(config, encoder, agent, policies)


class TestTrainingEpisode:
    def test_one_terminal_episode_updates_only_the_selected_state_action(self) -> None:
        scheduler = _scheduler()
        before = np.array(scheduler.agent.q_table, copy=True)
        decision = scheduler.run_training_episode(_workload(), episode=0)
        after = np.array(scheduler.agent.q_table)
        assert decision.learned is True
        assert decision.state_seen_in_training is False
        assert scheduler.agent.visit_counts[decision.state_index] == 1
        changed = np.argwhere(before != after)
        assert changed.tolist() == [[decision.state_index, decision.action]]

    def test_decision_reports_greedy_action_and_selected_policy_metrics(self) -> None:
        scheduler = _scheduler()
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert decision.policy_name == ACTION_NAMES[decision.action]
        assert decision.greedy_policy_name == ACTION_NAMES[decision.greedy_action]
        assert 0 <= decision.state_index < scheduler.encoder.n_states
        assert decision.metrics.policy_name == ACTION_NAMES[decision.action]

    def test_all_four_references_use_the_same_workload_fingerprint(self) -> None:
        scheduler = _scheduler()
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert sorted(decision.reference_metrics) == [0, 1, 2, 3]
        assert decision.metrics.workload_fingerprint == decision.workload_fingerprint
        assert all(
            metrics.workload_fingerprint == decision.workload_fingerprint
            for metrics in decision.reference_metrics.values()
        )

    def test_explored_means_the_random_branch_was_used(self) -> None:
        scheduler = _scheduler()
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert decision.epsilon == pytest.approx(1.0)
        assert decision.explored is True
        assert scheduler.agent.last_action_was_random_exploration is True

    def test_invalid_episode_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="episode"):
            _scheduler().run_training_episode(_workload(), episode=-1)


class TestEvaluationEpisode:
    def test_evaluation_never_updates_q_values_or_visit_counts(self) -> None:
        scheduler = _scheduler()
        scheduler.run_training_episode(_workload(), episode=0)
        before_q = np.array(scheduler.agent.q_table, copy=True)
        before_visits = np.array(scheduler.agent.visit_counts, copy=True)
        decision = scheduler.run_evaluation(_workload())
        assert decision.learned is False
        assert np.array_equal(before_q, np.array(scheduler.agent.q_table))
        assert np.array_equal(before_visits, np.array(scheduler.agent.visit_counts))
        assert decision.state_seen_in_training is True

    def test_evaluation_is_greedy_deterministic_and_reports_no_exploration(self) -> None:
        scheduler = _scheduler()
        first = scheduler.run_evaluation(_workload())
        second = scheduler.run_evaluation(_workload())
        assert first == second
        assert first.epsilon == 0.0
        assert first.explored is False
        assert first.action == scheduler.agent.greedy_action(first.state_index)
        assert first.state_seen_in_training is False

    def test_unseen_zero_initialized_state_has_documented_lowest_index_fallback(self) -> None:
        decision = _scheduler().run_evaluation(_workload())
        assert decision.action == ACTION_FCFS
        assert decision.greedy_action == ACTION_FCFS
        assert decision.state_seen_in_training is False

    def test_trained_q_values_determine_the_evaluation_action(self) -> None:
        scheduler = _scheduler()
        initial = scheduler.run_evaluation(_workload())
        scheduler.agent.update(initial.state_index, ACTION_ROUND_ROBIN, 5.0, next_state=None)
        learned = scheduler.run_evaluation(_workload())
        assert learned.action == ACTION_ROUND_ROBIN
        assert learned.state_seen_in_training is True

    def test_evaluation_rejects_non_workload_input(self) -> None:
        with pytest.raises(ValidationError, match="Workload"):
            _scheduler().run_evaluation("not a workload")  # type: ignore[arg-type]


class TestRewardAndMetrics:
    def test_selected_result_is_one_of_the_four_baseline_results(self) -> None:
        decision = _scheduler().run_evaluation(_workload())
        assert decision.metrics == decision.reference_metrics[decision.action]

    def test_decision_reward_matches_the_shared_four_policy_reward(self) -> None:
        from rl.reward import compute_reward

        scheduler = _scheduler()
        decision = scheduler.run_evaluation(_workload())
        expected = compute_reward(
            scheduler.config.reward, decision.reference_metrics, decision.action
        ).reward
        assert decision.reward == pytest.approx(expected)

    def test_decision_row_exposes_unseen_state_and_offline_action_metadata(self) -> None:
        row = _scheduler().run_evaluation(_workload()).as_row()
        assert "state_seen_in_training" in row
        assert "greedy_policy_name" in row


class TestReproducibility:
    def test_same_exploration_seed_gives_identical_training_decisions(self) -> None:
        first = _scheduler(agent_seed=12)
        second = _scheduler(agent_seed=12)
        workload = _workload()
        for episode in range(3):
            d1 = first.run_training_episode(workload, episode)
            d2 = second.run_training_episode(workload, episode)
            assert (d1.action, d1.reward, d1.state_index, d1.explored) == (
                d2.action,
                d2.reward,
                d2.state_index,
                d2.explored,
            )
        assert np.array_equal(np.array(first.agent.q_table), np.array(second.agent.q_table))

    def test_two_training_seeds_are_independent_streams(self) -> None:
        first = _scheduler(agent_seed=1)
        second = _scheduler(agent_seed=2)
        workload = _workload()
        first_actions = [first.run_training_episode(workload, i).action for i in range(3)]
        second_actions = [second.run_training_episode(workload, i).action for i in range(3)]
        assert first_actions != second_actions or not np.array_equal(
            first.agent.q_table, second.agent.q_table
        )
