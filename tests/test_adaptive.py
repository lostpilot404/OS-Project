"""The adaptive scheduler: learning, greedy evaluation and controller wiring."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import (
    ACTION_FCFS,
    ACTION_NAMES,
    ACTION_ROUND_ROBIN,
    ExperimentConfig,
    QLearningConfig,
    QuantumControllerConfig,
    RewardConfig,
    SchedulerConfig,
    StateConfig,
    TrainingConfig,
    EvaluationConfig,
    WorkloadFamilyConfig,
    derive_seed,
)
from errors import ValidationError
from rl.adaptive import AdaptiveScheduler
from rl.q_learning import QLearningAgent
from rl.quantum_controller import QuantumController
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from tests.helpers import workload_from_rows
from workload.models import Workload


def _config() -> ExperimentConfig:
    """A small, fast, fully specified experiment configuration for the tests."""
    family = WorkloadFamilyConfig(name="tiny", description="test family", num_processes=4)
    return ExperimentConfig(
        name="test",
        scheduler=SchedulerConfig(),
        families=(family,),
        state=StateConfig(),
        reward=RewardConfig(),
        q_learning=QLearningConfig(),
        quantum_controller=QuantumControllerConfig(),
        training=TrainingConfig(episodes=3, seed=7, family_cycle=("tiny",)),
        evaluation=EvaluationConfig(repetitions=1, seed=11),
    )


def _workload(name: str = "tiny", rows=None) -> Workload:
    """A deterministic workload for the tests."""
    rows = rows or [(1, 0, 5, 2), (2, 0, 3, 1), (3, 1, 7, 3), (4, 4, 2, 2)]
    return workload_from_rows(name, rows)


def _scheduler(
    config: ExperimentConfig | None = None,
    *,
    with_controller: bool = True,
    learning_rate: float = 0.1,
    agent_seed: int = 0,
    controller_seed: int = 0,
) -> AdaptiveScheduler:
    """Build an adaptive scheduler with a fixed agent and, optionally, a controller."""
    config = config or _config()
    if not with_controller:
        # Disabling the controller is part of the configuration, exactly as a user would
        # do it; the adaptive scheduler refuses a controller that the config disables.
        config = replace(
            config, quantum_controller=replace(config.quantum_controller, enabled=False)
        )
    encoder = StateEncoder(config.state)
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        QLearningConfig(learning_rate=learning_rate, initial_value=config.q_learning.initial_value),
        seed=agent_seed,
        initial_value=config.q_learning.initial_value,
    )
    controller = None
    if with_controller:
        controller = QuantumController(encoder, config.quantum_controller, seed=controller_seed)
    policies = [policy_class(config.scheduler) for policy_class in POLICY_CLASSES]
    return AdaptiveScheduler(config, encoder, agent, policies, controller)


class TestConstruction:
    def test_policies_must_cover_the_four_actions(self) -> None:
        config = _config()
        encoder = StateEncoder(config.state)
        agent = QLearningAgent(encoder.n_states, 4, config.q_learning, seed=0)
        policies = [POLICY_CLASSES[0](config.scheduler), POLICY_CLASSES[1](config.scheduler)]
        with pytest.raises(ValidationError, match="four actions"):
            AdaptiveScheduler(config, encoder, agent, policies)

    def test_agent_and_encoder_must_agree_on_the_state_space(self) -> None:
        config = _config()
        encoder = StateEncoder(StateConfig(state_variables=("burst_profile",)))
        agent = QLearningAgent(StateEncoder(StateConfig()).n_states, 4, config.q_learning, seed=0)
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        with pytest.raises(ValidationError, match="states but the encoder"):
            AdaptiveScheduler(config, encoder, agent, policies)

    def test_controller_is_required_when_enabled(self) -> None:
        config = _config()
        encoder = StateEncoder(config.state)
        agent = QLearningAgent(encoder.n_states, 4, config.q_learning, seed=0)
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        with pytest.raises(ValidationError, match="none was supplied"):
            AdaptiveScheduler(config, encoder, agent, policies, None)

    def test_controller_configuration_must_match(self) -> None:
        config = _config()
        encoder = StateEncoder(config.state)
        agent = QLearningAgent(encoder.n_states, 4, config.q_learning, seed=0)
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        other = QuantumControllerConfig(multipliers=(0.25, 1.0))
        controller = QuantumController(encoder, other, seed=0)
        with pytest.raises(ValidationError, match="different configuration"):
            AdaptiveScheduler(config, encoder, agent, policies, controller)


class TestTrainingEpisode:
    def test_training_updates_the_q_table(self) -> None:
        scheduler = _scheduler(with_controller=False)
        before = np.array(scheduler.agent.q_table, copy=True)
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert decision.learned is True
        assert not np.array_equal(before, np.array(scheduler.agent.q_table))
        assert scheduler.agent.visit_counts[decision.state_index] == 1

    def test_decision_reports_the_chosen_policy_and_its_metrics(self) -> None:
        scheduler = _scheduler(with_controller=False)
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert decision.policy_name == ACTION_NAMES[decision.action]
        assert 0 <= decision.state_index < scheduler.encoder.n_states
        assert decision.metrics.policy_name == ACTION_NAMES[decision.action]

    def test_every_conventional_policy_was_run_on_the_same_workload(self) -> None:
        scheduler = _scheduler(with_controller=False)
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert sorted(decision.reference_metrics) == [0, 1, 2, 3]
        assert decision.metrics.workload_fingerprint == decision.workload_fingerprint
        assert all(
            metrics.workload_fingerprint == decision.workload_fingerprint
            for metrics in decision.reference_metrics.values()
        )

    def test_exploration_flag_follows_the_epsilon_schedule(self) -> None:
        scheduler = _scheduler(with_controller=False)
        # With epsilon = 1.0 every episode explores, so the flag reflects the draw.
        decision = scheduler.run_training_episode(_workload(), episode=0)
        assert decision.epsilon == pytest.approx(1.0)
        greedy = scheduler.agent.greedy_action(decision.state_index)
        assert decision.explored == (decision.action != greedy)


class TestEvaluationEpisode:
    def test_evaluation_never_updates_the_q_table(self) -> None:
        scheduler = _scheduler()
        scheduler.run_training_episode(_workload(), episode=0)
        before = np.array(scheduler.agent.q_table, copy=True)
        before_controller = np.array(scheduler.quantum_controller.agent.q_table, copy=True)
        visits_before = np.array(scheduler.agent.visit_counts, copy=True)
        decision = scheduler.run_evaluation(_workload())
        assert decision.learned is False
        assert np.array_equal(before, np.array(scheduler.agent.q_table))
        assert np.array_equal(before_controller, np.array(scheduler.quantum_controller.agent.q_table))
        assert np.array_equal(visits_before, np.array(scheduler.agent.visit_counts))

    def test_evaluation_is_greedy_and_reports_no_exploration(self) -> None:
        scheduler = _scheduler()
        decision = scheduler.run_evaluation(_workload())
        assert decision.epsilon == 0.0
        assert decision.explored is False
        assert decision.action == scheduler.agent.greedy_action(decision.state_index)

    def test_evaluation_action_comes_from_the_q_table(self) -> None:
        # With an untouched table every action has the same value, so the tie-break gives
        # action 0; after rewarding action 2 for this state, the choice must change.
        scheduler = _scheduler(with_controller=False)
        untrained = scheduler.run_evaluation(_workload())
        assert untrained.action == ACTION_FCFS
        scheduler.agent.update(untrained.state_index, ACTION_ROUND_ROBIN, 5.0, next_state=None)
        retrained = scheduler.run_evaluation(_workload())
        assert retrained.action == ACTION_ROUND_ROBIN

    def test_repeated_evaluation_gives_identical_results(self) -> None:
        scheduler = _scheduler()
        first = scheduler.run_evaluation(_workload())
        second = scheduler.run_evaluation(_workload())
        assert first == second

    def test_learned_quantum_can_be_requested(self) -> None:
        scheduler = _scheduler()
        decision = scheduler.run_evaluation(_workload(), use_learned_quantum=True)
        assert decision.learned is False
        # Without training, the controller falls back to classic Round Robin.
        assert decision.controller_multiplier == pytest.approx(1.0)

    def test_learned_quantum_is_rejected_without_a_controller(self) -> None:
        scheduler = _scheduler(with_controller=False)
        with pytest.raises(ValidationError, match="controller is disabled"):
            scheduler.run_evaluation(_workload(), use_learned_quantum=True)


class TestChosenPolicyMetrics:
    def test_the_adaptive_result_equals_the_chosen_baseline_result(self) -> None:
        scheduler = _scheduler(with_controller=False)
        decision = scheduler.run_evaluation(_workload())
        chosen = decision.reference_metrics[decision.action]
        assert decision.metrics == chosen

    def test_reward_matches_the_reward_of_the_chosen_action(self) -> None:
        from rl.reward import compute_reward

        scheduler = _scheduler(with_controller=False)
        decision = scheduler.run_evaluation(_workload())
        expected = compute_reward(
            scheduler.config.reward, decision.reference_metrics, decision.action
        ).reward
        assert decision.reward == pytest.approx(expected)


class TestQuantumControllerWiring:
    def _forced_round_robin_scheduler(self) -> AdaptiveScheduler:
        """Build a scheduler whose agent always prefers the Round-Robin action."""
        config = _config()
        scheduler = _scheduler(
            config, with_controller=True, learning_rate=1.0, agent_seed=3, controller_seed=5
        )
        for state in range(scheduler.encoder.n_states):
            scheduler.agent.update(state, ACTION_ROUND_ROBIN, 10.0, next_state=None)
        return scheduler

    def test_training_updates_the_controller(self) -> None:
        scheduler = self._forced_round_robin_scheduler()
        before = np.array(scheduler.quantum_controller.agent.q_table, copy=True)
        scheduler.run_training_episode(_workload(), episode=1)
        assert not np.array_equal(before, np.array(scheduler.quantum_controller.agent.q_table))

    def test_learned_quantum_replaces_the_round_robin_result(self) -> None:
        scheduler = self._forced_round_robin_scheduler()
        workload = _workload()
        expected_metrics, multiplier = scheduler.run_round_robin_with_learned_quantum(workload)
        decision = scheduler.run_evaluation(workload, use_learned_quantum=True)
        assert decision.action == ACTION_ROUND_ROBIN
        assert decision.quantum_multiplier == pytest.approx(multiplier)
        assert decision.metrics == expected_metrics

    def test_classic_regime_ignores_the_learned_quantum(self) -> None:
        scheduler = self._forced_round_robin_scheduler()
        workload = _workload()
        decision = scheduler.run_evaluation(workload, use_learned_quantum=False)
        assert decision.quantum_multiplier == pytest.approx(1.0)
        assert decision.metrics == decision.reference_metrics[ACTION_ROUND_ROBIN]

    def test_controller_rejects_an_unknown_multiplier(self) -> None:
        scheduler = _scheduler()
        with pytest.raises(ValidationError):
            scheduler.quantum_controller.multiplier_for_action(99)

    def test_direct_controller_measurement_requires_the_controller(self) -> None:
        scheduler = _scheduler(with_controller=False)
        with pytest.raises(ValidationError, match="disabled"):
            scheduler.run_round_robin_with_learned_quantum(_workload())


class TestReproducibilityOfDecisions:
    def test_identical_seeds_give_identical_training_runs(self) -> None:
        first = _scheduler()
        second = _scheduler()
        workload = _workload()
        for episode in range(3):
            d1 = first.run_training_episode(workload, episode)
            d2 = second.run_training_episode(workload, episode)
            assert (d1.action, d1.reward, d1.state_index) == (d2.action, d2.reward, d2.state_index)
        assert np.array_equal(np.array(first.agent.q_table), np.array(second.agent.q_table))

    def test_the_agent_seed_is_derived_deterministically(self) -> None:
        assert derive_seed(42, 1) == derive_seed(42, 1)
        assert derive_seed(42, 1) != derive_seed(42, 2)
        assert derive_seed(42, 1) != derive_seed(43, 1)
