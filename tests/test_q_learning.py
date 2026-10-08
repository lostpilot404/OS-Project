"""Tabular Q-learning: update arithmetic, action selection and validation."""

from __future__ import annotations

import numpy as np
import pytest

from config import QLearningConfig
from errors import ValidationError
from rl.q_learning import EpsilonSchedule, QLearningAgent


def _agent(
    n_states: int = 4,
    n_actions: int = 3,
    initial_value: float = 0.0,
    seed: int = 0,
    **config_overrides,
) -> QLearningAgent:
    """Build an agent with the given hyperparameters."""
    config = QLearningConfig(**{"learning_rate": 0.5, "discount_factor": 0.9, **config_overrides})
    return QLearningAgent(n_states, n_actions, config, seed=seed, initial_value=initial_value)


class TestInitialisation:
    def test_table_shape_and_initial_value(self) -> None:
        agent = _agent(initial_value=0.25)
        assert agent.q_table.shape == (4, 3)
        assert np.allclose(agent.q_table, 0.25)

    def test_visit_counts_start_at_zero(self) -> None:
        agent = _agent()
        assert np.array_equal(agent.visit_counts, np.zeros(4, dtype=np.int64))

    def test_q_table_is_read_only(self) -> None:
        agent = _agent()
        with pytest.raises(ValueError):
            agent.q_table[0, 0] = 1.0

    @pytest.mark.parametrize("n_states, n_actions", [(0, 3), (3, 0), (-1, 2)])
    def test_invalid_sizes_are_rejected(self, n_states: int, n_actions: int) -> None:
        with pytest.raises(ValidationError):
            _agent(n_states=n_states, n_actions=n_actions)


class TestUpdateArithmetic:
    def test_terminal_update_matches_the_hand_computation(self) -> None:
        # Q = 0, reward = 2, alpha = 0.5, terminal -> Q = 0 + 0.5 * (2 - 0) = 1.0
        agent = _agent()
        error = agent.update(state=1, action=2, reward=2.0, next_state=None)
        assert error == pytest.approx(2.0)
        assert agent.state_action_value(1, 2) == pytest.approx(1.0)

    def test_discounted_update_matches_the_hand_computation(self) -> None:
        # alpha = 0.5, gamma = 0.9.
        # Step 1: Q(0, 1) <- 0 + 0.5 * (1 + 0) = 0.5           (terminal, reward 1)
        # Step 2: Q(1, 0) <- 0 + 0.5 * (2 + 0) = 1.0           (terminal, reward 2)
        # Step 3: target = 0 + 0.9 * max(Q(1)) = 0.9
        #         error  = 0.9 - 0.5 = 0.4 -> Q(0, 1) = 0.5 + 0.5 * 0.4 = 0.7
        agent = _agent()
        agent.update(0, 1, 1.0, next_state=None)
        agent.update(1, 0, 2.0, next_state=None)
        error = agent.update(0, 1, 0.0, next_state=1)
        assert error == pytest.approx(0.4)
        assert agent.state_action_value(0, 1) == pytest.approx(0.7)

    def test_greedy_next_state_value_is_used(self) -> None:
        # max over the next state's actions is the only value that enters the target.
        agent = _agent()
        agent.update(2, 0, 5.0, next_state=None)  # Q(2, 0) = 2.5
        agent.update(2, 2, 1.0, next_state=None)  # Q(2, 2) = 0.5
        agent.update(0, 0, 0.0, next_state=2)
        assert agent.state_action_value(0, 0) == pytest.approx(0.5 * 0.9 * 2.5)

    def test_visit_count_increases_once_per_update(self) -> None:
        agent = _agent()
        agent.update(3, 1, 1.0, next_state=None)
        agent.update(3, 0, 1.0, next_state=None)
        agent.update(0, 0, 1.0, next_state=None)
        assert agent.visit_counts[3] == 2
        assert agent.visit_counts[0] == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"state": 9, "action": 0, "reward": 1.0, "next_state": None},
            {"state": 0, "action": 9, "reward": 1.0, "next_state": None},
            {"state": 0, "action": 0, "reward": float("nan"), "next_state": None},
            {"state": 0, "action": 0, "reward": 1.0, "next_state": 42},
        ],
    )
    def test_invalid_updates_are_rejected(self, kwargs: dict) -> None:
        agent = _agent()
        with pytest.raises(ValidationError):
            agent.update(**kwargs)


class TestActionSelection:
    def test_greedy_action_breaks_ties_towards_the_lowest_index(self) -> None:
        agent = _agent(initial_value=0.0)
        assert agent.greedy_action(0) == 0

    def test_greedy_action_follows_the_table(self) -> None:
        agent = _agent()
        agent.update(1, 2, 3.0, next_state=None)
        assert agent.greedy_action(1) == 2

    def test_zero_epsilon_is_always_greedy(self) -> None:
        agent = _agent()
        agent.update(0, 1, 2.0, next_state=None)
        assert {agent.select_action(0, epsilon=0.0) for _ in range(20)} == {1}

    def test_full_epsilon_explores_every_action(self) -> None:
        agent = _agent()
        chosen = {agent.select_action(0, epsilon=1.0) for _ in range(200)}
        assert chosen == {0, 1, 2}

    def test_selection_is_reproducible_for_a_fixed_seed(self) -> None:
        first = _agent(seed=123)
        second = _agent(seed=123)
        assert [first.select_action(0, 0.7) for _ in range(30)] == [
            second.select_action(0, 0.7) for _ in range(30)
        ]

    def test_different_seeds_give_different_sequences(self) -> None:
        first = _agent(seed=1)
        second = _agent(seed=2)
        assert [first.select_action(0, 1.0) for _ in range(30)] != [
            second.select_action(0, 1.0) for _ in range(30)
        ]

    @pytest.mark.parametrize("epsilon", [-0.1, 1.1])
    def test_invalid_epsilon_is_rejected(self, epsilon: float) -> None:
        with pytest.raises(ValidationError, match="epsilon"):
            _agent().select_action(0, epsilon)

    def test_zero_epsilon_records_a_greedy_selection_not_exploration(self) -> None:
        agent = _agent()
        agent.update(0, 2, 5.0, next_state=None)
        assert agent.select_action(0, epsilon=0.0) == 2
        assert agent.last_action_was_random_exploration is False

    def test_random_branch_is_exploration_even_when_it_equals_greedy(self) -> None:
        agent = _agent(n_actions=1)
        assert agent.select_action(0, epsilon=1.0) == 0
        assert agent.greedy_action(0) == 0
        assert agent.last_action_was_random_exploration is True


class TestEpsilonSchedule:
    def test_decay_and_floor(self) -> None:
        schedule = EpsilonSchedule(start=1.0, minimum=0.1, decay_per_episode=0.5)
        assert schedule.epsilon_for_episode(0) == pytest.approx(1.0)
        assert schedule.epsilon_for_episode(1) == pytest.approx(0.5)
        assert schedule.epsilon_for_episode(2) == pytest.approx(0.25)
        assert schedule.epsilon_for_episode(3) == pytest.approx(0.125)
        assert schedule.epsilon_for_episode(4) == pytest.approx(0.1)
        assert schedule.epsilon_for_episode(1000) == pytest.approx(0.1)

    def test_agent_exposes_the_same_schedule(self) -> None:
        agent = _agent(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_per_episode=0.9)
        assert agent.epsilon_for_episode(0) == pytest.approx(1.0)
        assert agent.epsilon_for_episode(1) == pytest.approx(0.9)
        assert agent.epsilon_for_episode(50) == pytest.approx(0.05)

    def test_negative_episode_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EpsilonSchedule(1.0, 0.1, 0.9).epsilon_for_episode(-1)

    def test_invalid_schedule_values_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EpsilonSchedule(start=0.5, minimum=0.9, decay_per_episode=0.9)
        with pytest.raises(ValidationError):
            EpsilonSchedule(start=1.0, minimum=0.1, decay_per_episode=1.5)

    def test_epsilon_decays_to_the_configured_floor(self) -> None:
        config = QLearningConfig(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_per_episode=0.995)
        agent = QLearningAgent(2, 2, config, seed=0)
        assert agent.epsilon_for_episode(599) == pytest.approx(0.05)
