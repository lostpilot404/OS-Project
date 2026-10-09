"""Reusable tabular Q-learning agent for offline and sequential scheduling.

The legacy offline selector supplies terminal one-step transitions, while the runtime
controller supplies nonterminal transitions so the configured discount factor affects
future-state targets. State and state-action visits are tracked separately; runtime callers
can mask unvisited actions without changing the legacy agent defaults. Epsilon-greedy
action selection is deterministic at epsilon zero, with ties resolved by lowest index.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from config import QLearningConfig
from errors import ValidationError

__all__ = ["QLearningAgent", "EpsilonSchedule"]


class EpsilonSchedule:
    """Multiplicative epsilon decay, floored at a minimum.

    ``epsilon(episode) = max(epsilon_min, epsilon_start * decay ** episode)``
    """

    def __init__(self, start: float, minimum: float, decay_per_episode: float) -> None:
        """
        Args:
            start: Exploration rate at episode 0.
            minimum: Lower bound for the exploration rate.
            decay_per_episode: Multiplicative decay applied once per episode.

        Raises:
            ValidationError: If the values are outside their documented ranges.
        """
        if not 0.0 <= minimum <= start <= 1.0:
            raise ValidationError(
                f"require 0 <= minimum <= start <= 1, got {minimum}, {start}"
            )
        if not 0.0 < decay_per_episode <= 1.0:
            raise ValidationError(
                f"decay_per_episode must lie in (0, 1], got {decay_per_episode}"
            )
        self._start = float(start)
        self._minimum = float(minimum)
        self._decay = float(decay_per_episode)

    @property
    def start(self) -> float:
        """Exploration rate at episode 0."""
        return self._start

    @property
    def minimum(self) -> float:
        """Lower bound of the exploration rate."""
        return self._minimum

    def epsilon_for_episode(self, episode: int) -> float:
        """Return the exploration rate used in ``episode`` (0-based).

        Raises:
            ValidationError: If ``episode`` is negative.
        """
        if episode < 0:
            raise ValidationError(f"episode must be >= 0, got {episode}")
        return max(self._minimum, self._start * self._decay**episode)


class QLearningAgent:
    """A tabular Q-learning agent over a finite state space.

    Attributes are kept private; the learned table is exposed through
    :attr:`q_table` (a read-only view) and :meth:`state_action_value`.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: QLearningConfig,
        seed: int,
        initial_value: float = 0.0,
    ) -> None:
        """
        Args:
            n_states: Size of the discrete state space.
            n_actions: Size of the discrete action space.
            config: Learning-rate, discount-factor and exploration settings.
            seed: Seed of the agent's exploration random number generator.
            initial_value: Neutral starting value for every Q-entry. The default is zero;
                epsilon-greedy exploration, rather than optimistic initialization, samples
                other actions.

        Raises:
            ValidationError: If the table shape or the configuration is invalid.
        """
        if n_states < 1 or n_actions < 1:
            raise ValidationError(
                f"n_states and n_actions must be >= 1, got {n_states}, {n_actions}"
            )
        if not isinstance(config, QLearningConfig):
            raise ValidationError(f"config must be a QLearningConfig, got {type(config).__name__}")
        if not np.isfinite(initial_value):
            raise ValidationError(f"initial_value must be finite, got {initial_value}")
        self._config = config
        self._n_states = int(n_states)
        self._n_actions = int(n_actions)
        self._q_table = np.full((self._n_states, self._n_actions), float(initial_value))
        self._visit_counts = np.zeros(self._n_states, dtype=np.int64)
        self._state_action_visit_counts = np.zeros(
            (self._n_states, self._n_actions), dtype=np.int64
        )
        self._rng = np.random.default_rng(seed)
        self._epsilon = EpsilonSchedule(
            config.epsilon_start, config.epsilon_min, config.epsilon_decay_per_episode
        )
        self._last_action_was_random_exploration = False

    # -- introspection ---------------------------------------------------------------
    @property
    def config(self) -> QLearningConfig:
        """The agent's hyperparameters."""
        return self._config

    @property
    def n_states(self) -> int:
        """Size of the state space."""
        return self._n_states

    @property
    def n_actions(self) -> int:
        """Size of the action space."""
        return self._n_actions

    @property
    def q_table(self) -> np.ndarray:
        """A read-only view of the Q-table with shape ``(n_states, n_actions)``."""
        view = self._q_table.view()
        view.flags.writeable = False
        return view

    @property
    def visit_counts(self) -> np.ndarray:
        """Read-only view of how often each state has been updated."""
        view = self._visit_counts.view()
        view.flags.writeable = False
        return view

    @property
    def state_action_visit_counts(self) -> np.ndarray:
        """Read-only ``(state, action)`` counts for transitions used in learning."""
        view = self._state_action_visit_counts.view()
        view.flags.writeable = False
        return view

    def visited_actions(self, state: int) -> tuple[int, ...]:
        """Return actions tried in ``state``, ordered by action index."""
        self._check_state(state)
        return tuple(
            int(action)
            for action in np.flatnonzero(self._state_action_visit_counts[state] > 0)
        )

    def greedy_visited_action(self, state: int) -> Optional[int]:
        """Return the highest-valued tried action, or ``None`` if none was tried.

        This is intentionally different from :meth:`greedy_action`: runtime evaluation
        must not silently treat the initial value of an unvisited action as evidence.
        """
        actions = self.visited_actions(state)
        if not actions:
            return None
        return max(actions, key=lambda action: (self._q_table[state, action], -action))

    def state_action_value(self, state: int, action: int) -> float:
        """Return ``Q(state, action)``.

        Raises:
            ValidationError: If the state or action is out of range.
        """
        self._check_state(state)
        self._check_action(action)
        return float(self._q_table[state, action])

    def best_value(self, state: int) -> float:
        """Return ``max_a Q(state, a)``."""
        self._check_state(state)
        return float(self._q_table[state].max())

    def greedy_action(self, state: int) -> int:
        """Return the action with the highest Q-value, ties going to the lowest index."""
        self._check_state(state)
        return int(np.argmax(self._q_table[state]))

    # -- acting ----------------------------------------------------------------------
    def select_action(
        self,
        state: int,
        epsilon: float,
        greedy_actions: Optional[Sequence[int]] = None,
    ) -> int:
        """Choose an action with epsilon-greedy exploration.

        By default the greedy branch considers the full action space, preserving the
        legacy selector's behavior. Runtime callers pass the actions already tried in
        this state; random exploration still samples the full action space.

        Args:
            state: The current state index.
            epsilon: Probability of choosing a uniformly random action instead of the
                greedy one.
            greedy_actions: Optional subset used only for the greedy branch. This lets a
                runtime controller avoid treating unvisited action values as evidence.

        Returns:
            The chosen action index.

        Raises:
            ValidationError: If the state or epsilon is invalid.
        """
        self._check_state(state)
        if not 0.0 <= epsilon <= 1.0:
            raise ValidationError(f"epsilon must lie in [0, 1], got {epsilon}")
        candidates = tuple(range(self._n_actions)) if greedy_actions is None else tuple(
            int(action) for action in greedy_actions
        )
        if not candidates:
            raise ValidationError("greedy_actions must not be empty")
        for action in candidates:
            self._check_action(action)
        if epsilon > 0.0 and self._rng.random() < epsilon:
            self._last_action_was_random_exploration = True
            return int(self._rng.integers(0, self._n_actions))
        self._last_action_was_random_exploration = False
        return max(candidates, key=lambda action: (self._q_table[state, action], -action))

    @property
    def last_action_was_random_exploration(self) -> bool:
        """Whether the most recent selection used epsilon's random-action branch.

        A random draw can happen to equal the greedy action; it still counts as
        exploration because the action was sampled uniformly rather than selected greedily.
        """
        return self._last_action_was_random_exploration

    def epsilon_for_episode(self, episode: int) -> float:
        """Exploration rate of the given training episode (0-based)."""
        return self._epsilon.epsilon_for_episode(episode)

    # -- learning --------------------------------------------------------------------
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: Optional[int] = None,
        mask_unvisited_next_actions: bool = False,
    ) -> float:
        """Apply one Q-learning update and return the temporal-difference error.

        Args:
            state: State in which the action was taken.
            action: Action that was taken.
            reward: Reward received.
            next_state: State reached afterwards, or ``None`` for a terminal transition,
                in which case no discounted future value is added.
            mask_unvisited_next_actions: When true, bootstrap only from actions already
                tried in ``next_state``. This is used by runtime learning so zero-valued
                placeholders for unseen actions are not treated as estimates. The legacy
                offline learner keeps its original all-actions default.

        Returns:
            The temporal-difference error ``target - Q(state, action)`` *before* the
            update, which makes the arithmetic checkable in tests.

        Raises:
            ValidationError: If any argument is out of range or not finite.
        """
        self._check_state(state)
        self._check_action(action)
        if not np.isfinite(reward):
            raise ValidationError(f"reward must be finite, got {reward}")
        if next_state is not None:
            self._check_state(next_state)

        if not isinstance(mask_unvisited_next_actions, bool):
            raise ValidationError("mask_unvisited_next_actions must be a bool")
        target = reward
        if next_state is not None:
            if mask_unvisited_next_actions:
                known = self.visited_actions(next_state)
                next_value = (
                    max(float(self._q_table[next_state, candidate]) for candidate in known)
                    if known
                    else 0.0
                )
            else:
                next_value = self.best_value(next_state)
            target += self._config.discount_factor * next_value
        error = target - float(self._q_table[state, action])
        self._q_table[state, action] += self._config.learning_rate * error
        self._visit_counts[state] += 1
        self._state_action_visit_counts[state, action] += 1
        return float(error)

    # -- helpers ---------------------------------------------------------------------
    def _check_state(self, state: int) -> None:
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
            raise ValidationError(f"state must be an integer, got {state!r}")
        if not 0 <= int(state) < self._n_states:
            raise ValidationError(
                f"state must lie in [0, {self._n_states - 1}], got {state}"
            )

    def _check_action(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValidationError(f"action must be an integer, got {action!r}")
        if not 0 <= int(action) < self._n_actions:
            raise ValidationError(
                f"action must lie in [0, {self._n_actions - 1}], got {action}"
            )

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_states={self._n_states}, n_actions={self._n_actions}, "
            f"alpha={self._config.learning_rate}, gamma={self._config.discount_factor})"
        )
