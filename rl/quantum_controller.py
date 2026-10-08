"""Round-Robin quantum controller.

The policy-selection agent chooses *which* of the four policies runs.  The Round-Robin
quantum is itself a scheduling parameter whose best value depends on the workload, so a
second tabular Q-learning agent learns a quantum multiplier for the Round-Robin action.
This is a documented extension: the five learned policies of Hazarika et al.,
"Workload Aware Dynamic Scheduling Algorithm for Multi-core Systems" (ACM SIGSOFT SEN
43(4), 2018), which this module implements on a single CPU.  It is *not* one of the four
conventional baselines and it is disabled during evaluation by default
(:attr:`config.QuantumControllerConfig.use_during_evaluation`), so the headline
comparison uses classic Round Robin with the fixed configured quantum.

The controller is a single-step learner: its episode lasts one workload.  It observes the
same state as the policy agent, applies one multiplier for the whole workload, and
receives the reward of that workload's Round-Robin run relative to classic Round Robin on
the same workload.  Because the episode is single-step, the discount factor cannot
influence its values; the update still follows the generic rule with a terminal
transition.

A state whose Q-values have never been updated falls back to multiplier ``1.0`` (classic
Round Robin) rather than to an arbitrary unvisited action.
"""

from __future__ import annotations

from typing import Tuple

from config import QLearningConfig, QuantumControllerConfig
from errors import ValidationError
from rl.q_learning import QLearningAgent
from rl.state import StateEncoder

__all__ = ["QuantumController", "CLASSIC_MULTIPLIER"]

#: Multiplier that reproduces classic Round Robin with the configured quantum.
CLASSIC_MULTIPLIER: float = 1.0


class QuantumController:
    """Learns a Round-Robin quantum multiplier per workload state."""

    def __init__(self, encoder: StateEncoder, config: QuantumControllerConfig, seed: int) -> None:
        """
        Args:
            encoder: The state encoder shared with the policy-selection agent.
            config: Controller hyperparameters (multipliers, alpha, epsilon schedule).
            seed: Seed of the controller's exploration random number generator.

        Raises:
            ValidationError: If the encoder or configuration is invalid.
        """
        if not isinstance(encoder, StateEncoder):
            raise ValidationError(f"encoder must be a StateEncoder, got {type(encoder).__name__}")
        if not isinstance(config, QuantumControllerConfig):
            raise ValidationError(
                f"config must be a QuantumControllerConfig, got {type(config).__name__}"
            )
        self._config = config
        self._encoder = encoder
        self._agent = QLearningAgent(
            n_states=encoder.n_states,
            n_actions=len(config.multipliers),
            config=QLearningConfig(
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon_start=config.epsilon_start,
                epsilon_min=config.epsilon_min,
                epsilon_decay_per_episode=config.epsilon_decay_per_episode,
            ),
            seed=seed,
            initial_value=0.0,
        )

    @property
    def config(self) -> QuantumControllerConfig:
        """The controller's hyperparameters."""
        return self._config

    @property
    def agent(self) -> QLearningAgent:
        """The underlying single-step Q-learning agent."""
        return self._agent

    @property
    def multipliers(self) -> Tuple[float, ...]:
        """The candidate quantum multipliers, in action order."""
        return tuple(float(m) for m in self._config.multipliers)

    def multiplier_for_action(self, action: int) -> float:
        """Return the multiplier encoded by an action index.

        Raises:
            ValidationError: If the action is out of range.
        """
        if not 0 <= action < len(self.multipliers):
            raise ValidationError(
                f"controller action must lie in [0, {len(self.multipliers) - 1}], got {action}"
            )
        return self.multipliers[action]

    def greedy_action(self, state: int) -> int:
        """Return the best-known multiplier action for ``state``.

        States that have never been updated fall back to classic Round Robin.
        """
        if self._agent.visit_counts[state] == 0:
            return self.multipliers.index(CLASSIC_MULTIPLIER)
        return self._agent.greedy_action(state)

    def greedy_multiplier(self, state: int) -> float:
        """Return the multiplier chosen greedily for ``state``."""
        return self.multiplier_for_action(self.greedy_action(state))

    def select_action(self, state: int, epsilon: float) -> int:
        """Choose a multiplier action epsilon-greedily (training only)."""
        return self._agent.select_action(state, epsilon)

    def epsilon_for_episode(self, episode: int) -> float:
        """The controller's exploration rate in the given training episode."""
        return self._agent.epsilon_for_episode(episode)

    def update(self, state: int, action: int, reward: float) -> float:
        """Apply the single-step Q-learning update and return the TD error."""
        return self._agent.update(state, action, reward, next_state=None)

    def __repr__(self) -> str:
        return f"QuantumController(multipliers={self.multipliers}, alpha={self._config.learning_rate})"
