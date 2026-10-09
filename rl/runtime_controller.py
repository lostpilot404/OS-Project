"""Causal online Q-learning controller for the runtime simulator."""

from __future__ import annotations

from typing import Callable, Optional

from config import ACTION_FCFS, ACTION_NAMES, QLearningConfig
from errors import ValidationError
from rl.q_learning import QLearningAgent
from rl.runtime_state import RuntimeStateEncoder
from scheduler.runtime import (
    RuntimeActionDecision,
    RuntimeObservation,
)

__all__ = ["RuntimeQController"]


class RuntimeQController:
    """Tabular sequential Q learner operating only on causal observations.

    Training uses epsilon-greedy decisions at every epoch. The first visit to a state is
    forced to explore uniformly so every reachable state gets an action sample even after
    the episode-level epsilon schedule has decayed. Subsequent greedy choices and
    Q-learning bootstraps mask unvisited actions. Evaluation is deterministic, never
    explores, never updates, and falls back to an explicitly supplied causal heuristic
    (or FCFS when no fallback is supplied) in states with no learned action.
    """

    def __init__(
        self,
        encoder: RuntimeStateEncoder,
        config: QLearningConfig,
        seed: int,
        fallback_action: Optional[Callable[[RuntimeObservation], int]] = None,
    ) -> None:
        if not isinstance(encoder, RuntimeStateEncoder):
            raise ValidationError("encoder must be RuntimeStateEncoder")
        self.encoder = encoder
        self.agent = QLearningAgent(
            n_states=encoder.n_states,
            n_actions=len(ACTION_NAMES),
            config=config,
            seed=seed,
        )
        self._fallback_action = fallback_action or (lambda _observation: ACTION_FCFS)

    def encode_observation(self, observation: RuntimeObservation) -> int:
        """Encode a causal observation; this method accepts no workload argument."""
        return self.encoder.encode(observation)

    def select_action(
        self,
        observation: RuntimeObservation,
        *,
        training: bool = False,
        episode: int = 0,
    ) -> RuntimeActionDecision:
        """Select one of the four policies from currently observed work."""
        if not isinstance(observation, RuntimeObservation):
            raise ValidationError("RuntimeQController requires a RuntimeObservation")
        if not isinstance(training, bool):
            raise ValidationError("training must be a bool")
        if episode < 0:
            raise ValidationError("episode must be >= 0")
        state = self.encoder.encode(observation)
        known = self.agent.visited_actions(state)
        unvisited_count = self.agent.n_actions - len(known)

        if training:
            epsilon = self.agent.epsilon_for_episode(episode)
            if not known:
                # Deliberate state-coverage rule: an unseen state is always sampled,
                # rather than letting its zero-initialized Q values masquerade as data.
                action = self.agent.select_action(state, epsilon=1.0)
                explored = True
                reason = "forced-unseen-state-exploration"
            else:
                action = self.agent.select_action(
                    state,
                    epsilon=epsilon,
                    greedy_actions=known,
                )
                explored = self.agent.last_action_was_random_exploration
                reason = None
            return RuntimeActionDecision(
                action=action,
                state_index=state,
                epsilon=epsilon,
                explored=explored,
                fallback_reason=reason,
                unvisited_action_count=unvisited_count,
            )

        if known:
            action = self.agent.greedy_visited_action(state)
            assert action is not None
            return RuntimeActionDecision(
                action=action,
                state_index=state,
                epsilon=0.0,
                explored=False,
                unvisited_action_count=unvisited_count,
            )

        action = self._fallback_action(observation)
        return RuntimeActionDecision(
            action=action,
            state_index=state,
            epsilon=0.0,
            explored=False,
            fallback_reason="unseen-state-causal-fallback",
            unvisited_action_count=unvisited_count,
        )

    def observe_transition(
        self,
        decision: RuntimeActionDecision,
        reward: float,
        next_observation: Optional[RuntimeObservation],
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """Apply a Q-learning update only during an explicitly training run."""
        if not isinstance(training, bool) or not isinstance(terminal, bool):
            raise ValidationError("training and terminal flags must be bools")
        if not training:
            return
        if decision.state_index is None:
            raise ValidationError("training decision is missing its encoded state")
        if terminal:
            if next_observation is not None:
                raise ValidationError("terminal transition must not have a next observation")
            next_state = None
        else:
            if next_observation is None:
                raise ValidationError("nonterminal transition requires a next observation")
            next_state = self.encoder.encode(next_observation)
        self.agent.update(
            state=decision.state_index,
            action=decision.action,
            reward=reward,
            next_state=next_state,
            mask_unvisited_next_actions=True,
        )
