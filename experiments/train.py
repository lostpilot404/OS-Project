"""Training runs for the offline single-workload policy selector.

Each episode generates one complete workload, selects one of the four conventional
schedulers, runs it to completion, computes the four-policy-relative reward, and performs
one terminal tabular Q-learning update. This module writes Q-values; evaluation is kept in
:mod:`experiments.evaluate` and does not update them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import ACTION_NAMES, ExperimentConfig, derive_seed
from errors import ConfigurationError
from rl.adaptive import OfflinePolicySelector
from rl.q_learning import QLearningAgent
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator

__all__ = ["TrainingHistory", "TrainingResult", "train"]

_WORKLOAD_STREAM_TAG = 0
_AGENT_STREAM_TAG = 1
_SUMMARY_WINDOW = 100


@dataclass(frozen=True)
class TrainingHistory:
    """Per-episode training observations for one independent training seed."""

    episodes: int
    seed: int
    family_names: Tuple[str, ...]
    workload_fingerprints: Tuple[str, ...]
    state_indices: Tuple[int, ...]
    chosen_actions: Tuple[int, ...]
    greedy_actions: Tuple[int, ...]
    state_seen_before_action: Tuple[bool, ...]
    rewards: Tuple[float, ...]
    epsilons: Tuple[float, ...]
    explored: Tuple[bool, ...]
    visit_counts: Tuple[int, ...]

    @property
    def mean_reward(self) -> float:
        """Mean terminal reward over all training episodes."""
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def mean_reward_over_last(self, episodes: int = _SUMMARY_WINDOW) -> float:
        """Mean reward over the final ``episodes`` training episodes."""
        window = self.rewards[-episodes:]
        return sum(window) / len(window) if window else 0.0

    def mean_reward_over_first(self, episodes: int = _SUMMARY_WINDOW) -> float:
        """Mean reward over the first ``episodes`` training episodes."""
        window = self.rewards[:episodes]
        return sum(window) / len(window) if window else 0.0

    def action_counts(self) -> Dict[str, int]:
        """Number of episodes in which each policy action was selected."""
        counts = Counter(self.chosen_actions)
        return {ACTION_NAMES[action]: counts.get(action, 0) for action in range(len(ACTION_NAMES))}

    @property
    def random_exploration_rate(self) -> float:
        """Fraction of episodes that actually took epsilon's random-action branch."""
        if not self.explored:
            return 0.0
        return sum(self.explored) / len(self.explored)

    @property
    def greedy_action_match_rate(self) -> float:
        """Fraction of selected actions equal to the pre-update greedy action."""
        if not self.chosen_actions:
            return 0.0
        matches = sum(a == greedy for a, greedy in zip(self.chosen_actions, self.greedy_actions))
        return matches / len(self.chosen_actions)

    @property
    def visited_states(self) -> int:
        """Number of states that received at least one training update."""
        return sum(count > 0 for count in self.visit_counts)

    @property
    def n_states(self) -> int:
        """Number of possible discrete states in the Q-table."""
        return len(self.visit_counts)

    def summary(self) -> Dict[str, object]:
        """Return compact training statistics for generated reports."""
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "mean_reward_first_100": self.mean_reward_over_first(),
            "mean_reward_last_100": self.mean_reward_over_last(),
            "mean_reward_all": self.mean_reward,
            "action_counts": self.action_counts(),
            "random_exploration_episode_rate": self.random_exploration_rate,
            "greedy_action_match_rate": self.greedy_action_match_rate,
            "visited_states": self.visited_states,
            "n_states": self.n_states,
            "final_epsilon": self.epsilons[-1] if self.epsilons else 0.0,
        }

    def as_dict(self) -> Dict[str, object]:
        """Return the complete history as JSON-serializable data."""
        return {
            "summary": self.summary(),
            "episodes_detail": {
                "family": list(self.family_names),
                "workload_fingerprint": list(self.workload_fingerprints),
                "state_index": list(self.state_indices),
                "state_seen_before_action": list(self.state_seen_before_action),
                "action": list(self.chosen_actions),
                "action_name": [ACTION_NAMES[action] for action in self.chosen_actions],
                "greedy_action": list(self.greedy_actions),
                "greedy_action_name": [ACTION_NAMES[action] for action in self.greedy_actions],
                "reward": list(self.rewards),
                "epsilon": list(self.epsilons),
                "used_random_exploration": list(self.explored),
            },
        }

    def to_json(self, path: Path) -> Path:
        """Write the episode-level history and its summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class TrainingResult:
    """Trained policy-selection agent, its history, and associated objects."""

    config: ExperimentConfig
    agent: QLearningAgent
    history: TrainingHistory
    encoder: StateEncoder
    scheduler: OfflinePolicySelector

    @property
    def training_fingerprints(self) -> frozenset[str]:
        """Fingerprints of workloads presented during this training run."""
        return frozenset(self.history.workload_fingerprints)

    def q_table_rows(self) -> List[Dict[str, object]]:
        """Return every state/action value with coverage and the training seed."""
        rows: List[Dict[str, object]] = []
        table = self.agent.q_table
        for state in range(self.agent.n_states):
            row: Dict[str, object] = {
                "training_seed": self.history.seed,
                "state_index": state,
                "state_bins": self.encoder.decode(state),
                "visit_count": int(self.agent.visit_counts[state]),
            }
            for action, name in enumerate(ACTION_NAMES):
                row[f"q_{name}"] = float(table[state, action])
            row["greedy_action"] = ACTION_NAMES[self.agent.greedy_action(state)]
            rows.append(row)
        return rows


def build_agent(config: ExperimentConfig, encoder: StateEncoder) -> QLearningAgent:
    """Build the policy-selection Q-table and independent exploration RNG."""
    return QLearningAgent(
        n_states=encoder.n_states,
        n_actions=len(ACTION_NAMES),
        config=config.q_learning,
        seed=derive_seed(config.training.seed, _AGENT_STREAM_TAG),
        initial_value=config.q_learning.initial_value,
    )


def build_scheduler(
    config: ExperimentConfig,
    encoder: StateEncoder,
    agent: QLearningAgent,
) -> OfflinePolicySelector:
    """Build the offline selector with exactly the four reference schedulers."""
    policies = [policy_class(config.scheduler) for policy_class in POLICY_CLASSES]
    return OfflinePolicySelector(config, encoder, agent, policies)


def train(
    config: ExperimentConfig,
    generator: Optional[WorkloadGenerator] = None,
    encoder: Optional[StateEncoder] = None,
) -> TrainingResult:
    """Train one agent using ``config.training.seed``.

    The experiment runner invokes this function independently for each configured
    training seed. Evaluation workloads use a separate seed stream and are never used for
    a Q-table update.
    """
    config.validate()
    generator = generator or WorkloadGenerator(config.families)
    encoder = encoder or StateEncoder(config.state)
    agent = build_agent(config, encoder)
    scheduler = build_scheduler(config, encoder, agent)

    cycle: Sequence[str] = config.training.family_cycle or generator.family_names
    if not cycle:
        raise ConfigurationError("no workload families to train on")

    families: List[str] = []
    fingerprints: List[str] = []
    states: List[int] = []
    actions: List[int] = []
    greedy_actions: List[int] = []
    state_seen: List[bool] = []
    rewards: List[float] = []
    epsilons: List[float] = []
    explored: List[bool] = []

    for episode in range(config.training.episodes):
        family = cycle[episode % len(cycle)]
        workload = generator.generate(
            family, derive_seed(config.training.seed, _WORKLOAD_STREAM_TAG, episode)
        )
        decision = scheduler.run_training_episode(workload, episode)
        families.append(family)
        fingerprints.append(decision.workload_fingerprint)
        states.append(decision.state_index)
        actions.append(decision.action)
        greedy_actions.append(decision.greedy_action)
        state_seen.append(decision.state_seen_in_training)
        rewards.append(decision.reward)
        epsilons.append(decision.epsilon)
        explored.append(decision.explored)

    history = TrainingHistory(
        episodes=config.training.episodes,
        seed=config.training.seed,
        family_names=tuple(families),
        workload_fingerprints=tuple(fingerprints),
        state_indices=tuple(states),
        chosen_actions=tuple(actions),
        greedy_actions=tuple(greedy_actions),
        state_seen_before_action=tuple(state_seen),
        rewards=tuple(rewards),
        epsilons=tuple(epsilons),
        explored=tuple(explored),
        visit_counts=tuple(int(count) for count in agent.visit_counts),
    )
    return TrainingResult(
        config=config,
        agent=agent,
        history=history,
        encoder=encoder,
        scheduler=scheduler,
    )
