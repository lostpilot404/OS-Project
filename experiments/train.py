"""Training of the adaptive scheduler.

Training runs a fixed number of episodes, one workload per episode, cycling through the
configured workload families in round-robin order.  Every draw is seeded through
:func:`config.derive_seed`, so the whole run is reproducible from
:attr:`config.TrainingConfig.seed` alone.

Training is deliberately separated from evaluation (:mod:`experiments.evaluate`): the
Q-table is written here and only here, and the evaluation workloads come from a different
master seed, so no evaluation workload is ever trained on.  The pre-training class
verification (:mod:`experiments.verify_classes`) is likewise separate: it measures which
policy each workload class favours *before* the agent is trained, and it never touches
the agent.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import ACTION_NAMES, ExperimentConfig, derive_seed
from rl.adaptive import AdaptiveScheduler
from rl.q_learning import QLearningAgent
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator

__all__ = ["TrainingHistory", "TrainingResult", "train"]

#: Seed-derivation tags, so that the agent's exploration stream and the workload stream
#: can never collide (see :func:`config.derive_seed`).
_WORKLOAD_STREAM_TAG = 0
_AGENT_STREAM_TAG = 1

#: Window used for the "recent reward" summary statistics.
_SUMMARY_WINDOW = 100


@dataclass(frozen=True)
class TrainingHistory:
    """Per-episode record of a training run.

    Attributes:
        episodes: Number of episodes trained.
        seed: Master seed of the run.
        family_names: Workload family used in every episode, in episode order.
        workload_fingerprints: Fingerprint of the workload scheduled in every episode.
        state_indices: Discretised state observed in every episode.
        chosen_actions: Action selected in every episode.
        rewards: Reward received in every episode.
        epsilons: Exploration rate in force in every episode.
        explored: Whether the chosen action differed from the greedy action.
        visit_counts: Final per-state update counts (length = number of states).
    """

    episodes: int
    seed: int
    family_names: Tuple[str, ...]
    workload_fingerprints: Tuple[str, ...]
    state_indices: Tuple[int, ...]
    chosen_actions: Tuple[int, ...]
    rewards: Tuple[float, ...]
    epsilons: Tuple[float, ...]
    explored: Tuple[bool, ...]
    visit_counts: Tuple[int, ...]

    @property
    def mean_reward(self) -> float:
        """Mean reward over all episodes."""
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def mean_reward_over_last(self, episodes: int = _SUMMARY_WINDOW) -> float:
        """Mean reward over the final ``episodes`` episodes."""
        window = self.rewards[-episodes:]
        return sum(window) / len(window) if window else 0.0

    def mean_reward_over_first(self, episodes: int = _SUMMARY_WINDOW) -> float:
        """Mean reward over the first ``episodes`` episodes."""
        window = self.rewards[:episodes]
        return sum(window) / len(window) if window else 0.0

    def action_counts(self) -> Dict[str, int]:
        """Number of episodes in which each action was selected."""
        counts = Counter(self.chosen_actions)
        return {ACTION_NAMES[action]: counts.get(action, 0) for action in range(len(ACTION_NAMES))}

    def greedy_action_counts(self) -> Dict[str, int]:
        """Number of episodes whose action equalled the greedy action."""
        return {
            ACTION_NAMES[action]: sum(
                1 for a, was_explored in zip(self.chosen_actions, self.explored)
                if a == action and not was_explored
            )
            for action in range(len(ACTION_NAMES))
        }

    @property
    def exploration_rate(self) -> float:
        """Fraction of episodes in which the action was not the greedy action."""
        if not self.explored:
            return 0.0
        return sum(1 for flag in self.explored if flag) / len(self.explored)

    @property
    def visited_states(self) -> int:
        """Number of states that received at least one update."""
        return sum(1 for count in self.visit_counts if count > 0)

    @property
    def n_states(self) -> int:
        """Size of the state space."""
        return len(self.visit_counts)

    def summary(self) -> Dict[str, object]:
        """Return the summary statistics reported with the experiment."""
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "mean_reward_first_100": self.mean_reward_over_first(),
            "mean_reward_last_100": self.mean_reward_over_last(),
            "mean_reward_all": self.mean_reward,
            "action_counts": self.action_counts(),
            "greedy_action_counts": self.greedy_action_counts(),
            "exploration_rate": self.exploration_rate,
            "visited_states": self.visited_states,
            "n_states": self.n_states,
            "final_epsilon": self.epsilons[-1] if self.epsilons else 0.0,
        }

    def to_json(self, path: Path) -> Path:
        """Write the full per-episode history to ``path`` as JSON."""
        payload = {
            "summary": self.summary(),
            "episodes_detail": {
                "family": list(self.family_names),
                "workload_fingerprint": list(self.workload_fingerprints),
                "state_index": list(self.state_indices),
                "action": list(self.chosen_actions),
                "action_name": [ACTION_NAMES[a] for a in self.chosen_actions],
                "reward": list(self.rewards),
                "epsilon": list(self.epsilons),
                "explored": list(self.explored),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class TrainingResult:
    """Everything produced by a training run.

    Attributes:
        config: The configuration the run was made with.
        agent: The trained policy-selection agent.
        history: The per-episode record.
        encoder: The state encoder shared by agent and scheduler.
        scheduler: The adaptive scheduler used for training (kept for inspection).
    """

    config: ExperimentConfig
    agent: QLearningAgent
    history: TrainingHistory
    encoder: StateEncoder
    scheduler: AdaptiveScheduler

    @property
    def training_fingerprints(self) -> frozenset:
        """Fingerprints of all workloads seen during training."""
        return frozenset(self.history.workload_fingerprints)

    def q_table_rows(self) -> List[Dict[str, object]]:
        """The Q-table as a list of JSON-serialisable rows (one per state)."""
        rows: List[Dict[str, object]] = []
        table = self.agent.q_table
        for state in range(self.agent.n_states):
            row: Dict[str, object] = {
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
    """Build the policy-selection agent described by ``config``."""
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
) -> AdaptiveScheduler:
    """Build the adaptive scheduler from the configured conventional policies."""
    policies = [policy_class(config.scheduler) for policy_class in POLICY_CLASSES]
    return AdaptiveScheduler(config, encoder, agent, policies)


def train(
    config: ExperimentConfig,
    generator: Optional[WorkloadGenerator] = None,
    encoder: Optional[StateEncoder] = None,
) -> TrainingResult:
    """Train the adaptive scheduler and record the full episode history.

    Args:
        config: The experiment configuration; validated before training starts.
        generator: Optional pre-built workload generator (one is created if omitted).
        encoder: Optional pre-built state encoder (one is created if omitted).

    Returns:
        The trained agent, history and the objects it was built with.

    Raises:
        ConfigurationError: If the configuration is inconsistent.
    """
    config.validate()
    generator = generator or WorkloadGenerator(config.families)
    encoder = encoder or StateEncoder(config.state)
    agent = build_agent(config, encoder)
    scheduler = build_scheduler(config, encoder, agent)

    cycle: Sequence[str] = config.training.family_cycle or generator.family_names
    if not cycle:
        raise ValueError("no workload families to train on")

    families: List[str] = []
    fingerprints: List[str] = []
    states: List[int] = []
    actions: List[int] = []
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
