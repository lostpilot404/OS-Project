"""Offline single-workload policy selection using one tabular Q-learning agent.

The agent receives the complete workload description, encodes its pre-execution
characteristics, and chooses one of FCFS, SJF, Round Robin, or Priority for the whole
workload. The chosen scheduler runs to completion. There is no runtime switching, quantum
controller, or sequential control within a schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from config import ACTION_NAMES, ExperimentConfig
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from rl.q_learning import QLearningAgent
from rl.reward import compute_reward
from rl.state import StateEncoder, StateSnapshot, observe_workload_state
from scheduler.base import SchedulingPolicy
from workload.models import Workload

__all__ = ["PolicySelectionDecision", "OfflinePolicySelector"]


@dataclass(frozen=True)
class PolicySelectionDecision:
    """One offline decision and its complete-workload outcome.

    ``explored`` records whether epsilon-greedy selection actually took the random branch;
    it does not mean merely that the selected action differed from the greedy action.
    ``state_seen_in_training`` is meaningful during evaluation and identifies the
    deterministic initialization fallback for a state with no training updates.
    """

    workload_name: str
    workload_fingerprint: str
    snapshot: StateSnapshot
    state_index: int
    state_seen_in_training: bool
    action: int
    greedy_action: int
    policy_name: str
    greedy_policy_name: str
    explored: bool
    epsilon: float
    metrics: WorkloadMetrics
    reward: float
    reference_metrics: Mapping[int, WorkloadMetrics]
    learned: bool

    def as_row(self) -> Dict[str, object]:
        """Flatten the decision and selected-policy metrics for a CSV row."""
        row: Dict[str, object] = {
            "workload_name": self.workload_name,
            "workload_fingerprint": self.workload_fingerprint,
            "state_index": self.state_index,
            "state_seen_in_training": self.state_seen_in_training,
            **{f"state_{name}": value for name, value in self.snapshot.as_dict().items()},
            "action": self.action,
            "policy_name": self.policy_name,
            "greedy_action": self.greedy_action,
            "greedy_policy_name": self.greedy_policy_name,
            "explored": self.explored,
            "epsilon": self.epsilon,
            "reward": self.reward,
            "learned": self.learned,
        }
        row.update({f"selector_{key}": value for key, value in self.metrics.as_dict().items()})
        return row


class OfflinePolicySelector:
    """Select one conventional scheduler for a complete workload before execution.

    This class is an offline policy selector. The workload's full process list is available
    at decision time; all four reference schedulers run the same workload to calculate the
    single-step reward. Training performs one terminal Q update; evaluation is greedy and
    read-only with respect to the learned Q-table.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        encoder: StateEncoder,
        agent: QLearningAgent,
        policies: Sequence[SchedulingPolicy],
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise ValidationError(
                f"config must be an ExperimentConfig, got {type(config).__name__}"
            )
        config.validate()
        if not isinstance(encoder, StateEncoder):
            raise ValidationError(f"encoder must be a StateEncoder, got {type(encoder).__name__}")
        if not isinstance(agent, QLearningAgent):
            raise ValidationError(f"agent must be a QLearningAgent, got {type(agent).__name__}")
        if encoder.config != config.state:
            raise ValidationError("the state encoder configuration does not match the experiment")
        if agent.n_states != encoder.n_states or agent.n_actions != len(ACTION_NAMES):
            raise ValidationError("the agent table dimensions do not match the state/action spaces")

        by_action = {policy.action_index: policy for policy in policies}
        if set(by_action) != set(range(len(ACTION_NAMES))):
            raise ValidationError(
                f"policies must cover actions 0-{len(ACTION_NAMES) - 1}, got {sorted(by_action)}"
            )
        for action, policy in by_action.items():
            if policy.name != ACTION_NAMES[action]:
                raise ValidationError(
                    f"policy at action {action} must be {ACTION_NAMES[action]!r}, got {policy.name!r}"
                )
        self._config = config
        self._encoder = encoder
        self._agent = agent
        self._policies = by_action

    @property
    def config(self) -> ExperimentConfig:
        """The experiment configuration this selector runs with."""
        return self._config

    @property
    def agent(self) -> QLearningAgent:
        """The policy-selection agent."""
        return self._agent

    @property
    def encoder(self) -> StateEncoder:
        """The state encoder."""
        return self._encoder

    def run_training_episode(self, workload: Workload, episode: int) -> PolicySelectionDecision:
        """Select, execute, score, and update once for a training workload."""
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValidationError(f"episode must be a non-negative integer, got {episode!r}")
        return self._run(workload, episode=episode, learn=True)

    def run_evaluation(self, workload: Workload) -> PolicySelectionDecision:
        """Select greedily for one evaluation workload without mutating the Q-table."""
        return self._run(workload, episode=None, learn=False)

    def _run(self, workload: Workload, episode: int | None, learn: bool) -> PolicySelectionDecision:
        if not isinstance(workload, Workload):
            raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
        snapshot = observe_workload_state(workload)
        state = self._encoder.encode(snapshot)
        greedy_action = self._agent.greedy_action(state)
        state_seen_in_training = bool(self._agent.visit_counts[state] > 0)
        epsilon = self._agent.epsilon_for_episode(episode) if learn and episode is not None else 0.0
        action = self._agent.select_action(state, epsilon)
        explored = self._agent.last_action_was_random_exploration

        # All four references receive the identical, complete workload. These results are
        # also the only basis of the policy-selection reward.
        reference_metrics = {
            index: compute_metrics(policy.run(workload))
            for index, policy in self._policies.items()
        }
        breakdown = compute_reward(self._config.reward, reference_metrics, action)
        if learn:
            # Every episode is terminal after one policy has scheduled the whole batch;
            # no discounted next-state value is used.
            self._agent.update(state, action, breakdown.reward, next_state=None)

        selected_metrics = reference_metrics[action]
        return PolicySelectionDecision(
            workload_name=workload.name,
            workload_fingerprint=workload.fingerprint,
            snapshot=snapshot,
            state_index=state,
            state_seen_in_training=state_seen_in_training,
            action=action,
            greedy_action=greedy_action,
            policy_name=ACTION_NAMES[action],
            greedy_policy_name=ACTION_NAMES[greedy_action],
            explored=explored,
            epsilon=epsilon,
            metrics=selected_metrics,
            reward=breakdown.reward,
            reference_metrics=reference_metrics,
            learned=learn,
        )
