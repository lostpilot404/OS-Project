"""The adaptive scheduler: Q-learning selects the policy for a workload.

The adaptive system follows exactly the project's conceptual flow:

``Workload -> Observe State -> Q-Learning Agent -> Select Scheduling Policy ->
Execute Scheduling -> Measure Performance -> Calculate Reward -> Update Q-Table ->
Observe next state``

One episode is one workload: the agent observes the workload profile, selects one of the
four conventional policies, that policy schedules the workload, the six metrics are
computed, the reward is computed against the four conventional policies *on the same
workload* (a terminal transition -- the next state is observed when the next workload
arrives), and the Q-table is updated.

Training and evaluation are separate methods, so evaluation can never update the Q-table:
:meth:`run_training_episode` explores with the configured epsilon schedule and writes to
the Q-table, while :meth:`run_evaluation` selects the greedy action and writes nothing.

The four conventional policies are always run on the workload to provide the reward's
reference; their metrics travel with every decision, which is what allows the evaluation
to prove that every scheduler saw identical workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from config import ACTION_NAMES, ExperimentConfig
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from rl.q_learning import QLearningAgent
from rl.reward import compute_reward
from rl.state import StateEncoder, StateSnapshot, observe_workload_state
from scheduler.base import SchedulingPolicy
from workload.models import Workload

__all__ = ["AdaptiveDecision", "AdaptiveScheduler"]


@dataclass(frozen=True)
class AdaptiveDecision:
    """Everything that happened in one adaptive scheduling episode.

    Attributes:
        workload_name: Workload condition of the scheduled workload.
        workload_fingerprint: Fingerprint of the scheduled workload.
        snapshot: The observed (undiscretised) state.
        state_index: The discretised state index used by the Q-table.
        state_visit_count: Number of Q-table updates this state had received *before*
            this decision.  During evaluation this is exactly the training visit count,
            so a value of 0 marks a state the agent has never learned about.
        action: Action chosen by the agent.
        policy_name: Name of the chosen policy.
        explored: Whether the chosen action differs from the current greedy action.
        epsilon: Exploration rate in force when the action was chosen.
        metrics: Metrics of the scheduling run that was actually reported.
        reward: Reward of the decision.
        reference_metrics: Metrics of all four conventional policies on the same workload.
        learned: Whether the Q-table was updated by this decision.
    """

    workload_name: str
    workload_fingerprint: str
    snapshot: StateSnapshot
    state_index: int
    state_visit_count: int
    action: int
    policy_name: str
    explored: bool
    epsilon: float
    metrics: WorkloadMetrics
    reward: float
    reference_metrics: Mapping[int, WorkloadMetrics]
    learned: bool

    def as_row(self) -> Dict[str, object]:
        """Flatten the decision into a table row."""
        row: Dict[str, object] = {
            "workload_name": self.workload_name,
            "workload_fingerprint": self.workload_fingerprint,
            "state_index": self.state_index,
            "state_visit_count": self.state_visit_count,
            "state_burst_profile": self.snapshot.burst_profile,
            "state_burst_dispersion": self.snapshot.burst_dispersion,
            "state_long_job_share": self.snapshot.long_job_share,
            "state_arrival_concentration": self.snapshot.arrival_concentration,
            "state_offered_load": self.snapshot.offered_load,
            "state_priority_spread": self.snapshot.priority_spread,
            "state_priority_burst_alignment": self.snapshot.priority_burst_alignment,
            "action": self.action,
            "policy_name": self.policy_name,
            "explored": self.explored,
            "epsilon": self.epsilon,
            "reward": self.reward,
            "learned": self.learned,
        }
        for key, value in self.metrics.as_dict().items():
            if key in ("policy_name", "workload_name", "workload_fingerprint"):
                continue  # already present above, under the decision's own names
            row[f"adaptive_{key}"] = value
        return row


class AdaptiveScheduler:
    """Runs the Q-learning policy-selection loop over workloads."""

    def __init__(
        self,
        config: ExperimentConfig,
        encoder: StateEncoder,
        agent: QLearningAgent,
        policies: Sequence[SchedulingPolicy],
    ) -> None:
        """
        Args:
            config: The experiment configuration (reward and scheduler settings).
            encoder: State encoder shared with the agent.
            agent: The policy-selection agent.
            policies: The four conventional policies, one per action.

        Raises:
            ValidationError: If the policies do not cover the four actions exactly once.
        """
        if not isinstance(config, ExperimentConfig):
            raise ValidationError(
                f"config must be an ExperimentConfig, got {type(config).__name__}"
            )
        if not isinstance(agent, QLearningAgent):
            raise ValidationError(f"agent must be a QLearningAgent, got {type(agent).__name__}")
        if not isinstance(encoder, StateEncoder):
            raise ValidationError(f"encoder must be a StateEncoder, got {type(encoder).__name__}")
        actions = sorted(policy.action_index for policy in policies)
        if actions != list(range(len(ACTION_NAMES))):
            raise ValidationError(
                f"policies must cover the four actions exactly once, got action indices {actions}"
            )
        if agent.n_states != encoder.n_states:
            raise ValidationError(
                f"agent has {agent.n_states} states but the encoder produces {encoder.n_states}"
            )
        if agent.n_actions != len(ACTION_NAMES):
            raise ValidationError(
                f"agent has {agent.n_actions} actions but the project defines "
                f"{len(ACTION_NAMES)} actions"
            )

        self._config = config
        self._encoder = encoder
        self._agent = agent
        self._policies = {policy.action_index: policy for policy in policies}

    # -- public API -------------------------------------------------------------------
    @property
    def config(self) -> ExperimentConfig:
        """The experiment configuration this scheduler runs with."""
        return self._config

    @property
    def agent(self) -> QLearningAgent:
        """The policy-selection agent."""
        return self._agent

    @property
    def encoder(self) -> StateEncoder:
        """The state encoder."""
        return self._encoder

    def run_training_episode(self, workload: Workload, episode: int) -> AdaptiveDecision:
        """Run one training episode: explore, measure, reward and learn.

        Args:
            workload: The workload to schedule in this episode.
            episode: 0-based episode index, which drives the epsilon schedule.

        Returns:
            The decision, with ``learned=True``.
        """
        return self._run(workload, episode=episode, learn=True)

    def run_evaluation(self, workload: Workload) -> AdaptiveDecision:
        """Run the greedy policy on a workload without touching the Q-table.

        Args:
            workload: The workload to schedule.

        Returns:
            The decision, with ``learned=False``.
        """
        return self._run(workload, episode=None, learn=False)

    # -- internals --------------------------------------------------------------------
    def _run(
        self,
        workload: Workload,
        episode: Optional[int],
        learn: bool,
    ) -> AdaptiveDecision:
        """Shared implementation of the training and evaluation episodes."""
        snapshot = observe_workload_state(workload, self._config.state)
        state = self._encoder.encode(snapshot)

        epsilon = self._agent.epsilon_for_episode(episode) if learn else 0.0
        action = self._agent.select_action(state, epsilon)
        explored = self._agent.last_action_was_exploration(state, action)

        # Every conventional policy runs on the same workload: this is the reward's
        # reference and the evidence that no baseline saw a different input.
        attempts: Dict[int, WorkloadMetrics] = {
            index: compute_metrics(policy.run(workload))
            for index, policy in self._policies.items()
        }

        reward_breakdown = compute_reward(self._config.reward, attempts, action)
        # Captured before the update: during evaluation this is the number of training
        # updates the state has received, i.e. how much the agent knows about it.
        state_visit_count = int(self._agent.visit_counts[state])
        if learn:
            self._agent.update(state, action, reward_breakdown.reward, next_state=None)

        return AdaptiveDecision(
            workload_name=workload.name,
            workload_fingerprint=workload.fingerprint,
            snapshot=snapshot,
            state_index=state,
            state_visit_count=state_visit_count,
            action=action,
            policy_name=ACTION_NAMES[action],
            explored=explored,
            epsilon=epsilon,
            metrics=attempts[action],
            reward=reward_breakdown.reward,
            reference_metrics=attempts,
            learned=learn,
        )
