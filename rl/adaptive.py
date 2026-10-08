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

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

from config import ACTION_NAMES, ACTION_ROUND_ROBIN, ExperimentConfig, SchedulerConfig
from errors import ValidationError
from evaluation.metrics import WorkloadMetrics, compute_metrics
from rl.q_learning import QLearningAgent
from rl.quantum_controller import CLASSIC_MULTIPLIER, QuantumController
from rl.reward import compute_reward, compute_reward_against_reference
from rl.state import StateEncoder, StateSnapshot, observe_workload_state
from scheduler.base import SchedulingPolicy
from scheduler.round_robin import RoundRobin
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
        action: Action chosen by the agent.
        policy_name: Name of the chosen policy.
        explored: Whether the chosen action differs from the current greedy action.
        epsilon: Exploration rate in force when the action was chosen.
        quantum_multiplier: Quantum multiplier applied to the Round-Robin run reported in
            :attr:`metrics` (``1.0`` when classic Round Robin was used).
        controller_multiplier: Multiplier the quantum controller chose in this episode, or
            ``None`` when the controller is disabled.
        metrics: Metrics of the scheduling run that was actually reported.
        reward: Reward of the decision.
        reference_metrics: Metrics of all four conventional policies on the same workload.
        learned: Whether the Q-table was updated by this decision.
    """

    workload_name: str
    workload_fingerprint: str
    snapshot: StateSnapshot
    state_index: int
    action: int
    policy_name: str
    explored: bool
    epsilon: float
    quantum_multiplier: float
    controller_multiplier: Optional[float]
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
            "state_burst_profile": self.snapshot.burst_profile,
            "state_burst_dispersion": self.snapshot.burst_dispersion,
            "state_offered_load": self.snapshot.offered_load,
            "state_priority_spread": self.snapshot.priority_spread,
            "action": self.action,
            "policy_name": self.policy_name,
            "explored": self.explored,
            "epsilon": self.epsilon,
            "quantum_multiplier": self.quantum_multiplier,
            "controller_multiplier": self.controller_multiplier,
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
        quantum_controller: Optional[QuantumController] = None,
    ) -> None:
        """
        Args:
            config: The experiment configuration (reward, scheduler settings and the
                quantum-controller regime).
            encoder: State encoder shared with the agent.
            agent: The policy-selection agent.
            policies: The four conventional policies, one per action.
            quantum_controller: Round-Robin quantum controller; required when the
                configuration enables it.

        Raises:
            ValidationError: If the policies do not cover the four actions exactly once or
                the controller configuration is inconsistent.
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
        if config.quantum_controller.enabled:
            if quantum_controller is None:
                raise ValidationError(
                    "the configuration enables the quantum controller but none was supplied"
                )
            if quantum_controller.config != config.quantum_controller:
                raise ValidationError(
                    "the quantum controller was built with a different configuration than "
                    "the experiment configuration"
                )
        elif quantum_controller is not None:
            raise ValidationError(
                "a quantum controller was supplied but the configuration disables it"
            )

        self._config = config
        self._encoder = encoder
        self._agent = agent
        self._policies = {policy.action_index: policy for policy in policies}
        self._controller = quantum_controller

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

    @property
    def quantum_controller(self) -> Optional[QuantumController]:
        """The Round-Robin quantum controller, if the configuration enables it."""
        return self._controller

    def run_training_episode(self, workload: Workload, episode: int) -> AdaptiveDecision:
        """Run one training episode: explore, measure, reward and learn.

        Args:
            workload: The workload to schedule in this episode.
            episode: 0-based episode index, which drives the epsilon schedules.

        Returns:
            The decision, with ``learned=True``.
        """
        return self._run(workload, episode=episode, learn=True, use_controller_quantum=False)

    def run_evaluation(
        self, workload: Workload, use_learned_quantum: Optional[bool] = None
    ) -> AdaptiveDecision:
        """Run the greedy policy on a workload without touching the Q-table.

        Args:
            workload: The workload to schedule.
            use_learned_quantum: Whether Round Robin should use the controller's learned
                quantum.  ``None`` follows
                :attr:`config.QuantumControllerConfig.use_during_evaluation`.

        Returns:
            The decision, with ``learned=False``.

        Raises:
            ValidationError: If the learned quantum is requested while the quantum
                controller is disabled.
        """
        if use_learned_quantum is None:
            use_learned_quantum = bool(
                self._controller is not None
                and self._config.quantum_controller.use_during_evaluation
            )
        if use_learned_quantum and self._controller is None:
            raise ValidationError(
                "learned quantum requested but the quantum controller is disabled"
            )
        return self._run(
            workload, episode=None, learn=False, use_controller_quantum=bool(use_learned_quantum)
        )

    def run_round_robin_with_learned_quantum(self, workload: Workload) -> Tuple[WorkloadMetrics, float]:
        """Run Round Robin with the controller's greedy multiplier for this workload.

        This measures the quantum controller directly, independently of whether the
        policy-selection agent happens to choose the Round-Robin action.  It never writes
        to either Q-table.

        Args:
            workload: The workload to schedule.

        Returns:
            The metrics of the controlled Round-Robin run and the multiplier used.

        Raises:
            ValidationError: If the quantum controller is disabled.
        """
        if self._controller is None:
            raise ValidationError(
                "the quantum controller is disabled; there is no learned quantum to apply"
            )
        state = self._encoder.encode(observe_workload_state(workload))
        multiplier = self._controller.greedy_multiplier(state)
        return self._run_controlled_round_robin(workload, multiplier), multiplier

    # -- internals --------------------------------------------------------------------
    def _run(
        self,
        workload: Workload,
        episode: Optional[int],
        learn: bool,
        use_controller_quantum: bool,
    ) -> AdaptiveDecision:
        """Shared implementation of the training and evaluation episodes."""
        snapshot = observe_workload_state(workload)
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

        reported: Dict[int, WorkloadMetrics] = dict(attempts)
        quantum_multiplier = 1.0
        controller_multiplier: Optional[float] = None
        controller = self._controller
        if controller is not None:
            controller_action = self._choose_controller_action(controller, state, learn, episode)
            controller_multiplier = controller.multiplier_for_action(controller_action)
            # The controlled Round-Robin run is needed to train the controller, and to
            # report a learned-quantum result when that regime is requested.
            if learn or use_controller_quantum:
                controlled = (
                    attempts[ACTION_ROUND_ROBIN]
                    if controller_multiplier == CLASSIC_MULTIPLIER
                    else self._run_controlled_round_robin(workload, controller_multiplier)
                )
                if learn:
                    controller_reward = compute_reward_against_reference(
                        self._config.reward, attempts[ACTION_ROUND_ROBIN], controlled
                    )
                    controller.update(state, controller_action, controller_reward)
                if use_controller_quantum:
                    reported[ACTION_ROUND_ROBIN] = controlled
                    quantum_multiplier = controller_multiplier

        reward_breakdown = compute_reward(self._config.reward, reported, action)
        if learn:
            self._agent.update(state, action, reward_breakdown.reward, next_state=None)

        return AdaptiveDecision(
            workload_name=workload.name,
            workload_fingerprint=workload.fingerprint,
            snapshot=snapshot,
            state_index=state,
            action=action,
            policy_name=ACTION_NAMES[action],
            explored=explored,
            epsilon=epsilon,
            quantum_multiplier=quantum_multiplier,
            controller_multiplier=controller_multiplier,
            metrics=reported[action],
            reward=reward_breakdown.reward,
            reference_metrics=attempts,
            learned=learn,
        )

    @staticmethod
    def _choose_controller_action(
        controller: QuantumController,
        state: int,
        learn: bool,
        episode: Optional[int],
    ) -> int:
        """Select the controller's action: epsilon-greedy while learning, greedy otherwise."""
        if learn:
            return controller.select_action(state, controller.epsilon_for_episode(episode or 0))
        return controller.greedy_action(state)

    def _run_controlled_round_robin(self, workload: Workload, multiplier: float) -> WorkloadMetrics:
        """Run Round Robin with the quantum scaled by ``multiplier``.

        The quantum is rounded to the nearest whole time unit of at least one, which is the
        resolution the simulator schedules in.
        """
        base_quantum = self._config.scheduler.round_robin_quantum
        quantum = max(1, int(round(base_quantum * multiplier)))
        scheduler_config: SchedulerConfig = replace(
            self._config.scheduler, round_robin_quantum=quantum
        )
        return compute_metrics(RoundRobin(scheduler_config).run(workload))
