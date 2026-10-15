"""Offline, one-policy-per-workload selector with a strict decision boundary.

This module implements the system's decision function.  A :class:`SelectionDecision` is
computed from the **declared process table only** (via :mod:`workload.fingerprint` /
:mod:`rl.state`) and the learned Q-table.  :meth:`WorkloadPolicySelector.select` has no
access to schedulers and cannot execute any policy: counterfactual outcomes enter only
after the action is fixed, as the training reward (:meth:`train_episode`) or as post-hoc
evaluation evidence (:meth:`evaluate`).

One decision is made per complete workload; the selected policy then runs the whole
workload.  This is not a runtime scheduler and never switches policies during
execution.

Training episodes are terminal one-step Q-learning updates, exactly as in the preserved
legacy selector (:mod:`rl.adaptive`): the reward is computed from the four-policy
counterfactuals after the action is fixed, and no discounted next-state value is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from config import ACTION_NAMES, RewardConfig, RiskObjectiveConfig
from errors import ValidationError
from evaluation.counterfactual import (
    CounterfactualReport,
    RegretRecord,
    run_counterfactuals,
)
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import (
    OBJECTIVE_NAMES,
    OBJECTIVE_RISK_AWARE,
    ObjectiveBreakdown,
    compute_objective,
)
from rl.q_learning import QLearningAgent
from rl.state import StateEncoder, observe_workload_state
from scheduler.base import SchedulingPolicy
from workload.fingerprint import WorkloadFingerprint, fingerprint_workload
from workload.models import Workload

__all__ = [
    "SelectionDecision",
    "TrainingRecord",
    "EvaluationRecord",
    "WorkloadPolicySelector",
]

AnyEncoder = Union[StateEncoder, FingerprintStateEncoder]

#: How the action was obtained; explanations must not present fallbacks as learning.
SELECTION_GREEDY = "greedy"
SELECTION_EPSILON = "epsilon_exploration"
SELECTION_UNSEEN_FALLBACK = "unseen_state_fallback"


@dataclass(frozen=True)
class SelectionDecision:
    """One offline policy decision, computed without executing any policy.

    Attributes:
        workload_name: Name of the workload.
        workload_fingerprint: Content digest of the process table.
        fingerprint: Interpretable workload characteristics (process table only).
        state_index: Encoded tabular state.
        state_bins: Bin index of every state variable/feature.
        state_encoder: ``"legacy"`` for the original 4-feature state or ``"fingerprint"``
            for the workload-fingerprint state.
        state_seen_in_training: Whether the state received training updates before this
            decision (evaluation-time diagnostics).
        q_values: ``Q(state, a)`` for every action at decision time.
        action_visits: Training visit count of every ``state, action`` pair.
        action: Selected action index.
        policy_name: Name of the selected policy.
        greedy_action: Greedy action among visited actions at decision time (the
            lowest-index fallback on wholly unseen states); unvisited zero-initialised
            entries are never treated as estimates.
        explored: Whether epsilon-greedy took the random branch (training only).
        epsilon: Exploration rate used (0 during evaluation).
        objective: Declared objective the agent's values are trained/interpreted under.
        selection_rule: ``greedy``, ``epsilon_exploration`` or
            ``unseen_state_fallback`` (greedy action on a state with zero training
            visits; ties fall to the lowest index and are not learned evidence).
    """

    workload_name: str
    workload_fingerprint: str
    fingerprint: WorkloadFingerprint
    state_index: int
    state_bins: Mapping[str, int]
    state_encoder: str
    state_seen_in_training: bool
    q_values: Tuple[float, ...]
    action_visits: Tuple[int, ...]
    action: int
    policy_name: str
    greedy_action: int
    greedy_policy_name: str
    explored: bool
    epsilon: float
    objective: str
    selection_rule: str

    def as_row(self) -> Dict[str, object]:
        """Flatten the decision (without counterfactuals) for a CSV row."""
        row: Dict[str, object] = {
            "workload_name": self.workload_name,
            "workload_fingerprint": self.workload_fingerprint,
            "state_index": self.state_index,
            "state_encoder": self.state_encoder,
            "state_seen_in_training": self.state_seen_in_training,
            "action": self.action,
            "policy_name": self.policy_name,
            "greedy_action": self.greedy_action,
            "greedy_policy_name": self.greedy_policy_name,
            "explored": self.explored,
            "epsilon": self.epsilon,
            "objective": self.objective,
            "selection_rule": self.selection_rule,
        }
        row.update({f"state_{name}": value for name, value in self.state_bins.items()})
        row.update(
            {f"feature_{name}": value for name, value in self.fingerprint.as_dict().items()}
        )
        for index, name in enumerate(ACTION_NAMES):
            row[f"q_{name}"] = self.q_values[index]
            row[f"visits_{name}"] = self.action_visits[index]
        return row


@dataclass(frozen=True)
class TrainingRecord:
    """Everything one training episode produced.

    The decision is fixed first; the counterfactual report and reward are computed
    afterwards and are used only for the Q update and logging.
    """

    decision: SelectionDecision
    counterfactuals: CounterfactualReport
    breakdown: ObjectiveBreakdown
    regret: RegretRecord
    td_error: float

    def as_row(self) -> Dict[str, object]:
        """Flat row for training history tables (decision + objective evidence)."""
        row = self.decision.as_row()
        row.update(self.breakdown.as_dict())
        row.update(self.regret.as_dict())
        row["td_error"] = self.td_error
        return row


@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluated selection: decision, post-hoc counterfactuals and explanations."""

    decision: SelectionDecision
    counterfactuals: CounterfactualReport
    regret: RegretRecord
    regret_by_objective: Mapping[str, RegretRecord]

    def as_row(self) -> Dict[str, object]:
        """Flat row: decision, selected-policy outcome, objectives and regret."""
        row = self.decision.as_row()
        outcome = self.counterfactuals.outcomes[self.decision.action]
        row.update({f"selected_{k}": v for k, v in outcome.as_dict().items()})
        for objective in OBJECTIVE_NAMES:
            breakdown = self.counterfactuals.breakdowns[objective][self.decision.action]
            prefix = "risk_" if objective != OBJECTIVE_NAMES[0] else "orig_"
            row[f"{prefix}objective_cost"] = breakdown.cost
            row[f"{prefix}objective_reward"] = breakdown.reward
            for name, term in zip(breakdown.metric_names, breakdown.terms):
                row[f"{prefix}term_{name}"] = term
        row.update(self.regret.as_dict())
        for objective, record in self.regret_by_objective.items():
            if objective != self.regret.objective:
                row.update({f"alt_{k}": v for k, v in record.as_dict().items()})
        return row


class WorkloadPolicySelector:
    """Offline selector: fingerprint/state -> tabular Q-learning -> one policy.

    The selector holds no schedulers.  :meth:`select` is a pure decision function over
    the workload description and the Q-table; policies are passed to
    :meth:`train_episode` / :meth:`evaluate` only so the counterfactual evidence can be
    produced **after** the action is fixed.

    Unvisited actions are never treated as estimates: the greedy branch compares only
    actions with training visits (see :meth:`QLearningAgent.greedy_visited_action`),
    because the risk-aware terminal reward ``-J`` is non-positive and would otherwise
    lose every comparison against the zero initialisation of unvisited entries.  A
    state with no visits at all selects the documented lowest-index fallback
    (``unseen_state_fallback``); a training episode's first visit forces uniform
    exploration.  The preserved original selector (:mod:`rl.adaptive`) keeps its
    original full-space argmax behaviour; see ``docs/LIMITATIONS.md``.
    """

    def __init__(
        self,
        encoder: AnyEncoder,
        agent: QLearningAgent,
        objective: str = OBJECTIVE_RISK_AWARE,
        reward_config: RewardConfig | None = None,
        risk_config: RiskObjectiveConfig | None = None,
        mask_unvisited: bool = True,
    ) -> None:
        """
        Args:
            encoder: :class:`rl.state.StateEncoder` (original state) or
                :class:`rl.fingerprint_state.FingerprintStateEncoder`.
            agent: Tabular Q-learning agent sized to the encoder.
            objective: Declared training/evaluation objective (one of
                :data:`rl.objective.OBJECTIVE_NAMES`).
            reward_config: Weights of the original objective.
            risk_config: Weights of the risk-aware objective.
            mask_unvisited: When true (default) the greedy branch compares only
                actions with training visits.  ``False`` reproduces the preserved
                original selector's full-space argmax, where zero-initialised
                unvisited entries tie and fall to the lowest action index; the
                study's ``ql_orig`` arm uses that setting to pin legacy behaviour.

        Raises:
            ValidationError: If the components are inconsistent.
        """
        if not isinstance(encoder, (StateEncoder, FingerprintStateEncoder)):
            raise ValidationError(
                "encoder must be a StateEncoder or FingerprintStateEncoder, "
                f"got {type(encoder).__name__}"
            )
        if not isinstance(agent, QLearningAgent):
            raise ValidationError(f"agent must be a QLearningAgent, got {type(agent).__name__}")
        if objective not in OBJECTIVE_NAMES:
            raise ValidationError(
                f"objective must be one of {list(OBJECTIVE_NAMES)}, got {objective!r}"
            )
        if not isinstance(mask_unvisited, bool):
            raise ValidationError("mask_unvisited must be a bool")
        if agent.n_states != encoder.n_states or agent.n_actions != len(ACTION_NAMES):
            raise ValidationError("the agent table dimensions do not match the encoder")
        self._encoder = encoder
        self._agent = agent
        self._objective = objective
        self._reward_config = reward_config or RewardConfig()
        self._risk_config = risk_config or RiskObjectiveConfig()
        self._mask_unvisited = mask_unvisited

    # -- introspection ----------------------------------------------------------------
    @property
    def encoder(self) -> AnyEncoder:
        """The state encoder."""
        return self._encoder

    @property
    def agent(self) -> QLearningAgent:
        """The tabular Q-learning agent."""
        return self._agent

    @property
    def objective(self) -> str:
        """The declared objective of this selector."""
        return self._objective

    # -- decision --------------------------------------------------------------------
    def select(self, workload: Workload, episode: Optional[int] = None) -> SelectionDecision:
        """Choose one policy for a workload.  Executes nothing.

        Args:
            workload: The workload to decide for.
            episode: Training episode index (enables epsilon-greedy exploration), or
                ``None`` for a greedy evaluation decision.

        Returns:
            The :class:`SelectionDecision` computed from features and Q-values only.

        Raises:
            ValidationError: If the workload or episode is invalid.
        """
        if not isinstance(workload, Workload):
            raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
        if episode is not None and (
            isinstance(episode, bool) or not isinstance(episode, int) or episode < 0
        ):
            raise ValidationError(f"episode must be a non-negative integer, got {episode!r}")

        fingerprint = fingerprint_workload(workload)
        state, bins = self._encode(workload, fingerprint)
        q_values = tuple(self._agent.state_action_value(state, a) for a in range(len(ACTION_NAMES)))
        action_visits = tuple(
            int(self._agent.state_action_visit_counts[state, a]) for a in range(len(ACTION_NAMES))
        )
        state_seen = int(self._agent.visit_counts[state]) > 0
        visited = self._agent.visited_actions(state)
        # With masking, unvisited state-action values are never treated as estimates:
        # the greedy branch considers only actions with training updates.  With the
        # risk-aware objective the terminal reward is -J (non-positive), so the zero
        # initialisation would otherwise make unvisited actions dominate every
        # comparison.  Wholly unseen states use the documented lowest-index fallback.
        masked_greedy = self._agent.greedy_visited_action(state)

        if episode is None:
            epsilon = 0.0
            explored = False
            if self._mask_unvisited:
                if masked_greedy is None:
                    action = 0
                    greedy_action = 0
                    selection_rule = SELECTION_UNSEEN_FALLBACK
                else:
                    action = masked_greedy
                    greedy_action = masked_greedy
                    selection_rule = SELECTION_GREEDY
            else:
                action = self._agent.greedy_action(state)
                greedy_action = action
                selection_rule = (
                    SELECTION_GREEDY if state_seen else SELECTION_UNSEEN_FALLBACK
                )
        else:
            epsilon = self._agent.epsilon_for_episode(episode)
            if self._mask_unvisited:
                if not visited:
                    # First visit: force uniform exploration of the full action space.
                    action = self._agent.select_action(state, 1.0)
                    epsilon = 1.0
                else:
                    action = self._agent.select_action(
                        state, epsilon, greedy_actions=visited
                    )
            else:
                action = self._agent.select_action(state, epsilon)
            explored = self._agent.last_action_was_random_exploration
            greedy_action = (
                masked_greedy
                if self._mask_unvisited and masked_greedy is not None
                else self._agent.greedy_action(state)
            )
            if explored:
                selection_rule = SELECTION_EPSILON
            elif state_seen:
                selection_rule = SELECTION_GREEDY
            else:
                selection_rule = SELECTION_UNSEEN_FALLBACK

        return SelectionDecision(
            workload_name=workload.name,
            workload_fingerprint=workload.fingerprint,
            fingerprint=fingerprint,
            state_index=state,
            state_bins=bins,
            state_encoder=(
                "legacy" if isinstance(self._encoder, StateEncoder) else "fingerprint"
            ),
            state_seen_in_training=state_seen,
            q_values=q_values,
            action_visits=action_visits,
            action=action,
            policy_name=ACTION_NAMES[action],
            greedy_action=greedy_action,
            greedy_policy_name=ACTION_NAMES[greedy_action],
            explored=explored,
            epsilon=epsilon,
            objective=self._objective,
            selection_rule=selection_rule,
        )

    # -- learning --------------------------------------------------------------------
    def train_episode(
        self,
        workload: Workload,
        episode: int,
        policies: Sequence[SchedulingPolicy],
    ) -> TrainingRecord:
        """Select, then score with the four counterfactuals, then update Q once.

        The counterfactual runs happen strictly after the action is fixed; their only
        use is the terminal reward and the training log.

        Raises:
            ValidationError: If the workload, episode or policies are invalid.
        """
        decision = self.select(workload, episode=episode)
        report = run_counterfactuals(
            workload, policies, self._reward_config, self._risk_config
        )
        breakdown = compute_objective(
            self._objective,
            report.outcomes,
            decision.action,
            self._reward_config,
            self._risk_config,
        )
        regret = report.regret(self._objective, decision.action)
        td_error = self._agent.update(
            decision.state_index, decision.action, breakdown.reward, next_state=None
        )
        return TrainingRecord(
            decision=decision,
            counterfactuals=report,
            breakdown=breakdown,
            regret=regret,
            td_error=td_error,
        )

    # -- evaluation ------------------------------------------------------------------
    def evaluate(
        self,
        workload: Workload,
        policies: Sequence[SchedulingPolicy],
    ) -> EvaluationRecord:
        """Make a greedy decision, then produce post-hoc counterfactual evidence.

        The Q-table is not mutated.  The counterfactuals are computed after the
        decision and are used only for scoring, regret and explanations.

        Raises:
            ValidationError: If the workload or policies are invalid.
        """
        decision = self.select(workload, episode=None)
        report = run_counterfactuals(
            workload, policies, self._reward_config, self._risk_config
        )
        regret = report.regret(self._objective, decision.action)
        regret_by_objective = {
            objective: report.regret(objective, decision.action) for objective in OBJECTIVE_NAMES
        }
        return EvaluationRecord(
            decision=decision,
            counterfactuals=report,
            regret=regret,
            regret_by_objective=regret_by_objective,
        )

    # -- helpers ---------------------------------------------------------------------
    def _encode(
        self, workload: Workload, fingerprint: WorkloadFingerprint
    ) -> Tuple[int, Dict[str, int]]:
        if isinstance(self._encoder, StateEncoder):
            snapshot = observe_workload_state(workload)
            state = self._encoder.encode(snapshot)
            return state, self._encoder.decode(state)
        state = self._encoder.encode(fingerprint)
        return state, self._encoder.decode(state)

    def __repr__(self) -> str:
        return (
            f"WorkloadPolicySelector(objective={self._objective!r}, "
            f"n_states={self._agent.n_states})"
        )
