"""Recovery-action policies: predeclared rule, tabular Q-learning, and controls.

All three policies answer the same question at a detected degradation event: *which
recovery action should the scheduler apply now?*  They receive exactly the same causal
context -- the diagnosis produced from visible evidence, how far the rolling statistic
exceeded the threshold, how much work is ready, which policy is currently in force and how
noisy the trace has been -- and they have no access to the workload, the remaining
processes or the outcome.

* :class:`RuleHealingPolicy` is the predeclared non-learning comparison method: a fixed
  mapping from diagnosis to action with no fitted parameters. The mapping is declared
  before the held-out run and never tuned afterwards.
* :class:`QLearningHealingPolicy` is the reinforcement-learning controller: tabular
  Q-learning over the encoded context, with epsilon-greedy exploration during training,
  forced first-visit exploration so unseen contexts are sampled, masked bootstrapping over
  visited actions, and deterministic greedy decisions during evaluation.
* :class:`HoldOnlyPolicy` is the control: it never intervenes, so the arm that uses it
  measures the detection machinery's cost with no recovery at all.

The reward the learning controller maximises is the *verified* relative improvement of
the monitored waiting-time statistic over the observation window, minus an explicit
intervention cost, minus a penalty when the intervention had to be rolled back. Doing
nothing therefore scores better than a change that does not help, which is exactly the
incentive that unnecessary recovery lacks in a naive "always act" design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

from config import QLearningConfig
from errors import ValidationError
from healing.actions import (
    GROW_QUANTUM,
    HEALING_ACTION_NAMES,
    HOLD,
    SET_PRIORITY,
    SET_ROUND_ROBIN,
    SET_SJF,
    SHRINK_QUANTUM,
)
from healing.config import HealingConfig
from rl.q_learning import QLearningAgent

__all__ = [
    "FAIRNESS_RULE_MAPPING",
    "WAITING_TIME_RULE_MAPPING",
    "HealingContext",
    "HealingContextEncoder",
    "PolicyChoice",
    "HealingPolicy",
    "RuleHealingPolicy",
    "RuleWaitingTimePolicy",
    "MappedRulePolicy",
    "HoldOnlyPolicy",
    "QLearningHealingPolicy",
    "RULE_MAPPING",
]

#: Predeclared *fairness-oriented* diagnosis -> recovery mapping. It is a deterministic
#: function of the primary diagnosis and of the policy currently in force, and contains no
#: fitted parameter. ``SHRINK_QUANTUM`` is only reachable under Round Robin (the only
#: policy whose service length uses the quantum); under a non-preemptive policy the
#: declared recovery for starvation and congestion is to introduce time slicing first.
FAIRNESS_RULE_MAPPING: Tuple[Tuple[str, int], ...] = (
    ("head_of_line_blocking", SET_SJF),
    ("fairness_starvation", SET_ROUND_ROBIN),
    ("tail_dispersion", SET_SJF),
    ("priority_ordering_anomaly", SET_PRIORITY),
    ("queue_congestion", SET_ROUND_ROBIN),
    ("unexplained", HOLD),
)
#: Second choice when the declared first choice is already in force: with Round Robin
#: active, shortening the quantum is the remaining fairness lever.
FAIRNESS_RULE_FALLBACK: Tuple[Tuple[str, int], ...] = (
    ("fairness_starvation", SHRINK_QUANTUM),
    ("queue_congestion", SHRINK_QUANTUM),
)

#: Predeclared *waiting-time-oriented* diagnosis -> recovery mapping. It is declared with
#: the same information and the same audit trail as the fairness rule; the only difference
#: is which objective its author chose to serve, and the study reports both.
WAITING_TIME_RULE_MAPPING: Tuple[Tuple[str, int], ...] = (
    ("head_of_line_blocking", SET_SJF),
    ("fairness_starvation", SET_SJF),
    ("tail_dispersion", SET_SJF),
    ("priority_ordering_anomaly", SET_PRIORITY),
    ("queue_congestion", SET_SJF),
    ("unexplained", HOLD),
)
WAITING_TIME_RULE_FALLBACK: Tuple[Tuple[str, int], ...] = (
    ("head_of_line_blocking", GROW_QUANTUM),
    ("fairness_starvation", GROW_QUANTUM),
    ("tail_dispersion", GROW_QUANTUM),
    ("queue_congestion", GROW_QUANTUM),
)

#: Backwards-compatible alias for the fairness-oriented declaration.
RULE_MAPPING: Tuple[Tuple[str, int], ...] = FAIRNESS_RULE_MAPPING


def _already_in_force(action: int, current_policy: int) -> bool:
    """Whether a set-policy action is a no-op for the policy currently in force."""
    target = {
        SET_ROUND_ROBIN: 2,
        SET_SJF: 1,
        SET_PRIORITY: 3,
    }.get(action)
    return target is not None and target == current_policy


@dataclass(frozen=True)
class HealingContext:
    """Observable context of one recovery decision."""

    diagnosis_index: int
    excess_band_units: float
    ready_count: int
    current_policy: int
    relative_band: float
    remaining_work_ratio: float = 0.0

    def as_dict(self) -> dict:
        """Flat dictionary used for artifacts and the Q-table dump."""
        return {
            "diagnosis_index": self.diagnosis_index,
            "excess_band_units": self.excess_band_units,
            "ready_count": self.ready_count,
            "current_policy": self.current_policy,
            "relative_band": self.relative_band,
            "remaining_work_ratio": self.remaining_work_ratio,
        }


class HealingContextEncoder:
    """Coarse mixed-radix encoding of :class:`HealingContext` for tabular learning.

    The five features and their bins are declared here and are the only information the
    learning controller sees:

    ==========================  ==================================================
    Feature                     Bins
    ==========================  ==================================================
    Primary diagnosis           6 categories (see :mod:`healing.diagnosis`)
    Excess over threshold       ``< 1.5k``, ``< 2.5k``, ``>= 2.5k`` band widths,
                                where ``k`` is the detection multiplier
    Ready-queue size            1-2, 3-4, 5+
    Current policy              FCFS, SJF, Round Robin, Priority
    Visible remaining work      ``< 3``, ``< 8``, ``>= 8`` multiples of the reference
                                waiting time (how much work the decision can still affect)
    ==========================  ==================================================

    The state space is ``6 * 3 * 3 * 4 * 3 = 648``; with 8 recovery actions the table has
    5184 entries, small enough to be dumped in full as an artifact. The fifth feature is a
    *scale* feature on purpose: the value of changing the selection rule depends on how
    much work is still visible, and conditioning on it removed a large part of the
    reward variance that otherwise swamped the action-value differences.
    """

    RADICES: Tuple[int, ...] = (6, 3, 3, 4, 3)

    def __init__(self, config: HealingConfig) -> None:
        if not isinstance(config, HealingConfig):
            raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
        self._config = config
        total = 1
        for radix in self.RADICES:
            total *= radix
        self.n_states = total

    def feature_bins(self, context: HealingContext) -> Tuple[int, int, int, int, int]:
        """Return the five documented bins of a context."""
        if not isinstance(context, HealingContext):
            raise ValidationError("encoder accepts HealingContext only")
        if context.diagnosis_index not in range(self.RADICES[0]):
            raise ValidationError("diagnosis_index is outside the declared categories")
        if context.current_policy not in range(self.RADICES[3]):
            raise ValidationError("current_policy is outside the four-policy space")
        multiplier = self._config.threshold_band_multiplier
        excess = context.excess_band_units if context.excess_band_units == context.excess_band_units else 0.0
        if excess < 1.5 * multiplier:
            excess_bin = 0
        elif excess < 2.5 * multiplier:
            excess_bin = 1
        else:
            excess_bin = 2
        ready_count = max(1, int(context.ready_count))
        ready_bin = 0 if ready_count <= 2 else (1 if ready_count <= 4 else 2)
        work_ratio = context.remaining_work_ratio
        work_ratio = 0.0 if work_ratio != work_ratio else work_ratio
        work_bin = 0 if work_ratio < 3.0 else (1 if work_ratio < 8.0 else 2)
        return (
            int(context.diagnosis_index),
            excess_bin,
            ready_bin,
            int(context.current_policy),
            work_bin,
        )

    def encode(self, context: HealingContext) -> int:
        """Map a context to a stable mixed-radix state index."""
        state = 0
        for value, radix in zip(self.feature_bins(context), self.RADICES):
            state = state * radix + value
        return state

    def __call__(self, context: HealingContext) -> int:
        return self.encode(context)


def _normalise_allowed(allowed_actions: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Validate a masked action set (``None`` means the full declared action space)."""
    if allowed_actions is None:
        return tuple(range(len(HEALING_ACTION_NAMES)))
    actions = tuple(int(action) for action in allowed_actions)
    if HOLD not in actions:
        raise ValidationError("HOLD must always be available")
    for action in actions:
        if action not in range(len(HEALING_ACTION_NAMES)):
            raise ValidationError(f"masked action {action!r} is outside the action space")
    return tuple(sorted(set(actions)))


@dataclass(frozen=True)
class PolicyChoice:
    """One recovery-action choice with audit metadata."""

    action: int
    epsilon: float = 0.0
    explored: bool = False
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in range(len(HEALING_ACTION_NAMES)):
            raise ValidationError(f"invalid healing action {self.action!r}")

    @property
    def action_name(self) -> str:
        """Human-readable action label."""
        return HEALING_ACTION_NAMES[self.action]


@runtime_checkable
class HealingPolicy(Protocol):
    """Interface of a recovery-action policy."""

    name: str

    def select(
        self,
        context: HealingContext,
        *,
        training: bool,
        episode: int,
        allowed_actions: Optional[Tuple[int, ...]] = None,
    ) -> PolicyChoice:
        """Choose a recovery action from the causal context.

        ``allowed_actions`` is the set of actions that can actually change the live
        configuration right now (the controller always includes ``HOLD``). Actions outside
        it are not decisions at all, so a learning controller must not spend samples on
        them.
        """

    def observe_outcome(
        self,
        context: HealingContext,
        choice: PolicyChoice,
        reward: float,
        next_context: HealingContext,
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """Learn from the verified outcome; evaluation must not update."""


class MappedRulePolicy:
    """Predeclared diagnosis-to-action mapping with no fitted parameters.

    Args:
        name: Arm name recorded in the artifacts.
        mapping: Declared ``(diagnosis, action)`` pairs.
        fallback: Declared ``(diagnosis, action)`` pairs used when the first-choice action
            is already in force and would therefore be a no-op.
    """

    def __init__(
        self,
        name: str,
        mapping: Tuple[Tuple[str, int], ...],
        fallback: Tuple[Tuple[str, int], ...] = (),
    ) -> None:
        if not name:
            raise ValidationError("a rule policy needs a non-empty name")
        categories = {category for category, _ in mapping}
        if len(categories) != len(mapping):
            raise ValidationError("rule mapping contains duplicate diagnoses")
        self.name = name
        self._mapping = dict(mapping)
        self._fallback = dict(fallback)

    @property
    def mapping(self) -> Tuple[Tuple[str, str], ...]:
        """Declared mapping with human-readable action names."""
        return tuple(
            (category, HEALING_ACTION_NAMES[action])
            for category, action in sorted(self._mapping.items())
        )

    def select(
        self,
        context: HealingContext,
        *,
        training: bool,
        episode: int,
        allowed_actions: Optional[Tuple[int, ...]] = None,
    ) -> PolicyChoice:
        """Return the declared action for the primary diagnosis.

        The declared action is used when it is available and would actually change the
        configuration. Otherwise the arm falls back to its declared alternative for that
        diagnosis, and finally to ``HOLD``. Masking is honoured rather than ignored: a
        request that cannot take effect must not be recorded as a recovery attempt.
        """
        del training, episode
        if not isinstance(context, HealingContext):
            raise ValidationError("a rule policy requires a HealingContext")
        from healing.diagnosis import DIAGNOSIS_CATEGORIES

        category = DIAGNOSIS_CATEGORIES[context.diagnosis_index]
        if category not in self._mapping:
            raise ValidationError(f"rule policy {self.name!r} declares no action for {category}")
        allowed = _normalise_allowed(allowed_actions)
        declared = self._mapping[category]
        candidates = (declared, self._fallback.get(category, HOLD), HOLD)
        action = HOLD
        note = f"{self.name}: declared action for diagnosis {category}"
        for index, candidate in enumerate(candidates):
            if candidate not in allowed or _already_in_force(candidate, context.current_policy):
                continue
            action = candidate
            if index == 0:
                break
            reason = (
                "is already in force"
                if _already_in_force(declared, context.current_policy)
                else "is not available here"
            )
            note = (
                f"{self.name}: declared {HEALING_ACTION_NAMES[declared]} for diagnosis "
                f"{category} {reason}; using {HEALING_ACTION_NAMES[candidate]}"
                if candidate != HOLD
                else f"{self.name}: declared {HEALING_ACTION_NAMES[declared]} for diagnosis "
                f"{category} {reason}; holding"
            )
            break
        return PolicyChoice(action=action, note=note)

    def observe_outcome(
        self,
        context: HealingContext,
        choice: PolicyChoice,
        reward: float,
        next_context: HealingContext,
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """No-op: a declared rule has nothing to learn."""
        del context, choice, reward, next_context, terminal, training


def RuleHealingPolicy(config: HealingConfig) -> MappedRulePolicy:  # noqa: N802 - factory
    """Return the predeclared fairness-oriented rule arm."""
    if not isinstance(config, HealingConfig):
        raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
    return MappedRulePolicy(
        "predeclared_fairness_rule",
        FAIRNESS_RULE_MAPPING,
        FAIRNESS_RULE_FALLBACK,
    )


def RuleWaitingTimePolicy(config: HealingConfig) -> MappedRulePolicy:  # noqa: N802 - factory
    """Return the predeclared waiting-time-oriented rule arm."""
    if not isinstance(config, HealingConfig):
        raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")
    return MappedRulePolicy(
        "predeclared_waiting_time_rule",
        WAITING_TIME_RULE_MAPPING,
        WAITING_TIME_RULE_FALLBACK,
    )


class HoldOnlyPolicy:
    """Control policy that never intervenes (detector-only arm)."""

    name = "hold_only_control"

    def select(
        self,
        context: HealingContext,
        *,
        training: bool,
        episode: int,
        allowed_actions: Optional[Tuple[int, ...]] = None,
    ) -> PolicyChoice:
        """Always hold."""
        del context, training, episode, allowed_actions
        return PolicyChoice(action=HOLD, note="control policy never intervenes")

    def observe_outcome(
        self,
        context: HealingContext,
        choice: PolicyChoice,
        reward: float,
        next_context: HealingContext,
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """No-op control."""
        del context, choice, reward, next_context, terminal, training


class QLearningHealingPolicy:
    """Tabular Q-learning over the encoded recovery context.

    Training uses epsilon-greedy action selection with episode-decayed epsilon, and forces
    a uniformly random action the first time a context is seen so that an unseen row of
    the table is never mistaken for evidence. Bootstrapping is masked to actions that have
    actually been tried in the successor context. Evaluation is greedy, deterministic and
    read-only.
    """

    name = "q_learning_recovery"

    def __init__(
        self,
        encoder: HealingContextEncoder,
        q_config: QLearningConfig,
        seed: int,
    ) -> None:
        if not isinstance(encoder, HealingContextEncoder):
            raise ValidationError("encoder must be a HealingContextEncoder")
        if not isinstance(q_config, QLearningConfig):
            raise ValidationError("q_config must be a QLearningConfig")
        self.encoder = encoder
        self.agent = QLearningAgent(
            n_states=encoder.n_states,
            n_actions=len(HEALING_ACTION_NAMES),
            config=q_config,
            seed=seed,
        )
        self._last_choice_explored = False

    def select(
        self,
        context: HealingContext,
        *,
        training: bool,
        episode: int,
        allowed_actions: Optional[Tuple[int, ...]] = None,
    ) -> PolicyChoice:
        """Choose a recovery action; exploration happens only when ``training``.

        Only actions that can change the configuration are offered (see
        :func:`healing.controller.available_actions`), so the learned values are not spent
        on requests that are indistinguishable from holding.
        """
        if not isinstance(context, HealingContext):
            raise ValidationError("QLearningHealingPolicy requires a HealingContext")
        if not isinstance(training, bool):
            raise ValidationError("training must be a bool")
        if episode < 0:
            raise ValidationError("episode must be >= 0")
        state = self.encoder.encode(context)
        allowed = _normalise_allowed(allowed_actions)
        known = tuple(action for action in self.agent.visited_actions(state) if action in allowed)
        if training:
            epsilon = self.agent.epsilon_for_episode(episode)
            if not known:
                # An unseen context is always sampled, uniformly over the actions that can
                # actually change something.
                action = self.agent.select_action(
                    state, epsilon=1.0, exploration_actions=allowed
                )
                return PolicyChoice(
                    action=action,
                    epsilon=epsilon,
                    explored=True,
                    note="forced-unseen-context-exploration",
                )
            action = self.agent.select_action(
                state,
                epsilon=epsilon,
                greedy_actions=known,
                exploration_actions=allowed,
            )
            return PolicyChoice(
                action=action,
                epsilon=epsilon,
                explored=self.agent.last_action_was_random_exploration,
            )
        if not known:
            return PolicyChoice(
                action=HOLD,
                epsilon=0.0,
                explored=False,
                note="unseen-context-causal-hold",
            )
        action = max(known, key=lambda item: (self.agent.state_action_value(state, item), -item))
        return PolicyChoice(action=action, epsilon=0.0, explored=False)

    def observe_outcome(
        self,
        context: HealingContext,
        choice: PolicyChoice,
        reward: float,
        next_context: HealingContext,
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """Apply the Q-learning update for one verified recovery decision."""
        if not isinstance(training, bool) or not isinstance(terminal, bool):
            raise ValidationError("training and terminal flags must be bools")
        if not training:
            return
        if not isinstance(context, HealingContext) or not isinstance(next_context, HealingContext):
            raise ValidationError("learning requires healing contexts on both sides")
        state = self.encoder.encode(context)
        next_state = None if terminal else self.encoder.encode(next_context)
        self.agent.update(
            state=state,
            action=choice.action,
            reward=reward,
            next_state=next_state,
            mask_unvisited_next_actions=True,
        )


def describe_rule_mapping() -> Tuple[Tuple[str, str], ...]:
    """Return the fairness rule mapping with human-readable action names."""
    return tuple(
        (category, HEALING_ACTION_NAMES[action])
        for category, action in sorted(dict(FAIRNESS_RULE_MAPPING).items())
    )
