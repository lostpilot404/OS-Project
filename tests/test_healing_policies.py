"""Tests of the recovery policies: declared rules, controls, masking and Q-learning."""

from __future__ import annotations

import pytest

from config import ACTION_ROUND_ROBIN, QLearningConfig
from errors import ValidationError
from healing.actions import (
    GROW_QUANTUM,
    HEALING_ACTION_NAMES,
    HOLD,
    SET_FCFS,
    SET_PRIORITY,
    SET_ROUND_ROBIN,
    SET_SJF,
    SHRINK_QUANTUM,
    SchedulerSettings,
)
from healing.config import HealingConfig
from healing.diagnosis import DIAGNOSIS_CATEGORIES
from healing.policies import (
    HoldOnlyPolicy,
    HealingContext,
    HealingContextEncoder,
    QLearningHealingPolicy,
    RuleHealingPolicy,
    RuleWaitingTimePolicy,
    describe_rule_mapping,
)


def _context(**overrides) -> HealingContext:
    values = {
        "diagnosis_index": 0,
        "excess_band_units": 2.0,
        "ready_count": 3,
        "current_policy": 0,
        "relative_band": 0.2,
        "remaining_work_ratio": 4.0,
    }
    values.update(overrides)
    return HealingContext(**values)


def test_context_encoding_is_stable_and_bounded() -> None:
    """The declared 648-state encoding is deterministic and covers the declared bins."""
    encoder = HealingContextEncoder(HealingConfig())
    assert encoder.n_states == 648
    indices = set()
    for diagnosis in range(len(DIAGNOSIS_CATEGORIES)):
        for excess in (0.0, 2.0, 9.0):
            for ready in (1, 3, 9):
                for policy in range(4):
                    for ratio in (0.0, 4.0, 20.0):
                        state = encoder.encode(
                            _context(
                                diagnosis_index=diagnosis,
                                excess_band_units=excess,
                                ready_count=ready,
                                current_policy=policy,
                                remaining_work_ratio=ratio,
                            )
                        )
                        assert 0 <= state < encoder.n_states
                        indices.add(state)
    assert len(indices) == 648, "every declared bin combination must be reachable"
    assert encoder.encode(_context()) == encoder.encode(_context())
    # A non-finite excess must not create an out-of-range state.
    assert encoder.encode(_context(excess_band_units=float("nan"))) < encoder.n_states
    with pytest.raises(ValidationError):
        encoder.encode(_context(diagnosis_index=99))
    with pytest.raises(ValidationError):
        encoder.encode("not-a-context")  # type: ignore[arg-type]


def test_declared_rules_are_fixed_and_complete() -> None:
    """Both rule arms map every diagnosis to a declared action and never change by itself."""
    config = HealingConfig()
    fairness = RuleHealingPolicy(config)
    waiting = RuleWaitingTimePolicy(config)
    mapping = dict(describe_rule_mapping())
    assert len(mapping) == len(DIAGNOSIS_CATEGORIES)
    assert fairness.name == "predeclared_fairness_rule"
    assert waiting.name == "predeclared_waiting_time_rule"
    for index in range(len(DIAGNOSIS_CATEGORIES)):
        first = fairness.select(_context(diagnosis_index=index), training=False, episode=0)
        second = fairness.select(_context(diagnosis_index=index), training=False, episode=0)
        assert first == second, "rule policies must be deterministic"
        assert first.action in range(len(HEALING_ACTION_NAMES))
        assert first.epsilon == 0.0 and first.explored is False
        assert waiting.select(_context(diagnosis_index=index), training=False, episode=0).action in range(
            len(HEALING_ACTION_NAMES)
        )
    # The controls the rules are built from differ on at least one diagnosis, so the two
    # arms are not the same experiment twice.
    assert any(
        fairness.select(_context(diagnosis_index=i), training=False, episode=0).action
        != waiting.select(_context(diagnosis_index=i), training=False, episode=0).action
        for i in range(len(DIAGNOSIS_CATEGORIES))
    )


def test_rule_never_requests_a_change_that_is_already_in_force() -> None:
    """A set-policy rule asked for the policy already running must not claim a change."""
    config = HealingConfig()
    waiting = RuleWaitingTimePolicy(config)
    # The waiting-time declaration maps fairness starvation onto SJF.
    starvation = DIAGNOSIS_CATEGORIES.index("fairness_starvation")
    choice = waiting.select(
        _context(diagnosis_index=starvation, current_policy=1), training=False, episode=0
    )
    assert choice.note is not None and "already in force" in choice.note.lower()
    assert choice.action != SET_SJF


def test_hold_only_control_never_intervenes() -> None:
    """The detection control holds for every context, including during training."""
    control = HoldOnlyPolicy()
    for diagnosis in range(len(DIAGNOSIS_CATEGORIES)):
        choice = control.select(_context(diagnosis_index=diagnosis), training=True, episode=5)
        assert choice.action == HOLD
        assert control.select(
            _context(), training=False, episode=0, allowed_actions=(HOLD, SET_SJF)
        ).action == HOLD
    control.observe_outcome(None, None, 1.0, None, terminal=True, training=True)


def test_masked_action_set_is_validated_and_respected() -> None:
    """Unavailable actions are never selected, and HOLD must always remain available."""
    config = HealingConfig()
    policy = RuleHealingPolicy(config)
    only_hold = policy.select(_context(), training=False, episode=0, allowed_actions=(HOLD,))
    assert only_hold.action == HOLD, "a rule arm must hold rather than request an unavailable action"
    assert "holding" in (only_hold.note or "").lower()
    no_change_possible = policy.select(
        _context(), training=False, episode=0, allowed_actions=(HOLD, SET_SJF)
    )
    assert no_change_possible.action in (HOLD, SET_SJF)
    with pytest.raises(ValidationError, match="HOLD must always be available"):
        policy.select(_context(), training=False, episode=0, allowed_actions=(SET_SJF,))
    with pytest.raises(ValidationError, match="outside the action space"):
        policy.select(_context(), training=False, episode=0, allowed_actions=(HOLD, 99))
    # A rule that declares SJF must not request it while SJF is already in force.
    starving = _context(
        diagnosis_index=DIAGNOSIS_CATEGORIES.index("fairness_starvation"), current_policy=1
    )
    assert RuleWaitingTimePolicy(config).select(
        starving, training=False, episode=0, allowed_actions=(HOLD, SET_SJF)
    ).action == HOLD
    assert SchedulerSettings(policy=1, quantum=4).policy_name == "SJF"


def test_q_learning_policy_explores_then_uses_what_it_learned() -> None:
    """Forced first visits, masked exploration, deterministic greedy evaluation, and a
    real value update on the visited state-action pair."""
    config = HealingConfig()
    encoder = HealingContextEncoder(config)
    policy = QLearningHealingPolicy(
        encoder,
        QLearningConfig(
            learning_rate=0.5,
            discount_factor=1.0,
            epsilon_start=0.0,
            epsilon_min=0.0,
        ),
        seed=11,
    )
    context = _context(diagnosis_index=DIAGNOSIS_CATEGORIES.index("fairness_starvation"))
    allowed = (HOLD, SET_SJF, SET_ROUND_ROBIN)
    first = policy.select(context, training=True, episode=0, allowed_actions=allowed)
    assert first.explored is True
    assert first.action in allowed

    # Deterministic greedy evaluation: repeated calls return the same action.
    policy.observe_outcome(context, first, reward=1.0, next_context=context, terminal=True, training=True)
    table_before = policy.agent.q_table.copy()
    greedy = policy.select(context, training=False, episode=0, allowed_actions=allowed)
    assert greedy == policy.select(context, training=False, episode=0, allowed_actions=allowed)
    assert greedy.explored is False
    assert (policy.agent.q_table == table_before).all(), "evaluation must not learn"

    # A terminal update on a visited action must move the action value.
    state = encoder.encode(context)
    assert policy.agent.visit_counts[state] > 0
    assert policy.agent.state_action_visit_counts[state][first.action] > 0
    assert policy.agent.q_table[state][first.action] == pytest.approx(0.5)
    assert first.action in policy.agent.visited_actions(state)

    with pytest.raises(ValidationError, match="HealingContext"):
        policy.select("nope", training=False, episode=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="episode"):
        policy.select(context, training=True, episode=-1)


def test_q_learning_is_deterministic_for_a_given_seed() -> None:
    """Two agents with the same seed produce the same decisions and the same table."""
    def train(seed: int):
        policy = QLearningHealingPolicy(
            HealingContextEncoder(HealingConfig()),
            QLearningConfig(learning_rate=0.3, discount_factor=1.0, epsilon_start=1.0),
            seed=seed,
        )
        contexts = [
            _context(diagnosis_index=index % len(DIAGNOSIS_CATEGORIES), ready_count=1 + index % 7)
            for index in range(40)
        ]
        actions = []
        for episode in range(6):
            for context in contexts:
                choice = policy.select(context, training=True, episode=episode)
                policy.observe_outcome(
                    context,
                    choice,
                    reward=0.5 if choice.action == HOLD else -0.25,
                    next_context=context,
                    terminal=False,
                    training=True,
                )
                actions.append(choice.action)
        return actions, policy.agent.q_table.copy()

    first_actions, first_table = train(7)
    second_actions, second_table = train(7)
    third_actions, _ = train(8)
    assert first_actions == second_actions
    assert (first_table == second_table).all()
    assert first_actions != third_actions, "the declared seed must be the only source of variation"


def test_grow_and_shrink_are_distinct_actions_under_round_robin() -> None:
    """The quantum actions exist in the declared space and act on the quantum only."""
    from healing.actions import apply_healing_action

    rr = SchedulerSettings(policy=ACTION_ROUND_ROBIN, quantum=4)
    grown = apply_healing_action(GROW_QUANTUM, current=rr, last_accepted=None, min_quantum=1, max_quantum=16)
    shrunk = apply_healing_action(SHRINK_QUANTUM, current=rr, last_accepted=None, min_quantum=1, max_quantum=16)
    assert grown.settings.policy == ACTION_ROUND_ROBIN and grown.settings.quantum == 8
    assert shrunk.settings.policy == ACTION_ROUND_ROBIN and shrunk.settings.quantum == 2
    assert HEALING_ACTION_NAMES[SET_FCFS] == "set_fcfs"
    assert HEALING_ACTION_NAMES[SET_PRIORITY] == "set_priority"
