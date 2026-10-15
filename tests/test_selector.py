"""Tests for the offline workload policy selector (decision path and training)."""

from __future__ import annotations

import pytest

from config import (
    ACTION_NAMES,
    FingerprintStateConfig,
    QLearningConfig,
    RewardConfig,
    RiskObjectiveConfig,
    StateConfig,
)
from errors import ValidationError
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE, compute_objective
from rl.q_learning import QLearningAgent
from rl.selector import (
    SELECTION_EPSILON,
    SELECTION_GREEDY,
    SELECTION_UNSEEN_FALLBACK,
    WorkloadPolicySelector,
)
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, workload_from_rows

ROWS = [(1, 0, 3, 1), (2, 1, 9, 2), (3, 2, 2, 3), (4, 6, 5, 1)]


def policies():
    return [cls(DEFAULT_SCHEDULER_CONFIG) for cls in POLICY_CLASSES]


def make_selector(
    objective: str = OBJECTIVE_RISK_AWARE,
    encoder_kind: str = "fingerprint",
    epsilon_start: float = 0.0,
    seed: int = 7,
) -> WorkloadPolicySelector:
    if encoder_kind == "fingerprint":
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
    else:
        encoder = StateEncoder(StateConfig())
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        QLearningConfig(epsilon_start=epsilon_start, epsilon_min=0.0),
        seed=seed,
    )
    return WorkloadPolicySelector(encoder, agent, objective=objective)


class TestDecisionPath:
    def test_decision_contains_features_and_q_values(self) -> None:
        selector = make_selector()
        decision = selector.select(workload_from_rows("d", ROWS))
        assert decision.objective == OBJECTIVE_RISK_AWARE
        assert len(decision.q_values) == 4
        assert len(decision.action_visits) == 4
        assert decision.fingerprint.process_count == 4
        assert decision.state_encoder == "fingerprint"
        assert decision.selection_rule == SELECTION_UNSEEN_FALLBACK

    def test_unseen_state_is_marked_as_fallback_not_learning(self) -> None:
        selector = make_selector()
        decision = selector.select(workload_from_rows("d2", ROWS), episode=None)
        assert decision.state_seen_in_training is False
        assert decision.selection_rule == SELECTION_UNSEEN_FALLBACK
        # deterministic tie-break to the lowest action index
        assert decision.action == 0
        assert decision.policy_name == "FCFS"

    def test_greedy_selection_follows_learned_values(self) -> None:
        selector = make_selector()
        state = selector.select(workload_from_rows("d3", ROWS)).state_index
        # teach the agent that SJF is best in this state
        for _ in range(5):
            selector.agent.update(state, 1, reward=1.0, next_state=None)
        decision = selector.select(workload_from_rows("d3", ROWS), episode=None)
        assert decision.state_seen_in_training is True
        assert decision.selection_rule == SELECTION_GREEDY
        assert decision.action == 1
        assert decision.policy_name == "SJF"
        assert decision.greedy_action == 1

    def test_training_selection_records_exploration(self) -> None:
        selector = make_selector(epsilon_start=1.0, seed=3)
        decision = selector.select(workload_from_rows("d4", ROWS), episode=0)
        assert decision.epsilon == 1.0
        assert decision.explored is True
        assert decision.selection_rule == SELECTION_EPSILON

    def test_legacy_state_encoder_is_supported(self) -> None:
        selector = make_selector(encoder_kind="legacy")
        decision = selector.select(workload_from_rows("d5", ROWS))
        assert decision.state_encoder == "legacy"
        assert set(decision.state_bins) == set(StateConfig().state_variables)


class TestTrainingEpisodes:
    def test_q_update_uses_negative_risk_cost(self) -> None:
        selector = make_selector(objective=OBJECTIVE_RISK_AWARE)
        workload = workload_from_rows("t", ROWS)
        decision = selector.select(workload, episode=0)
        record = selector.train_episode(workload, episode=0, policies=policies())
        state = decision.state_index
        action = record.decision.action
        # terminal update: Q <- alpha * reward with Q starting at 0
        expected_q = 0.1 * record.breakdown.reward
        assert record.breakdown.reward == pytest.approx(-record.breakdown.cost)
        assert selector.agent.state_action_value(state, action) == pytest.approx(expected_q)
        assert record.td_error == pytest.approx(record.breakdown.reward)

    def test_original_objective_training_uses_original_reward(self) -> None:
        selector = make_selector(objective=OBJECTIVE_ORIGINAL)
        workload = workload_from_rows("t2", ROWS)
        record = selector.train_episode(workload, episode=0, policies=policies())
        assert record.breakdown.objective == OBJECTIVE_ORIGINAL
        assert record.breakdown.reward > -100
        assert record.breakdown.reward == pytest.approx(-record.breakdown.cost)

    def test_training_record_row_has_objective_and_regret(self) -> None:
        selector = make_selector()
        record = selector.train_episode(
            workload_from_rows("t3", ROWS), episode=1, policies=policies()
        )
        row = record.as_row()
        assert "objective_cost" in row
        assert "regret" in row
        assert "td_error" in row
        assert "feature_burst_cv" in row

    def test_reproducible_training_with_fixed_seed(self) -> None:
        results = []
        for _ in range(2):
            selector = make_selector(epsilon_start=0.5, seed=11)
            for episode in range(8):
                selector.train_episode(
                    workload_from_rows("t4", ROWS), episode=episode, policies=policies()
                )
            results.append(selector.agent.q_table.copy())
        assert (results[0] == results[1]).all()


class TestValidation:
    def test_rejects_inconsistent_components(self) -> None:
        encoder = FingerprintStateEncoder(FingerprintStateConfig())
        agent = QLearningAgent(5, 4, QLearningConfig(), seed=0)
        with pytest.raises(ValidationError):
            WorkloadPolicySelector(encoder, agent)
        with pytest.raises(ValidationError):
            WorkloadPolicySelector(object(), QLearningAgent(243, 4, QLearningConfig(), seed=0))
        with pytest.raises(ValidationError):
            WorkloadPolicySelector(
                encoder, QLearningAgent(243, 4, QLearningConfig(), seed=0), objective="nope"
            )

    def test_rejects_bad_workload_or_episode(self) -> None:
        selector = make_selector()
        with pytest.raises(ValidationError):
            selector.select("not a workload")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            selector.select(workload_from_rows("v", ROWS), episode=-1)
        with pytest.raises(ValidationError):
            selector.select(workload_from_rows("v", ROWS), episode=True)
