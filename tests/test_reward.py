"""Reward function: arithmetic, clipping, references and validation."""

from __future__ import annotations

import pytest

from config import ACTION_FCFS, ACTION_PRIORITY, ACTION_ROUND_ROBIN, ACTION_SJF, RewardConfig
from errors import ConfigurationError, ValidationError
from rl.reward import (
    BENEFIT_METRICS,
    COST_METRICS,
    REWARD_METRICS,
    compute_reward,
)
from tests.helpers import metrics_stub

#: Four runs that are identical in every metric: the reference is then the run itself.
IDENTICAL = {
    action: metrics_stub(policy_name=str(action)) for action in range(4)
}


def _attempts(**per_action: dict) -> dict:
    """Build four runs on the same workload, overriding metrics per action."""
    return {
        action: metrics_stub(policy_name=str(action), **per_action.get(str(action), {}))
        for action in range(4)
    }


class TestReferenceSemantics:
    def test_a_run_equal_to_the_four_policy_mean_scores_zero(self) -> None:
        breakdown = compute_reward(RewardConfig(), IDENTICAL, ACTION_SJF)
        assert breakdown.reward == pytest.approx(0.0)
        assert all(term == pytest.approx(1.0) for term in breakdown.normalised_terms)

    def test_the_best_of_the_four_scores_positive(self) -> None:
        attempts = _attempts(**{"1": {"avg_waiting_time": 0.5, "avg_turnaround_time": 0.5}})
        assert compute_reward(RewardConfig(), attempts, ACTION_SJF).reward > 0

    def test_the_worst_of_the_four_scores_negative(self) -> None:
        attempts = _attempts(**{"1": {"avg_waiting_time": 4.0, "avg_turnaround_time": 4.0}})
        assert compute_reward(RewardConfig(), attempts, ACTION_SJF).reward < 0

    def test_reference_values_are_the_means_of_the_four_runs(self) -> None:
        attempts = _attempts(**{"2": {"cpu_utilization": 50.0}})
        breakdown = compute_reward(RewardConfig(), attempts, ACTION_FCFS)
        utilisation_index = REWARD_METRICS.index("cpu_utilization")
        assert breakdown.reference_values[utilisation_index] == pytest.approx(
            (100.0 * 3 + 50.0) / 4
        )

    def test_all_metrics_appear_in_a_fixed_order(self) -> None:
        breakdown = compute_reward(RewardConfig(), IDENTICAL, ACTION_FCFS)
        assert len(breakdown.chosen_values) == len(COST_METRICS) + len(BENEFIT_METRICS)
        assert REWARD_METRICS[: len(COST_METRICS)] == COST_METRICS


class TestArithmetic:
    def test_single_weight_isolates_one_metric(self) -> None:
        # Only the waiting-time weight is non-zero: the reward must be exactly
        # weight * (1 - waiting / reference_waiting).
        config = RewardConfig(
            weight_waiting_time=1.0,
            weight_turnaround_time=0.0,
            weight_response_time=0.0,
            weight_context_switches=0.0,
            weight_cpu_utilization=0.0,
            weight_throughput=0.0,
        )
        attempts = _attempts(**{"3": {"avg_waiting_time": 3.0}})
        reference = (1.0 * 3 + 3.0) / 4  # three runs at 1.0 plus the chosen run at 3.0
        expected = 1.0 * (1.0 - 3.0 / reference)
        assert compute_reward(config, attempts, ACTION_PRIORITY).reward == pytest.approx(expected)

    def test_benefit_metric_above_the_reference_is_rewarded(self) -> None:
        config = RewardConfig(
            weight_waiting_time=0.0,
            weight_turnaround_time=0.0,
            weight_response_time=0.0,
            weight_context_switches=0.0,
            weight_cpu_utilization=0.0,
            weight_throughput=1.0,
        )
        attempts = _attempts(**{"0": {"throughput": 0.2}})
        # reference throughput = (0.2 + 0.1 * 3) / 4 = 0.125 -> term = 1.6
        expected = 1.0 * (0.2 / 0.125 - 1.0)
        assert compute_reward(config, attempts, ACTION_FCFS).reward == pytest.approx(expected)

    def test_ratios_are_clipped(self) -> None:
        config = RewardConfig(reference_clip=1.5)
        attempts = _attempts(**{"2": {"avg_waiting_time": 100.0}})
        breakdown = compute_reward(config, attempts, ACTION_ROUND_ROBIN)
        assert max(breakdown.normalised_terms) == pytest.approx(1.5)

    def test_zero_reference_terms_are_neutral(self) -> None:
        # Every metric is zero: an empty workload.  All terms are neutral, so the reward
        # is exactly zero regardless of the weights.
        zero = {action: metrics_stub(**{metric: 0.0 for metric in REWARD_METRICS}) for action in range(4)}
        assert compute_reward(RewardConfig(), zero, ACTION_SJF).reward == pytest.approx(0.0)

    def test_switches_are_compared_per_process(self) -> None:
        attempts = _attempts(**{"1": {"context_switches_per_process": 2.0}})
        breakdown = compute_reward(RewardConfig(), attempts, ACTION_SJF)
        index = REWARD_METRICS.index("context_switches_per_process")
        assert breakdown.chosen_values[index] == 2.0


class TestValidation:
    def test_missing_actions_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="missing actions"):
            compute_reward(RewardConfig(), {0: metrics_stub()}, ACTION_FCFS)

    def test_unknown_chosen_action_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no metrics"):
            compute_reward(RewardConfig(), IDENTICAL, 9)

    def test_different_workloads_are_rejected(self) -> None:
        attempts = dict(IDENTICAL)
        attempts[ACTION_SJF] = metrics_stub(workload_fingerprint="other")
        with pytest.raises(ValidationError, match="same workload"):
            compute_reward(RewardConfig(), attempts, ACTION_FCFS)

    def test_invalid_configuration_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="RewardConfig"):
            compute_reward("weights", IDENTICAL, ACTION_FCFS)  # type: ignore[arg-type]

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigurationError, match="sum to 1"):
            RewardConfig(weight_waiting_time=0.9, weight_cpu_utilization=0.075)

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must be >= 0"):
            RewardConfig(weight_waiting_time=-0.5, weight_throughput=0.075)
