"""Tests for the objective module: original reward preservation and ``risk_aware_v1``.

Hand-computed expected values pin the normalisation, weights and risk criterion; the
original objective must reproduce :func:`rl.reward.compute_reward` exactly.
"""

from __future__ import annotations

import math

import pytest

from config import ACTION_NAMES, RewardConfig, RiskObjectiveConfig
from errors import ValidationError
from rl.objective import (
    OBJECTIVE_NAMES,
    OBJECTIVE_ORIGINAL,
    OBJECTIVE_RISK_AWARE,
    RISK_OBJECTIVE_METRICS,
    ObjectiveBreakdown,
    PolicyOutcome,
    compute_all_objectives,
    compute_objective,
    outcome_from_result,
    percentile_waiting_time,
)
from rl.reward import REWARD_METRICS, compute_reward
from scheduler import FCFS
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, make_metrics, workload_from_rows


def make_outcome(
    wait: float,
    response: float,
    switches: float,
    p95_wait: float,
    fingerprint: str = "f" * 16,
) -> PolicyOutcome:
    return PolicyOutcome(
        metrics=make_metrics(
            avg_waiting_time=wait,
            avg_response_time=response,
            avg_turnaround_time=wait + 2.0,
            context_switches_per_process=switches,
            cpu_utilization=90.0,
            throughput=0.1,
            fingerprint=fingerprint,
        ),
        p95_waiting_time=p95_wait,
    )


def four_attempts() -> dict:
    """Four synthetic policy outcomes with clean four-policy references.

    References: waiting 10, response 7, switches/proc 1.125, p95 waiting 14.5.
    """
    return {
        0: make_outcome(10.0, 10.0, 0.5, 12.0),  # FCFS
        1: make_outcome(6.0, 6.0, 0.5, 20.0),  # SJF: good mean, bad tail
        2: make_outcome(14.0, 2.0, 3.0, 14.0),  # Round Robin
        3: make_outcome(10.0, 10.0, 0.5, 12.0),  # Priority
    }


class TestRiskObjective:
    def test_hand_computed_cost_and_terms(self) -> None:
        attempts = four_attempts()
        breakdown = compute_objective(OBJECTIVE_RISK_AWARE, attempts, chosen_action=1)

        assert breakdown.metric_names == RISK_OBJECTIVE_METRICS
        assert breakdown.reference_values == pytest.approx((10.0, 7.0, 1.125, 14.5))
        expected_terms = (
            6.0 / 10.0,
            6.0 / 7.0,
            0.5 / 1.125,
            20.0 / 14.5,
        )
        assert breakdown.terms == pytest.approx(expected_terms)
        expected_cost = (
            0.35 * expected_terms[0]
            + 0.25 * expected_terms[1]
            + 0.15 * expected_terms[2]
            + 0.25 * expected_terms[3]
        )
        assert breakdown.cost == pytest.approx(expected_cost)
        assert breakdown.reward == pytest.approx(-breakdown.cost)

    def test_reference_match_scores_exactly_one(self) -> None:
        """A policy equal to the four-policy mean on every metric scores J = 1."""
        identical = make_outcome(10.0, 7.0, 1.125, 14.5)
        attempts = {action: identical for action in range(4)}
        breakdown = compute_objective(OBJECTIVE_RISK_AWARE, attempts, chosen_action=3)
        assert breakdown.cost == pytest.approx(1.0)

    def test_terms_are_clipped_at_the_configured_maximum(self) -> None:
        attempts = four_attempts()
        attempts[2] = make_outcome(50.0, 50.0, 20.0, 50.0)  # every ratio clips
        breakdown = compute_objective(OBJECTIVE_RISK_AWARE, attempts, chosen_action=2)
        assert set(breakdown.terms) == {2.0}
        assert breakdown.cost == pytest.approx(2.0)

    def test_zero_reference_maps_to_neutral_term(self) -> None:
        attempts = {
            0: make_outcome(10.0, 10.0, 0.0, 12.0),
            1: make_outcome(6.0, 6.0, 0.0, 8.0),
            2: make_outcome(14.0, 2.0, 0.0, 14.0),
            3: make_outcome(10.0, 10.0, 0.0, 12.0),
        }
        breakdown = compute_objective(OBJECTIVE_RISK_AWARE, attempts, chosen_action=1)
        # reference for context switches per process is 0 -> term is exactly 1
        assert breakdown.terms[2] == 1.0

    def test_risk_criterion_penalises_bad_tail(self) -> None:
        """Lower mean but worse p95 can lose to a balanced policy under risk_aware_v1."""
        attempts = four_attempts()
        sjf = compute_objective(OBJECTIVE_RISK_AWARE, attempts, 1)
        fcfs = compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0)
        assert attempts[1].metrics.avg_waiting_time < attempts[0].metrics.avg_waiting_time
        assert attempts[1].p95_waiting_time > attempts[0].p95_waiting_time
        # With the default weights the SJF arm still wins overall; the tail term must
        # nevertheless be the only term where FCFS beats SJF among the cost terms.
        assert sjf.cost < fcfs.cost
        assert sjf.terms[3] > fcfs.terms[3]

    def test_tail_weight_changes_the_ranking(self) -> None:
        """Raising the risk weight flips the SJF-vs-FCFS ranking on these outcomes."""
        attempts = four_attempts()
        tail_heavy = RiskObjectiveConfig(
            weight_waiting_time=0.1,
            weight_response_time=0.1,
            weight_context_switches=0.1,
            weight_tail_waiting=0.7,
        )
        sjf = compute_objective(OBJECTIVE_RISK_AWARE, attempts, 1, risk_config=tail_heavy)
        fcfs = compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0, risk_config=tail_heavy)
        assert fcfs.cost < sjf.cost

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError):
            RiskObjectiveConfig(weight_tail_waiting=0.5)


class TestOriginalObjectivePreserved:
    def test_matches_compute_reward_exactly(self) -> None:
        attempts = four_attempts()
        for action in range(4):
            breakdown = compute_objective(OBJECTIVE_ORIGINAL, attempts, action)
            reference = compute_reward(
                RewardConfig(),
                {a: o.metrics for a, o in attempts.items()},
                action,
            )
            assert breakdown.reward == reference.reward
            assert breakdown.cost == -reference.reward
            assert breakdown.terms == reference.normalised_terms
            assert breakdown.chosen_values == reference.chosen_values
            assert breakdown.reference_values == reference.reference_values
            assert breakdown.metric_names == REWARD_METRICS

    def test_custom_reward_config_is_used(self) -> None:
        attempts = four_attempts()
        config = RewardConfig(
            weight_waiting_time=1.0,
            weight_turnaround_time=0.0,
            weight_response_time=0.0,
            weight_context_switches=0.0,
            weight_cpu_utilization=0.0,
            weight_throughput=0.0,
        )
        breakdown = compute_objective(OBJECTIVE_ORIGINAL, attempts, 1, reward_config=config)
        assert breakdown.reward == pytest.approx(1.0 - 6.0 / 10.0)


class TestBothObjectivesTogether:
    def test_all_objectives_cover_every_action(self) -> None:
        attempts = four_attempts()
        everything = compute_all_objectives(attempts)
        assert set(everything) == set(OBJECTIVE_NAMES)
        for objective, per_action in everything.items():
            assert set(per_action) == set(range(len(ACTION_NAMES)))
            for action, breakdown in per_action.items():
                assert breakdown.objective == objective
                assert breakdown.action == action

    def test_objective_values_are_comparable_costs(self) -> None:
        attempts = four_attempts()
        everything = compute_all_objectives(attempts)
        for per_action in everything.values():
            costs = [per_action[a].cost for a in range(4)]
            assert all(math.isfinite(cost) for cost in costs)


class TestValidation:
    def test_unknown_objective_rejected(self) -> None:
        with pytest.raises(ValidationError):
            compute_objective("nope", four_attempts(), 0)

    def test_missing_or_extra_actions_rejected(self) -> None:
        attempts = four_attempts()
        del attempts[2]
        with pytest.raises(ValidationError):
            compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0)
        attempts = four_attempts()
        attempts[4] = make_outcome(1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValidationError):
            compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0)

    def test_mixed_workloads_rejected(self) -> None:
        attempts = four_attempts()
        attempts[2] = make_outcome(1.0, 1.0, 1.0, 1.0, fingerprint="a" * 16)
        with pytest.raises(ValidationError):
            compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0)

    def test_chosen_action_without_outcome_rejected(self) -> None:
        with pytest.raises(ValidationError):
            compute_objective(OBJECTIVE_RISK_AWARE, four_attempts(), 9)

    def test_non_policy_outcomes_rejected(self) -> None:
        attempts = four_attempts()
        attempts[1] = "not an outcome"  # type: ignore[assignment]
        with pytest.raises(ValidationError):
            compute_objective(OBJECTIVE_RISK_AWARE, attempts, 0)


class TestTailStatistic:
    def test_nearest_rank_on_a_real_schedule(self) -> None:
        workload = workload_from_rows(
            "waits", [(1, 0, 1, 1), (2, 0, 2, 1), (3, 0, 3, 1), (4, 0, 4, 1), (5, 0, 5, 1)]
        )
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        waits = sorted(outcome.waiting_time for outcome in result.outcomes)
        assert waits == [0, 1, 3, 6, 10]
        assert percentile_waiting_time(result, 0.95) == 10.0
        assert percentile_waiting_time(result, 0.5) == 3.0

    def test_outcome_from_result_pairs_metrics_and_tail(self) -> None:
        workload = workload_from_rows("pair", [(1, 0, 2, 1), (2, 0, 2, 1)])
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        outcome = outcome_from_result(result)
        # FCFS waits are [0, 2] -> mean 1.0; nearest-rank p95 (index 1) -> 2.0
        assert outcome.metrics.avg_waiting_time == pytest.approx(1.0)
        assert outcome.p95_waiting_time == pytest.approx(2.0)

    def test_invalid_tail_requests_rejected(self) -> None:
        workload = workload_from_rows("bad", [(1, 0, 2, 1)])
        result = FCFS(DEFAULT_SCHEDULER_CONFIG).run(workload)
        with pytest.raises(ValidationError):
            percentile_waiting_time(result, 1.0)
        with pytest.raises(ValidationError):
            percentile_waiting_time(result, 0.0)
        with pytest.raises(ValidationError):
            percentile_waiting_time("not a result", 0.5)  # type: ignore[arg-type]


class TestBreakdownRecord:
    def test_as_dict_contains_every_term(self) -> None:
        breakdown = compute_objective(OBJECTIVE_RISK_AWARE, four_attempts(), 1)
        record = breakdown.as_dict()
        assert record["objective"] == OBJECTIVE_RISK_AWARE
        assert record["action"] == 1
        for name in RISK_OBJECTIVE_METRICS:
            assert f"term_{name}" in record
            assert f"chosen_{name}" in record
            assert f"reference_{name}" in record
        assert isinstance(breakdown, ObjectiveBreakdown)
