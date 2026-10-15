"""Tests for post-decision counterfactual evaluation, regret, and isolation.

The isolation tests are the executable form of the contract: the selector's decision
function cannot run policies or read counterfactual outcomes, and the counterfactual
phase happens strictly after the decision.
"""

from __future__ import annotations

import pytest

from config import ACTION_NAMES, FingerprintStateConfig, QLearningConfig, StateConfig
from errors import ValidationError
from evaluation.counterfactual import (
    CounterfactualReport,
    RegretRecord,
    run_counterfactuals,
    selection_regret,
)
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import (
    OBJECTIVE_ORIGINAL,
    OBJECTIVE_RISK_AWARE,
    outcome_from_result,
    compute_objective,
)
from rl.q_learning import QLearningAgent
from rl.selector import (
    SELECTION_GREEDY,
    SELECTION_UNSEEN_FALLBACK,
    WorkloadPolicySelector,
)
from scheduler import POLICY_CLASSES
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, workload_from_rows

WORKLOAD_ROWS = [
    (1, 0, 3, 1),
    (2, 1, 9, 2),
    (3, 2, 2, 3),
    (4, 6, 5, 1),
    (5, 8, 1, 4),
]


def policies():
    return [cls(DEFAULT_SCHEDULER_CONFIG) for cls in POLICY_CLASSES]


def make_selector(objective: str = OBJECTIVE_RISK_AWARE) -> WorkloadPolicySelector:
    encoder = FingerprintStateEncoder(FingerprintStateConfig())
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        QLearningConfig(epsilon_start=0.0, epsilon_min=0.0),
        seed=1,
    )
    return WorkloadPolicySelector(encoder, agent, objective=objective)


class TestCounterfactualReport:
    def test_runs_all_four_policies_on_the_same_workload(self) -> None:
        workload = workload_from_rows("cf", WORKLOAD_ROWS)
        report = run_counterfactuals(workload, policies())
        assert set(report.outcomes) == set(range(4))
        assert report.workload_fingerprint == workload.fingerprint
        for outcome in report.outcomes.values():
            assert outcome.workload_fingerprint == workload.fingerprint

    def test_outcomes_match_direct_computation(self) -> None:
        workload = workload_from_rows("cf2", WORKLOAD_ROWS)
        report = run_counterfactuals(workload, policies())
        for action, policy in enumerate(policies()):
            direct = outcome_from_result(policy.run(workload))
            assert report.outcomes[action].metrics == direct.metrics
            assert report.outcomes[action].p95_waiting_time == direct.p95_waiting_time

    def test_both_objectives_are_scored_for_every_policy(self) -> None:
        workload = workload_from_rows("cf3", WORKLOAD_ROWS)
        report = run_counterfactuals(workload, policies())
        for objective in (OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE):
            for action in range(4):
                expected = compute_objective(
                    objective, report.outcomes, action
                )
                assert report.cost(objective, action) == pytest.approx(expected.cost)

    def test_as_rows_cover_every_policy(self) -> None:
        report = run_counterfactuals(workload_from_rows("cf4", WORKLOAD_ROWS), policies())
        rows = report.as_rows()
        assert len(rows) == 4
        assert {row["policy_name"] for row in rows} == set(ACTION_NAMES)
        for row in rows:
            assert "orig_objective_cost" in row
            assert "risk_objective_cost" in row
            assert "p95_waiting_time" in row

    def test_rejects_bad_inputs(self) -> None:
        workload = workload_from_rows("cf5", WORKLOAD_ROWS)
        with pytest.raises(ValidationError):
            run_counterfactuals("not a workload", policies())  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            run_counterfactuals(workload, policies()[:3])
        wrong = policies()
        wrong[0] = policies()[1]
        with pytest.raises(ValidationError):
            run_counterfactuals(workload, wrong)


class TestRegret:
    def test_regret_is_cost_gap_to_the_hindsight_best(self) -> None:
        workload = workload_from_rows("reg", WORKLOAD_ROWS)
        report = run_counterfactuals(workload, policies())
        for objective in (OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE):
            best = report.best_action(objective)
            for action in range(4):
                record = report.regret(objective, action)
                assert isinstance(record, RegretRecord)
                assert record.best_action == best
                assert record.regret == pytest.approx(
                    record.chosen_cost - record.best_cost
                )
                assert record.regret >= -1e-12
                assert record.is_hindsight_best == (action == best)

    def test_hindsight_best_regret_is_zero(self) -> None:
        report = run_counterfactuals(workload_from_rows("reg2", WORKLOAD_ROWS), policies())
        best = report.best_action(OBJECTIVE_RISK_AWARE)
        record = selection_regret(report, OBJECTIVE_RISK_AWARE, best)
        assert record.regret == pytest.approx(0.0)
        assert record.is_hindsight_best

    def test_regret_uses_declared_objective_costs(self) -> None:
        report = run_counterfactuals(workload_from_rows("reg3", WORKLOAD_ROWS), policies())
        record = report.regret(OBJECTIVE_ORIGINAL, 0)
        assert record.chosen_cost == pytest.approx(
            report.breakdowns[OBJECTIVE_ORIGINAL][0].cost
        )

    def test_rejects_unknown_objective_or_action(self) -> None:
        report = run_counterfactuals(workload_from_rows("reg4", WORKLOAD_ROWS), policies())
        with pytest.raises(ValidationError):
            report.regret("nope", 0)
        with pytest.raises(ValidationError):
            report.regret(OBJECTIVE_RISK_AWARE, 9)
        with pytest.raises(ValidationError):
            selection_regret("not a report", OBJECTIVE_RISK_AWARE, 0)  # type: ignore[arg-type]


class TestSelectionIsolation:
    def test_select_executes_no_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The decision path must not call any scheduler's run()."""
        selector = make_selector()
        workload = workload_from_rows("iso", WORKLOAD_ROWS)
        for policy in policies():
            monkeypatch.setattr(
                policy,
                "run",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("select() must not execute policies")
                ),
            )
        decision = selector.select(workload, episode=None)
        assert decision.policy_name in ACTION_NAMES

    def test_select_does_not_call_the_counterfactual_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import evaluation.counterfactual as counterfactual_module

        def explode(*args, **kwargs):
            raise AssertionError("select() must not compute counterfactuals")

        monkeypatch.setattr(counterfactual_module, "run_counterfactuals", explode)
        selector = make_selector()
        decision = selector.select(workload_from_rows("iso2", WORKLOAD_ROWS), episode=0)
        assert decision.action in range(4)

    def test_counterfactuals_cannot_change_a_decision(self) -> None:
        """Two selects around a counterfactual run return identical decisions."""
        selector = make_selector()
        workload = workload_from_rows("iso3", WORKLOAD_ROWS)
        before = selector.select(workload, episode=None)
        run_counterfactuals(workload, policies())
        after = selector.select(workload, episode=None)
        assert before == after

    def test_training_computes_counterfactuals_only_after_the_action(self) -> None:
        """In train_episode the decision must be complete before policies run."""
        selector = make_selector()
        workload = workload_from_rows("iso4", WORKLOAD_ROWS)
        call_order = []
        original_select = selector.select

        def tracking_select(*args, **kwargs):
            decision = original_select(*args, **kwargs)
            call_order.append(("select", decision.action))
            return decision

        recorded_policies = policies()
        for policy in recorded_policies:
            original_run = policy.run

            def tracking_run(workload, _original=original_run, _name=policy.name):
                call_order.append(("run", _name))
                return _original(workload)

            policy.run = tracking_run  # type: ignore[method-assign]

        selector.select = tracking_select  # type: ignore[method-assign]
        record = selector.train_episode(workload, episode=0, policies=recorded_policies)
        assert call_order[0][0] == "select"
        assert all(kind == "run" for kind, _ in call_order[1:])
        assert record.decision.action in range(4)
        assert len([entry for entry in call_order if entry[0] == "run"]) == 4


class TestEvaluationRecordRows:
    def test_evaluation_row_contains_objectives_and_regret(self) -> None:
        selector = make_selector()
        record = selector.evaluate(workload_from_rows("row", WORKLOAD_ROWS), policies())
        row = record.as_row()
        assert row["policy_name"] == record.decision.policy_name
        assert "risk_objective_cost" in row
        assert "orig_objective_cost" in row
        assert "regret" in row
        assert row["workload_fingerprint"] == record.decision.workload_fingerprint

    def test_evaluation_does_not_mutate_the_q_table(self) -> None:
        selector = make_selector()
        before = selector.agent.q_table.copy()
        selector.evaluate(workload_from_rows("ro", WORKLOAD_ROWS), policies())
        assert (selector.agent.q_table == before).all()
