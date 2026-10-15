"""Tests for structured decision explanations.

The core property: every claim in an explanation is derived from recomputable
evidence.  These tests recompute the evidence independently (from the counterfactual
report and the decision) and assert that the explanation's statements, evidence fields
and verdicts agree with it - including the negative case where the selector did not
pick the hindsight-best policy.
"""

from __future__ import annotations

import json

import pytest

from config import ACTION_NAMES, FingerprintStateConfig, QLearningConfig, RiskObjectiveConfig
from errors import ValidationError
from evaluation.counterfactual import run_counterfactuals
from rl.explain import (
    EXPLANATION_SCHEMA_VERSION,
    INFORMATION_BOUNDARY,
    build_explanation,
    explain_evaluation_record,
)
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE
from rl.q_learning import QLearningAgent
from rl.selector import SELECTION_GREEDY, SELECTION_UNSEEN_FALLBACK, WorkloadPolicySelector
from scheduler import POLICY_CLASSES
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, workload_from_rows
from workload.fingerprint import FINGERPRINT_FEATURES

ROWS = [(1, 0, 3, 1), (2, 1, 9, 2), (3, 2, 2, 3), (4, 6, 5, 1)]


def policies():
    return [cls(DEFAULT_SCHEDULER_CONFIG) for cls in POLICY_CLASSES]


def evaluate_rows():
    encoder = FingerprintStateEncoder(FingerprintStateConfig())
    agent = QLearningAgent(
        encoder.n_states,
        len(ACTION_NAMES),
        QLearningConfig(epsilon_start=0.0, epsilon_min=0.0),
        seed=5,
    )
    selector = WorkloadPolicySelector(encoder, agent, objective=OBJECTIVE_RISK_AWARE)
    workload = workload_from_rows("exp", ROWS)
    record = selector.evaluate(workload, policies())
    return selector, record


class TestExplanationStructure:
    def test_contains_all_required_sections(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        assert explanation.schema_version == EXPLANATION_SCHEMA_VERSION
        assert explanation.information_boundary == INFORMATION_BOUNDARY
        assert set(explanation.workload_characteristics) == set(FINGERPRINT_FEATURES)
        assert set(explanation.counterfactual_objectives) == set(ACTION_NAMES)
        assert explanation.decision["policy_name"] == record.decision.policy_name
        assert explanation.declared_objective["name"] == OBJECTIVE_RISK_AWARE
        assert "risk_criterion" in explanation.declared_objective
        assert explanation.reasons

    def test_json_serialisable(self) -> None:
        _, record = evaluate_rows()
        payload = explain_evaluation_record(record).as_dict()
        text = json.dumps(payload)
        assert json.loads(text) == payload

    def test_decision_time_and_post_hoc_evidence_are_labelled(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        kinds = {reason.kind for reason in explanation.reasons}
        assert "selection_basis" in kinds
        assert "objective_rank" in kinds
        assert "workload_characteristics" in kinds
        rank = next(r for r in explanation.reasons if r.kind == "objective_rank")
        assert "Post-hoc" in rank.statement


class TestEvidenceConsistency:
    def test_counterfactual_costs_match_the_report(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        for action, name in enumerate(ACTION_NAMES):
            entry = explanation.counterfactual_objectives[name]
            for objective, cost in entry["costs"].items():
                expected = record.counterfactuals.cost(objective, action)
                assert cost == pytest.approx(expected)

    def test_regret_matches_recomputation(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        regret = record.counterfactuals.regret(
            record.decision.objective, record.decision.action
        )
        assert explanation.regret["regret"] == pytest.approx(regret.regret)
        assert explanation.regret["regret_best_action"] == regret.best_action

    def test_q_value_evidence_matches_the_decision(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        basis = next(r for r in explanation.reasons if r.kind == "selection_basis")
        for name, value in zip(ACTION_NAMES, record.decision.q_values):
            assert basis.evidence["q_values"][name] == pytest.approx(value)

    def test_objective_rank_claims_match_the_costs(self) -> None:
        """The verdict in the statement must agree with the computed costs."""
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        rank = next(r for r in explanation.reasons if r.kind == "objective_rank")
        chosen = record.decision.action
        best = record.counterfactuals.best_action(record.decision.objective)
        if chosen == best:
            assert "the lowest cost" in rank.statement
            assert rank.evidence["regret"] == pytest.approx(0.0)
        else:
            assert "was not the hindsight minimum" in rank.statement
            assert rank.evidence["regret"] > 0.0
            assert "selector did not know" in rank.statement

    def test_term_comparison_reasons_only_claim_real_minima(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        objective = record.decision.objective
        chosen = record.decision.action
        breakdowns = record.counterfactuals.breakdowns[objective]
        claimed = None
        for reason in explanation.reasons:
            if reason.kind == "term_comparison":
                claimed = set(reason.evidence["favorable_terms"])
        truly_best = set()
        for position, metric in enumerate(breakdowns[chosen].metric_names):
            terms = {a: breakdowns[a].terms[position] for a in breakdowns}
            if terms[chosen] < min(terms[a] for a in terms if a != chosen):
                truly_best.add(metric)
        assert (claimed or set()) == truly_best

    def test_term_tradeoff_lists_only_real_losses(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        objective = record.decision.objective
        chosen = record.decision.action
        breakdowns = record.counterfactuals.breakdowns[objective]
        for reason in explanation.reasons:
            if reason.kind == "term_tradeoff":
                for metric, entry in reason.evidence["tradeoff_terms"].items():
                    position = breakdowns[chosen].metric_names.index(metric)
                    terms = {a: breakdowns[a].terms[position] for a in breakdowns}
                    assert terms[chosen] == pytest.approx(entry["chosen_term"])
                    assert entry["chosen_term"] > entry["best_term"]

    def test_deterministic_output(self) -> None:
        _, record = evaluate_rows()
        first = explain_evaluation_record(record).as_dict()
        second = explain_evaluation_record(record).as_dict()
        assert first == second


class TestNoFabricatedClaims:
    def test_fallback_selection_is_never_described_as_learned(self) -> None:
        _, record = evaluate_rows()
        if record.decision.selection_rule != SELECTION_UNSEEN_FALLBACK:
            pytest.skip("decision was not a fallback on this workload")
        explanation = explain_evaluation_record(record)
        basis = next(r for r in explanation.reasons if r.kind == "selection_basis")
        assert "not learned evidence" in basis.statement

    def test_workload_characteristics_values_are_exact(self) -> None:
        _, record = evaluate_rows()
        explanation = explain_evaluation_record(record)
        raw = record.decision.fingerprint.as_dict()
        for name in FINGERPRINT_FEATURES:
            assert explanation.workload_characteristics[name]["value"] == raw[name]
        reason = next(
            r for r in explanation.reasons if r.kind == "workload_characteristics"
        )
        for name, bin_index in record.decision.state_bins.items():
            assert reason.evidence["state_bins"][name] == bin_index

    def test_baseline_explanations_carry_no_q_claims(self) -> None:
        workload = workload_from_rows("base", ROWS)
        report = run_counterfactuals(workload, policies())
        explanation = build_explanation(
            workload_name=workload.name,
            workload_fingerprint=workload.fingerprint,
            fingerprint=record_fingerprint(report, workload),
            selection={
                "policy_name": "SJF",
                "action": 1,
                "selection_rule": "fingerprint_rule",
                "selection_kind": "baseline",
                "selection_evidence": {"rule_branch": "otherwise -> SJF"},
            },
            counterfactuals=report,
            objective=OBJECTIVE_RISK_AWARE,
        )
        basis = next(r for r in explanation.reasons if r.kind == "selection_basis")
        assert "Non-learning selection rule" in basis.statement
        assert "q_values" not in basis.evidence

    def test_rejects_mismatched_inputs(self) -> None:
        _, record = evaluate_rows()
        with pytest.raises(ValidationError):
            build_explanation(
                workload_name="x",
                workload_fingerprint="0" * 16,
                fingerprint=record.decision.fingerprint,
                selection={
                    "policy_name": "FCFS",
                    "action": 1,
                    "selection_rule": SELECTION_GREEDY,
                    "selection_kind": "q_learning",
                },
                counterfactuals=record.counterfactuals,
                objective=OBJECTIVE_RISK_AWARE,
            )
        with pytest.raises(ValidationError):
            build_explanation(
                workload_name="x",
                workload_fingerprint=record.decision.workload_fingerprint,
                fingerprint=record.decision.fingerprint,
                selection={
                    "policy_name": "FCFS",
                    "action": 0,
                    "selection_rule": SELECTION_GREEDY,
                    "selection_kind": "q_learning",
                },
                counterfactuals=record.counterfactuals,
                objective="unknown objective",
            )


def record_fingerprint(report, workload):
    from workload.fingerprint import fingerprint_workload

    return fingerprint_workload(workload)
