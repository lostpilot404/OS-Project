"""Structured, evidence-backed explanations for offline policy selections.

Every selection of the study produces one :class:`Explanation`.  The explanation
contains:

* the workload characteristics (fingerprint features with values and state bins),
* the selected policy and how it was selected (greedy / exploration / unseen-state
  fallback),
* the objective values of **all four** fixed policies under both declared objectives
  (computed post hoc from the counterfactual runs),
* hindsight regret, and
* a list of :class:`Reason` records whose ``evidence`` fields are the exact computed
  numbers behind every claim in the statement text.

No natural-language justification is invented: statements are deterministic templates
over the evidence, so each claim can be re-derived from the recorded numbers.  A reason
is emitted only when its claim is true of the computed evidence (for example, a
"lowest term" reason exists only if the selected policy really has the lowest
normalised term).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from config import ACTION_NAMES, RewardConfig, RiskObjectiveConfig
from errors import ValidationError
from evaluation.counterfactual import CounterfactualReport
from rl.objective import (
    OBJECTIVE_ORIGINAL,
    OBJECTIVE_RISK_AWARE,
    RISK_OBJECTIVE_METRICS,
)
from rl.selector import (
    SELECTION_EPSILON,
    SELECTION_GREEDY,
    SELECTION_UNSEEN_FALLBACK,
    EvaluationRecord,
)
from workload.fingerprint import FINGERPRINT_FEATURES, WorkloadFingerprint

__all__ = [
    "EXPLANATION_SCHEMA_VERSION",
    "INFORMATION_BOUNDARY",
    "Reason",
    "Explanation",
    "build_explanation",
    "explain_evaluation_record",
]

#: Schema version of the explanation JSON.
EXPLANATION_SCHEMA_VERSION = "1.0"

#: The declared input contract, copied verbatim into every explanation.
INFORMATION_BOUNDARY = (
    "The offline selector decides from the declared process table only: "
    "(pid, arrival_time, burst_time, priority) per process. Schedule outcomes, "
    "metrics and counterfactual results are never decision inputs; counterfactual "
    "objective values in this explanation are computed after the decision and are "
    "evaluation evidence only."
)

_OBJECTIVE_TEXT = {
    OBJECTIVE_ORIGINAL: (
        "Original six-metric reward: chosen-policy metric over the four-policy "
        "arithmetic mean per metric, combined with fixed cost/benefit weights "
        "(rl.reward). Cost is -reward (lower is better)."
    ),
    OBJECTIVE_RISK_AWARE: (
        "risk_aware_v1: J = w_waiting*n(mean waiting) + w_response*n(mean response) "
        "+ w_switch*n(context switches per process) + w_tail*n(p95 per-process "
        "waiting); n(x) = clip(x/reference, 0, clip), reference = four-policy "
        "arithmetic mean on the same workload. Cost is J (lower is better)."
    ),
}

_RISK_CRITERION_TEXT = (
    "Explicit tail-risk term: nearest-rank p95 of per-process waiting times, weight "
    "w_tail. The objective is mean-tail and risk-averse with respect to waiting-time "
    "outliers."
)


@dataclass(frozen=True)
class Reason:
    """One machine-checkable justification.

    Attributes:
        kind: Reason category (``selection_basis``, ``objective_rank``,
            ``term_comparison``, ``term_tradeoff``, ``workload_characteristics``).
        statement: Deterministic template text filled exclusively from ``evidence``.
        evidence: The computed numbers behind the statement.
    """

    kind: str
    statement: str
    evidence: Mapping[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "statement": self.statement, "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class Explanation:
    """Complete structured explanation of one policy selection."""

    schema_version: str
    workload_name: str
    workload_fingerprint: str
    information_boundary: str
    workload_characteristics: Mapping[str, Mapping[str, object]]
    decision: Mapping[str, object]
    declared_objective: Mapping[str, object]
    counterfactual_objectives: Mapping[str, Mapping[str, object]]
    regret: Mapping[str, object]
    reasons: Tuple[Reason, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, object]:
        """JSON-ready dictionary of the full explanation."""
        return {
            "schema_version": self.schema_version,
            "workload_name": self.workload_name,
            "workload_fingerprint": self.workload_fingerprint,
            "information_boundary": self.information_boundary,
            "workload_characteristics": {
                name: dict(values) for name, values in self.workload_characteristics.items()
            },
            "decision": dict(self.decision),
            "declared_objective": dict(self.declared_objective),
            "counterfactual_objectives": {
                policy: dict(values)
                for policy, values in self.counterfactual_objectives.items()
            },
            "regret": dict(self.regret),
            "reasons": [reason.as_dict() for reason in self.reasons],
        }


def explain_evaluation_record(
    record: EvaluationRecord,
    reward_config: RewardConfig | None = None,
    risk_config: RiskObjectiveConfig | None = None,
) -> Explanation:
    """Build the explanation of one Q-selector evaluation decision.

    The counterfactual objectives in the explanation come from ``record`` and were
    computed strictly after the decision (see :mod:`rl.selector`).
    """
    decision = record.decision
    return build_explanation(
        workload_name=decision.workload_name,
        workload_fingerprint=decision.workload_fingerprint,
        fingerprint=decision.fingerprint,
        selection={
            "policy_name": decision.policy_name,
            "action": decision.action,
            "selection_rule": decision.selection_rule,
            "selection_kind": "q_learning",
            "q_values": dict(zip(ACTION_NAMES, decision.q_values)),
            "action_visits": dict(zip(ACTION_NAMES, decision.action_visits)),
            "greedy_action": decision.greedy_action,
            "greedy_policy_name": decision.greedy_policy_name,
            "explored": decision.explored,
            "epsilon": decision.epsilon,
        },
        counterfactuals=record.counterfactuals,
        objective=decision.objective,
        state_bins=dict(decision.state_bins),
        state_features=tuple(decision.state_bins),
        reward_config=reward_config,
        risk_config=risk_config,
    )


def build_explanation(
    workload_name: str,
    workload_fingerprint: str,
    fingerprint: WorkloadFingerprint,
    selection: Mapping[str, object],
    counterfactuals: CounterfactualReport,
    objective: str,
    state_bins: Optional[Mapping[str, int]] = None,
    state_features: Sequence[str] = (),
    reward_config: RewardConfig | None = None,
    risk_config: RiskObjectiveConfig | None = None,
) -> Explanation:
    """Assemble a full explanation from computed decision and evaluation evidence.

    Args:
        workload_name: Name of the workload.
        workload_fingerprint: Content digest of the process table.
        fingerprint: Workload characteristics (process table only).
        selection: How the policy was selected; must contain ``policy_name``,
            ``action``, ``selection_rule`` and ``selection_kind``, plus the
            selection-specific evidence (Q-values and visits for the Q-learning kind).
        counterfactuals: Post-decision outcomes of the four fixed policies.
        objective: Declared objective name.
        state_bins: Bin index of every state feature/variable, if a tabular state was
            used.
        state_features: Names of the features/variables that form the state.
        reward_config: Weights of the original objective.
        risk_config: Weights of the risk-aware objective.

    Returns:
        The :class:`Explanation`.

    Raises:
        ValidationError: If the inputs are inconsistent (e.g. the selected action is
            missing from the counterfactual report).
    """
    if objective not in (OBJECTIVE_ORIGINAL, OBJECTIVE_RISK_AWARE):
        raise ValidationError(f"unknown objective {objective!r}")
    action = int(selection["action"])
    policy_name = str(selection["policy_name"])
    if policy_name != ACTION_NAMES[action]:
        raise ValidationError(
            f"selection policy {policy_name!r} does not match action {action}"
        )
    if counterfactuals.workload_fingerprint != workload_fingerprint:
        raise ValidationError(
            "counterfactual report belongs to a different workload "
            f"({counterfactuals.workload_fingerprint} != {workload_fingerprint})"
        )
    breakdowns = counterfactuals.breakdowns[objective]
    costs = {index: breakdown.cost for index, breakdown in breakdowns.items()}
    best_action = min(sorted(costs), key=lambda index: costs[index])
    sorted_costs = sorted(costs.items(), key=lambda item: (item[1], item[0]))

    characteristics: Dict[str, Dict[str, object]] = {}
    for name in FINGERPRINT_FEATURES:
        value = fingerprint.as_dict()[name]
        entry: Dict[str, object] = {
            "value": value,
            "in_state": name in state_features,
        }
        if state_bins is not None and name in state_bins:
            entry["state_bin"] = state_bins[name]
        characteristics[name] = entry

    reasons = _build_reasons(
        selection=selection,
        counterfactuals=counterfactuals,
        objective=objective,
        costs=costs,
        best_action=best_action,
        sorted_costs=sorted_costs,
        state_bins=state_bins or {},
        fingerprint=fingerprint,
    )

    declared_objective: Dict[str, object] = {
        "name": objective,
        "description": _OBJECTIVE_TEXT[objective],
        "normalization": (
            "per-metric ratio to the four-policy arithmetic mean on the same workload, "
            "clipped above (zero reference maps to 1)"
        ),
    }
    if objective == OBJECTIVE_RISK_AWARE:
        config = risk_config or RiskObjectiveConfig()
        declared_objective["weights"] = {
            "waiting": config.weight_waiting_time,
            "response": config.weight_response_time,
            "context_switches": config.weight_context_switches,
            "tail_waiting": config.weight_tail_waiting,
        }
        declared_objective["risk_criterion"] = _RISK_CRITERION_TEXT
        declared_objective["tail_quantile"] = config.tail_quantile
    else:
        config = reward_config or RewardConfig()
        declared_objective["weights"] = {
            "waiting": config.weight_waiting_time,
            "turnaround": config.weight_turnaround_time,
            "response": config.weight_response_time,
            "context_switches": config.weight_context_switches,
            "cpu_utilization": config.weight_cpu_utilization,
            "throughput": config.weight_throughput,
        }

    counterfactual_objectives: Dict[str, Dict[str, object]] = {}
    for index in sorted(counterfactuals.outcomes):
        outcome = counterfactuals.outcomes[index]
        entry = {
            "policy_name": ACTION_NAMES[index],
            "costs": {
                name: counterfactuals.breakdowns[name][index].cost
                for name in counterfactuals.breakdowns
            },
            "metrics": {
                "avg_waiting_time": outcome.metrics.avg_waiting_time,
                "avg_response_time": outcome.metrics.avg_response_time,
                "avg_turnaround_time": outcome.metrics.avg_turnaround_time,
                "context_switches_per_process": outcome.metrics.context_switches_per_process,
                "cpu_utilization": outcome.metrics.cpu_utilization,
                "throughput": outcome.metrics.throughput,
                "p95_waiting_time": outcome.p95_waiting_time,
            },
            "terms": {
                name: dict(
                    zip(
                        counterfactuals.breakdowns[name][index].metric_names,
                        counterfactuals.breakdowns[name][index].terms,
                    )
                )
                for name in counterfactuals.breakdowns
            },
        }
        counterfactual_objectives[ACTION_NAMES[index]] = entry

    regret = counterfactuals.regret(objective, action)
    return Explanation(
        schema_version=EXPLANATION_SCHEMA_VERSION,
        workload_name=workload_name,
        workload_fingerprint=workload_fingerprint,
        information_boundary=INFORMATION_BOUNDARY,
        workload_characteristics=characteristics,
        decision=dict(selection),
        declared_objective=declared_objective,
        counterfactual_objectives=counterfactual_objectives,
        regret=regret.as_dict(),
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------------------
# reason generation (every statement is a template over computed evidence)
# --------------------------------------------------------------------------------------


def _build_reasons(
    selection: Mapping[str, object],
    counterfactuals: CounterfactualReport,
    objective: str,
    costs: Mapping[int, float],
    best_action: int,
    sorted_costs: Sequence[Tuple[int, float]],
    state_bins: Mapping[str, int],
    fingerprint: WorkloadFingerprint,
) -> Sequence[Reason]:
    action = int(selection["action"])
    reasons = [_selection_basis_reason(selection)]
    reasons.append(
        _objective_rank_reason(
            selection, counterfactuals, objective, costs, best_action, sorted_costs
        )
    )
    reasons.extend(_term_reasons(selection, counterfactuals, objective))
    reasons.append(_characteristics_reason(fingerprint, state_bins))
    return [reason for reason in reasons if reason is not None]


def _selection_basis_reason(selection: Mapping[str, object]) -> Reason:
    rule = str(selection["selection_rule"])
    policy_name = str(selection["policy_name"])
    evidence: Dict[str, object] = {
        "selection_rule": rule,
        "policy_name": policy_name,
        "selection_kind": selection.get("selection_kind"),
    }
    if selection.get("selection_kind") == "q_learning":
        q_values = dict(selection["q_values"])  # type: ignore[arg-type]
        visits = dict(selection["action_visits"])  # type: ignore[arg-type]
        evidence.update({"q_values": q_values, "action_visits": visits})
        evidence["greedy_policy_name"] = selection["greedy_policy_name"]
        if rule == SELECTION_UNSEEN_FALLBACK:
            statement = (
                f"State had zero training visits; the deterministic tie-break selected "
                f"{policy_name} (lowest action index). This is a documented fallback, "
                f"not learned evidence."
            )
        elif rule == SELECTION_EPSILON:
            evidence["epsilon"] = selection["epsilon"]
            statement = (
                f"Training exploration (epsilon={selection['epsilon']:.4f}) selected "
                f"{policy_name}; the greedy action at that time was "
                f"{selection['greedy_policy_name']}."
            )
        else:
            best_q = q_values[policy_name]
            runner = max(
                (name for name in q_values if name != policy_name),
                key=lambda name: (q_values[name], name),
            )
            evidence["runner_up_policy"] = runner
            evidence["runner_up_q"] = q_values[runner]
            evidence["q_margin"] = best_q - q_values[runner]
            statement = (
                f"Greedy selection: Q(state, {policy_name})={best_q:.6f} was the highest "
                f"learned value (visits={visits[policy_name]}); margin over {runner} "
                f"was {best_q - q_values[runner]:.6f}."
            )
    else:
        evidence["selection_evidence"] = dict(selection.get("selection_evidence", {}))
        statement = (
            f"Non-learning selection rule '{rule}' selected {policy_name}; no learned "
            f"values were consulted."
        )
    return Reason(kind="selection_basis", statement=statement, evidence=evidence)


def _objective_rank_reason(
    selection: Mapping[str, object],
    counterfactuals: CounterfactualReport,
    objective: str,
    costs: Mapping[int, float],
    best_action: int,
    sorted_costs: Sequence[Tuple[int, float]],
) -> Reason:
    action = int(selection["action"])
    policy_name = str(selection["policy_name"])
    chosen_cost = costs[action]
    best_policy = ACTION_NAMES[best_action]
    best_cost = costs[best_action]
    evidence: Dict[str, object] = {
        "objective": objective,
        "policy_name": policy_name,
        "chosen_cost": chosen_cost,
        "best_policy": best_policy,
        "best_cost": best_cost,
        "regret": chosen_cost - best_cost,
        "costs": {ACTION_NAMES[index]: cost for index, cost in costs.items()},
    }
    if action == best_action:
        runner_up_policy = ACTION_NAMES[sorted_costs[1][0]]
        runner_up_cost = sorted_costs[1][1]
        evidence["runner_up_policy"] = runner_up_policy
        evidence["runner_up_cost"] = runner_up_cost
        evidence["margin"] = runner_up_cost - chosen_cost
        statement = (
            f"Post-hoc counterfactual evidence under {objective}: {policy_name} achieved "
            f"the lowest cost ({chosen_cost:.6f}) of the four policies; runner-up "
            f"{runner_up_policy} scored {runner_up_cost:.6f} "
            f"(margin {runner_up_cost - chosen_cost:.6f})."
        )
    else:
        statement = (
            f"Post-hoc counterfactual evidence under {objective}: {policy_name} scored "
            f"{chosen_cost:.6f}, which was not the hindsight minimum; {best_policy} "
            f"scored {best_cost:.6f} (regret {chosen_cost - best_cost:.6f}). The "
            f"selector did not know these outcomes at decision time."
        )
    return Reason(kind="objective_rank", statement=statement, evidence=evidence)


def _term_reasons(
    selection: Mapping[str, object],
    counterfactuals: CounterfactualReport,
    objective: str,
) -> Sequence[Reason]:
    action = int(selection["action"])
    policy_name = str(selection["policy_name"])
    breakdowns = counterfactuals.breakdowns[objective]
    metric_names = breakdowns[action].metric_names
    reasons = []
    favorable: Dict[str, object] = {}
    tradeoffs: Dict[str, object] = {}
    for position, metric in enumerate(metric_names):
        terms = {
            index: breakdown.terms[position] for index, breakdown in breakdowns.items()
        }
        chosen_term = terms[action]
        best_index = min(sorted(terms), key=lambda index: terms[index])
        worst_of_best = {ACTION_NAMES[index]: terms[index] for index in terms}
        if action == best_index and all(
            terms[index] >= chosen_term for index in terms
        ) and sum(1 for index in terms if terms[index] == chosen_term) == 1:
            favorable[metric] = {
                "chosen_term": chosen_term,
                "terms": worst_of_best,
            }
        elif chosen_term > terms[best_index]:
            tradeoffs[metric] = {
                "chosen_term": chosen_term,
                "best_term": terms[best_index],
                "best_policy": ACTION_NAMES[best_index],
            }
    if favorable:
        names = ", ".join(sorted(favorable))
        statement = (
            f"Under the declared objective, {policy_name} had the strictly lowest "
            f"normalised term for: {names}."
        )
        reasons.append(
            Reason(
                kind="term_comparison",
                statement=statement,
                evidence={
                    "objective": objective,
                    "policy_name": policy_name,
                    "favorable_terms": favorable,
                },
            )
        )
    if tradeoffs:
        parts = []
        for metric in sorted(tradeoffs):
            entry = tradeoffs[metric]
            parts.append(
                f"{metric}: {entry['chosen_term']:.6f} vs best "
                f"{entry['best_term']:.6f} ({entry['best_policy']})"
            )
        statement = (
            f"Trade-offs: terms where {policy_name} was not the lowest under "
            f"{objective} -- " + "; ".join(parts) + "."
        )
        reasons.append(
            Reason(
                kind="term_tradeoff",
                statement=statement,
                evidence={
                    "objective": objective,
                    "policy_name": policy_name,
                    "tradeoff_terms": tradeoffs,
                },
            )
        )
    return reasons


def _characteristics_reason(
    fingerprint: WorkloadFingerprint, state_bins: Mapping[str, int]
) -> Reason:
    values = fingerprint.as_dict()
    evidence: Dict[str, object] = {
        "features": {name: values[name] for name in FINGERPRINT_FEATURES},
        "state_bins": dict(state_bins),
    }
    binned = [
        f"{name}={values[name]:.4g} (bin {state_bins[name]})"
        for name in state_bins
    ]
    statement = (
        "Workload characteristics used for the decision: "
        + (", ".join(binned) if binned else "no tabular state features")
        + "."
    )
    return Reason(
        kind="workload_characteristics", statement=statement, evidence=evidence
    )
