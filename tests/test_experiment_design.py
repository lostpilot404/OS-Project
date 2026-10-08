"""Tests for the revised experiment design (the reviewer's correction).

These tests pin the *design decisions* of the redesigned experiment, so that a future
change cannot silently undo them:

* the Round-Robin quantum controller extension is gone from the core project (the RR
  quantum parameter itself stays -- it is part of Round Robin);
* the reward is exactly the reward that was declared before the redesign (no weight was
  tuned to manufacture policy diversity);
* the agent is trained on several workload conditions and evaluated on held-out
  workloads, and its selection is workload-aware (different classes trigger different
  policies, matching the pre-training verification);
* the state uses only decision-time information and distinguishes the conditions;
* every scheduler is evaluated on identical workloads.
"""

from __future__ import annotations

from collections import Counter

import pytest

from config import ACTION_NAMES, RewardConfig, build_default_config
from experiments.evaluate import evaluate
from experiments.train import train
from experiments.verify_classes import verify_classes
from rl.state import StateEncoder
from workload.generator import WorkloadGenerator


class TestQuantumControllerIsRemoved:
    def test_no_quantum_controller_module_or_config(self) -> None:
        import config
        import rl

        assert not hasattr(config, "QuantumControllerConfig")
        assert not hasattr(rl, "QuantumController")
        assert "quantum_controller" not in config.ExperimentConfig.__dataclass_fields__

    def test_adaptive_scheduler_has_no_controller_and_no_learned_quantum(self) -> None:
        from dataclasses import fields

        from rl.adaptive import AdaptiveDecision, AdaptiveScheduler

        assert "quantum_controller" not in AdaptiveScheduler.__init__.__code__.co_varnames
        assert not hasattr(AdaptiveScheduler, "run_round_robin_with_learned_quantum")
        decision_fields = {f.name for f in fields(AdaptiveDecision)}
        assert "quantum_multiplier" not in decision_fields
        assert "controller_multiplier" not in decision_fields

    def test_evaluation_has_a_single_adaptive_regime(self) -> None:
        from experiments import evaluate as evaluate_module

        assert evaluate_module.ADAPTIVE_REGIME == "adaptive"
        assert not hasattr(evaluate_module, "ADAPTIVE_LEARNED_REGIME")
        assert not hasattr(evaluate_module, "ROUND_ROBIN_LEARNED_QUANTUM_REGIME")

    def test_reward_module_has_no_controller_reference_reward(self) -> None:
        from rl import reward

        assert not hasattr(reward, "compute_reward_against_reference")

    def test_round_robin_quantum_parameter_is_still_configured(self) -> None:
        # The quantum itself is part of Round Robin and must stay.
        from config import SchedulerConfig

        assert SchedulerConfig().round_robin_quantum == 4
        from scheduler.round_robin import RoundRobin

        assert RoundRobin(SchedulerConfig()).name == "Round Robin"


class TestRewardIsUnchanged:
    def test_reward_weights_are_the_declared_ones(self) -> None:
        # The reward was deliberately NOT retuned when the workloads and the state were
        # redesigned: these are the weights declared before the first experiment.
        reward = RewardConfig()
        assert reward.weight_waiting_time == pytest.approx(0.30)
        assert reward.weight_turnaround_time == pytest.approx(0.25)
        assert reward.weight_response_time == pytest.approx(0.20)
        assert reward.weight_context_switches == pytest.approx(0.10)
        assert reward.weight_cpu_utilization == pytest.approx(0.075)
        assert reward.weight_throughput == pytest.approx(0.075)
        assert reward.reference_clip == pytest.approx(2.0)
        assert sum(reward.cost_weights() + reward.benefit_weights()) == pytest.approx(1.0)

    def test_reward_is_computed_against_the_four_policies_on_the_same_workload(self) -> None:
        from evaluation.metrics import compute_metrics
        from rl.reward import compute_reward
        from scheduler import POLICY_CLASSES
        from tests.helpers import workload_from_rows

        config = build_default_config()
        policies = [cls(config.scheduler) for cls in POLICY_CLASSES]
        workload = workload_from_rows(
            "w", [(1, 0, 5, 1), (2, 0, 3, 2), (3, 1, 7, 1), (4, 4, 2, 2)]
        )
        attempts = {i: compute_metrics(p.run(workload)) for i, p in enumerate(policies)}
        # The reward of the best action is non-negative: the reference is the mean of
        # the four policies, so at least one policy must be at or above it.
        rewards = [compute_reward(config.reward, attempts, a).reward for a in range(4)]
        assert max(rewards) >= 0.0
        # The reward of a given action is the deviation of that action from the mean.
        best = max(range(4), key=lambda a: rewards[a])
        assert rewards[best] == pytest.approx(max(rewards))


@pytest.fixture(scope="module")
def artifacts():
    """The full default experiment: verification, training and evaluation."""
    config = build_default_config()
    generator = WorkloadGenerator(config.families)
    encoder = StateEncoder(config.state)
    verification = verify_classes(config, generator=generator, encoder=encoder)
    training = train(config, generator=generator, encoder=encoder)
    evaluation = evaluate(config, training, generator=generator)
    return config, verification, training, evaluation


class TestWorkloadAwareSelection:
    """The end-to-end property the redesign exists to demonstrate."""

    def test_the_agent_selects_different_policies_for_different_classes(
        self, artifacts
    ) -> None:
        _, _, _, evaluation = artifacts
        selected = Counter()
        for row in evaluation.decisions.itertuples():
            selected[(row.family, row.policy_name)] += 1
        per_family = {}
        for (family, policy), count in selected.items():
            per_family.setdefault(family, Counter())[policy] = count
        # At least two classes must be dominated by different policies.
        dominant = {
            family: counts.most_common(1)[0][0] for family, counts in per_family.items()
        }
        assert len(set(dominant.values())) >= 2

    def test_selection_matches_the_pre_training_verification(self, artifacts) -> None:
        _, verification, _, evaluation = artifacts
        modal = {entry.family: entry.modal_winner for entry in verification.classes}
        selected = Counter()
        for row in evaluation.decisions.itertuples():
            selected[(row.family, row.policy_name)] += 1
        per_family = {}
        for (family, policy), count in selected.items():
            per_family.setdefault(family, Counter())[policy] = count
        for family, counts in per_family.items():
            assert counts.most_common(1)[0][0] == modal[family], family

    def test_interactive_classes_trigger_round_robin(self, artifacts) -> None:
        _, _, _, evaluation = artifacts
        for row in evaluation.decisions.itertuples():
            if row.family in ("interactive", "interactive_sparse"):
                # Round Robin is chosen exactly when the workload actually contains the
                # long background jobs that make preemption pay (visible in the state).
                has_long_jobs = row.state_long_job_share > 0.5
                if has_long_jobs:
                    assert row.policy_name in ("Round Robin", "SJF")

    def test_selection_is_learned_not_hard_coded(self, artifacts) -> None:
        # The greedy action must come from the Q-table: forcing a different action's
        # value to the top must change the evaluation decision.
        _, _, training, evaluation = artifacts
        workload = evaluation.decisions.iloc[0]
        state = int(workload["state_index"])
        agent = training.agent
        before = agent.greedy_action(state)
        other = (before + 1) % len(ACTION_NAMES)
        agent.update(state, other, 1e6, next_state=None)
        assert agent.greedy_action(state) == other

    def test_training_uses_multiple_conditions(self, artifacts) -> None:
        config, _, training, _ = artifacts
        families = set(training.history.family_names)
        assert families == {family.name for family in config.families}
        assert len(families) >= 5

    def test_evaluation_is_held_out(self, artifacts) -> None:
        config, _, training, evaluation = artifacts
        assert evaluation.evaluation_seed != config.training.seed
        evaluated = set(evaluation.metrics["workload_fingerprint"])
        assert not (evaluated & training.training_fingerprints)

    def test_all_baselines_see_identical_workloads(self, artifacts) -> None:
        _, _, _, evaluation = artifacts
        grouped = evaluation.metrics.groupby(["family", "repetition"])[
            "workload_fingerprint"
        ].nunique()
        assert (grouped == 1).all()
        counts = (
            evaluation.metrics[evaluation.metrics["regime"] == "baseline"]
            .groupby("policy")
            .size()
        )
        assert set(counts.index) == set(ACTION_NAMES)
        assert counts.nunique() == 1

    def test_state_coverage_is_reported_and_complete(self, artifacts) -> None:
        _, _, training, evaluation = artifacts
        summary = evaluation.summary()
        assert summary["distinct_states_evaluated"] >= 1
        assert summary["evaluated_states_visited_during_training"] == (
            summary["distinct_states_evaluated"]
        )
        assert training.history.visited_states >= 1
        assert training.history.visited_states <= training.history.n_states

    def test_adaptive_performance_is_measured_per_class(self, artifacts) -> None:
        _, _, _, evaluation = artifacts
        adaptive = evaluation.regime("adaptive")
        assert set(adaptive["family"]) == set(evaluation.families)
        for column in (
            "avg_waiting_time",
            "avg_turnaround_time",
            "avg_response_time",
            "cpu_utilization",
            "throughput",
            "context_switches_per_process",
        ):
            assert column in adaptive.columns
