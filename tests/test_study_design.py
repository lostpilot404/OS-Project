"""Tests for the study design: splits, ablation matrix, baselines and legacy pinning.

Covers the executable claims of ``experiments/study_config.py``: held-out families
never enter training, the held-out test split is disjoint from training workloads, the
ablation matrix is complete, baselines use training data only, the predeclared rule is
deterministic, and the ``ql_orig`` arm reproduces the preserved legacy selector.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import (
    ACTION_NAMES,
    QLearningConfig,
    build_default_config,
    derive_seed,
)
from errors import ValidationError
from experiments.baselines import (
    RULE_BRANCHES,
    best_fixed_action,
    fingerprint_rule_action,
    random_action,
)
from experiments.study_config import (
    BASELINE_ARMS,
    HELD_OUT_FAMILIES,
    Q_ARMS,
    TRAINING_FAMILIES,
    build_study_config,
    build_study_design,
)
from rl.adaptive import OfflinePolicySelector
from rl.fingerprint_state import FingerprintStateEncoder
from rl.objective import OBJECTIVE_ORIGINAL
from rl.q_learning import QLearningAgent
from rl.selector import WorkloadPolicySelector
from rl.state import StateEncoder
from scheduler import POLICY_CLASSES
from tests.helpers import DEFAULT_SCHEDULER_CONFIG, workload_from_rows
from workload.fingerprint import fingerprint_workload
from workload.generator import WorkloadGenerator

from config import FingerprintStateConfig, StateConfig


class TestStudyDesign:
    def test_held_out_families_never_enter_training(self) -> None:
        design = build_study_design()
        config = build_study_config()
        cycle = config.training.family_cycle
        assert set(design.training_families) == set(cycle)
        for family in design.held_out_families:
            assert family not in cycle
            assert family in {f.name for f in config.families}

    def test_test_split_is_disjoint_from_training_workloads(self) -> None:
        from experiments.study import _build_split, _check_split_separation

        config = build_study_config()
        design = build_study_design()
        generator = WorkloadGenerator(config.families)
        training_fingerprints = set()
        for seed in design.training_seeds:
            for episode in range(30):
                family = cycle_family(config, episode)
                workload = generator.generate(family, derive_seed(seed, 0, episode))
                training_fingerprints.add(workload.fingerprint)
        test_split = _build_split(config, generator, design.test_seed, 3)
        validation_split = _build_split(config, generator, design.validation_seed, 2)
        _check_split_separation(
            training_fingerprints,
            {"validation": validation_split, "test": test_split},
        )

    def test_overlapping_splits_are_rejected(self) -> None:
        from experiments.study import _build_split, _check_split_separation

        config = build_study_config()
        design = build_study_design()
        generator = WorkloadGenerator(config.families)
        split = _build_split(config, generator, design.test_seed, 2)
        with pytest.raises(ValidationError):
            _check_split_separation(set(), {"test": split, "again": split})

    def test_ablation_matrix_is_complete(self) -> None:
        assert set(Q_ARMS) == {
            "ql_orig",
            "ql_risk",
            "ql_risk_origstate",
            "ql_orig_fingerprint",
        }
        assert Q_ARMS["ql_orig"] == ("legacy", "original")
        assert Q_ARMS["ql_risk"] == ("fingerprint", "risk_aware_v1")
        assert Q_ARMS["ql_risk_origstate"] == ("legacy", "risk_aware_v1")
        assert Q_ARMS["ql_orig_fingerprint"] == ("fingerprint", "original")

    def test_design_matches_the_frozen_constants(self) -> None:
        design = build_study_design()
        assert design.training_seeds == (42, 43, 44, 45, 46)
        assert design.test_repetitions == 30
        assert design.primary_objective == "risk_aware_v1"
        assert set(design.baseline_arms) == set(BASELINE_ARMS)


def cycle_family(config, episode: int) -> str:
    cycle = config.training.family_cycle
    return cycle[episode % len(cycle)]


class TestBaselines:
    def test_best_fixed_uses_lowest_mean_training_cost(self) -> None:
        assert best_fixed_action({0: 1.2, 1: 0.8, 2: 1.0, 3: 0.9}) == 1
        assert best_fixed_action({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}) == 0  # ties -> FCFS
        with pytest.raises(ValidationError):
            best_fixed_action({0: 1.0, 1: 1.0})

    def test_fingerprint_rule_branches(self) -> None:
        # long head-of-line job + variable bursts -> Round Robin
        rr = fingerprint_from_values(
            first_arrival_burst_ratio=8.0, burst_cv=1.2,
            priority_burst_correlation=0.0,
        )
        action, branch, evidence = fingerprint_rule_action(rr)
        assert action == 2 and branch == RULE_BRANCHES[0]
        assert evidence["first_arrival_burst_ratio"] == 8.0

        # priority tracks bursts -> Priority
        prio = fingerprint_from_values(
            first_arrival_burst_ratio=1.0, burst_cv=0.8,
            priority_burst_correlation=0.7,
        )
        action, branch, _ = fingerprint_rule_action(prio)
        assert action == 3 and branch == RULE_BRANCHES[1]

        # near-uniform bursts -> FCFS
        uniform = fingerprint_from_values(
            first_arrival_burst_ratio=1.0, burst_cv=0.1,
            priority_burst_correlation=0.0,
        )
        action, branch, _ = fingerprint_rule_action(uniform)
        assert action == 0 and branch == RULE_BRANCHES[2]

        # otherwise -> SJF
        mixed = fingerprint_from_values(
            first_arrival_burst_ratio=1.0, burst_cv=0.8,
            priority_burst_correlation=0.0,
        )
        action, branch, _ = fingerprint_rule_action(mixed)
        assert action == 1 and branch == RULE_BRANCHES[3]

    def test_fingerprint_rule_is_deterministic(self) -> None:
        fingerprint = fingerprint_from_values(
            first_arrival_burst_ratio=5.0, burst_cv=1.0, priority_burst_correlation=0.1
        )
        assert fingerprint_rule_action(fingerprint) == fingerprint_rule_action(fingerprint)

    def test_rule_never_reads_counterfactuals(self) -> None:
        """The rule input is the fingerprint type only (enforced by signature)."""
        import inspect

        from experiments import baselines

        parameters = inspect.signature(baselines.fingerprint_rule_action).parameters
        assert list(parameters) == ["fingerprint"]

    def test_random_action_is_seeded(self) -> None:
        first = [random_action(np.random.default_rng(7))[0] for _ in range(5)]
        second = [random_action(np.random.default_rng(7))[0] for _ in range(5)]
        assert first == second
        with pytest.raises(ValidationError):
            random_action("not a generator")  # type: ignore[arg-type]


def fingerprint_from_values(**overrides) -> object:
    from workload.fingerprint import WorkloadFingerprint

    values = {
        "workload_name": "synthetic",
        "digest": "0" * 16,
        "process_count": 15,
        "arrival_span": 20,
        "arrival_rate": 0.75,
        "arrival_gap_cv": 1.0,
        "burst_mean": 10.0,
        "burst_cv": 0.5,
        "burst_skew": 0.0,
        "first_arrival_burst_ratio": 1.0,
        "priority_std": 1.0,
        "priority_unique_fraction": 0.4,
        "priority_burst_correlation": 0.0,
    }
    values.update(overrides)
    return WorkloadFingerprint(**values)


class TestLegacyPinning:
    def test_ql_orig_arm_reproduces_the_legacy_selector(self) -> None:
        """Same state, reward, agent and workload stream -> identical decisions."""
        config = build_default_config()
        policies = [cls(DEFAULT_SCHEDULER_CONFIG) for cls in POLICY_CLASSES]
        generator = WorkloadGenerator(config.families)

        legacy_agent = QLearningAgent(
            StateEncoder(config.state).n_states,
            len(ACTION_NAMES),
            QLearningConfig(),
            seed=42,
        )
        legacy = OfflinePolicySelector(
            config, StateEncoder(config.state), legacy_agent, policies
        )

        study_agent = QLearningAgent(
            StateEncoder(config.state).n_states,
            len(ACTION_NAMES),
            QLearningConfig(),
            seed=42,
        )
        study_selector = WorkloadPolicySelector(
            StateEncoder(config.state),
            study_agent,
            objective=OBJECTIVE_ORIGINAL,
            mask_unvisited=False,
        )

        for episode in range(40):
            family = cycle_family(config, episode)
            workload = generator.generate(family, derive_seed(42, 0, episode))
            legacy_decision = legacy.run_training_episode(workload, episode)
            study_record = study_selector.train_episode(workload, episode, policies)
            assert study_record.decision.action == legacy_decision.action
            assert study_record.decision.state_index == legacy_decision.state_index
            assert study_record.breakdown.reward == pytest.approx(legacy_decision.reward)
        assert (study_agent.q_table == legacy_agent.q_table).all()
