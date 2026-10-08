"""Guardrails for the offline policy-selection experiment design."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from config import ACTION_NAMES, SELECTOR_LABEL, RewardConfig, SchedulerConfig
from experiments.evaluate import BASELINE_REGIME, SELECTOR_REGIME
from rl.adaptive import OfflinePolicySelector, PolicySelectionDecision


def test_round_robin_quantum_is_a_baseline_parameter_not_a_second_controller() -> None:
    assert SchedulerConfig().round_robin_quantum == 4
    assert ACTION_NAMES == ("FCFS", "SJF", "Round Robin", "Priority")
    assert SELECTOR_LABEL == "Offline Policy Selector (Q-Learning)"
    assert not Path("rl/quantum_controller.py").exists()


def test_quantum_controller_is_absent_from_the_core_api() -> None:
    import config
    import rl

    assert not hasattr(config, "QuantumControllerConfig")
    assert not hasattr(rl, "QuantumController")
    assert "quantum_controller" not in config.ExperimentConfig.__dataclass_fields__
    assert "quantum_controller" not in OfflinePolicySelector.__init__.__code__.co_varnames
    decision_fields = {field.name for field in fields(PolicySelectionDecision)}
    assert "quantum_multiplier" not in decision_fields
    assert "controller_multiplier" not in decision_fields


def test_evaluation_has_one_baseline_and_one_offline_selector_regime() -> None:
    assert BASELINE_REGIME == "baseline"
    assert SELECTOR_REGIME == "selector"


def test_default_reward_is_fixed_and_sums_to_one() -> None:
    reward = RewardConfig()
    assert reward.weight_waiting_time == 0.30
    assert reward.weight_turnaround_time == 0.25
    assert reward.weight_response_time == 0.20
    assert reward.weight_context_switches == 0.10
    assert reward.weight_cpu_utilization == 0.075
    assert reward.weight_throughput == 0.075
    assert reward.reference_clip == 2.0
    assert sum(reward.cost_weights() + reward.benefit_weights()) == pytest.approx(1.0)
