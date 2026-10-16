"""Reward/probe alignment: the replay probe must match the controller's reward."""

import pandas as pd

from healing.canonical_numbers import HELPFUL, TOTAL_INTERVENTIONS
from healing.config import HealingConfig
from healing.reward import controller_reward, cost_in_band_units


def _band_scaled(reference: float, band: float, cost: float) -> float:
    return cost * max(reference, 1e-9) / max(band, 1e-9)


def test_controller_reward_centred_on_residual() -> None:
    cfg = HealingConfig()
    # Helpful residual above boundary minus cost should be positive, harmful negative
    assert controller_reward(residual=0.5, config=cfg, cost_band_units=0.05, changed=True, confirmed_failure=False) > 0
    assert controller_reward(residual=-0.5, config=cfg, cost_band_units=0.05, changed=True, confirmed_failure=True) < 0
    # Held carries no cost, so its reward is just residual - baseline
    held = controller_reward(residual=0.2, config=cfg, changed=False, held=True)
    intervened = controller_reward(residual=0.2, config=cfg, cost_band_units=0.1, changed=True, held=False)
    assert held > intervened


def test_cost_scaling_is_band_units() -> None:
    cfg = HealingConfig(intervention_cost=0.01, rollback_penalty=0.02)
    ref, band = 120.0, 30.0
    cost = cost_in_band_units(ref, band, cfg.intervention_cost)
    penalty = cost_in_band_units(ref, band, cfg.rollback_penalty)
    assert cost == _band_scaled(ref, band, 0.01)
    assert penalty == _band_scaled(ref, band, 0.02)
    # Cost must be in same units as residual (dimensionless), not waiting-time units
    reward_with = controller_reward(residual=0.3, config=cfg, cost_band_units=cost, changed=True, confirmed_failure=False)
    reward_without = controller_reward(residual=0.3, config=cfg, cost_band_units=0.0, changed=True, confirmed_failure=False)
    assert reward_with == reward_without - cost


def test_rollback_penalty_only_on_confirmed_failure() -> None:
    cfg = HealingConfig()
    cost = 0.05
    penalty = 0.10
    inc_rejected = controller_reward(residual=0.0, config=cfg, cost_band_units=cost, rollback_penalty_band=penalty, changed=True, confirmed_failure=True, held=False)
    inc_inconclusive = controller_reward(residual=0.0, config=cfg, cost_band_units=cost, rollback_penalty_band=penalty, changed=True, confirmed_failure=False, held=False)
    assert inc_rejected == inc_inconclusive - penalty


def test_healing_action_probe_produces_authoritative_reward_columns() -> None:
    from experiments.healing_experiment import action_effect_probe
    from experiments.healing_config import HealingExperimentConfig
    from experiments.healing_experiment import build_healing_workloads
    from workload.generator import WorkloadGenerator
    from config import build_default_config

    config = HealingExperimentConfig(
        training_seeds=(7101,),
        training_episodes_per_seed=2,
        validation_seed=8201,
        validation_repetitions=2,
        final_test_seed=23819,
        final_test_repetitions=2,
    )
    gen = WorkloadGenerator(build_default_config().families)
    workloads = build_healing_workloads("validation", config, gen)[:2]
    probe = action_effect_probe(config, workloads)
    assert "controller_reward" in probe.columns, "probe must carry controller_reward"
    assert "raw_load_improvement_band_units" in probe.columns
    assert "reward_band_units" in probe.columns  # backward compat alias
    assert "closure_residual" in probe.columns
    assert "verdict" in probe.columns
    # Held rows must have identical controller_reward and raw proxy only when trivial; otherwise they diverge.
    diff = (probe["controller_reward"] - probe["raw_load_improvement_band_units"]).abs()
    assert (diff > 1e-9).any(), "authoritative reward and raw proxy should differ for intervening actions"
