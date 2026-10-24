"""Probe schema regression: legacy raw proxy must not be mistaken for controller reward."""

import pathlib

import pandas as pd
import pytest

from errors import ValidationError


def _legacy_frame() -> pd.DataFrame:
    # Minimal legacy schema simulation: reward_band_units present, controller_reward absent
    return pd.DataFrame(
        {
            "action_name": ["hold", "set_sjf"],
            "reward_band_units": [0.07, 0.27],  # legacy raw proxy
            "true_effect_per_process": [0.0, 2.5],
            "diagnosis": ["fairness_starvation", "fairness_starvation"],
        }
    )


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_name": ["hold", "set_sjf"],
            "controller_reward": [-0.30, -0.25],
            "raw_load_improvement_band_units": [0.07, 0.27],
            "reward_band_units": [-0.30, -0.25],  # alias to controller_reward in new schema
            "closure_residual": [0.0, 0.12],
            "verdict": ["held", "accepted"],
            "confirmed_failure": [False, False],
            "window_decisions": [5, 5],
            "true_effect_per_process": [0.0, 2.5],
            "diagnosis": ["fairness_starvation", "fairness_starvation"],
            "paired_closure_change": [0.0, 0.22],
            "paired_raw_change": [0.0, 0.20],
            "paired_residual_change": [0.0, 0.12],
        }
    )


def test_canonical_probe_is_not_confused_with_legacy():
    legacy = _legacy_frame()
    assert "controller_reward" not in legacy.columns
    assert "raw_load_improvement_band_units" not in legacy.columns
    canonical = _canonical_frame()
    assert "controller_reward" in canonical.columns
    assert "raw_load_improvement_band_units" in canonical.columns
    # Legacy values are numerically close to raw proxy, but canonical controller_reward is distinct
    assert legacy["reward_band_units"].iloc[1] != canonical["controller_reward"].iloc[1]
    # In canonical file, reward_band_units must equal controller_reward (alias), not raw proxy
    assert canonical["reward_band_units"].iloc[1] == canonical["controller_reward"].iloc[1]
    assert canonical["raw_load_improvement_band_units"].iloc[1] == legacy["reward_band_units"].iloc[1]


def test_healing_plots_rejects_legacy_schema(tmp_path):
    from visualization.healing_plots import plot_healing_figures

    legacy = _legacy_frame()
    # Need minimal required columns for other figs to not error before probe check; we provide empty metrics etc but probe check happens first
    empty = pd.DataFrame()
    # Provide minimal probe-like frame missing controller_reward -> should raise ValidationError
    with pytest.raises(ValidationError, match="legacy schema.*missing 'controller_reward'"):
        plot_healing_figures(
            figures_dir=tmp_path / "invalid-output",
            metrics=pd.DataFrame({"arm": [], "family": [], "avg_waiting_time": [], "p95_waiting_time": []}),
            comparisons=empty,
            detection=pd.DataFrame({"split": [], "arm": [], "interventions": [], "accepted": [], "rejected": [], "inconclusive": [], "held": []}),
            audit=empty,
            probe=legacy,
        )
    assert not (tmp_path / "invalid-output").exists()


def test_healing_plots_accepts_canonical_schema(tmp_path, monkeypatch):
    from visualization.healing_plots import plot_healing_figures

    canonical = _canonical_frame()
    # Build minimal surrounding frames so plot_healing_figures can complete
    metrics = pd.DataFrame(
        {
            "arm": ["fixed_fcfs", "fixed_sjf"],
            "family": ["short_jobs", "short_jobs"],
            "avg_waiting_time": [100.0, 90.0],
            "p95_waiting_time": [150.0, 130.0],
        }
    )
    detection = pd.DataFrame(
        {
            "split": ["final_test", "final_test"],
            "arm": ["rule_wait", "q_recovery"],
            "incidents": [5, 5],
            "interventions": [3, 3],
            "accepted": [1, 1],
            "rejected": [1, 1],
            "inconclusive": [1, 1],
            "held": [0, 0],
            "episodes_with_fallback": [0, 0],
        }
    )
    audit = pd.DataFrame({"split": [], "arm": [], "matched_gain": []})
    comparisons = pd.DataFrame()
    # Inspect actual figure objects rather than assuming a saved PNG is correct.
    from matplotlib.figure import Figure
    saved = {}
    original_savefig = Figure.savefig

    def capture_figure(self, path, *args, **kwargs):
        saved[pathlib.Path(path).name] = self
        return original_savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    # Should not raise
    paths = plot_healing_figures(tmp_path, metrics, comparisons, detection, audit, canonical)
    assert "figure_action_effects" in paths
    assert pathlib.Path(paths["figure_action_effects"]).exists()
    action_figure = saved["healing_action_effects.png"]
    recovery_figure = saved["healing_recovery_outcomes.png"]
    assert action_figure is not recovery_figure
    assert not action_figure.axes[0].patches, "Recovery bars must not leak into reward scatter"
    assert len(action_figure.axes[0].collections) == 1
    assert list(action_figure.axes[0].lines[0].get_xdata()) == [-0.30, -0.30]
    assert action_figure.axes[0].get_xlabel().startswith("Controller reward")


def test_legacy_plot_labels_explicitly(tmp_path):
    from visualization.healing_plots import plot_healing_figures_legacy

    legacy = _legacy_frame()
    # Must succeed with legacy data and label as legacy; the output directory is a real
    # temporary directory rather than a hard-coded POSIX path, so this also runs on Windows.
    out = plot_healing_figures_legacy(tmp_path, legacy)
    assert "figure_action_effects_legacy" in out


def test_on_disk_canonical_is_not_legacy():
    # The committed canonical file must be the new schema after the fix
    probe_path = pathlib.Path("results/healing/healing_action_probe.csv")
    assert probe_path.exists(), "canonical probe must exist"
    probe = pd.read_csv(probe_path)
    assert "controller_reward" in probe.columns, "canonical must contain controller_reward"
    assert "raw_load_improvement_band_units" in probe.columns
    assert "closure_residual" in probe.columns
    assert "verdict" in probe.columns
    # reward_band_units in canonical must equal controller_reward (alias), not raw proxy
    # Check at least one row where they differ from raw
    diff = (probe["controller_reward"] - probe["raw_load_improvement_band_units"]).abs()
    assert (diff > 1e-6).any(), "canonical controller_reward must differ from raw proxy for intervening actions"
    # Legacy file preserved for traceability
    legacy_path = pathlib.Path("results/healing/healing_action_probe.legacy_raw_proxy.csv")
    assert legacy_path.exists(), "legacy file must be preserved"
    legacy = pd.read_csv(legacy_path)
    assert "controller_reward" not in legacy.columns
    assert "reward_band_units" in legacy.columns
