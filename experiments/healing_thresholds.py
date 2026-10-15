"""Derivation of the declared detection and verification thresholds.

The study declares constants whose values are not free parameters: the congestion gate
``min_load_factor``, the *unconditional* reporting threshold ``accept_closure_factor``, the
phase model ``closure_baseline`` and the two operative verification boundaries
``accept_residual_factor`` / ``reject_residual_factor``. All of them are derived here from the
*no-action control* (the full monitoring, detection and diagnosis loop with a policy that
never intervenes) on the training split, and the derivation is stored as an artifact of the
study so that the declared numbers can be checked against data rather than trusted. Nothing
in this module ever looks at the final-test split (it raises if asked).

What is measured
----------------
* **False triggers.** How many degradation events does the loop declare when the scheduler
  configuration never changes? Every one of them is, by construction, either the workload's
  own behaviour or noise, so the rate is an upper bound on the detector's false-positive
  rate. The sweep over ``min_load_factor`` shows what each declared gate removes.
* **Null verification statistic.** For those events, the *normalised load closure*
  ``(trigger_load - post_load) / max(trigger_load - reference_load, load_band)`` of the
  observation window. Since nothing was changed, this is exactly the distribution the
  verification rule sees when an intervention has no effect at all. Declaring
  ``accept_closure_factor`` at its 95th percentile caps the rule's false-accept rate at 5 %
  *under the control*, which is the only quantity that is measurable without an oracle.

* **Phase-aware baseline.** The closure statistic's no-action distribution depends strongly
  on the load geometry observed at the trigger, so a single marginal percentile wastes most
  of the available discrimination. The derivation therefore fits a least-squares baseline
  ``E[closure | trigger phase]`` on the control incidents (five features, all of them
  visible at the trigger epoch), declares its coefficients, and takes the verification
  boundaries from the *residual* distribution instead. Conditioning is a power improvement
  at an unchanged false-accept budget, not a relaxation of the rule: both boundaries stay at
  the 95th/5th percentile of the control, now of the residual.
* **Calibration check.** The same rates are recomputed on the *validation* control (a split
  the coefficients were not fitted on), so the declared false-accept and false-reject rates
  are checked out of sample and reported either way.

The high percentile (above 1) of the raw statistic is itself the study's main methodological
finding: a trigger is a local maximum of the monitored statistic, so the statistic decays
afterwards whatever the scheduler does. The measured decay is much larger than the measured
effect of the best recovery action (on the validation probe ``set_sjf`` moves the statistic by
+0.152 while the control spread is 0.797 raw and 0.339 after conditioning — 0.45 residual sd,
so about 39 windows of the same action would be needed for 80 % power), so an in-run verdict
can only ever accept a small share of the interventions that help. The
counterfactual replay audit in :mod:`experiments.healing_experiment` is what measures the
true effect; the verifier is reported as a safety filter with a measured accept rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import numpy as np

from errors import ValidationError
from healing.config import HealingConfig
from healing.verifier import closure_residual, expected_closure
from healing.controller import SelfHealingController
from healing.policies import HoldOnlyPolicy
from scheduler import RuntimeSimulator
from experiments.healing_experiment import build_healing_workloads
from workload.generator import WorkloadGenerator

__all__ = [
    "GATE_SWEEP",
    "BASELINE_FEATURES",
    "ThresholdDerivation",
    "collect_control_incidents",
    "derive_thresholds",
    "derivation_rows",
    "fit_closure_baseline",
    "residuals",
]

#: Feature order of the declared phase-aware baseline (intercept first). A single place
#: defines what "the trigger phase" means, and ``tests/test_healing_thresholds.py`` pins the
#: mapping against the verifier's own :class:`healing.verifier.TriggerPhase`.
BASELINE_FEATURES: Tuple[str, ...] = (
    "intercept",
    "excess_band_units",
    "relative_band",
    "load_band_at_trigger",
    "trigger_load_level",
    "reference_load_level",
)

#: Step the declared residual boundaries are rounded outward to.
BOUNDARY_STEP = 0.05

#: Congestion-gate values swept to document what the declared gate removes.
GATE_SWEEP: Tuple[float, ...] = (0.0, 0.5, 0.8, 1.0, 1.25)

#: Fraction of the control traces used for the (more expensive) sweep.
SWEEP_STRIDE = 10


def _design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the declared phase features of every control incident, in order."""
    return np.column_stack(
        [
            np.ones(len(frame)),
            frame["excess_band_units"].astype(float).to_numpy(),
            frame["relative_band"].astype(float).to_numpy(),
            frame["load_band_at_trigger"].astype(float).to_numpy(),
            frame["trigger_load_level"].astype(float).to_numpy(),
            frame["reference_load_level"].astype(float).to_numpy(),
        ]
    )


def fit_closure_baseline(incidents: pd.DataFrame) -> Tuple[float, ...]:
    """Fit the phase-aware baseline of the closure statistic on control incidents.

    Args:
        incidents: Control incident ledger with the declared phase columns and a
            ``closure_fraction`` column.

    Returns:
        ``(intercept, excess, relative_band, load_band, trigger_load, reference_load)``,
        the least-squares coefficients in :data:`BASELINE_FEATURES` order.

    Raises:
        ValidationError: If there is not enough evidence to fit a baseline.
    """
    usable = incidents[incidents["closure_fraction"].notna()]
    if len(usable) < len(BASELINE_FEATURES) + 1:
        raise ValidationError("not enough control incidents to fit the phase baseline")
    design = _design_matrix(usable)
    target = usable["closure_fraction"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return tuple(float(value) for value in beta)


def residuals(incidents: pd.DataFrame, baseline: Sequence[float]) -> np.ndarray:
    """Return the phase-aware residuals of a control incident ledger."""
    usable = incidents[incidents["closure_fraction"].notna()]
    if usable.empty:
        return np.asarray([], dtype=float)
    expected = _design_matrix(usable) @ np.asarray(list(baseline), dtype=float)
    return usable["closure_fraction"].astype(float).to_numpy() - expected


def _round_outward(quantile: float) -> float:
    """Round a quantile to the next boundary step *away* from the acceptance region."""
    steps = np.floor(quantile / BOUNDARY_STEP) if quantile < 0 else np.ceil(quantile / BOUNDARY_STEP)
    return float(round(steps * BOUNDARY_STEP, 4))


@dataclass(frozen=True)
class ThresholdDerivation:
    """Derived evidence behind the declared detector and verification thresholds."""

    split: str
    traces: int
    incidents: int
    incidents_per_trace: float
    null_p50: float
    null_p90: float
    null_p95: float
    derived_accept_closure_factor: float
    declared_accept_closure_factor: float
    declared_false_accept_rate: float
    #: Phase-aware baseline fitted on the control and the coefficients actually declared.
    fitted_baseline: Tuple[float, ...]
    declared_baseline: Tuple[float, ...]
    baseline_r2: float
    residual_sd: float
    residual_reduction: float
    residual_p05: float
    residual_p95: float
    derived_accept_residual_factor: float
    derived_reject_residual_factor: float
    declared_accept_residual_factor: float
    declared_reject_residual_factor: float
    false_accept_residual_rate: float
    false_reject_residual_rate: float
    #: Out-of-sample calibration of the *declared* rule on the validation control.
    validation_traces: int
    validation_incidents: int
    validation_false_accept_rate: float
    validation_false_reject_rate: float
    gate_sweep: Tuple[Dict[str, Any], ...]

    def as_dict(self) -> Dict[str, Any]:
        """Flat dictionary for the JSON summary."""
        return {
            "split": self.split,
            "traces": self.traces,
            "control_incidents": self.incidents,
            "control_incidents_per_trace": self.incidents_per_trace,
            "null_closure_p50": self.null_p50,
            "null_closure_p90": self.null_p90,
            "null_closure_p95": self.null_p95,
            "derived_accept_closure_factor": self.derived_accept_closure_factor,
            "declared_accept_closure_factor": self.declared_accept_closure_factor,
            "declared_false_accept_rate_on_control": self.declared_false_accept_rate,
            "baseline_features": list(BASELINE_FEATURES),
            "fitted_closure_baseline": list(self.fitted_baseline),
            "declared_closure_baseline": list(self.declared_baseline),
            "baseline_r2": self.baseline_r2,
            "baseline_residual_sd": self.residual_sd,
            "residual_sd_reduction": self.residual_reduction,
            "control_residual_p05": self.residual_p05,
            "control_residual_p95": self.residual_p95,
            "derived_accept_residual_factor": self.derived_accept_residual_factor,
            "derived_reject_residual_factor": self.derived_reject_residual_factor,
            "declared_accept_residual_factor": self.declared_accept_residual_factor,
            "declared_reject_residual_factor": self.declared_reject_residual_factor,
            "false_accept_residual_rate_on_control": self.false_accept_residual_rate,
            "false_reject_residual_rate_on_control": self.false_reject_residual_rate,
            "validation_traces": self.validation_traces,
            "validation_control_incidents": self.validation_incidents,
            "validation_false_accept_rate": self.validation_false_accept_rate,
            "validation_false_reject_rate": self.validation_false_reject_rate,
            "gate_sweep": [dict(row) for row in self.gate_sweep],
        }


def collect_control_incidents(
    config,
    split: str = "training",
    *,
    healing: Optional[HealingConfig] = None,
    stride: int = 1,
) -> pd.DataFrame:
    """Run the no-action control over one split and return its incident ledger.

    Args:
        config: The study configuration (families, split seeds and repetitions).
        split: Which split to use. Only ``training`` and ``validation`` are valid here: the
            final-test split must never inform a declared constant.
        healing: Optional override of the declared thresholds (used by the gate sweep).
        stride: Keep every ``stride``-th trace (deterministic subsetting).

    Returns:
        One row per declared degradation event, with the observed null statistics.
    """
    if split not in ("training", "validation"):
        raise ValidationError(f"thresholds may only be derived from training/validation, got {split!r}")
    if stride < 1:
        raise ValidationError(f"stride must be >= 1, got {stride!r}")
    thresholds = healing or config.healing
    workloads = build_healing_workloads(split, config, WorkloadGenerator(config.families))
    simulator = RuntimeSimulator(config.scheduler)
    rows: List[Dict[str, Any]] = []
    evaluated = 0
    for index, item in enumerate(workloads):
        if index % stride:
            continue
        evaluated += 1
        controller = SelfHealingController(
            thresholds,
            config.scheduler,
            HoldOnlyPolicy(),
            initial_policy=0,
            label="control",
        )
        controller.begin_episode(index)
        simulator.run(item.workload, controller, policy_name="control")
        episode = controller.end_episode()
        for incident in episode.incidents:
            rows.append(
                {
                    "split": split,
                    "trace_index": index,
                    "family": item.family,
                    "decision_index": incident.decision_index,
                    "window_mean_at_trigger": incident.window_mean_at_trigger,
                    "trigger_load_level": incident.trigger_load_level,
                    "reference_load_level": incident.pre_load_level,
                    "post_load_level": incident.post_load_level,
                    "load_band_at_trigger": incident.load_band_at_trigger,
                    "closure_fraction": incident.closure_fraction,
                    "excess_band_units": incident.excess_band_units,
                    "relative_band": (
                        incident.band_at_trigger / incident.reference_at_trigger
                        if incident.reference_at_trigger > 0.0
                        else float("nan")
                    ),
                    "verdict": incident.verdict,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["traces"] = evaluated
    return frame


def derive_thresholds(config, split: str = "training") -> Tuple[ThresholdDerivation, pd.DataFrame]:
    """Derive the declared thresholds from the no-action control on one split.

    Returns:
        The :class:`ThresholdDerivation` and the incident-level evidence behind it.
    """
    incidents = collect_control_incidents(config, split)
    traces = int(incidents.attrs.get("traces", 0))
    closure = incidents["closure_fraction"].dropna() if not incidents.empty else pd.Series(dtype=float)
    null_p50 = float(np.percentile(closure, 50)) if len(closure) else float("nan")
    null_p90 = float(np.percentile(closure, 90)) if len(closure) else float("nan")
    null_p95 = float(np.percentile(closure, 95)) if len(closure) else float("nan")
    derived = round(null_p95, 1) if len(closure) else float(config.healing.accept_closure_factor)
    declared = float(config.healing.accept_closure_factor)
    false_accept = float((closure >= declared).mean()) if len(closure) else float("nan")

    if len(closure):
        fitted = fit_closure_baseline(incidents)
        residual = residuals(incidents, fitted)
        predicted = closure.to_numpy() - residual
        spread = float(closure.std(ddof=0))
        r2 = float(1.0 - residual.var() / closure.var()) if closure.var() > 0 else float("nan")
        residual_sd = float(residual.std(ddof=0))
        residual_p05 = float(np.percentile(residual, 5))
        residual_p95 = float(np.percentile(residual, 95))
    else:  # pragma: no cover - an empty control is a broken experiment, reported, not hidden
        fitted = tuple(config.healing.closure_baseline)
        residual = np.asarray([], dtype=float)
        residual_sd = float("nan")
        r2 = float("nan")
        residual_p05 = residual_p95 = float("nan")
        spread = float("nan")
    derived_accept_residual = (
        _round_outward(residual_p95) if len(residual) else float(config.healing.accept_residual_factor)
    )
    derived_reject_residual = (
        _round_outward(residual_p05) if len(residual) else float(config.healing.reject_residual_factor)
    )
    declared_baseline = tuple(float(value) for value in config.healing.closure_baseline)
    declared_residual = residuals(incidents, declared_baseline) if len(closure) else np.asarray([], dtype=float)
    false_accept_residual = (
        float((declared_residual >= config.healing.accept_residual_factor).mean())
        if len(declared_residual)
        else float("nan")
    )
    false_reject_residual = (
        float((declared_residual <= config.healing.reject_residual_factor).mean())
        if len(declared_residual)
        else float("nan")
    )
    # Out-of-sample calibration check on the validation control (never used for fitting).
    validation = collect_control_incidents(config, "validation")
    validation_residual = residuals(validation, declared_baseline) if not validation.empty else np.asarray([], dtype=float)
    validation_false_accept = (
        float((validation_residual >= config.healing.accept_residual_factor).mean())
        if len(validation_residual)
        else float("nan")
    )
    validation_false_reject = (
        float((validation_residual <= config.healing.reject_residual_factor).mean())
        if len(validation_residual)
        else float("nan")
    )

    sweep: List[Dict[str, Any]] = []
    for factor in GATE_SWEEP:
        subset = collect_control_incidents(
            config,
            split,
            healing=replace(config.healing, min_load_factor=factor),
            stride=SWEEP_STRIDE,
        )
        sweep_traces = max(1, int(subset.attrs.get("traces", 0)))
        sweep.append(
            {
                "min_load_factor": factor,
                "traces": sweep_traces,
                "incidents": int(len(subset)),
                "incidents_per_trace": len(subset) / sweep_traces,
            }
        )

    derivation = ThresholdDerivation(
        split=split,
        traces=traces,
        incidents=int(len(incidents)),
        incidents_per_trace=(len(incidents) / traces) if traces else float("nan"),
        null_p50=null_p50,
        null_p90=null_p90,
        null_p95=null_p95,
        derived_accept_closure_factor=derived,
        declared_accept_closure_factor=declared,
        declared_false_accept_rate=false_accept,
        fitted_baseline=fitted,
        declared_baseline=declared_baseline,
        baseline_r2=r2,
        residual_sd=residual_sd,
        residual_reduction=(1.0 - residual_sd / spread) if spread and spread > 0 else float("nan"),
        residual_p05=residual_p05,
        residual_p95=residual_p95,
        derived_accept_residual_factor=derived_accept_residual,
        derived_reject_residual_factor=derived_reject_residual,
        declared_accept_residual_factor=float(config.healing.accept_residual_factor),
        declared_reject_residual_factor=float(config.healing.reject_residual_factor),
        false_accept_residual_rate=false_accept_residual,
        false_reject_residual_rate=false_reject_residual,
        validation_traces=int(validation.attrs.get("traces", 0)),
        validation_incidents=int(len(validation)),
        validation_false_accept_rate=validation_false_accept,
        validation_false_reject_rate=validation_false_reject,
        gate_sweep=tuple(sweep),
    )
    return derivation, incidents


def derivation_rows(derivation: ThresholdDerivation) -> List[Dict[str, Any]]:
    """Rows of the machine-readable derivation artifact (one per quantity)."""
    rows = [
        {"quantity": "traces", "value": derivation.traces, "source": derivation.split},
        {"quantity": "control_incidents", "value": derivation.incidents, "source": derivation.split},
        {
            "quantity": "control_incidents_per_trace",
            "value": round(derivation.incidents_per_trace, 4),
            "source": derivation.split,
        },
        {"quantity": "null_closure_p50", "value": round(derivation.null_p50, 4), "source": derivation.split},
        {"quantity": "null_closure_p90", "value": round(derivation.null_p90, 4), "source": derivation.split},
        {"quantity": "null_closure_p95", "value": round(derivation.null_p95, 4), "source": derivation.split},
        {
            "quantity": "derived_accept_closure_factor",
            "value": derivation.derived_accept_closure_factor,
            "source": f"{derivation.split} (95th percentile of the null closure)",
        },
        {
            "quantity": "declared_accept_closure_factor",
            "value": derivation.declared_accept_closure_factor,
            "source": "healing/config.py HealingConfig",
        },
        {
            "quantity": "declared_false_accept_rate_on_control",
            "value": round(derivation.declared_false_accept_rate, 4),
            "source": f"{derivation.split} (control)",
        },
        {
            "quantity": "baseline_r2",
            "value": round(derivation.baseline_r2, 4),
            "source": f"{derivation.split} (control, least squares)",
        },
        {
            "quantity": "baseline_residual_sd",
            "value": round(derivation.residual_sd, 4),
            "source": f"{derivation.split} (control residual)",
        },
        {
            "quantity": "control_residual_p95",
            "value": round(derivation.residual_p95, 4),
            "source": f"{derivation.split} (95th percentile of the residual)",
        },
        {
            "quantity": "control_residual_p05",
            "value": round(derivation.residual_p05, 4),
            "source": f"{derivation.split} (5th percentile of the residual)",
        },
        {
            "quantity": "derived_accept_residual_factor",
            "value": derivation.derived_accept_residual_factor,
            "source": f"{derivation.split} (control residual p95, rounded outward)",
        },
        {
            "quantity": "derived_reject_residual_factor",
            "value": derivation.derived_reject_residual_factor,
            "source": f"{derivation.split} (control residual p05, rounded outward)",
        },
        {
            "quantity": "declared_accept_residual_factor",
            "value": derivation.declared_accept_residual_factor,
            "source": "healing/config.py HealingConfig",
        },
        {
            "quantity": "declared_reject_residual_factor",
            "value": derivation.declared_reject_residual_factor,
            "source": "healing/config.py HealingConfig",
        },
        {
            "quantity": "declared_false_accept_residual_rate_on_control",
            "value": round(derivation.false_accept_residual_rate, 4),
            "source": f"{derivation.split} (control, declared rule)",
        },
        {
            "quantity": "declared_false_reject_residual_rate_on_control",
            "value": round(derivation.false_reject_residual_rate, 4),
            "source": f"{derivation.split} (control, declared rule)",
        },
        {
            "quantity": "validation_false_accept_residual_rate",
            "value": round(derivation.validation_false_accept_rate, 4),
            "source": "validation (control, out of sample)",
        },
        {
            "quantity": "validation_false_reject_residual_rate",
            "value": round(derivation.validation_false_reject_rate, 4),
            "source": "validation (control, out of sample)",
        },
    ]
    for name, coefficient in zip(BASELINE_FEATURES, derivation.fitted_baseline):
        rows.append(
            {
                "quantity": f"fitted_closure_baseline[{name}]",
                "value": round(coefficient, 6),
                "source": f"{derivation.split} (control, least squares)",
            }
        )
    for name, coefficient in zip(BASELINE_FEATURES, derivation.declared_baseline):
        rows.append(
            {
                "quantity": f"declared_closure_baseline[{name}]",
                "value": round(coefficient, 6),
                "source": "healing/config.py HealingConfig",
            }
        )
    for row in derivation.gate_sweep:
        rows.append(
            {
                "quantity": f"control_incidents_per_trace@min_load_factor={row['min_load_factor']}",
                "value": round(row["incidents_per_trace"], 4),
                "source": f"{derivation.split} (stride {SWEEP_STRIDE})",
            }
        )
    return rows


def _main() -> int:
    """Print the derivation for the documentation (``python -m experiments.healing_thresholds``)."""
    from experiments.healing_config import HealingExperimentConfig

    config = HealingExperimentConfig()
    derivation, _ = derive_thresholds(config, "training")
    print(f"[training] traces={derivation.traces} control incidents={derivation.incidents} "
          f"({derivation.incidents_per_trace:.3f}/trace)")
    print(f"[training] raw null closure p50={derivation.null_p50:.3f} p90={derivation.null_p90:.3f} "
          f"p95={derivation.null_p95:.3f} -> derived={derivation.derived_accept_closure_factor} "
          f"declared={derivation.declared_accept_closure_factor} "
          f"false-accept={derivation.declared_false_accept_rate:.3f}")
    print("[training] fitted phase baseline "
          + ", ".join(f"{name}={value:+.4f}" for name, value in zip(BASELINE_FEATURES, derivation.fitted_baseline)))
    print(f"[training] baseline R2={derivation.baseline_r2:.3f} residual sd={derivation.residual_sd:.3f} "
          f"(raw sd reduction {derivation.residual_reduction:.1%})")
    print(f"[training] residual p05={derivation.residual_p05:+.3f} p95={derivation.residual_p95:+.3f} "
          f"-> derived accept={derivation.derived_accept_residual_factor:+.2f} "
          f"reject={derivation.derived_reject_residual_factor:+.2f} | declared "
          f"accept={derivation.declared_accept_residual_factor:+.2f} "
          f"reject={derivation.declared_reject_residual_factor:+.2f} "
          f"| false accept={derivation.false_accept_residual_rate:.3f} "
          f"false reject={derivation.false_reject_residual_rate:.3f}")
    print(f"[validation] control incidents={derivation.validation_incidents} "
          f"out-of-sample false accept={derivation.validation_false_accept_rate:.3f} "
          f"false reject={derivation.validation_false_reject_rate:.3f}")
    for row in derivation.gate_sweep:
        print(f"[training] min_load_factor={row['min_load_factor']:.2f} "
              f"incidents/trace={row['incidents_per_trace']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - developer entry point
    raise SystemExit(_main())
