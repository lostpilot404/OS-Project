"""Derivation of the declared detection and verification thresholds.

The study declares two constants whose values are not free parameters: the congestion gate
``min_load_factor`` and the verification threshold ``accept_closure_factor``. Both are
derived here from the *no-action control* (the full monitoring, detection and diagnosis loop
with a policy that never intervenes) on the training split, and the derivation is stored as
an artifact of the study so that the declared numbers can be checked against data rather
than trusted.

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

The high percentile (above 1) is itself the study's main methodological finding: a trigger
is a local maximum of the monitored statistic, so the statistic decays afterwards whatever
the scheduler does. The measured decay is much larger than the measured effect of any
recovery action, so an in-run verdict cannot attribute causality; the counterfactual replay
audit in :mod:`experiments.healing_experiment` is what measures the true effect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from errors import ValidationError
from healing.config import HealingConfig
from healing.controller import SelfHealingController
from healing.policies import HoldOnlyPolicy
from scheduler import RuntimeSimulator
from experiments.healing_experiment import build_healing_workloads
from workload.generator import WorkloadGenerator

__all__ = [
    "GATE_SWEEP",
    "ThresholdDerivation",
    "collect_control_incidents",
    "derive_thresholds",
    "derivation_rows",
]

#: Congestion-gate values swept to document what the declared gate removes.
GATE_SWEEP: Tuple[float, ...] = (0.0, 0.5, 0.8, 1.0, 1.25)

#: Fraction of the control traces used for the (more expensive) sweep.
SWEEP_STRIDE = 10


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
    ]
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
    for split in ("training", "validation"):
        derivation, _ = derive_thresholds(config, split)
        print(f"[{split}] traces={derivation.traces} control incidents={derivation.incidents} "
              f"({derivation.incidents_per_trace:.3f}/trace)")
        print(f"[{split}] null closure p50={derivation.null_p50:.3f} p90={derivation.null_p90:.3f} "
              f"p95={derivation.null_p95:.3f} -> derived={derivation.derived_accept_closure_factor} "
              f"declared={derivation.declared_accept_closure_factor} "
              f"false-accept={derivation.declared_false_accept_rate:.3f}")
        for row in derivation.gate_sweep:
            print(f"[{split}] min_load_factor={row['min_load_factor']:.2f} "
                  f"incidents/trace={row['incidents_per_trace']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - developer entry point
    raise SystemExit(_main())
