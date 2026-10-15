"""Derivation of the declared detection and verification thresholds.

The study declares constants whose values are not free parameters: the congestion gate
``min_load_factor``, the *unconditional* reporting threshold ``accept_closure_factor``, the
phase model ``post_load_baseline`` and the two operative verification boundaries
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
  of the available discrimination. The derivation therefore fits a least-squares baseline for
  the *no-action post-window load* on the control incidents (three physically observed
  trigger features plus an intercept, all visible at the trigger epoch), forms the closure
  expectation with the statistic's own denominator, declares those coefficients, and takes
  the verification boundaries from the *residual* distribution instead. Conditioning is a
  power improvement
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "BASELINE_FEATURES",
    "ThresholdDerivation",
    "collect_control_incidents",
    "derive_thresholds",
    "derivation_rows",
    "fit_post_load_baseline",
    "expected_closure_column",
    "judgeable",
    "verification_model_comparison",
    "residuals",
]

#: Regressor order of the declared structural baseline (intercept first). A single place
#: defines what "the trigger phase" means, and the test suite pins the mapping against the
#: verifier's own :meth:`healing.verifier.TriggerPhase.features`.
BASELINE_FEATURES: Tuple[str, ...] = (
    "intercept",
    "trigger_load_level",
    "ready_count_at_trigger",
    "log1p_visible_work",
)

#: Regressor sets that were measured on the control *before* the declared one was chosen.
#: They are fitted and reported by :func:`verification_model_comparison`, so the choice of
#: the declared specification is reproducible instead of a narrative. The first entry is the
#: free fit of the closure statistic itself that the structural specification replaced.
ALTERNATIVE_FEATURES: Tuple[Tuple[str, ...], ...] = (
    # The free fit of the closure statistic that the structural specification replaced.
    (
        "intercept",
        "excess_band_units",
        "relative_band",
        "load_band_at_trigger",
        "trigger_load_level",
        "reference_load_level",
    ),
    # The trigger's two queue levels and nothing else: the reference point the structural
    # specification has to beat.
    (
        "intercept",
        "trigger_load_level",
        "reference_load_level",
    ),
    # Levels plus a clock term, i.e. "the trace has been running for a while".
    (
        "intercept",
        "trigger_load_level",
        "reference_load_level",
        "log1p_simulated_time",
    ),
    # Levels plus the controller's own decision counter and the clock. This was the first
    # structural specification tried; it is kept in the table because its coefficients are a
    # warning: the decision counter alone replaced most of the ready-set information in the
    # training control, which is an artefact of a population whose triggers all happen in the
    # first ten decisions, not a property of the workload.
    (
        "intercept",
        "trigger_load_level",
        "reference_load_level",
        "decision_index",
        "log1p_simulated_time",
    ),
    # The declared specification plus the trace's own normal level (collinear with the
    # trigger level; the coefficient changes sign between control halves).
    (
        "intercept",
        "trigger_load_level",
        "reference_load_level",
        "ready_count_at_trigger",
        "log1p_visible_work",
    ),
    # The declared specification reduced to the ready set alone.
    (
        "intercept",
        "ready_count_at_trigger",
    ),
    # The declared specification plus the monitor window's own size (an instrumentation
    # constant rather than a workload quantity).
    (
        "intercept",
        "trigger_load_level",
        "ready_count_at_trigger",
        "log1p_visible_work",
        "window_size_at_trigger",
    ),
    # The declared specification plus the simulated clock.
    (
        "intercept",
        "trigger_load_level",
        "ready_count_at_trigger",
        "log1p_visible_work",
        "log1p_simulated_time",
    ),
)

#: Step the declared residual boundaries are rounded outward to.
BOUNDARY_STEP = 0.05

#: Congestion-gate values swept to document what the declared gate removes.
GATE_SWEEP: Tuple[float, ...] = (0.0, 0.5, 0.8, 1.0, 1.25)

#: Fraction of the control traces used for the (more expensive) sweep.
SWEEP_STRIDE = 10


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return one regressor column by its declared name.

    Raises:
        ValidationError: If the column is missing from the ledger.
    """
    if name == "intercept":
        return np.ones(len(frame))
    if name == "gap":
        return (
            frame["trigger_load_level"].astype(float) - frame["reference_load_level"].astype(float)
        ).to_numpy()
    if name == "log1p_visible_work":
        return np.log1p(np.maximum(frame["visible_remaining_work_at_trigger"].astype(float), 0.0))
    if name == "log1p_simulated_time":
        return np.log1p(np.maximum(frame["simulated_time"].astype(float), 0.0))
    for candidate in (name, f"{name}_at_trigger"):
        if candidate in frame.columns:
            return frame[candidate].astype(float).to_numpy()
    raise ValidationError(f"control ledger has no column for regressor {name!r}")


def _design_matrix(frame: pd.DataFrame, features: Sequence[str] = BASELINE_FEATURES) -> np.ndarray:
    """Return the requested regressors of every control incident, in order."""
    return np.column_stack([_column(frame, name) for name in features])


def judgeable(incidents: pd.DataFrame) -> pd.DataFrame:
    """Return the control windows the verifier would actually score.

    The boundaries are percentiles of the no-action residual, so the population they are
    taken over has to be the population the rule can judge. A window that was truncated by
    the end of the trace, that observed no interval at all, or that never reached the
    declared completion floor is concluded as inconclusive without any comparison against
    the boundaries (:func:`healing.verifier.verify`), so including those windows would
    calibrate the rule on evidence it never sees and make every declared rate an
    overstatement of its strictness.
    """
    frame = incidents[incidents["closure_fraction"].notna()].copy()
    if "window_truncated" in frame.columns:
        frame = frame[~frame["window_truncated"].astype(bool)]
    if "verification_decisions" in frame.columns:
        frame = frame[frame["verification_decisions"].astype(int) > 0]
    if "verification_completions" in frame.columns and "verify_completions" in frame.attrs:
        floor = int(frame.attrs["verify_completions"])
        frame = frame[frame["verification_completions"].astype(int) >= floor]
    return frame


def _closure_denominator(frame: pd.DataFrame) -> np.ndarray:
    """Return ``max(trigger - reference, max(load_band, 1e-9))`` for every incident row."""
    return np.maximum(
        frame["trigger_load_level"].astype(float).to_numpy()
        - frame["reference_load_level"].astype(float).to_numpy(),
        np.maximum(frame["load_band_at_trigger"].astype(float).to_numpy(), 1e-9),
    )


def _rank_auc(scores, positive) -> float:
    """Return the rank AUC of ``scores`` separating ``positive`` from the rest.

    Ties contribute half, which is the convention of the reported discrimination metric.
    """
    scores = pd.Series(np.asarray(scores, dtype=float))
    label = pd.Series(np.asarray(positive, dtype=bool))
    positives = int(label.sum())
    negatives = int((~label).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = scores.rank()
    return float((ranks[label.to_numpy()].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def fit_post_load_baseline(
    incidents: pd.DataFrame, features: Sequence[str] = BASELINE_FEATURES
) -> Tuple[float, ...]:
    """Fit the structural phase-aware reference on the no-action control.

    Args:
        incidents: Control incident ledger with the regressor columns, the observed
            ``post_load_level`` and a ``closure_fraction`` column.
        features: Regressor names to fit (defaults to the declared set).

    Returns:
        ``(intercept, ...)``, the least-squares coefficients in ``features`` order.

    Raises:
        ValidationError: If there is not enough evidence to fit a baseline.
    """
    usable = judgeable(incidents)
    usable = usable[usable["post_load_level"].notna()]
    if len(usable) < len(features) + 1:
        raise ValidationError("not enough control incidents to fit the phase baseline")
    design = _design_matrix(usable, features)
    target = usable["post_load_level"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return tuple(float(value) for value in beta)


def expected_closure_column(
    incidents: pd.DataFrame, baseline: Sequence[float], features: Sequence[str] = BASELINE_FEATURES
) -> np.ndarray:
    """Return the structural expectation of the closure statistic for every row.

    The expectation is formed exactly like the observed statistic: the fitted queue level is
    turned into a closure with ``(trigger_load - expected_post_load) / max(trigger_load -
    reference_load, load_band)``, so the dispersion floor and the ratio stay in the
    reference instead of being linearised away.
    """
    usable = incidents
    trigger = usable["trigger_load_level"].astype(float).to_numpy()
    reference = usable["reference_load_level"].astype(float).to_numpy()
    band = np.maximum(usable["load_band_at_trigger"].astype(float).to_numpy(), 1e-9)
    denominator = np.maximum(trigger - reference, band)
    expected_post = _design_matrix(usable, features) @ np.asarray(list(baseline), dtype=float)
    return (trigger - expected_post) / denominator


def residuals(
    incidents: pd.DataFrame, baseline: Sequence[float], features: Sequence[str] = BASELINE_FEATURES
) -> np.ndarray:
    """Return the phase-aware residuals of the control windows the rule can judge."""
    usable = judgeable(incidents)
    if usable.empty:
        return np.asarray([], dtype=float)
    return usable["closure_fraction"].astype(float).to_numpy() - expected_closure_column(
        usable, baseline, features
    )


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
    #: Calibration population: how many declared events the verifier can actually judge, and
    #: how many were excluded because the trace end truncated them or their telemetry was
    #: below the declared completion floor.
    judged_incidents: int
    excluded_truncated: int
    excluded_thin: int
    #: The same null statistics over *every* declared event, for continuity with the earlier
    #: derivation and to make the effect of the population choice visible.
    all_events_incidents: int
    all_events_null_p95: float
    all_events_residual_sd: float
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
            "fitted_post_load_baseline": list(self.fitted_baseline),
            "declared_post_load_baseline": list(self.declared_baseline),
            "judged_incidents": self.judged_incidents,
            "excluded_truncated": self.excluded_truncated,
            "excluded_thin": self.excluded_thin,
            "all_events_incidents": self.all_events_incidents,
            "all_events_null_p95": self.all_events_null_p95,
            "all_events_residual_sd": self.all_events_residual_sd,
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
                    "simulated_time": incident.simulated_time,
                    "window_mean_at_trigger": incident.window_mean_at_trigger,
                    "window_size_at_trigger": incident.window_size_at_trigger,
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
                    "ready_count_at_trigger": incident.ready_count_at_trigger,
                    "visible_remaining_work_at_trigger": (
                        incident.visible_remaining_work_at_trigger
                    ),
                    "verification_decisions": incident.verification_decisions,
                    "verification_completions": incident.verification_completions,
                    "window_truncated": incident.window_truncated,
                    "verdict": incident.verdict,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["traces"] = evaluated
    frame.attrs["verify_completions"] = int(thresholds.verify_completions)
    return frame


def verification_model_comparison(
    config,
    *,
    control: Optional[pd.DataFrame] = None,
    validation_control: Optional[pd.DataFrame] = None,
    audit: Optional[pd.DataFrame] = None,
    incidents: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compare the candidate phase-aware references honestly and return the evidence table.

    The declared specification was chosen by **control fit only**: each candidate's
    out-of-sample residual spread is estimated by five-fold cross-validation grouped by
    trace (folds never share a trace, so the estimate cannot be inflated by the control's own
    within-trace structure), and the decision-relevant behaviour of each candidate is then
    *reported* rather than used for the choice. When a validation audit frame and the
    matching incident ledger are supplied, the table also gives the share of genuinely
    helpful and genuinely harmful interventions whose window would clear that candidate's
    95th-percentile boundary.

    Args:
        config: The study configuration (splits, families, declared constants).
        control: Training control ledger (collected here when not supplied).
        validation_control: Validation control ledger, used for the calibration check.
        audit: Optional intervention-level audit frame (``healing_audit.csv``); only
            ``split == "validation"`` rows are used.
        incidents: Optional validation incident ledger holding the trigger-time regressors
            of the audited interventions (``healing_incidents_validation.csv``).

    Returns:
        One row per candidate specification.
    """
    control = collect_control_incidents(config, "training") if control is None else control
    validation_control = (
        collect_control_incidents(config, "validation")
        if validation_control is None
        else validation_control
    )
    judged = judgeable(control).reset_index(drop=True)
    folds = pd.factorize(judged["trace_index"].to_numpy())[0] % 5
    joined: Optional[pd.DataFrame] = None
    if audit is not None and incidents is not None and not audit.empty:
        held = audit[
            (audit["split"].astype(str) == "validation")
            & (audit["verdict"].astype(str) != "episode_level")
        ]
        key = ["arm", "training_seed", "family", "repetition", "decision_index"]
        ledger = incidents[
            key
            + [
                "trigger_load_level",
                "load_band_at_trigger",
                "simulated_time",
                "ready_count_at_trigger",
                "visible_remaining_work_at_trigger",
                "window_size_at_trigger",
                "excess_band_units",
                "band_at_trigger",
                "reference_at_trigger",
            ]
            + (["reference_load_level"] if "reference_load_level" in incidents.columns else ["pre_load_level"])
        ].copy()
        ledger["relative_band"] = (
            ledger["band_at_trigger"].astype(float)
            / ledger["reference_at_trigger"].astype(float).where(
                ledger["reference_at_trigger"].astype(float) > 0.0
            )
        )
        if "reference_load_level" not in ledger.columns:
            # The incident ledger names the trace's own normal level ``pre_load_level``; the
            # control ledger (and every regressor name) calls it ``reference_load_level``.
            ledger = ledger.rename(columns={"pre_load_level": "reference_load_level"})
        joined = held.merge(ledger, on=key, how="inner")
    specs: List[Tuple[str, Tuple[str, ...]]] = [("declared", BASELINE_FEATURES)] + [
        (f"alternative_{index}", features)
        for index, features in enumerate(ALTERNATIVE_FEATURES, start=1)
    ]
    # The audited windows are the same rows for every candidate: the population the verifier
    # is judged on must not change with the specification being compared.
    scored = None
    helped_mask: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    if joined is not None and len(joined):
        scored = joined[joined["closure_fraction"].notna()].reset_index(drop=True)
        if len(scored):
            helped_mask = scored["matched_gain"].astype(float).to_numpy() > 0
            clusters = scored.groupby(["family", "repetition"]).ngroup().to_numpy()
    rows: List[Dict[str, Any]] = []
    certified_by_spec: Dict[str, np.ndarray] = {}
    for name, features in specs:
        design = _design_matrix(judged, features)
        target = judged["post_load_level"].astype(float).to_numpy()
        out_of_fold = np.empty(len(judged))
        for fold in range(5):
            mask = folds != fold
            coefficients, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
            out_of_fold[~mask] = target[~mask] - design[~mask] @ coefficients
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = residuals(control, coefficients, features)
        boundary = float(np.percentile(residual, 95)) if len(residual) else float("nan")
        validation_residual = residuals(validation_control, coefficients, features)
        row: Dict[str, Any] = {
            "specification": name,
            "features": ",".join(features),
            "parameters": len(features),
            "judged_incidents": int(len(judged)),
            "out_of_sample_post_load_sd": float(out_of_fold.std(ddof=0)),
            "out_of_sample_closure_residual_sd": float(
                (-(out_of_fold / _closure_denominator(judged))).std(ddof=0)
            ),
            "residual_sd": float(residual.std(ddof=0)) if len(residual) else float("nan"),
            "residual_p95_boundary": boundary,
            "validation_false_accept": (
                float((validation_residual >= boundary).mean())
                if len(validation_residual)
                else float("nan")
            ),
            "audited_interventions": int(len(joined)) if joined is not None else 0,
            "certified_share_of_helpful": float("nan"),
            "certified_share_of_harmful": float("nan"),
        }
        if scored is not None and len(scored):
            window_residual = scored["closure_fraction"].astype(float).to_numpy() - (
                expected_closure_column(scored, coefficients, features)
            )
            certified = window_residual >= boundary
            helped = helped_mask
            harmful = scored["matched_gain"].astype(float).to_numpy() < 0
            certified_by_spec[name] = certified
            if helped.any():
                row["certified_share_of_helpful"] = float(certified[helped].mean())
            if harmful.any():
                row["certified_share_of_harmful"] = float(certified[harmful].mean())
            row["rank_auc_helpful_over_harmful"] = _rank_auc(window_residual, helped)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _attach_paired_uncertainty(frame, certified_by_spec, helped_mask, clusters)
    return frame


def _attach_paired_uncertainty(
    frame: pd.DataFrame,
    certified_by_spec: Dict[str, np.ndarray],
    helped: Optional[np.ndarray],
    clusters: Optional[np.ndarray],
    *,
    reference: str = "declared",
    draws: int = 2000,
    seed: int = 20261015,
) -> None:
    """Add the paired uncertainty of the decision-relevant metric to the comparison table.

    ``certified_share_of_helpful`` is a share over *interventions*, but interventions are
    clustered: several of them belong to the same workload trace, and the traces differ in how
    much a recovery window can possibly close. The difference between two candidate
    specifications is therefore resampled over the workload clusters (paired: the same clusters
    are used for both specifications in every draw), which is the only comparison the study
    makes for the choice of the declared reference. The table is written even when the interval
    excludes zero for a candidate that was not adopted — the numbers are recorded, not hidden.
    """
    columns = (
        "paired_delta_certified_helpful",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
        "paired_delta_includes_zero",
    )
    for column in columns:
        frame[column] = float("nan")
    frame["paired_delta_includes_zero"] = pd.Series(
        [pd.NA] * len(frame), dtype="boolean"
    )
    if (
        helped is None
        or clusters is None
        or reference not in certified_by_spec
        or not helped.any()
    ):
        return
    unique = np.unique(clusters)
    index_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    # Draw once and reuse the same clusters for every candidate: the comparison is paired.
    rng = np.random.default_rng(seed)
    resamples = [
        np.concatenate([index_by_cluster[cluster] for cluster in rng.choice(unique, size=len(unique), replace=True)])
        for _ in range(draws)
    ]
    baseline = certified_by_spec[reference]
    for position, name in enumerate(frame["specification"]):
        if name not in certified_by_spec:
            continue
        if name == reference:
            frame.loc[position, "paired_delta_certified_helpful"] = 0.0
            frame.loc[position, "paired_delta_ci_low"] = 0.0
            frame.loc[position, "paired_delta_ci_high"] = 0.0
            frame.loc[position, "paired_delta_includes_zero"] = True
            continue
        candidate = certified_by_spec[name]
        deltas = np.asarray(
            [
                candidate[index][helped[index]].mean() - baseline[index][helped[index]].mean()
                for index in resamples
            ]
        )
        point = float(
            candidate[helped].mean() - baseline[helped].mean()
        )
        low, high = (float(value) for value in np.percentile(deltas, [2.5, 97.5]))
        frame.loc[position, "paired_delta_certified_helpful"] = point
        frame.loc[position, "paired_delta_ci_low"] = low
        frame.loc[position, "paired_delta_ci_high"] = high
        frame.loc[position, "paired_delta_includes_zero"] = bool(low <= 0.0 <= high)


def derive_thresholds(config, split: str = "training") -> Tuple[ThresholdDerivation, pd.DataFrame]:
    """Derive the declared thresholds from the no-action control on one split.

    Returns:
        The :class:`ThresholdDerivation` and the incident-level evidence behind it.
    """
    incidents = collect_control_incidents(config, split)
    traces = int(incidents.attrs.get("traces", 0))
    # Two populations are reported side by side: every declared event (the detector's
    # behaviour) and the windows the verifier can actually judge (the rule's calibration).
    all_closure = (
        incidents["closure_fraction"].dropna() if not incidents.empty else pd.Series(dtype=float)
    )
    judged = judgeable(incidents)
    judged_closure = judged["closure_fraction"].astype(float)
    closure = judged_closure
    excluded_truncated = (
        int(incidents["window_truncated"].astype(bool).sum())
        if "window_truncated" in incidents.columns
        else 0
    )
    excluded_thin = int(len(all_closure) - len(judged_closure) - excluded_truncated)
    null_p50 = float(np.percentile(closure, 50)) if len(closure) else float("nan")
    null_p90 = float(np.percentile(closure, 90)) if len(closure) else float("nan")
    null_p95 = float(np.percentile(closure, 95)) if len(closure) else float("nan")
    derived = round(null_p95, 1) if len(closure) else float(config.healing.accept_closure_factor)
    declared = float(config.healing.accept_closure_factor)
    false_accept = float((closure >= declared).mean()) if len(closure) else float("nan")

    if len(closure):
        fitted = fit_post_load_baseline(incidents)
        residual = residuals(incidents, fitted)
        spread = float(closure.std(ddof=0))
        r2 = float(1.0 - residual.var() / closure.var()) if closure.var() > 0 else float("nan")
        residual_sd = float(residual.std(ddof=0))
        residual_p05 = float(np.percentile(residual, 5))
        residual_p95 = float(np.percentile(residual, 95))
    else:  # pragma: no cover - an empty control is a broken experiment, reported, not hidden
        fitted = tuple(config.healing.post_load_baseline)
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
    declared_baseline = tuple(float(value) for value in config.healing.post_load_baseline)
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
        judged_incidents=int(len(judged)),
        excluded_truncated=excluded_truncated,
        excluded_thin=max(0, excluded_thin),
        all_events_incidents=int(len(all_closure)),
        all_events_null_p95=(
            float(np.percentile(all_closure, 95)) if len(all_closure) else float("nan")
        ),
        all_events_residual_sd=(
            float(residuals(incidents, declared_baseline).std(ddof=0))
            if len(all_closure)
            else float("nan")
        ),
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
                "quantity": f"fitted_post_load_baseline[{name}]",
                "value": round(coefficient, 6),
                "source": f"{derivation.split} (control, least squares)",
            }
        )
    for name, coefficient in zip(BASELINE_FEATURES, derivation.declared_baseline):
        rows.append(
            {
                "quantity": f"declared_post_load_baseline[{name}]",
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
