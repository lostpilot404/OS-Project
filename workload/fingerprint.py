"""Interpretable workload fingerprinting for the offline policy selector.

Information boundary
--------------------
Every feature in this module is a pure function of the **declared process table** of a
batch workload: ``(pid, arrival_time, burst_time, priority)`` per process.  This is the
complete information the offline selector is allowed to use (see
``docs/METHODOLOGY.md``, "Information boundary").  The module deliberately imports
nothing from the schedulers, the metrics, the RL stack or the experiment layer, so no
schedule outcome, metric, counterfactual result or reward can reach a feature.  Tests
enforce both the import boundary and behavioural purity.

The fingerprint is used in two places:

* the selector's discretised state (a declared subset of these features; see
  :mod:`rl.fingerprint_state`), and
* decision explanations, where every feature value is reported with its bin.

Feature definitions
-------------------
===============================  ==========================================================
Feature                          Definition
===============================  ==========================================================
``process_count``                Number of processes.
``arrival_span``                 ``max(arrival) - min(arrival)`` (0 for fewer than 2).
``arrival_rate``                 ``process_count / max(1, arrival_span)``.
``arrival_gap_cv``               Population std / mean of the successive gaps between
                                 sorted arrival times (0 when the mean gap is 0).
``burst_mean``                   Mean burst time.
``burst_cv``                     Population std / mean of burst times (0 when the mean
                                 burst is 0).
``burst_skew``                   Bowley (quantile) skew of bursts:
                                 ``(q90 + q10 - 2*q50) / (q90 - q10)`` with nearest-rank
                                 quantiles; 0 when ``q90 == q10``.
``priority_std``                 Population standard deviation of priorities.
``priority_unique_fraction``     Distinct priority values / process count.
``priority_burst_correlation``   Pearson correlation of (priority, burst); 0 when
                                 degenerate (fewer than 2 processes or zero variance).
===============================  ==========================================================

All features are deterministic and reproducible: the same process table always yields
the same fingerprint, independent of process insertion order (the workload model already
canonicalises order, and the computations are order-invariant anyway).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from errors import ValidationError
from workload.models import Workload

__all__ = [
    "FINGERPRINT_FEATURES",
    "WorkloadFingerprint",
    "fingerprint_workload",
]

#: Every fingerprint feature, in documentation order.
FINGERPRINT_FEATURES: Tuple[str, ...] = (
    "process_count",
    "arrival_span",
    "arrival_rate",
    "arrival_gap_cv",
    "burst_mean",
    "burst_cv",
    "burst_skew",
    "priority_std",
    "priority_unique_fraction",
    "priority_burst_correlation",
)


def _population_std(values: Sequence[float]) -> float:
    """Return the population standard deviation of ``values`` (0 for fewer than 2)."""
    count = len(values)
    if count < 2:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return math.sqrt(variance)


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank quantile of ``values`` at ``fraction`` in [0, 1].

    For a non-empty list of size ``n`` the zero-based index is
    ``max(0, ceil(fraction * n) - 1)`` of the ascending list, matching
    :meth:`workload.models.Workload.percentile_burst_time`.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _bowley_skew(values: Sequence[float]) -> float:
    """Return the Bowley (quantile) skew of ``values`` in ``[-1, 1]`` (0 if degenerate)."""
    q10 = _nearest_rank(values, 0.10)
    q50 = _nearest_rank(values, 0.50)
    q90 = _nearest_rank(values, 0.90)
    spread = q90 - q10
    if spread <= 0.0:
        return 0.0
    return (q90 + q10 - 2.0 * q50) / spread


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the Pearson correlation of ``xs`` and ``ys`` (0 when degenerate)."""
    count = len(xs)
    if count < 2 or count != len(ys):
        return 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(var_x * var_y)
    if denominator <= 0.0:
        return 0.0
    return cov / denominator


@dataclass(frozen=True)
class WorkloadFingerprint:
    """Interpretable characteristics of one workload's declared process table.

    Attributes:
        workload_name: Name of the workload the fingerprint describes.
        digest: The workload model's content digest (SHA-256 of the process table),
            used to prove pairing across policies and splits.
        process_count: Number of processes.
        arrival_span: Latest minus earliest arrival time.
        arrival_rate: Processes per unit of arrival span (see module docstring).
        arrival_gap_cv: Coefficient of variation of successive arrival gaps.
        burst_mean: Mean burst time.
        burst_cv: Coefficient of variation of burst times.
        burst_skew: Bowley skew of burst times.
        priority_std: Population standard deviation of priorities.
        priority_unique_fraction: Distinct priority values divided by process count.
        priority_burst_correlation: Pearson correlation of priority and burst.
    """

    workload_name: str
    digest: str
    process_count: int
    arrival_span: int
    arrival_rate: float
    arrival_gap_cv: float
    burst_mean: float
    burst_cv: float
    burst_skew: float
    priority_std: float
    priority_unique_fraction: float
    priority_burst_correlation: float

    def as_dict(self) -> Dict[str, float]:
        """Return the feature values in :data:`FINGERPRINT_FEATURES` order."""
        return {
            "process_count": float(self.process_count),
            "arrival_span": float(self.arrival_span),
            "arrival_rate": self.arrival_rate,
            "arrival_gap_cv": self.arrival_gap_cv,
            "burst_mean": self.burst_mean,
            "burst_cv": self.burst_cv,
            "burst_skew": self.burst_skew,
            "priority_std": self.priority_std,
            "priority_unique_fraction": self.priority_unique_fraction,
            "priority_burst_correlation": self.priority_burst_correlation,
        }

    def as_row(self) -> Dict[str, object]:
        """Return the fingerprint as a flat table row (features plus identity)."""
        return {
            "workload_name": self.workload_name,
            "workload_fingerprint": self.digest,
            **{name: self.as_dict()[name] for name in FINGERPRINT_FEATURES},
        }


def fingerprint_workload(workload: Workload) -> WorkloadFingerprint:
    """Compute the workload fingerprint from the declared process table only.

    Args:
        workload: The workload to characterise.  Only the process attributes
            ``(arrival_time, burst_time, priority)`` are read; nothing about any
            schedule is consulted.

    Returns:
        The deterministic :class:`WorkloadFingerprint` of the process table.

    Raises:
        ValidationError: If ``workload`` is not a :class:`workload.models.Workload`.
    """
    if not isinstance(workload, Workload):
        raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")

    processes = workload.processes
    count = len(processes)
    if count == 0:
        return WorkloadFingerprint(
            workload_name=workload.name,
            digest=workload.fingerprint,
            process_count=0,
            arrival_span=0,
            arrival_rate=0.0,
            arrival_gap_cv=0.0,
            burst_mean=0.0,
            burst_cv=0.0,
            burst_skew=0.0,
            priority_std=0.0,
            priority_unique_fraction=0.0,
            priority_burst_correlation=0.0,
        )

    arrivals = [float(p.arrival_time) for p in processes]
    bursts = [float(p.burst_time) for p in processes]
    priorities = [float(p.priority) for p in processes]

    arrival_span = int(workload.arrival_span)
    sorted_arrivals = sorted(arrivals)
    gaps = [
        sorted_arrivals[index + 1] - sorted_arrivals[index] for index in range(count - 1)
    ]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    arrival_gap_cv = _population_std(gaps) / mean_gap if mean_gap > 0.0 else 0.0

    burst_mean = sum(bursts) / count
    burst_cv = _population_std(bursts) / burst_mean if burst_mean > 0.0 else 0.0

    return WorkloadFingerprint(
        workload_name=workload.name,
        digest=workload.fingerprint,
        process_count=count,
        arrival_span=arrival_span,
        arrival_rate=count / max(1, arrival_span),
        arrival_gap_cv=arrival_gap_cv,
        burst_mean=burst_mean,
        burst_cv=burst_cv,
        burst_skew=_bowley_skew(bursts),
        priority_std=_population_std(priorities),
        priority_unique_fraction=len(set(priorities)) / count,
        priority_burst_correlation=_pearson_correlation(priorities, bursts),
    )
