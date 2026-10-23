"""Reliability metrics beyond mean waiting time.

The original project reported only mean waiting time.  Review 2 recommended
measuring reliability: recovery delay, false interventions, regression rate,
rollback rate, P95 waiting time, starvation, fairness and context-switch
overhead.  This module computes those from the same artifacts the
healing study already produces, so no new runs are needed for existing
results — but the metrics become available for any new experiment that
adds workload drift or burst-prediction uncertainty.

All metrics are defined on the finished schedule (``ScheduleResult``) and
on the incident ledger (``IncidentRecord``); they are deterministic and
have no hidden parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from workload.models import ScheduleResult

__all__ = [
    "ReliabilityMetrics",
    "compute_reliability_metrics",
    "reliability_summary_for_arms",
    "jains_fairness_index",
]


def jains_fairness_index(waiting_times: Sequence[float]) -> float:
    """Jain's fairness index for waiting times (1 = perfectly fair, 1/n = worst)."""
    if not waiting_times:
        return 1.0
    n = len(waiting_times)
    s = sum(waiting_times)
    s2 = sum(w * w for w in waiting_times)
    if s2 == 0:
        return 1.0
    return (s * s) / (n * s2)


def _p95(values: Sequence[float]) -> float:
    """Nearest-rank P95: ceil(0.95*n)-th ordered value (1-indexed).

    Note: with n=15 processes per trace, ceil(0.95*15)=15 → index 14 → the
    maximum. Equality of per-trace P95 and max at n=15 is therefore a
    definition artifact, not evidence of light-tailed distribution. For an
    interpretable tail metric over an arm, use the pooled P95 over all
    process waits (see experiments/drift_uncertainty_experiment.py).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[idx])


def _starvation_count(waiting_times: Sequence[float], threshold_factor: float = 3.0) -> int:
    """Processes whose waiting time exceeds threshold_factor × median."""
    if not waiting_times:
        return 0
    ordered = sorted(waiting_times)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    if median == 0:
        return 0
    return sum(1 for w in waiting_times if w > threshold_factor * median)


@dataclass(frozen=True)
class ReliabilityMetrics:
    """Reliability and tail metrics for one (arm, workload) evaluation.

    Attributes mirror the Recommendation §3.4 and §4:
        mean_waiting, p95_waiting, max_waiting — latency summary.
        starved_processes — count of starved jobs (> 3× median wait).
        fairness_index — Jain's index over waiting times (higher = fairer).
        throughput, cpu_utilization — efficiency (from base metrics).
        context_switches, switching_overhead — cost of adaptation.
        recovery_delay_decisions — decisions between detector trigger and
            verification close (where measurable).
        recovery_delay_time — simulated time between trigger and verification.
        incidents, interventions, accepted, rejected, inconclusive — ledger counts.
        false_intervention_rate — share of interventions with matched_gain ≤ 0
            (requires audit; None when audit not available).
        regression_rate — share of traces where arm is worse than fixed FCFS.
        rollback_rate — share of interventions rolled back (rejected + inconclusive).
        fallback_triggered — whether safe fallback fired on this trace.
    """

    mean_waiting: float
    p95_waiting: float
    max_waiting: float
    starved_processes: int
    fairness_index: float
    throughput: float
    cpu_utilization: float
    context_switches: int
    switching_overhead: int
    recovery_delay_decisions: Optional[float]
    recovery_delay_time: Optional[float]
    incidents: int
    interventions: int
    accepted: int
    rejected: int
    inconclusive: int
    false_intervention_rate: Optional[float]
    regression_rate: Optional[float]
    rollback_rate: Optional[float]
    fallback_triggered: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean_waiting": self.mean_waiting,
            "p95_waiting": self.p95_waiting,
            "max_waiting": self.max_waiting,
            "starved_processes": self.starved_processes,
            "fairness_index": self.fairness_index,
            "throughput": self.throughput,
            "cpu_utilization": self.cpu_utilization,
            "context_switches": self.context_switches,
            "switching_overhead": self.switching_overhead,
            "recovery_delay_decisions": self.recovery_delay_decisions,
            "recovery_delay_time": self.recovery_delay_time,
            "incidents": self.incidents,
            "interventions": self.interventions,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "inconclusive": self.inconclusive,
            "false_intervention_rate": self.false_intervention_rate,
            "regression_rate": self.regression_rate,
            "rollback_rate": self.rollback_rate,
            "fallback_triggered": self.fallback_triggered,
        }


def compute_reliability_metrics(
    result: ScheduleResult,
    *,
    incidents: Sequence[Any] = (),
    audit_gains: Sequence[float] = (),
) -> ReliabilityMetrics:
    """Compute reliability metrics for one trace.

    Args:
        result: Finished schedule.
        incidents: IncidentRecords for this trace (ledger slice).
        audit_gains: Per-intervention counterfactual gains (matched_gain)
            for this trace; empty when audit not available.

    Returns:
        ReliabilityMetrics for this trace.
    """
    waiting_times = [float(o.waiting_time) for o in result.outcomes]
    n = len(waiting_times)
    mean_wait = sum(waiting_times) / n if n else 0.0
    p95 = _p95(waiting_times)
    max_wait = max(waiting_times) if waiting_times else 0.0
    starved = _starvation_count(waiting_times)
    fairness = jains_fairness_index(waiting_times)

    # Schedule-level efficiency (same denominators as evaluation.metrics).
    total_time = result.total_elapsed_time
    busy = result.cpu_busy_time
    throughput = (n / total_time) if total_time else 0.0
    cpu_util = (100.0 * busy / total_time) if total_time else 0.0

    # Incident-derived counts.
    inc_list = list(incidents)
    incident_count = len(inc_list)
    intervention_count = sum(1 for r in inc_list if getattr(r, "action_changed", False))
    accepted = sum(1 for r in inc_list if getattr(r, "verdict", None) == "accepted")
    rejected = sum(1 for r in inc_list if getattr(r, "verdict", None) == "rejected")
    inconclusive = sum(1 for r in inc_list if getattr(r, "verdict", None) == "inconclusive")
    fallback = any(getattr(r, "fallback_triggered", False) for r in inc_list)

    # Recovery delay: average verification window length where available.
    delays_d = [getattr(r, "verification_decisions", None) for r in inc_list if getattr(r, "verification_decisions", None) is not None]
    # Simulated time delay would need window_end_time - simulated_time;
    # fall back to None when not recorded.
    time_delays = []
    for r in inc_list:
        start = getattr(r, "simulated_time", None)
        end = getattr(r, "window_end_time", None)
        if start is not None and end is not None:
            time_delays.append(float(end - start))

    avg_delay_d = (sum(delays_d) / len(delays_d)) if delays_d else None
    avg_delay_t = (sum(time_delays) / len(time_delays)) if time_delays else None

    # False intervention rate from audit (ground truth).
    if audit_gains:
        false = sum(1 for g in audit_gains if g <= 0)
        false_rate = false / len(audit_gains)
    else:
        false_rate = None

    # Rollback rate = (rejected + inconclusive) / interventions
    if intervention_count:
        rollback_rate = (rejected + inconclusive) / intervention_count
    else:
        rollback_rate = None

    # regression_rate is per-arm across traces; caller aggregates.
    return ReliabilityMetrics(
        mean_waiting=mean_wait,
        p95_waiting=p95,
        max_waiting=max_wait,
        starved_processes=starved,
        fairness_index=fairness,
        throughput=throughput,
        cpu_utilization=cpu_util,
        context_switches=result.context_switches,
        switching_overhead=result.switching_overhead,
        recovery_delay_decisions=avg_delay_d,
        recovery_delay_time=avg_delay_t,
        incidents=incident_count,
        interventions=intervention_count,
        accepted=accepted,
        rejected=rejected,
        inconclusive=inconclusive,
        false_intervention_rate=false_rate,
        regression_rate=None,  # filled by per-arm aggregation
        rollback_rate=rollback_rate,
        fallback_triggered=fallback,
    )


def reliability_summary_for_arms(
    per_trace: Dict[str, List[ReliabilityMetrics]],
    *,
    baseline_arm: str = "fixed_fcfs",
) -> List[Dict[str, Any]]:
    """Aggregate per-trace reliability into a per-arm summary.

    Args:
        per_trace: Mapping arm_name -> list of trace metrics.
        baseline_arm: Arm to compare against for regression_rate.

    Returns:
        One row per arm with mean metrics and regression rate.
    """
    baseline_means = None
    if baseline_arm in per_trace:
        baseline_means = [m.mean_waiting for m in per_trace[baseline_arm]]

    rows: List[Dict[str, Any]] = []
    for arm, traces in per_trace.items():
        if not traces:
            continue
        n = len(traces)
        mean_vals = {
            "mean_waiting": sum(t.mean_waiting for t in traces) / n,
            "p95_waiting": sum(t.p95_waiting for t in traces) / n,
            "max_waiting": sum(t.max_waiting for t in traces) / n,
            "starved_processes": sum(t.starved_processes for t in traces) / n,
            "fairness_index": sum(t.fairness_index for t in traces) / n,
            "throughput": sum(t.throughput for t in traces) / n,
            "cpu_utilization": sum(t.cpu_utilization for t in traces) / n,
            "context_switches": sum(t.context_switches for t in traces) / n,
            "switching_overhead": sum(t.switching_overhead for t in traces) / n,
            "incidents": sum(t.incidents for t in traces) / n,
            "interventions": sum(t.interventions for t in traces) / n,
            "accepted": sum(t.accepted for t in traces) / n,
            "rejected": sum(t.rejected for t in traces) / n,
            "inconclusive": sum(t.inconclusive for t in traces) / n,
            "fallback_rate": sum(1 for t in traces if t.fallback_triggered) / n,
        }
        # Recovery delay averaged over traces that had incidents.
        delays = [t.recovery_delay_decisions for t in traces if t.recovery_delay_decisions is not None]
        mean_vals["mean_recovery_delay_decisions"] = (sum(delays) / len(delays)) if delays else None
        delays_t = [t.recovery_delay_time for t in traces if t.recovery_delay_time is not None]
        mean_vals["mean_recovery_delay_time"] = (sum(delays_t) / len(delays_t)) if delays_t else None
        # False intervention rate averaged where audit exists.
        false_rates = [t.false_intervention_rate for t in traces if t.false_intervention_rate is not None]
        mean_vals["false_intervention_rate"] = (sum(false_rates) / len(false_rates)) if false_rates else None
        rollback_rates = [t.rollback_rate for t in traces if t.rollback_rate is not None]
        mean_vals["rollback_rate"] = (sum(rollback_rates) / len(rollback_rates)) if rollback_rates else None

        # Regression rate: share of traces where this arm's mean_waiting > baseline trace at same index.
        if arm != baseline_arm and baseline_means is not None and len(baseline_means) == n:
            worse = sum(1 for a, b in zip((t.mean_waiting for t in traces), baseline_means) if a > b)
            mean_vals["regression_rate"] = worse / n
        else:
            mean_vals["regression_rate"] = None

        rows.append({"arm": arm, **mean_vals, "traces": n})
    return rows
