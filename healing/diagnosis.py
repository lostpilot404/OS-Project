"""Evidence-backed diagnosis of a detected degradation event.

A detection answers *"the recent completed-work statistic is significantly worse than
this trace's own stable reference"*. It does not say why. The diagnoser turns the visible
state at the trigger epoch into one primary category plus the matched categories and the
raw numbers behind every match, using only arrived work and already-completed processes.

Categories (precedence is fixed and declared, highest first):

``head_of_line_blocking``
    The ready-queue head requires at least ``head_block_factor`` times the median
    remaining burst of the ready set, with at least ``head_block_min_ready`` jobs ready:
    the classic non-preemptive pathology in which one long job blocks the short work
    behind it, and the direct observable cause of waiting-time inflation.
``fairness_starvation``
    Some ready process has been waiting at least ``starvation_age_factor`` times the
    stable reference waiting time (age includes earlier service, so this is a lower bound
    on its accumulated wait).
``tail_dispersion``
    The worst wait in the window is at least ``tail_dispersion_factor`` times the window
    median: a few jobs carry the whole degradation.
``priority_ordering_anomaly``
    A process with worse priority is running while better-priority work is ready. This is
    a policy/ordering mismatch rather than a waiting-time-inflation mechanism in itself,
    so it ranks below the mechanisms above; it is *not* hardware priority inversion,
    which cannot occur in a simulator without shared resources or blocking.
``queue_congestion``
    At least ``congestion_ready_count`` processes are ready at once.
``unexplained``
    The statistic degraded but the visible composition does not match any of the above;
    reported as such instead of being forced into a category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from errors import ValidationError
from healing.config import HealingConfig
from healing.monitor import MonitorSnapshot
from scheduler.runtime import RuntimeObservation

__all__ = [
    "DIAGNOSIS_CATEGORIES",
    "ReadyEvidence",
    "Diagnosis",
    "build_ready_evidence",
    "diagnose",
]

DIAGNOSIS_CATEGORIES: Tuple[str, ...] = (
    "head_of_line_blocking",
    "fairness_starvation",
    "tail_dispersion",
    "priority_ordering_anomaly",
    "queue_congestion",
    "unexplained",
)
_CATEGORY_INDEX = {name: index for index, name in enumerate(DIAGNOSIS_CATEGORIES)}


@dataclass(frozen=True)
class ReadyEvidence:
    """Observable composition of the ready set at the trigger epoch."""

    ready_count: int
    head_remaining: int
    median_remaining: float
    max_remaining: int
    min_remaining: int
    max_arrival_age: int
    mean_arrival_age: float
    priority_spread: int
    running_priority: Optional[int]
    best_ready_priority: int
    running_is_worse_than_best_ready: bool
    visible_remaining_work: int

    def as_dict(self) -> dict:
        """Flat dictionary used as diagnosis evidence."""
        return {
            "ready_count": self.ready_count,
            "head_remaining": self.head_remaining,
            "median_remaining": self.median_remaining,
            "max_remaining": self.max_remaining,
            "min_remaining": self.min_remaining,
            "max_arrival_age": self.max_arrival_age,
            "mean_arrival_age": self.mean_arrival_age,
            "priority_spread": self.priority_spread,
            "visible_remaining_work": self.visible_remaining_work,
        }


@dataclass(frozen=True)
class Diagnosis:
    """Primary category, matched categories and the numbers behind them."""

    category: str
    category_index: int
    matched: Tuple[str, ...]
    evidence: Tuple[Tuple[str, float], ...]
    statement: str

    def as_dict(self) -> dict:
        """Flat dictionary used by the incident ledger and arguments."""
        return {
            "diagnosis": self.category,
            "diagnosis_index": self.category_index,
            "diagnosis_matched": ",".join(self.matched),
            "diagnosis_statement": self.statement,
            **dict(self.evidence),
        }


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count % 2:
        return float(ordered[count // 2])
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def build_ready_evidence(
    observation: RuntimeObservation,
    *,
    running_priority: Optional[int],
    lower_priority_number_is_higher_priority: bool,
) -> ReadyEvidence:
    """Summarize the observable ready set of one decision epoch.

    Args:
        observation: The causal observation handed to the controller.
        running_priority: Priority of the process dispatched at the previous epoch, or
            ``None`` before the first dispatch. Used for inversion evidence only.
        lower_priority_number_is_higher_priority: Priority convention of the run.

    Raises:
        ValidationError: If the observation has no ready process.
    """
    if not isinstance(observation, RuntimeObservation):
        raise ValidationError("build_ready_evidence requires a RuntimeObservation")
    ready = observation.ready_processes
    if not ready:
        raise ValidationError("a decision epoch requires a ready process")
    remaining = [process.remaining_burst for process in ready]
    priorities = [process.priority for process in ready]
    ages = [observation.current_time - process.arrival_time for process in ready]
    if lower_priority_number_is_higher_priority:
        best_priority = min(priorities)
    else:
        best_priority = max(priorities)
    if running_priority is None:
        running_is_worse = False
    elif lower_priority_number_is_higher_priority:
        running_is_worse = running_priority > best_priority
    else:
        running_is_worse = running_priority < best_priority
    return ReadyEvidence(
        ready_count=len(ready),
        head_remaining=ready[0].remaining_burst,
        median_remaining=_median(remaining),
        max_remaining=max(remaining),
        min_remaining=min(remaining),
        max_arrival_age=max(ages),
        mean_arrival_age=sum(ages) / len(ages),
        priority_spread=max(priorities) - min(priorities),
        running_priority=running_priority,
        best_ready_priority=best_priority,
        running_is_worse_than_best_ready=running_is_worse,
        visible_remaining_work=sum(remaining),
    )


def diagnose(
    evidence: ReadyEvidence,
    snapshot: MonitorSnapshot,
    config: HealingConfig,
) -> Diagnosis:
    """Classify a detected degradation from observable evidence.

    Args:
        evidence: Ready-set composition at the trigger epoch.
        snapshot: Monitor snapshot at the trigger epoch (reference and window statistics).
        config: Declared diagnosis thresholds.

    Returns:
        The diagnosis with its primary category, all matched categories and the evidence
        behind each comparison.
    """
    if not isinstance(evidence, ReadyEvidence):
        raise ValidationError("diagnose requires ReadyEvidence")
    if not isinstance(snapshot, MonitorSnapshot):
        raise ValidationError("diagnose requires a MonitorSnapshot")
    if not isinstance(config, HealingConfig):
        raise ValidationError(f"config must be a HealingConfig, got {type(config).__name__}")

    matched: list[str] = []
    values: list[tuple[str, float]] = []

    if evidence.running_priority is not None:
        values.append(("running_priority", float(evidence.running_priority)))
        values.append(("best_ready_priority", float(evidence.best_ready_priority)))
        values.append(
            ("priority_ordering_anomaly_flag", float(evidence.running_is_worse_than_best_ready))
        )
        if evidence.priority_spread > 0 and evidence.running_is_worse_than_best_ready:
            matched.append("priority_ordering_anomaly")

    head_ratio = (
        evidence.head_remaining / evidence.median_remaining
        if evidence.median_remaining > 0
        else float("inf")
    )
    values.append(("head_to_median_ratio", head_ratio))
    if (
        evidence.ready_count >= config.head_block_min_ready
        and head_ratio >= config.head_block_factor
    ):
        matched.append("head_of_line_blocking")

    age_ratio = evidence.max_arrival_age / max(1.0, snapshot.reference_wait)
    values.append(("max_age_to_reference_ratio", age_ratio))
    if age_ratio >= config.starvation_age_factor:
        matched.append("fairness_starvation")

    tail_ratio = snapshot.window_max_wait / max(1.0, snapshot.window_median_wait)
    values.append(("window_tail_ratio", tail_ratio))
    if tail_ratio >= config.tail_dispersion_factor:
        matched.append("tail_dispersion")

    values.append(("ready_count_vs_congestion", float(evidence.ready_count)))
    if evidence.ready_count >= config.congestion_ready_count:
        matched.append("queue_congestion")

    primary = next(
        (category for category in DIAGNOSIS_CATEGORIES if category in matched),
        "unexplained",
    )
    values.append(("window_mean_wait", snapshot.window_mean_wait))
    values.append(("reference_wait", snapshot.reference_wait))
    values.append(("band", snapshot.band))
    statement = _statement(primary, matched, evidence, snapshot)
    return Diagnosis(
        category=primary,
        category_index=_CATEGORY_INDEX[primary],
        matched=tuple(matched),
        evidence=tuple(values),
        statement=statement,
    )


def _statement(
    primary: str,
    matched: list[str],
    evidence: ReadyEvidence,
    snapshot: MonitorSnapshot,
) -> str:
    """Build a human-readable, evidence-backed explanation of the diagnosis."""
    base = (
        f"recent completed-work wait {snapshot.window_mean_wait:.1f} vs reference "
        f"{snapshot.reference_wait:.1f} (band {snapshot.band:.1f}, window "
        f"{snapshot.window_size} completions)"
    )
    detail = {
        "priority_ordering_anomaly": (
            f"running priority {evidence.running_priority} while best ready priority is "
            f"{evidence.best_ready_priority}"
        ),
        "head_of_line_blocking": (
            f"ready head needs {evidence.head_remaining} vs median "
            f"{evidence.median_remaining:.1f} with {evidence.ready_count} ready"
        ),
        "fairness_starvation": (
            f"oldest ready job has waited at least {evidence.max_arrival_age} time units"
        ),
        "tail_dispersion": (
            f"worst window wait {snapshot.window_max_wait:.1f} vs median "
            f"{snapshot.window_median_wait:.1f}"
        ),
        "queue_congestion": f"{evidence.ready_count} processes ready at once",
        "unexplained": (
            "no visible composition evidence matched; the excess is unexplained by the "
            "observable state"
        ),
    }[primary]
    others = f"; also matched: {', '.join(matched)}" if len(matched) > 1 else ""
    return f"{primary}: {detail} ({base}{others})"
