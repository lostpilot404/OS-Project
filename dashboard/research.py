"""Committed research findings, read from the generated study artifacts.

The teacher dashboard separates two kinds of evidence:

* the two demo fixtures (``demo.json``), which are illustrative individual scenarios;
* the frozen held-out study (``results/healing/healing_summary.json`` and
  ``results/healing/healing_comparisons.csv``), which is the research result.

Every number in this module comes from those files. The sentences that summarise them are
built from the values (which arm is lowest, whether a paired interval excludes zero), so a
re-run with different numbers cannot leave a stale claim on the page.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from dashboard.model import DashboardError, _int, _number, _text

__all__ = [
    "ResearchArm",
    "PairedComparison",
    "ResearchSummary",
    "load_research",
]

#: Arms the page compares explicitly, with the wording used on the page.
PRIMARY_ARMS: Tuple[str, ...] = ("fixed_fcfs", "fixed_sjf", "rule_wait", "q_recovery")

#: Additional study arms shown for context (only if the artifact contains them).
CONTEXT_ARMS: Tuple[str, ...] = (
    "fixed_round_robin",
    "fixed_priority",
    "detector_only",
    "rule_fairness",
    "rule_wait_from_sjf",
    "q_recovery_from_sjf",
)

LABELS: Mapping[str, str] = {
    "fixed_fcfs": "FCFS",
    "fixed_sjf": "SJF",
    "fixed_round_robin": "Round Robin",
    "fixed_priority": "Priority",
    "detector_only": "Detector only (no intervention)",
    "rule_fairness": "Fairness rule",
    "rule_wait": "Waiting-time rule",
    "q_recovery": "Q-learning recovery",
    "rule_wait_from_sjf": "Waiting-time rule, from SJF",
    "q_recovery_from_sjf": "Q-learning recovery, from SJF",
}

#: Grouping used for the legend and the bar colours.
KINDS: Mapping[str, str] = {
    "fixed_fcfs": "fixed",
    "fixed_sjf": "fixed",
    "fixed_round_robin": "fixed",
    "fixed_priority": "fixed",
    "detector_only": "control",
    "rule_fairness": "rule",
    "rule_wait": "rule",
    "rule_wait_from_sjf": "rule",
    "q_recovery": "learned",
    "q_recovery_from_sjf": "learned",
}

_SHORT_LABELS: Mapping[str, str] = {
    "fixed_fcfs": "FCFS",
    "fixed_sjf": "SJF",
    "fixed_round_robin": "Round Robin",
    "fixed_priority": "Priority",
    "detector_only": "Detector only",
    "rule_fairness": "Fairness rule",
    "rule_wait": "Waiting-time rule",
    "q_recovery": "Q-learning recovery",
    "rule_wait_from_sjf": "Waiting rule (from SJF)",
    "q_recovery_from_sjf": "Q recovery (from SJF)",
}


def _fail(where: str, message: str) -> DashboardError:
    return DashboardError(f"{where}: {message}")


@dataclass(frozen=True)
class ResearchArm:
    """One arm of the held-out study, as recorded in ``healing_summary.json``."""

    arm: str
    mean_waiting_time: float
    p95_waiting_time: Optional[float] = None
    mean_response_time: Optional[float] = None
    context_switches: Optional[float] = None
    incidents: Optional[float] = None
    interventions: Optional[float] = None

    @property
    def label(self) -> str:
        return LABELS.get(self.arm, self.arm)

    @property
    def short_label(self) -> str:
        return _SHORT_LABELS.get(self.arm, self.arm)

    @property
    def kind(self) -> str:
        return KINDS.get(self.arm, "other")

    @property
    def is_primary(self) -> bool:
        return self.arm in PRIMARY_ARMS


@dataclass(frozen=True)
class PairedComparison:
    """One predeclared paired comparison from ``healing_comparisons.csv``."""

    target: str
    reference: str
    metric: str
    difference: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def label(self) -> str:
        return (
            f"{LABELS.get(self.target, self.target)} vs {LABELS.get(self.reference, self.reference)}"
        )


@dataclass(frozen=True)
class ResearchSummary:
    """Typed view of the frozen study summary plus its paired comparisons."""

    source_name: str
    title: str
    split: str
    workloads: Optional[int]
    final_test_seed: Optional[int]
    training_seeds: Tuple[int, ...]
    arms: Tuple[ResearchArm, ...]
    comparisons: Tuple[PairedComparison, ...]
    comparisons_source: Optional[str]

    # -- lookups -----------------------------------------------------------------
    def arm(self, name: str) -> ResearchArm:
        for arm in self.arms:
            if arm.arm == name:
                return arm
        raise _fail(self.source_name, f"the study summary has no arm {name!r}")

    def has_arm(self, name: str) -> bool:
        return any(arm.arm == name for arm in self.arms)

    def comparison(
        self, target: str, reference: str, metric: str = "avg_waiting_time"
    ) -> Optional[PairedComparison]:
        for row in self.comparisons:
            if row.target == target and row.reference == reference and row.metric == metric:
                return row
        return None

    @property
    def primary_arms(self) -> Tuple[ResearchArm, ...]:
        return tuple(self.arm(name) for name in PRIMARY_ARMS if self.has_arm(name))

    @property
    def context_arms(self) -> Tuple[ResearchArm, ...]:
        return tuple(self.arm(name) for name in CONTEXT_ARMS if self.has_arm(name))

    @property
    def chart_arms(self) -> Tuple[ResearchArm, ...]:
        """The arms drawn on the research chart, in a stable reading order."""
        fixed = [
            arm
            for name in ("fixed_fcfs", "fixed_sjf", "fixed_round_robin", "fixed_priority")
            if self.has_arm(name)
            for arm in (self.arm(name),)
        ]
        extra = [
            arm
            for name in ("detector_only", "rule_fairness", "rule_wait", "rule_wait_from_sjf", "q_recovery", "q_recovery_from_sjf")
            if self.has_arm(name)
            for arm in (self.arm(name),)
        ]
        return tuple(fixed + extra)

    # -- derived findings ---------------------------------------------------------
    @property
    def best_primary(self) -> ResearchArm:
        return min(self.primary_arms, key=lambda arm: arm.mean_waiting_time)

    def delta(self, target: str, reference: str) -> float:
        return self.arm(target).mean_waiting_time - self.arm(reference).mean_waiting_time

    @property
    def q_vs_rule(self) -> Optional[PairedComparison]:
        return self.comparison("q_recovery", "rule_wait")

    @property
    def rule_vs_fcfs(self) -> Optional[PairedComparison]:
        return self.comparison("rule_wait", "fixed_fcfs")

    @property
    def q_vs_fcfs(self) -> Optional[PairedComparison]:
        return self.comparison("q_recovery", "fixed_fcfs")

    @property
    def sjf_vs_rule(self) -> Optional[PairedComparison]:
        return self.comparison("rule_wait", "fixed_sjf")

    def finding_lines(self) -> Tuple[str, ...]:
        """Return the three headline findings, generated from the recorded numbers."""
        best = self.best_primary
        rule = self.arm("rule_wait")
        fcfs = self.arm("fixed_fcfs")
        learned = self.arm("q_recovery")
        lines = [
            (
                f"{best.label} has the lowest mean waiting time of the four compared "
                f"policies ({best.mean_waiting_time:.2f} simulated time units per process)."
            ),
            (
                f"Both recovery approaches improve on FCFS: the waiting-time rule "
                f"{rule.mean_waiting_time - fcfs.mean_waiting_time:+.2f} and Q-learning "
                f"recovery {learned.mean_waiting_time - fcfs.mean_waiting_time:+.2f} "
                f"simulated time units against {fcfs.mean_waiting_time:.2f} for FCFS."
            ),
        ]
        comparison = self.q_vs_rule
        if comparison is None:
            lines.append(
                "No paired Q-versus-rule interval is present in this artifact, so the "
                f"observed difference ({learned.mean_waiting_time - rule.mean_waiting_time:+.2f}) "
                "is not evidence that either policy is better."
            )
        elif comparison.excludes_zero:
            direction = "lower" if comparison.difference < 0 else "higher"
            lines.append(
                f"The paired Q-versus-rule difference is {comparison.difference:+.2f} with a "
                f"95% interval [{comparison.ci_low:+.2f}, {comparison.ci_high:+.2f}] "
                f"(Q-learning {direction}), so this artifact does support a difference."
            )
        else:
            lines.append(
                f"Q-learning recovery is not measurably better than the waiting-time rule: "
                f"the paired difference is {comparison.difference:+.2f} with a 95% interval "
                f"[{comparison.ci_low:+.2f}, {comparison.ci_high:+.2f}], which includes zero."
            )
        return tuple(lines)


def _arm_from_row(raw: Mapping[str, Any], where: str) -> ResearchArm:
    if "arm" not in raw:
        raise _fail(where, "missing required field 'arm'")
    arm = _text(raw["arm"], f"{where}.arm")
    if "avg_waiting_time" not in raw:
        raise _fail(where, "missing required field 'avg_waiting_time'")
    def optional(name: str) -> Optional[float]:
        value = raw.get(name)
        return None if value is None else _number(value, f"{where}.{name}")
    return ResearchArm(
        arm=arm,
        mean_waiting_time=_number(raw["avg_waiting_time"], f"{where}.avg_waiting_time"),
        p95_waiting_time=optional("p95_waiting_time"),
        mean_response_time=optional("avg_response_time"),
        context_switches=optional("context_switches"),
        incidents=optional("incidents"),
        interventions=optional("interventions"),
    )


def _csv_number(raw: Optional[str], where: str) -> float:
    if raw is None or raw.strip() == "":
        raise _fail(where, "expected a number, got an empty cell")
    try:
        value = float(raw)
    except ValueError as exc:
        raise _fail(where, f"expected a number, got {raw!r}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise _fail(where, f"expected a finite number, got {raw!r}")
    return value


def _load_comparisons(path: Path) -> Tuple[PairedComparison, ...]:
    rows: List[PairedComparison] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ("target", "reference", "metric", "mean_difference", "ci95_low", "ci95_high")
        header = tuple(reader.fieldnames or ())
        missing = [name for name in required if name not in header]
        if missing:
            raise _fail(f"{path.name}", f"missing column(s): {', '.join(missing)}")
        for index, raw in enumerate(reader):
            where = f"{path.name}: row {index + 2}"
            rows.append(
                PairedComparison(
                    target=_text(raw["target"], f"{where}.target"),
                    reference=_text(raw["reference"], f"{where}.reference"),
                    metric=_text(raw["metric"], f"{where}.metric"),
                    difference=_csv_number(raw["mean_difference"], f"{where}.mean_difference"),
                    ci_low=_csv_number(raw["ci95_low"], f"{where}.ci95_low"),
                    ci_high=_csv_number(raw["ci95_high"], f"{where}.ci95_high"),
                )
            )
    return tuple(rows)


def load_research(path: Path) -> ResearchSummary:
    """Load the committed research summary and its optional paired-comparison table.

    Args:
        path: Path to ``healing_summary.json``.

    Returns:
        The typed research summary.

    Raises:
        DashboardError: If the file is missing, unreadable, or lacks the four arms the
            page compares (FCFS, SJF, waiting-time rule, Q-learning recovery).
    """
    path = Path(path)
    if not path.exists():
        raise DashboardError(f"research summary not found: {path}")
    if not path.is_file():
        raise DashboardError(f"research summary is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(path.name, f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        raise DashboardError(f"cannot read research summary {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _fail(path.name, "expected the study summary to be a JSON object")
    if "means" not in payload:
        raise _fail(path.name, "missing required field 'means' (per-arm mean metrics)")
    raw_means = payload["means"]
    if not isinstance(raw_means, Sequence) or isinstance(raw_means, (str, bytes)) or not raw_means:
        raise _fail(path.name, "'means' must be a non-empty list")
    arms = tuple(
        _arm_from_row(raw, f"{path.name}: means[{index}]") for index, raw in enumerate(raw_means)
    )
    names = [arm.arm for arm in arms]
    if len(set(names)) != len(names):
        raise _fail(path.name, f"duplicate arm names in 'means': {names}")
    missing_arms = [name for name in PRIMARY_ARMS if name not in names]
    if missing_arms:
        raise _fail(
            path.name,
            "the study summary is missing the arm(s) this panel compares: "
            + ", ".join(missing_arms),
        )

    comparisons_path = path.with_name("healing_comparisons.csv")
    comparisons: Tuple[PairedComparison, ...] = ()
    comparisons_source: Optional[str] = None
    if comparisons_path.is_file():
        comparisons = _load_comparisons(comparisons_path)
        comparisons_source = comparisons_path.name

    split = payload.get("split")
    workloads = payload.get("workloads")
    final_test_seed = payload.get("final_test_seed")
    training_seeds_raw = payload.get("training_seeds") or ()
    training_seeds: Tuple[int, ...] = ()
    if training_seeds_raw:
        if not isinstance(training_seeds_raw, Sequence) or isinstance(training_seeds_raw, (str, bytes)):
            raise _fail(path.name, "'training_seeds' must be a list")
        training_seeds = tuple(
            _int(value, f"{path.name}: training_seeds[{index}]")
            for index, value in enumerate(training_seeds_raw)
        )

    return ResearchSummary(
        source_name=path.name,
        title=str(payload.get("project_title") or "Self-healing CPU scheduling study"),
        split=str(split) if split is not None else "unspecified",
        workloads=_int(workloads, f"{path.name}: workloads") if workloads is not None else None,
        final_test_seed=(
            _int(final_test_seed, f"{path.name}: final_test_seed")
            if final_test_seed is not None
            else None
        ),
        training_seeds=training_seeds,
        arms=arms,
        comparisons=comparisons,
        comparisons_source=comparisons_source,
    )
