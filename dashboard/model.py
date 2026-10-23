"""Typed, validated view of the saved teacher-demo trace.

The dashboard is a *consumer* of simulator output. Nothing in this module re-runs a
scheduler, re-invents a metric definition or fills a missing value with something that
merely looks plausible:

* execution intervals, waiting-time increments, incidents and metrics are read from
  ``demo.json`` (written by :func:`experiments.demo.run_demo` from the real engine);
* every reading is cross-checked against the recorded per-process outcomes, so a
  hand-edited or truncated file is rejected instead of being drawn;
* the per-process waiting / turnaround / response values are recomputed from
  ``completion_time``, ``arrival_time`` and ``burst_time`` with the definitions of
  :class:`workload.models.ProcessOutcome`.

Errors are :class:`DashboardError` (a ``ValueError``) and always name the offending
location, e.g. ``fixtures[0] (seed 3): metrics.avg_waiting_time``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config import ACTION_NAMES
from healing.actions import HEALING_ACTION_NAMES
from healing.verifier import ALL_VERDICTS

__all__ = [
    "DashboardError",
    "Process",
    "Outcome",
    "Interval",
    "PolicySegment",
    "Metrics",
    "Incident",
    "Fixture",
    "DemoDocument",
    "parse_demo",
    "parse_fixture",
    "POLICY_NAMES",
    "VERDICTS",
]

#: Policy display names, in the project's fixed action order.
POLICY_NAMES: Tuple[str, ...] = tuple(ACTION_NAMES)

#: Verdicts a concluded incident may carry (see :mod:`healing.verifier`).
VERDICTS: Tuple[str, ...] = tuple(ALL_VERDICTS)

#: Numeric tolerance used when a recorded aggregate is compared with the per-process rows.
_TOLERANCE = 1e-6


class DashboardError(ValueError):
    """Raised when dashboard input is missing, malformed or internally inconsistent."""


# --------------------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------------------


def _fail(where: str, message: str) -> "DashboardError":
    return DashboardError(f"{where}: {message}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(where, f"expected an object, got {type(value).__name__}")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(where, f"expected a list, got {type(value).__name__}")
    return value


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise _fail(where, f"missing required field {key!r}")
    return mapping[key]


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any, where: str, *, integral: bool = False) -> float:
    if not _is_number(value):
        raise _fail(where, f"expected a finite number, got {value!r}")
    if integral and float(value) != float(int(value)):
        raise _fail(where, f"expected an integer, got {value!r}")
    return float(value)


def _int(value: Any, where: str) -> int:
    return int(_number(value, where, integral=True))


def _optional_number(value: Any, where: str) -> Optional[float]:
    if value is None:
        return None
    return _number(value, where)


def _optional_int(value: Any, where: str) -> Optional[int]:
    if value is None:
        return None
    return _int(value, where)


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(where, f"expected a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise _fail(where, "expected a non-empty string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(where, f"expected true/false, got {value!r}")
    return value


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= _TOLERANCE * max(1.0, abs(float(left)), abs(float(right)))


# --------------------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Process:
    """One workload process as recorded in the demo input."""

    pid: int
    arrival_time: int
    burst_time: int
    priority: int


@dataclass(frozen=True)
class Outcome:
    """Recorded timing of one completed process.

    ``waiting_time``, ``turnaround_time`` and ``response_time`` use the single project
    definition implemented by :class:`workload.models.ProcessOutcome`.
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int
    start_time: int
    completion_time: int

    @property
    def waiting_time(self) -> int:
        return self.completion_time - self.arrival_time - self.burst_time

    @property
    def turnaround_time(self) -> int:
        return self.completion_time - self.arrival_time

    @property
    def response_time(self) -> int:
        return self.start_time - self.arrival_time


@dataclass(frozen=True)
class Interval:
    """One recorded dispatch interval (a real execution segment of the simulator)."""

    index: int
    pid: int
    policy: str
    start: int
    end: int
    switch_before: int
    idle_before: int
    waiting_increment: Optional[int]
    completed_pids: Tuple[int, ...]

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PolicySegment:
    """A maximal span during which one policy was in force for the dispatches."""

    policy: str
    start: int
    end: int
    decisions: int


@dataclass(frozen=True)
class Metrics:
    """Recorded aggregate metrics of one scheduling run.

    ``total_waiting_time``, ``num_processes`` and the three means are required because the
    dashboard's central comparison is built from them. Totals that only feed the detail
    table (busy/idle time, throughput) are optional and shown as ``n/a`` when absent.
    """

    policy_name: str
    workload_name: str
    workload_fingerprint: str
    num_processes: int
    total_waiting_time: int
    avg_waiting_time: float
    avg_turnaround_time: float
    avg_response_time: float
    cpu_utilization: float
    context_switches: int
    total_elapsed_time: int
    total_turnaround_time: Optional[int] = None
    total_response_time: Optional[int] = None
    cpu_busy_time: Optional[int] = None
    cpu_idle_time: Optional[int] = None
    throughput: Optional[float] = None

    @property
    def has_runtime_detail(self) -> bool:
        return all(
            value is not None
            for value in (self.cpu_busy_time, self.cpu_idle_time, self.throughput)
        )


@dataclass(frozen=True)
class Incident:
    """One concluded self-healing incident, with its trigger, verification and decision."""

    episode_index: int
    incident_index: int
    decision_index: int
    simulated_time: int
    detector_reason: str
    suspicion_evaluations: int
    trigger_threshold: float
    reference_at_trigger: float
    band_at_trigger: float
    window_mean_at_trigger: float
    window_size_at_trigger: int
    window_median_at_trigger: float
    window_max_at_trigger: float
    excess_band_units: float
    ready_count_at_trigger: int
    visible_remaining_work_at_trigger: int
    diagnosis: str
    diagnosis_statement: str
    diagnosis_matched: str
    context_state_index: int
    feature_bins: Tuple[int, ...]
    offered_actions: str
    action: int
    action_name: str
    action_changed: bool
    action_note: str
    request_note: str
    settings_before_policy: str
    settings_before_quantum: int
    settings_after_policy: str
    settings_after_quantum: int
    epsilon: float
    explored: bool
    verdict: str
    verifier_reason: str
    confirmed_failure: bool
    consecutive_failures: int
    window_truncated: bool
    forced_hold: bool
    verification_decisions: int
    verification_completions: int
    post_window_mean: Optional[float]
    improvement: Optional[float]
    improvement_band_units: Optional[float]
    pre_load_level: float
    trigger_load_level: float
    post_load_level: Optional[float]
    load_band_at_trigger: float
    load_improvement: Optional[float]
    load_improvement_band_units: Optional[float]
    closure_fraction: Optional[float]
    closure_residual: Optional[float]
    expected_closure: float
    required_closure_factor: float
    reject_closure_factor: float
    accept_residual_factor: float
    reject_residual_factor: float
    cost_band_units: float
    reward: Optional[float]
    rollback_target_policy: str
    rollback_target_quantum: Optional[int]
    switch_overhead_units: int
    window_end_time: Optional[int]
    fallback_triggered: bool
    relative_band: Optional[float]
    remaining_work_ratio: Optional[float]

    @property
    def is_intervention(self) -> bool:
        return self.action_changed

    @property
    def is_error(self) -> bool:
        """Whether the verdict is a *confirmed* failure (a rejected intervention)."""
        return self.verdict == "rejected"


@dataclass(frozen=True)
class Fixture:
    """One demo scenario: workload, recorded dispatch intervals, incident and metrics."""

    family: str
    seed: int
    fingerprint: str
    processes: Tuple[Process, ...]
    decisions: Tuple[Mapping[str, Any], ...]
    intervals: Tuple[Interval, ...]
    policy_segments: Tuple[PolicySegment, ...]
    incidents: Tuple[Incident, ...]
    outcomes: Tuple[Outcome, ...]
    metrics: Metrics
    fcfs_metrics: Metrics

    @property
    def primary_incident(self) -> Optional[Incident]:
        return self.incidents[0] if self.incidents else None

    @property
    def final_policy(self) -> str:
        """Policy in force at the last recorded dispatch (what actually ended the trace)."""
        if not self.intervals:
            return "n/a"
        return self.intervals[-1].policy

    @property
    def total_idle(self) -> int:
        return sum(interval.idle_before for interval in self.intervals)

    @property
    def total_switch_time(self) -> int:
        return sum(interval.switch_before for interval in self.intervals)

    @property
    def domain_end(self) -> int:
        return max(
            self.metrics.total_elapsed_time,
            max((outcome.completion_time for outcome in self.outcomes), default=0),
        )

    @property
    def mean_waiting_difference(self) -> float:
        return self.metrics.avg_waiting_time - self.fcfs_metrics.avg_waiting_time

    @property
    def mean_waiting_reduction_fraction(self) -> Optional[float]:
        baseline = self.fcfs_metrics.avg_waiting_time
        if baseline == 0:
            return None
        return (baseline - self.metrics.avg_waiting_time) / baseline

    def process_row(self, process: Process) -> Outcome:
        for outcome in self.outcomes:
            if outcome.pid == process.pid:
                return outcome
        raise _fail(f"seed {self.seed}", f"no outcome recorded for pid {process.pid}")

    @property
    def total_waiting_from_outcomes(self) -> int:
        return sum(outcome.waiting_time for outcome in self.outcomes)

    @property
    def mean_waiting_from_outcomes(self) -> float:
        if not self.outcomes:
            return 0.0
        return self.total_waiting_from_outcomes / len(self.outcomes)


@dataclass(frozen=True)
class DemoDocument:
    """The whole ``demo.json`` file."""

    scope: str
    scheduler: Mapping[str, Any]
    healing: Mapping[str, Any]
    fixtures: Tuple[Fixture, ...]
    source_name: str = "demo.json"


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _parse_process(raw: Any, where: str) -> Process:
    mapping = _mapping(raw, where)
    return Process(
        pid=_int(_require(mapping, "pid", where), f"{where}.pid"),
        arrival_time=_int(_require(mapping, "arrival_time", where), f"{where}.arrival_time"),
        burst_time=_int(_require(mapping, "burst_time", where), f"{where}.burst_time"),
        priority=_int(_require(mapping, "priority", where), f"{where}.priority"),
    )


def _parse_outcome(raw: Any, where: str) -> Outcome:
    mapping = _mapping(raw, where)
    return Outcome(
        pid=_int(_require(mapping, "pid", where), f"{where}.pid"),
        arrival_time=_int(_require(mapping, "arrival_time", where), f"{where}.arrival_time"),
        burst_time=_int(_require(mapping, "burst_time", where), f"{where}.burst_time"),
        priority=_int(_require(mapping, "priority", where), f"{where}.priority"),
        start_time=_int(_require(mapping, "start_time", where), f"{where}.start_time"),
        completion_time=_int(
            _require(mapping, "completion_time", where), f"{where}.completion_time"
        ),
    )


_METRIC_REQUIRED = (
    "policy_name",
    "workload_name",
    "workload_fingerprint",
    "num_processes",
    "total_waiting_time",
    "avg_waiting_time",
    "avg_turnaround_time",
    "avg_response_time",
    "cpu_utilization",
    "context_switches",
    "total_elapsed_time",
)
_METRIC_OPTIONAL = (
    "total_turnaround_time",
    "total_response_time",
    "cpu_busy_time",
    "cpu_idle_time",
    "throughput",
)


def _parse_metrics(raw: Any, where: str) -> Metrics:
    mapping = _mapping(raw, where)
    missing = [name for name in _METRIC_REQUIRED if name not in mapping]
    if missing:
        raise _fail(where, f"missing metric field(s): {', '.join(missing)}")
    values: Dict[str, Any] = {
        "policy_name": _text(mapping["policy_name"], f"{where}.policy_name"),
        "workload_name": _text(mapping["workload_name"], f"{where}.workload_name"),
        "workload_fingerprint": _text(
            mapping["workload_fingerprint"], f"{where}.workload_fingerprint"
        ),
        "num_processes": _int(mapping["num_processes"], f"{where}.num_processes"),
        "total_waiting_time": _int(
            mapping["total_waiting_time"], f"{where}.total_waiting_time"
        ),
        "avg_waiting_time": _number(mapping["avg_waiting_time"], f"{where}.avg_waiting_time"),
        "avg_turnaround_time": _number(
            mapping["avg_turnaround_time"], f"{where}.avg_turnaround_time"
        ),
        "avg_response_time": _number(
            mapping["avg_response_time"], f"{where}.avg_response_time"
        ),
        "cpu_utilization": _number(mapping["cpu_utilization"], f"{where}.cpu_utilization"),
        "context_switches": _int(mapping["context_switches"], f"{where}.context_switches"),
        "total_elapsed_time": _int(
            mapping["total_elapsed_time"], f"{where}.total_elapsed_time"
        ),
    }
    for name in _METRIC_OPTIONAL:
        if name not in mapping or mapping[name] is None:
            continue
        if name == "throughput":
            values[name] = _number(mapping[name], f"{where}.{name}")
        elif name in ("cpu_busy_time", "cpu_idle_time"):
            values[name] = _int(mapping[name], f"{where}.{name}")
        else:
            values[name] = _int(mapping[name], f"{where}.{name}")
    return Metrics(**values)


_INCIDENT_POLICY_FIELDS = ("settings_before_policy", "settings_after_policy")


def _parse_incident(raw: Any, where: str) -> Incident:
    mapping = _mapping(raw, where)

    def number(name: str) -> float:
        return _number(_require(mapping, name, where), f"{where}.{name}")

    def integer(name: str) -> int:
        return _int(_require(mapping, name, where), f"{where}.{name}")

    def optional_number(name: str) -> Optional[float]:
        return _optional_number(mapping.get(name), f"{where}.{name}")

    def text(name: str, *, allow_empty: bool = False) -> str:
        return _text(_require(mapping, name, where), f"{where}.{name}", allow_empty=allow_empty)

    verdict = text("verdict")
    if verdict not in VERDICTS:
        raise _fail(f"{where}.verdict", f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
    action_name = text("action_name")
    if action_name not in HEALING_ACTION_NAMES:
        raise _fail(
            f"{where}.action_name",
            f"unknown recovery action {action_name!r}; expected one of {HEALING_ACTION_NAMES}",
        )
    for name in _INCIDENT_POLICY_FIELDS:
        policy = text(name)
        if policy not in POLICY_NAMES:
            raise _fail(f"{where}.{name}", f"unknown policy {policy!r}; expected one of {POLICY_NAMES}")

    context = _mapping(mapping.get("context") or {}, f"{where}.context")
    rollback_policy = _text(
        _require(mapping, "rollback_target_policy", where),
        f"{where}.rollback_target_policy",
        allow_empty=True,
    )
    if rollback_policy and rollback_policy not in POLICY_NAMES:
        raise _fail(
            f"{where}.rollback_target_policy",
            f"unknown policy {rollback_policy!r}; expected one of {POLICY_NAMES} or an empty string",
        )

    window_end = _optional_int(mapping.get("window_end_time"), f"{where}.window_end_time")
    simulated_time = integer("simulated_time")
    if window_end is not None and window_end < simulated_time:
        raise _fail(
            f"{where}.window_end_time",
            f"window end {window_end} precedes the trigger time {simulated_time}",
        )

    incident = Incident(
        episode_index=integer("episode_index"),
        incident_index=integer("incident_index"),
        decision_index=integer("decision_index"),
        simulated_time=simulated_time,
        detector_reason=text("detector_reason"),
        suspicion_evaluations=integer("suspicion_evaluations"),
        trigger_threshold=number("trigger_threshold"),
        reference_at_trigger=number("reference_at_trigger"),
        band_at_trigger=number("band_at_trigger"),
        window_mean_at_trigger=number("window_mean_at_trigger"),
        window_size_at_trigger=integer("window_size_at_trigger"),
        window_median_at_trigger=number("window_median_at_trigger"),
        window_max_at_trigger=number("window_max_at_trigger"),
        excess_band_units=number("excess_band_units"),
        ready_count_at_trigger=integer("ready_count_at_trigger"),
        visible_remaining_work_at_trigger=integer("visible_remaining_work_at_trigger"),
        diagnosis=text("diagnosis"),
        diagnosis_statement=text("diagnosis_statement"),
        diagnosis_matched=_text(
            _require(mapping, "diagnosis_matched", where),
            f"{where}.diagnosis_matched",
            allow_empty=True,
        ),
        context_state_index=integer("context_state_index"),
        feature_bins=tuple(
            _int(value, f"{where}.feature_bins[{index}]")
            for index, value in enumerate(_sequence(_require(mapping, "feature_bins", where), f"{where}.feature_bins"))
        ),
        offered_actions=text("offered_actions"),
        action=integer("action"),
        action_name=action_name,
        action_changed=_boolean(
            _require(mapping, "action_changed", where), f"{where}.action_changed"
        ),
        action_note=text("action_note", allow_empty=True),
        request_note=text("request_note", allow_empty=True),
        settings_before_policy=text("settings_before_policy"),
        settings_before_quantum=integer("settings_before_quantum"),
        settings_after_policy=text("settings_after_policy"),
        settings_after_quantum=integer("settings_after_quantum"),
        epsilon=number("epsilon"),
        explored=_boolean(_require(mapping, "explored", where), f"{where}.explored"),
        verdict=verdict,
        verifier_reason=text("verifier_reason"),
        confirmed_failure=_boolean(
            _require(mapping, "confirmed_failure", where), f"{where}.confirmed_failure"
        ),
        consecutive_failures=integer("consecutive_failures"),
        window_truncated=_boolean(
            _require(mapping, "window_truncated", where), f"{where}.window_truncated"
        ),
        forced_hold=_boolean(
            _require(mapping, "forced_hold", where), f"{where}.forced_hold"
        ),
        verification_decisions=integer("verification_decisions"),
        verification_completions=integer("verification_completions"),
        post_window_mean=optional_number("post_window_mean"),
        improvement=optional_number("improvement"),
        improvement_band_units=optional_number("improvement_band_units"),
        pre_load_level=number("pre_load_level"),
        trigger_load_level=number("trigger_load_level"),
        post_load_level=optional_number("post_load_level"),
        load_band_at_trigger=number("load_band_at_trigger"),
        load_improvement=optional_number("load_improvement"),
        load_improvement_band_units=optional_number("load_improvement_band_units"),
        closure_fraction=optional_number("closure_fraction"),
        closure_residual=optional_number("closure_residual"),
        expected_closure=number("expected_closure"),
        required_closure_factor=number("required_closure_factor"),
        reject_closure_factor=number("reject_closure_factor"),
        accept_residual_factor=number("accept_residual_factor"),
        reject_residual_factor=number("reject_residual_factor"),
        cost_band_units=number("cost_band_units"),
        reward=optional_number("reward"),
        rollback_target_policy=rollback_policy,
        rollback_target_quantum=_optional_int(
            # ``as_row`` serialises an absent rollback target as an empty string.
            None
            if mapping.get("rollback_target_quantum") in ("", None)
            else mapping.get("rollback_target_quantum"),
            f"{where}.rollback_target_quantum",
        ),
        switch_overhead_units=integer("switch_overhead_units"),
        window_end_time=window_end,
        fallback_triggered=_boolean(
            _require(mapping, "fallback_triggered", where), f"{where}.fallback_triggered"
        ),
        relative_band=_optional_number(context.get("relative_band"), f"{where}.context.relative_band"),
        remaining_work_ratio=_optional_number(
            context.get("remaining_work_ratio"), f"{where}.context.remaining_work_ratio"
        ),
    )
    if incident.confirmed_failure != (incident.verdict == "rejected"):
        raise _fail(
            where,
            "inconsistent incident: confirmed_failure must be true exactly for a rejected "
            "verdict (an inconclusive window is not a confirmed failure)",
        )
    if incident.accept_residual_factor <= incident.reject_residual_factor:
        raise _fail(where, "accept_residual_factor must exceed reject_residual_factor")
    return incident


def _parse_intervals(decisions: Sequence[Mapping[str, Any]], where: str) -> Tuple[Interval, ...]:
    intervals: List[Interval] = []
    previous_end: Optional[int] = None
    for index, raw in enumerate(decisions):
        decision_where = f"{where}.decisions[{index}]"
        mapping = _mapping(raw, decision_where)
        start = _int(_require(mapping, "start", decision_where), f"{decision_where}.start")
        end = _int(_require(mapping, "end", decision_where), f"{decision_where}.end")
        pid = _int(_require(mapping, "pid", decision_where), f"{decision_where}.pid")
        policy = _text(_require(mapping, "policy", decision_where), f"{decision_where}.policy")
        if policy not in POLICY_NAMES:
            raise _fail(f"{decision_where}.policy", f"unknown policy {policy!r}")
        switch = _int(
            _require(mapping, "switch_overhead", decision_where),
            f"{decision_where}.switch_overhead",
        )
        waiting = _optional_int(
            mapping.get("waiting_increment"), f"{decision_where}.waiting_increment"
        )
        completed = tuple(
            _int(value, f"{decision_where}.completed_pids[{position}]")
            for position, value in enumerate(
                _sequence(mapping.get("completed_pids") or [], f"{decision_where}.completed_pids")
            )
        )
        if end <= start:
            raise _fail(decision_where, f"empty dispatch interval [{start}, {end})")
        if switch < 0:
            raise _fail(decision_where, f"negative switch overhead {switch}")
        idle = 0
        if previous_end is not None:
            if start < previous_end:
                raise _fail(
                    decision_where,
                    f"dispatch at {start} starts before the previous interval ended ({previous_end})",
                )
            gap = start - previous_end
            if switch > gap:
                raise _fail(
                    decision_where,
                    f"recorded switch overhead {switch} exceeds the {gap}-unit gap before the dispatch",
                )
            idle = gap - switch
        elif start > 0:
            idle = start
        intervals.append(
            Interval(
                index=index,
                pid=pid,
                policy=policy,
                start=start,
                end=end,
                switch_before=switch,
                idle_before=idle,
                waiting_increment=waiting,
                completed_pids=completed,
            )
        )
        previous_end = end
    return tuple(intervals)


def _policy_segments(intervals: Sequence[Interval]) -> Tuple[PolicySegment, ...]:
    segments: List[PolicySegment] = []
    for interval in intervals:
        if segments and segments[-1].policy == interval.policy:
            last = segments[-1]
            segments[-1] = PolicySegment(
                policy=last.policy,
                start=last.start,
                end=interval.end,
                decisions=last.decisions + 1,
            )
        else:
            segments.append(
                PolicySegment(
                    policy=interval.policy,
                    start=interval.start,
                    end=interval.end,
                    decisions=1,
                )
            )
    return tuple(segments)


def _validate_fixture(fixture: Fixture, where: str) -> None:
    """Cross-check the trace, the per-process outcomes and the recorded metrics."""
    processes = {process.pid: process for process in fixture.processes}
    outcomes = {outcome.pid: outcome for outcome in fixture.outcomes}
    if len(processes) != len(fixture.processes):
        raise _fail(where, "duplicate pid in the process list")
    if len(outcomes) != len(fixture.outcomes):
        raise _fail(where, "duplicate pid in the outcome list")
    if set(processes) != set(outcomes):
        missing = sorted(set(processes) - set(outcomes))
        extra = sorted(set(outcomes) - set(processes))
        raise _fail(
            where,
            f"process/outcome pid mismatch (no outcome: {missing}; unknown outcome: {extra})",
        )

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise _fail(where, message)

    # Every process must have arrived, run for exactly its burst, and been dispatched by
    # the recorded decisions -- otherwise the Gantt would show intervals the engine never
    # produced.
    executed: Dict[int, int] = {}
    first_start: Dict[int, int] = {}
    last_end: Dict[int, int] = {}
    for interval in fixture.intervals:
        if interval.pid not in processes:
            raise _fail(where, f"dispatch interval references unknown pid {interval.pid}")
        executed[interval.pid] = executed.get(interval.pid, 0) + interval.duration
        first_start.setdefault(interval.pid, interval.start)
        last_end[interval.pid] = interval.end
    for pid, process in processes.items():
        outcome = outcomes[pid]
        check(
            executed.get(pid, 0) == process.burst_time,
            f"pid {pid} executed {executed.get(pid, 0)} units but has burst {process.burst_time}",
        )
        check(
            first_start.get(pid) == outcome.start_time,
            f"pid {pid} first dispatch at {first_start.get(pid)} disagrees with the recorded "
            f"outcome start {outcome.start_time}",
        )
        check(
            last_end.get(pid) == outcome.completion_time,
            f"pid {pid} last dispatch ends at {last_end.get(pid)} but the outcome completes at "
            f"{outcome.completion_time}",
        )
        check(
            outcome.completion_time - outcome.arrival_time - outcome.burst_time
            == outcome.waiting_time,
            f"pid {pid} has an inconsistent waiting time",
        )
        check(
            outcome.arrival_time == process.arrival_time
            and outcome.burst_time == process.burst_time
            and outcome.priority == process.priority,
            f"pid {pid} differs between the workload table and the outcome row",
        )

    metrics = fixture.metrics
    check(
        metrics.workload_fingerprint == fixture.fingerprint,
        f"metrics.workload_fingerprint {metrics.workload_fingerprint!r} does not match the "
        f"fixture fingerprint {fixture.fingerprint!r}",
    )
    check(
        fixture.fcfs_metrics.workload_fingerprint == fixture.fingerprint,
        "the FCFS comparison was run on a different workload "
        f"({fixture.fcfs_metrics.workload_fingerprint!r} != {fixture.fingerprint!r})",
    )
    check(
        metrics.num_processes == len(fixture.outcomes),
        f"metrics.num_processes {metrics.num_processes} != {len(fixture.outcomes)} recorded outcomes",
    )
    check(
        fixture.fcfs_metrics.num_processes == metrics.num_processes,
        "the FCFS comparison completed a different number of processes",
    )

    total_wait = fixture.total_waiting_from_outcomes
    check(
        metrics.total_waiting_time == total_wait,
        f"metrics.total_waiting_time {metrics.total_waiting_time} does not equal the sum of "
        f"the per-process waiting times ({total_wait})",
    )
    check(
        _close(metrics.avg_waiting_time, total_wait / len(fixture.outcomes)),
        f"metrics.avg_waiting_time {metrics.avg_waiting_time!r} does not equal "
        f"total_waiting_time / num_processes ({total_wait / len(fixture.outcomes)!r})",
    )
    total_turnaround = sum(outcome.turnaround_time for outcome in fixture.outcomes)
    total_response = sum(outcome.response_time for outcome in fixture.outcomes)
    if metrics.total_turnaround_time is not None:
        check(
            metrics.total_turnaround_time == total_turnaround,
            "metrics.total_turnaround_time disagrees with the per-process outcomes",
        )
    if metrics.total_response_time is not None:
        check(
            metrics.total_response_time == total_response,
            "metrics.total_response_time disagrees with the per-process outcomes",
        )
    check(
        _close(metrics.avg_turnaround_time, total_turnaround / len(fixture.outcomes)),
        "metrics.avg_turnaround_time is inconsistent with the per-process outcomes",
    )
    check(
        _close(metrics.avg_response_time, total_response / len(fixture.outcomes)),
        "metrics.avg_response_time is inconsistent with the per-process outcomes",
    )

    elapsed = max(outcome.completion_time for outcome in fixture.outcomes)
    check(
        metrics.total_elapsed_time == elapsed,
        f"metrics.total_elapsed_time {metrics.total_elapsed_time} != last completion {elapsed}",
    )
    busy = sum(interval.duration for interval in fixture.intervals)
    if metrics.cpu_busy_time is not None:
        check(
            metrics.cpu_busy_time == busy,
            f"metrics.cpu_busy_time {metrics.cpu_busy_time} != dispatches' {busy} busy units",
        )
    if metrics.cpu_idle_time is not None:
        check(
            metrics.cpu_idle_time == fixture.total_idle,
            f"metrics.cpu_idle_time {metrics.cpu_idle_time} != the {fixture.total_idle} "
            "recorded idle units",
        )
    switches = sum(
        1 for previous, current in zip(fixture.intervals, fixture.intervals[1:]) if previous.pid != current.pid
    )
    check(
        metrics.context_switches == switches,
        f"metrics.context_switches {metrics.context_switches} != {switches} changes of the "
        "running process in the trace",
    )
    if metrics.total_elapsed_time and (metrics.cpu_busy_time is not None or metrics.cpu_idle_time is not None):
        check(
            _close(metrics.cpu_utilization, 100.0 * busy / metrics.total_elapsed_time),
            "metrics.cpu_utilization is inconsistent with busy time and elapsed time",
        )
    if metrics.throughput is not None and metrics.total_elapsed_time:
        check(
            _close(metrics.throughput, len(fixture.outcomes) / metrics.total_elapsed_time),
            "metrics.throughput is inconsistent with the process count and elapsed time",
        )

    wait_increments = [interval.waiting_increment for interval in fixture.intervals]
    if all(value is not None for value in wait_increments):
        check(
            sum(int(value) for value in wait_increments) == total_wait,
            "the recorded waiting_cost_increment values do not sum to the total waiting time",
        )

    for incident in fixture.incidents:
        if incident.decision_index < 0 or incident.decision_index >= len(fixture.intervals):
            raise _fail(
                where,
                f"incident at decision {incident.decision_index} is outside the recorded "
                f"trace (0..{len(fixture.intervals) - 1})",
            )
        dispatch = fixture.intervals[incident.decision_index]
        check(
            dispatch.start >= incident.simulated_time,
            f"incident trigger time {incident.simulated_time} follows its own dispatch "
            f"({dispatch.start})",
        )
        if incident.window_end_time is not None:
            check(
                incident.window_end_time <= metrics.total_elapsed_time,
                f"verification window ends at {incident.window_end_time}, after the trace "
                f"ends at {metrics.total_elapsed_time}",
            )
        if incident.action_changed:
            check(
                (incident.settings_before_policy, incident.settings_before_quantum)
                != (incident.settings_after_policy, incident.settings_after_quantum),
                "incident reports a configuration change that did not change any setting",
            )
        if incident.rollback_target_policy:
            check(
                fixture.final_policy == incident.rollback_target_policy,
                f"the incident records a rollback to {incident.rollback_target_policy} but the "
                f"trace ends under {fixture.final_policy}",
            )


def _parse_fixture(raw: Any, index: int, source: str) -> Fixture:
    where = f"{source}: fixtures[{index}]"
    mapping = _mapping(raw, where)
    seed = _int(_require(mapping, "seed", where), f"{where}.seed")
    where = f"{source}: fixtures[{index}] (seed {seed})"
    family = _text(_require(mapping, "family", where), f"{where}.family")
    fingerprint = _text(_require(mapping, "fingerprint", where), f"{where}.fingerprint")
    processes_raw = _sequence(_require(mapping, "processes", where), f"{where}.processes")
    if not processes_raw:
        raise _fail(f"{where}.processes", "the workload is empty")
    outcomes_raw = _sequence(_require(mapping, "outcomes", where), f"{where}.outcomes")
    if not outcomes_raw:
        raise _fail(f"{where}.outcomes", "no completed-process outcomes are recorded")
    decisions_raw = _sequence(_require(mapping, "decisions", where), f"{where}.decisions")
    if not decisions_raw:
        raise _fail(f"{where}.decisions", "no dispatch decisions are recorded")
    incidents_raw = _sequence(_require(mapping, "incidents", where), f"{where}.incidents")

    processes = tuple(
        _parse_process(item, f"{where}.processes[{position}]")
        for position, item in enumerate(processes_raw)
    )
    outcomes = tuple(
        _parse_outcome(item, f"{where}.outcomes[{position}]")
        for position, item in enumerate(outcomes_raw)
    )
    intervals = _parse_intervals(decisions_raw, where)
    incidents = tuple(
        _parse_incident(item, f"{where}.incidents[{position}]")
        for position, item in enumerate(incidents_raw)
    )
    metrics = _parse_metrics(_require(mapping, "metrics", where), f"{where}.metrics")
    if "fcfs_metrics" not in mapping:
        raise _fail(where, "missing required field 'fcfs_metrics' (the FCFS comparison run)")
    fcfs_metrics = _parse_metrics(mapping["fcfs_metrics"], f"{where}.fcfs_metrics")
    if fcfs_metrics.policy_name != "FCFS":
        raise _fail(
            f"{where}.fcfs_metrics.policy_name",
            f"the comparison run must be FCFS, got {fcfs_metrics.policy_name!r}",
        )

    fixture = Fixture(
        family=family,
        seed=seed,
        fingerprint=fingerprint,
        processes=processes,
        decisions=tuple(_mapping(item, f"{where}.decisions") for item in decisions_raw),
        intervals=intervals,
        policy_segments=_policy_segments(intervals),
        incidents=incidents,
        outcomes=outcomes,
        metrics=metrics,
        fcfs_metrics=fcfs_metrics,
    )
    _validate_fixture(fixture, where)
    return fixture


def parse_fixture(payload: Any, source: str = "engine") -> Fixture:
    """Validate a single fixture record (one entry of ``demo.json``'s ``fixtures``).

    The live dashboard runs the engine itself, so it receives fixture records without the
    surrounding document.  Parsing them through the same validator the file reader uses
    keeps both consumers on one schema and one set of consistency checks.

    Args:
        payload: One fixture mapping, exactly as :func:`experiments.demo.run_healing_workload`
            returns it.
        source: Name used in error messages and provenance lines.

    Returns:
        The validated :class:`Fixture`.

    Raises:
        DashboardError: If the record is incomplete or contradicts itself.
    """
    return _parse_fixture(payload, 0, source)


def parse_demo(payload: Any, source: str = "demo.json") -> DemoDocument:
    """Validate a parsed ``demo.json`` payload and return its typed view.

    Args:
        payload: The object returned by :func:`json.loads`.
        source: Name used in error messages and provenance lines.

    Returns:
        The validated :class:`DemoDocument`.

    Raises:
        DashboardError: If a required field is missing, a value has the wrong type, or the
            trace contradicts the recorded outcomes/metrics.
    """
    mapping = _mapping(payload, source)
    fixtures_raw = _sequence(_require(mapping, "fixtures", source), f"{source}: fixtures")
    if not fixtures_raw:
        raise _fail(f"{source}: fixtures", "the demo file contains no scenarios")
    fixtures = tuple(
        _parse_fixture(item, index, source) for index, item in enumerate(fixtures_raw)
    )
    seeds = [fixture.seed for fixture in fixtures]
    if len(set(seeds)) != len(seeds):
        raise _fail(f"{source}: fixtures", f"duplicate scenario seeds {seeds}")
    scope = _text(_require(mapping, "scope", source), f"{source}: scope")
    scheduler = dict(_mapping(_require(mapping, "scheduler", source), f"{source}: scheduler"))
    healing = dict(_mapping(_require(mapping, "healing", source), f"{source}: healing"))
    for name in ("round_robin_quantum", "switching_cost"):
        if name not in scheduler:
            raise _fail(f"{source}: scheduler", f"missing required field {name!r}")
        _int(scheduler[name], f"{source}: scheduler.{name}")
    for name in (
        "accept_residual_factor",
        "reject_residual_factor",
        "verify_decisions",
        "verify_completions",
        "verify_max_decisions",
        "max_consecutive_failures",
        "inconclusive_rollback",
        "threshold_band_multiplier",
        "hysteresis_evaluations",
    ):
        if name not in healing:
            raise _fail(f"{source}: healing", f"missing declared threshold {name!r}")
    return DemoDocument(
        scope=scope,
        scheduler=scheduler,
        healing=healing,
        fixtures=fixtures,
        source_name=source,
    )
