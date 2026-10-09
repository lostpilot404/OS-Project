"""Event-driven single-CPU runtime scheduling with a causal controller boundary.

Unlike the standalone batch schedulers, :class:`RuntimeSimulator` repeatedly asks a
controller to choose one of four conventional policies at dispatch/quantum epochs. It
owns the full workload and advances the event queue, but passes controllers only an
immutable :class:`RuntimeObservation` containing work that has arrived by that instant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Optional, Protocol, Sequence, runtime_checkable

from config import (
    ACTION_FCFS,
    ACTION_NAMES,
    ACTION_PRIORITY,
    ACTION_ROUND_ROBIN,
    ACTION_SJF,
    SchedulerConfig,
)
from errors import ValidationError
from scheduler.base import Timeline
from workload.models import (
    ExecutionSlice,
    Process,
    ScheduleResult,
    Workload,
    build_schedule_result,
)

__all__ = [
    "ObservedProcess",
    "RuntimeObservation",
    "RuntimeActionDecision",
    "RuntimeDecisionRecord",
    "RuntimeRun",
    "RuntimeController",
    "RuntimeSimulator",
]


@dataclass(frozen=True)
class ObservedProcess:
    """Causal, read-only view of one process that has arrived and is still ready."""

    pid: int
    arrival_time: int
    burst_time: int
    remaining_burst: int
    priority: int


@dataclass(frozen=True)
class RuntimeObservation:
    """Information exposed to a controller at a single decision epoch.

    No field refers to a :class:`Workload`, unarrived process, future event, or completed
    schedule. Burst lengths for arrived processes are exact, matching the simulator's
    explicit assumption that burst estimates are known when a process enters the system
    (the same assumption required by SJF). Aggregates cover only work observed so far.
    """

    current_time: int
    ready_processes: tuple[ObservedProcess, ...]
    completed_count: int
    arrived_count: int
    observed_mean_burst: float
    observed_priority_spread: int
    mean_ready_wait: float
    previous_policy: Optional[int]

    def __post_init__(self) -> None:
        if self.current_time < 0:
            raise ValidationError("observation current_time must be >= 0")
        if self.completed_count < 0 or self.arrived_count < 0:
            raise ValidationError("observation counts must be >= 0")
        if self.arrived_count < self.completed_count + len(self.ready_processes):
            raise ValidationError("arrived_count cannot be smaller than visible work")
        if any(p.arrival_time > self.current_time for p in self.ready_processes):
            raise ValidationError("observation contains a process that has not arrived")
        if any(p.remaining_burst <= 0 or p.burst_time <= 0 for p in self.ready_processes):
            raise ValidationError("ready process bursts must be positive")
        if self.previous_policy is not None and self.previous_policy not in range(len(ACTION_NAMES)):
            raise ValidationError("previous_policy is outside the four-action space")


@dataclass(frozen=True)
class RuntimeActionDecision:
    """Controller response plus auditable exploration/fallback metadata."""

    action: int
    state_index: Optional[int] = None
    epsilon: float = 0.0
    explored: bool = False
    fallback_reason: Optional[str] = None
    unvisited_action_count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.action, bool)
            or not isinstance(self.action, int)
            or self.action not in range(len(ACTION_NAMES))
        ):
            raise ValidationError(f"runtime action must be an integer in 0..3, got {self.action}")
        if not math.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValidationError(f"epsilon must be finite and in [0, 1], got {self.epsilon}")
        if isinstance(self.unvisited_action_count, bool) or self.unvisited_action_count < 0:
            raise ValidationError("unvisited_action_count must be >= 0")


@dataclass(frozen=True)
class RuntimeDecisionRecord:
    """One realized decision, transition reward and next causal state."""

    decision_index: int
    observation: RuntimeObservation
    decision: RuntimeActionDecision
    chosen_pid: int
    start_time: int
    end_time: int
    switch_overhead: int
    waiting_cost_increment: int
    reward: float
    next_state_index: Optional[int]
    terminal: bool
    policy_switch: bool

    @property
    def policy_name(self) -> str:
        """Human-readable policy selected at this epoch."""
        return ACTION_NAMES[self.decision.action]


@dataclass(frozen=True)
class RuntimeRun:
    """A validated schedule and controller/environment timing diagnostics."""

    result: ScheduleResult
    decisions: tuple[RuntimeDecisionRecord, ...]
    total_reward: float
    observation_ns: int
    selection_ns: int
    update_ns: int
    wall_ns: int

    @property
    def policy_switch_count(self) -> int:
        """Number of within-trace changes between selected policy labels."""
        return sum(record.policy_switch for record in self.decisions)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)


@runtime_checkable
class RuntimeController(Protocol):
    """Minimal interface available to the runtime environment."""

    def select_action(
        self,
        observation: RuntimeObservation,
        *,
        training: bool = False,
        episode: int = 0,
    ) -> RuntimeActionDecision:
        """Choose a policy from the causal observation only."""

    def observe_transition(
        self,
        decision: RuntimeActionDecision,
        reward: float,
        next_observation: Optional[RuntimeObservation],
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """Optionally learn from the transition; evaluation must not update."""


class RuntimeSimulator:
    """Event-driven scheduler that preserves one evolving ready queue.

    At an epoch, FCFS/SJF/Priority run the chosen process non-preemptively; Round Robin
    runs it for ``min(quantum, remaining_burst)``. Arrivals are admitted in chronological
    order and appended to the existing FIFO ready queue. A partially executed RR process
    is appended at the tail after arrivals at the quantum endpoint are admitted.

    Switch cost is charged only when the running PID changes, matching the standalone
    schedulers and ``build_schedule_result``. A policy change never resets or reconstructs
    the ready queue.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        if not isinstance(config, SchedulerConfig):
            raise ValidationError(f"config must be SchedulerConfig, got {type(config).__name__}")
        self.config = config

    def run(
        self,
        workload: Workload,
        controller: RuntimeController,
        *,
        training: bool = False,
        episode: int = 0,
        policy_name: str = "Runtime Adaptive",
    ) -> RuntimeRun:
        """Run one workload while exposing only observations to ``controller``.

        The workload is consumed privately by this method. Its object, complete process
        list, future arrivals, and schedule metrics are never passed to a controller
        method. Transition rewards are also computed from the causal elapsed interval.
        """
        if not isinstance(workload, Workload):
            raise ValidationError(f"workload must be Workload, got {type(workload).__name__}")
        if not callable(getattr(controller, "select_action", None)):
            raise ValidationError("controller must define select_action(observation, ...)" )
        if not isinstance(training, bool):
            raise ValidationError("training must be a bool")
        if episode < 0:
            raise ValidationError("episode must be >= 0")

        wall_start = perf_counter_ns()
        timeline = Timeline()
        processes = workload.processes  # private environment state; never passed onward
        remaining = {process.pid: process.burst_time for process in processes}
        ready: list[Process] = []
        observed: list[Process] = []
        completed: set[int] = set()
        next_process = 0
        current_time = 0
        previous_pid: Optional[int] = None
        previous_policy: Optional[int] = None
        records: list[RuntimeDecisionRecord] = []
        observation_ns = 0
        selection_ns = 0
        update_ns = 0
        total_reward = 0.0
        quantum = self.config.round_robin_quantum
        reward_scale = float(max(1, quantum))

        def admit_until(time: int) -> None:
            nonlocal next_process
            while (
                next_process < len(processes)
                and processes[next_process].arrival_time <= time
            ):
                process = processes[next_process]
                ready.append(process)
                observed.append(process)
                next_process += 1

        def wait_increment(start: int, end: int, waiting_pids: Sequence[int]) -> int:
            """Waiting-time area in [start,end), including newly arriving processes."""
            if end <= start:
                return 0
            duration = end - start
            delta = len(waiting_pids) * duration
            # Only the environment consults the future event queue. A process that
            # arrives exactly at end has accumulated no wait in this interval.
            for process in processes[next_process:]:
                if start < process.arrival_time < end:
                    delta += end - process.arrival_time
            return delta

        def make_observation() -> RuntimeObservation:
            nonlocal observation_ns
            started = perf_counter_ns()
            if not ready:
                raise ValidationError("controller observation requires a non-empty ready queue")
            visible = tuple(
                ObservedProcess(
                    pid=p.pid,
                    arrival_time=p.arrival_time,
                    burst_time=p.burst_time,
                    remaining_burst=remaining[p.pid],
                    priority=p.priority,
                )
                for p in ready
            )
            observed_priorities = [process.priority for process in observed]
            observed_bursts = [process.burst_time for process in observed]
            result = RuntimeObservation(
                current_time=current_time,
                ready_processes=visible,
                completed_count=len(completed),
                arrived_count=len(observed),
                observed_mean_burst=(sum(observed_bursts) / len(observed_bursts)),
                observed_priority_spread=(
                    max(observed_priorities) - min(observed_priorities)
                    if observed_priorities
                    else 0
                ),
                mean_ready_wait=(
                    sum(current_time - p.arrival_time for p in ready) / len(ready)
                ),
                previous_policy=previous_policy,
            )
            observation_ns += perf_counter_ns() - started
            return result

        def wait_for_decision() -> None:
            """Advance through an idle interval until an arrived process is ready."""
            nonlocal current_time
            if ready or len(completed) == len(processes):
                return
            if next_process >= len(processes):
                raise ValidationError("unfinished workload has no ready or unarrived process")
            current_time = processes[next_process].arrival_time
            admit_until(current_time)

        # Empty workloads are supported by the same validated result builder.
        if not processes:
            result = build_schedule_result(workload, (), policy_name, self.config.switching_cost)
            return RuntimeRun(result, (), 0.0, 0, 0, 0, perf_counter_ns() - wall_start)

        admit_until(current_time)
        wait_for_decision()
        observation = make_observation()
        decision_index = 0

        while len(completed) < len(processes):
            selection_started = perf_counter_ns()
            decision = controller.select_action(
                observation, training=training, episode=episode
            )
            selection_ns += perf_counter_ns() - selection_started
            if not isinstance(decision, RuntimeActionDecision):
                raise ValidationError(
                    "controller.select_action must return RuntimeActionDecision"
                )

            if decision.action == ACTION_FCFS:
                chosen = min(ready, key=lambda p: (p.arrival_time, p.pid))
            elif decision.action == ACTION_SJF:
                chosen = min(ready, key=lambda p: (remaining[p.pid], p.arrival_time, p.pid))
            elif decision.action == ACTION_ROUND_ROBIN:
                chosen = ready[0]
            elif decision.action == ACTION_PRIORITY:
                def priority_key(process: Process) -> tuple[int, int, int]:
                    priority = process.priority
                    if not self.config.lower_priority_number_is_higher_priority:
                        priority = -priority
                    return (priority, process.arrival_time, process.pid)
                chosen = min(ready, key=priority_key)
            else:  # RuntimeActionDecision already validates, retain a defensive guard.
                raise ValidationError(f"unsupported runtime action {decision.action}")

            ready.remove(chosen)
            switch_cost = (
                self.config.switching_cost
                if previous_pid is not None and previous_pid != chosen.pid
                else 0
            )
            start_time = current_time + switch_cost
            waiting_at_dispatch = [chosen.pid, *(process.pid for process in ready)]
            waiting_delta = wait_increment(current_time, start_time, waiting_at_dispatch)
            admit_until(start_time)
            current_time = start_time

            service = (
                min(quantum, remaining[chosen.pid])
                if decision.action == ACTION_ROUND_ROBIN
                else remaining[chosen.pid]
            )
            end_time = start_time + service
            waiting_delta += wait_increment(
                start_time, end_time, [process.pid for process in ready]
            )
            timeline.execute(chosen.pid, start_time, end_time)
            remaining[chosen.pid] -= service
            current_time = end_time
            # Arrivals at the quantum/completion endpoint enter the queue before a
            # preempted process is re-enqueued, preserving FIFO RR tie semantics.
            admit_until(current_time)
            if remaining[chosen.pid] == 0:
                completed.add(chosen.pid)
            elif decision.action == ACTION_ROUND_ROBIN:
                ready.append(chosen)
            else:
                raise ValidationError("a non-RR action unexpectedly preempted a process")

            terminal = len(completed) == len(processes)
            next_observation: Optional[RuntimeObservation] = None
            if not terminal:
                wait_for_decision()
                next_observation = make_observation()

            reward = -float(waiting_delta) / reward_scale
            total_reward += reward
            next_state_index = None
            if next_observation is not None:
                # RuntimeQController annotates and encodes states; the simulator itself
                # remains agnostic to the state representation.
                encode = getattr(controller, "encode_observation", None)
                if callable(encode):
                    next_state_index = int(encode(next_observation))

            policy_switch = (
                previous_policy is not None and decision.action != previous_policy
            )
            record = RuntimeDecisionRecord(
                decision_index=decision_index,
                observation=observation,
                decision=decision,
                chosen_pid=chosen.pid,
                start_time=start_time,
                end_time=end_time,
                switch_overhead=switch_cost,
                waiting_cost_increment=waiting_delta,
                reward=reward,
                next_state_index=next_state_index,
                terminal=terminal,
                policy_switch=policy_switch,
            )
            records.append(record)

            transition_hook = getattr(controller, "observe_transition", None)
            if training and callable(transition_hook):
                update_started = perf_counter_ns()
                transition_hook(
                    decision,
                    reward,
                    next_observation,
                    terminal=terminal,
                    training=True,
                )
                update_ns += perf_counter_ns() - update_started

            previous_pid = chosen.pid
            previous_policy = decision.action
            observation = next_observation  # type: ignore[assignment]
            decision_index += 1

        result = build_schedule_result(
            workload=workload,
            slices=timeline.slices,
            policy_name=policy_name,
            switching_cost=self.config.switching_cost,
        )
        expected_reward = -sum(outcome.waiting_time for outcome in result.outcomes) / reward_scale
        # Mean waiting * process count equals total waiting. The incremental causal reward
        # must reconstruct that quantity exactly (up to binary float representation).
        if abs(total_reward - expected_reward) > 1e-9:
            raise ValidationError(
                f"runtime reward accounting mismatch: accumulated={total_reward}, "
                f"schedule={expected_reward}"
            )
        return RuntimeRun(
            result=result,
            decisions=tuple(records),
            total_reward=total_reward,
            observation_ns=observation_ns,
            selection_ns=selection_ns,
            update_ns=update_ns,
            wall_ns=perf_counter_ns() - wall_start,
        )
