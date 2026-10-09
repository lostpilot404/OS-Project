"""Correctness, causality, and sequential-learning tests for runtime scheduling."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from config import (
    ACTION_FCFS,
    ACTION_NAMES,
    QLearningConfig,
    SchedulerConfig,
)
from errors import ValidationError
from evaluation.metrics import compute_metrics
from rl.q_learning import QLearningAgent
from rl.runtime_controller import RuntimeQController
from rl.runtime_heuristic import CausalHeuristic
from rl.runtime_state import RuntimeStateEncoder
from scheduler import POLICY_CLASSES, FCFS, RoundRobin, RuntimeSimulator
from scheduler.runtime import RuntimeActionDecision, RuntimeObservation
from workload.models import Process, Workload


class FixedController:
    """Test adapter for a constant policy action."""

    def __init__(self, action: int) -> None:
        self.action = action

    def select_action(self, observation, *, training=False, episode=0):
        assert isinstance(observation, RuntimeObservation)
        return RuntimeActionDecision(self.action)

    def observe_transition(self, *args, **kwargs):
        return None


class RecordingFCFSController(FixedController):
    def __init__(self) -> None:
        super().__init__(ACTION_FCFS)
        self.observations: list[RuntimeObservation] = []
        self.actions: list[int] = []

    def select_action(self, observation, *, training=False, episode=0):
        self.observations.append(observation)
        action = super().select_action(observation, training=training, episode=episode).action
        self.actions.append(action)
        return RuntimeActionDecision(action)


class EncodedActionController:
    """Deterministic observation-driven test policy with nonconstant actions."""

    def __init__(self, quantum: int) -> None:
        self.encoder = RuntimeStateEncoder(quantum)

    def select_action(self, observation, *, training=False, episode=0):
        state = self.encoder.encode(observation)
        return RuntimeActionDecision(action=state % len(ACTION_NAMES), state_index=state)

    def observe_transition(self, *args, **kwargs):
        return None


def _workload(rows: list[tuple[int, int, int, int]], name: str = "runtime-test") -> Workload:
    return Workload(name, [Process(*row) for row in rows])


def _assert_schedule_equivalent(actual, expected) -> None:
    assert actual.trace == expected.trace
    assert actual.outcomes == expected.outcomes
    assert actual.context_switches == expected.context_switches
    assert actual.switching_overhead == expected.switching_overhead
    assert actual.total_elapsed_time == expected.total_elapsed_time
    assert actual.cpu_busy_time == expected.cpu_busy_time
    assert actual.idle_time == expected.idle_time
    left = compute_metrics(actual)
    right = compute_metrics(expected)
    for metric in (
        "avg_waiting_time",
        "avg_turnaround_time",
        "avg_response_time",
        "cpu_utilization",
        "throughput",
        "context_switches",
    ):
        assert getattr(left, metric) == pytest.approx(getattr(right, metric))


class TestRuntimeAgainstFixedSchedulers:
    @pytest.mark.parametrize("action,policy_class", list(enumerate(POLICY_CLASSES)))
    def test_constant_runtime_policy_matches_its_standalone_implementation(
        self, action, policy_class
    ) -> None:
        config = SchedulerConfig(round_robin_quantum=3, switching_cost=2)
        workload = _workload(
            [
                (7, 0, 8, 2),
                (3, 0, 3, 1),
                (12, 4, 4, 5),
                (9, 13, 2, 2),
                (20, 17, 6, 3),
            ]
        )
        runtime = RuntimeSimulator(config).run(
            workload,
            FixedController(action),
            policy_name=policy_class.name,
        )
        standalone = policy_class(config).run(workload)
        _assert_schedule_equivalent(runtime.result, standalone)

    def test_round_robin_arrival_at_quantum_boundary_is_enqueued_before_preempted_job(self) -> None:
        config = SchedulerConfig(round_robin_quantum=2, switching_cost=0)
        workload = _workload([(10, 0, 4, 1), (20, 2, 1, 2), (30, 5, 1, 3)])
        result = RuntimeSimulator(config).run(workload, FixedController(2))
        standalone = RoundRobin(config).run(workload)
        assert result.result.trace == standalone.trace
        assert result.result.trace[1].pid == 20
        assert result.result.trace[2].pid == 10

    def test_switch_cost_after_idle_arrival_is_accounted_without_false_idle_violation(self) -> None:
        config = SchedulerConfig(round_robin_quantum=3, switching_cost=2)
        workload = _workload([(1, 0, 1, 1), (2, 20, 1, 1)])
        runtime = RuntimeSimulator(config).run(workload, FixedController(0), policy_name="FCFS")
        standalone = FCFS(config).run(workload)
        _assert_schedule_equivalent(runtime.result, standalone)
        assert runtime.result.switching_overhead == runtime.result.context_switches * 2
        assert runtime.result.idle_time == 19

    def test_seeded_random_workloads_match_all_four_reference_policies(self) -> None:
        rng = np.random.default_rng(101)
        for case in range(30):
            count = int(rng.integers(1, 10))
            rows = [
                (
                    int(100 + index * 3),
                    int(rng.integers(0, 25)),
                    int(rng.integers(1, 16)),
                    int(rng.integers(1, 6)),
                )
                for index in range(count)
            ]
            config = SchedulerConfig(
                round_robin_quantum=int(rng.integers(1, 6)),
                switching_cost=int(rng.integers(0, 3)),
                lower_priority_number_is_higher_priority=(case % 2 == 0),
            )
            workload = _workload(rows, name=f"random-{case}")
            for action, policy_class in enumerate(POLICY_CLASSES):
                runtime = RuntimeSimulator(config).run(
                    workload,
                    FixedController(action),
                    policy_name=policy_class.name,
                )
                standalone = policy_class(config).run(workload)
                _assert_schedule_equivalent(runtime.result, standalone)


class TestCausalBoundary:
    def test_hidden_future_changes_do_not_change_prefix_observations_or_actions(self) -> None:
        # The first two processes and their observable history are byte-identical. Only
        # the third process's future arrival, burst and priority differ.
        early_future = _workload(
            [(1, 0, 2, 1), (2, 50, 1, 2), (3, 100, 3, 4)],
            name="hidden-a",
        )
        late_future = _workload(
            [(1, 0, 2, 1), (2, 50, 1, 2), (3, 200, 99, 0)],
            name="hidden-b",
        )
        config = SchedulerConfig(round_robin_quantum=4, switching_cost=1)
        first_controller = EncodedActionController(config.round_robin_quantum)
        second_controller = EncodedActionController(config.round_robin_quantum)
        first = RuntimeSimulator(config).run(early_future, first_controller)
        second = RuntimeSimulator(config).run(late_future, second_controller)

        first_prefix = [
            record
            for record in first.decisions
            if record.observation.arrived_count <= 2
        ]
        second_prefix = [
            record
            for record in second.decisions
            if record.observation.arrived_count <= 2
        ]
        assert [record.observation for record in first_prefix] == [
            record.observation for record in second_prefix
        ]
        assert [record.decision.action for record in first_prefix] == [
            record.decision.action for record in second_prefix
        ]
        assert [record.decision.state_index for record in first_prefix] == [
            record.decision.state_index for record in second_prefix
        ]
        assert len(first_prefix) == 2
        assert first_prefix[1].observation.current_time == 50
        assert len({record.decision.action for record in first_prefix}) > 1
        assert all(not hasattr(record.observation, "workload") for record in first_prefix)
        assert first.result.workload_fingerprint != second.result.workload_fingerprint

    def test_controller_interfaces_accept_observations_not_workloads(self) -> None:
        parameters = inspect.signature(RuntimeQController.select_action).parameters
        assert "observation" in parameters
        assert "workload" not in parameters
        assert "next_observation" in inspect.signature(
            RuntimeQController.observe_transition
        ).parameters
        assert "workload" not in inspect.signature(
            RuntimeQController.observe_transition
        ).parameters

    def test_state_encoder_rejects_a_complete_workload(self) -> None:
        encoder = RuntimeStateEncoder(quantum=4)
        with pytest.raises(ValidationError, match="RuntimeObservation"):
            encoder.encode(_workload([(1, 0, 5, 1)]))  # type: ignore[arg-type]

    def test_identical_observations_have_identical_encoded_states(self) -> None:
        config = SchedulerConfig()
        workload = _workload([(1, 0, 5, 1), (2, 0, 9, 2)])
        encoder = RuntimeStateEncoder(config.round_robin_quantum)
        observation = RuntimeSimulator(config).run(
            workload, RecordingFCFSController()
        ).decisions[0].observation
        assert encoder.encode(observation) == encoder.encode(observation)
        assert 0 <= encoder.encode(observation) < encoder.n_states


class TestSequentialQRuntime:
    def test_nonterminal_gamma_changes_the_update_target(self) -> None:
        zero_gamma = QLearningAgent(
            2,
            4,
            QLearningConfig(learning_rate=1.0, discount_factor=0.0),
            seed=1,
            initial_value=10.0,
        )
        positive_gamma = QLearningAgent(
            2,
            4,
            QLearningConfig(learning_rate=1.0, discount_factor=0.9),
            seed=1,
            initial_value=10.0,
        )
        for agent in (zero_gamma, positive_gamma):
            agent.update(1, 2, 2.0, next_state=None)
        zero_gamma.update(0, 1, 0.0, next_state=1, mask_unvisited_next_actions=True)
        positive_gamma.update(0, 1, 0.0, next_state=1, mask_unvisited_next_actions=True)
        assert zero_gamma.state_action_value(0, 1) == 0.0
        assert positive_gamma.state_action_value(0, 1) == pytest.approx(1.8)
        assert positive_gamma.state_action_visit_counts[0, 1] == 1

    def test_terminal_and_nonterminal_updates_increment_state_action_counts(self) -> None:
        agent = QLearningAgent(3, 4, QLearningConfig(), seed=3)
        agent.update(0, 2, -1.0, next_state=1)
        agent.update(0, 1, -2.0, next_state=None)
        counts = agent.state_action_visit_counts
        assert counts[0, 2] == 1
        assert counts[0, 1] == 1
        assert counts.sum() == 2
        assert not counts.flags.writeable
        assert agent.visited_actions(0) == (1, 2)
        assert agent.greedy_visited_action(2) is None

    def test_training_is_sequential_and_evaluation_is_read_only(self) -> None:
        workload = _workload(
            [(1, 0, 8, 3), (2, 1, 2, 1), (3, 2, 5, 4), (4, 6, 1, 2)]
        )
        config = SchedulerConfig(round_robin_quantum=2, switching_cost=1)
        encoder = RuntimeStateEncoder(config.round_robin_quantum)
        heuristic = CausalHeuristic(config.round_robin_quantum)
        controller = RuntimeQController(
            encoder,
            QLearningConfig(
                learning_rate=0.5,
                discount_factor=0.8,
                epsilon_start=1.0,
                epsilon_min=0.2,
                epsilon_decay_per_episode=0.9,
            ),
            seed=55,
            fallback_action=heuristic.choose_action,
        )
        simulator = RuntimeSimulator(config)
        trained = simulator.run(workload, controller, training=True, episode=4)
        assert trained.decision_count >= len(workload)
        assert trained.decisions[-1].terminal
        assert all(record.next_state_index is not None for record in trained.decisions[:-1])
        assert sum(record.reward for record in trained.decisions) == pytest.approx(
            -sum(outcome.waiting_time for outcome in trained.result.outcomes)
            / config.round_robin_quantum
        )
        assert controller.agent.visit_counts.sum() == trained.decision_count
        assert controller.agent.state_action_visit_counts.sum() == trained.decision_count

        q_before = np.array(controller.agent.q_table, copy=True)
        state_counts_before = np.array(controller.agent.visit_counts, copy=True)
        pair_counts_before = np.array(controller.agent.state_action_visit_counts, copy=True)
        evaluated = simulator.run(workload, controller, training=False)
        assert np.array_equal(q_before, controller.agent.q_table)
        assert np.array_equal(state_counts_before, controller.agent.visit_counts)
        assert np.array_equal(pair_counts_before, controller.agent.state_action_visit_counts)
        assert evaluated.policy_switch_count >= 0

    def test_unseen_eval_state_uses_causal_fallback_and_reports_unvisited_actions(self) -> None:
        config = SchedulerConfig()
        heuristic = CausalHeuristic(config.round_robin_quantum)
        controller = RuntimeQController(
            RuntimeStateEncoder(config.round_robin_quantum),
            QLearningConfig(),
            seed=77,
            fallback_action=heuristic.choose_action,
        )
        workload = _workload([(1, 0, 5, 1), (2, 0, 15, 2)])
        run = RuntimeSimulator(config).run(workload, controller, training=False)
        assert run.decisions
        assert run.decisions[0].decision.fallback_reason == "unseen-state-causal-fallback"
        assert run.decisions[0].decision.unvisited_action_count == len(ACTION_NAMES)

    def test_runtime_action_validation_rejects_invalid_action(self) -> None:
        with pytest.raises(ValidationError):
            RuntimeActionDecision(action=9)
