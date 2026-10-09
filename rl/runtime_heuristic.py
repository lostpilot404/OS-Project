"""Predeclared causal, non-RL runtime policy heuristic."""

from __future__ import annotations

from config import ACTION_FCFS, ACTION_PRIORITY, ACTION_ROUND_ROBIN, ACTION_SJF
from errors import ValidationError
from scheduler.runtime import RuntimeActionDecision, RuntimeObservation

__all__ = ["CausalHeuristic"]


class CausalHeuristic:
    """Deterministic adaptive rule using only the same arrived-work observations.

    The rule is declared before evaluation and has no fitted parameters:

    1. choose Round Robin if at least three processes are ready and mean ready age is at
       least one quantum (fairness under a visibly aged queue);
    2. otherwise choose Priority if at least two ready processes have differing priority;
    3. otherwise choose SJF if the largest remaining burst is at least twice the
       smallest (visible burst heterogeneity);
    4. otherwise choose FCFS.

    Lower priority number is treated as more important, matching the default scheduler
    configuration. The runtime simulator applies the actual configured priority ordering.
    """

    def __init__(self, quantum: int) -> None:
        if isinstance(quantum, bool) or not isinstance(quantum, int) or quantum < 1:
            raise ValidationError(f"quantum must be an integer >= 1, got {quantum!r}")
        self.quantum = quantum

    def choose_action(self, observation: RuntimeObservation) -> int:
        """Return a policy index based only on the immutable observation."""
        if not isinstance(observation, RuntimeObservation):
            raise ValidationError("CausalHeuristic requires RuntimeObservation")
        ready = observation.ready_processes
        if len(ready) >= 3 and observation.mean_ready_wait >= self.quantum:
            return ACTION_ROUND_ROBIN
        priorities = {process.priority for process in ready}
        if len(ready) >= 2 and len(priorities) > 1:
            return ACTION_PRIORITY
        remaining = [process.remaining_burst for process in ready]
        if len(remaining) >= 2 and max(remaining) >= 2 * min(remaining):
            return ACTION_SJF
        return ACTION_FCFS

    def select_action(
        self,
        observation: RuntimeObservation,
        *,
        training: bool = False,
        episode: int = 0,
    ) -> RuntimeActionDecision:
        """Adapt to current work while conforming to the runtime controller protocol."""
        del training, episode
        return RuntimeActionDecision(
            action=self.choose_action(observation),
            epsilon=0.0,
            explored=False,
            unvisited_action_count=0,
        )

    def observe_transition(
        self,
        decision: RuntimeActionDecision,
        reward: float,
        next_observation: RuntimeObservation | None,
        *,
        terminal: bool,
        training: bool,
    ) -> None:
        """No-op: the predeclared heuristic has no learning updates."""
        del decision, reward, next_observation, terminal, training
