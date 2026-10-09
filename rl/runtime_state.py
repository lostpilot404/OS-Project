"""Discrete, causal state encoder for online runtime scheduling."""

from __future__ import annotations

from statistics import median

from errors import ValidationError
from scheduler.runtime import RuntimeObservation

__all__ = ["RuntimeStateEncoder"]


class RuntimeStateEncoder:
    """Encode only an arrived-work :class:`RuntimeObservation` into a tabular state.

    Five coarse features deliberately keep the Q table small enough to cover during
    episodic simulation:

    * ready-queue size: 1, 2--3, or 4+;
    * completed work: 0, 1--3, or 4+ processes;
    * median remaining burst among ready work: at most one quantum, at most four
      quanta, or larger;
    * mean arrival age of currently ready jobs: at most one quantum, at most four
      quanta, or larger (this is not accumulated ready-queue waiting time);
    * observed priority spread among all processes that have arrived: zero or non-zero.

    The encoder has no workload argument and uses no total process count, unarrived
    process, future arrival, or completed-schedule metric. Encoding is deterministic:
    identical observations always yield the same state index.
    """

    RADICES = (3, 3, 3, 3, 2)

    def __init__(self, quantum: int) -> None:
        if isinstance(quantum, bool) or not isinstance(quantum, int) or quantum < 1:
            raise ValidationError(f"quantum must be an integer >= 1, got {quantum!r}")
        self.quantum = quantum
        total = 1
        for radix in self.RADICES:
            total *= radix
        self.n_states = total

    def feature_bins(self, observation: RuntimeObservation) -> tuple[int, int, int, int, int]:
        """Return the five documented bins, useful for audit and tests."""
        if not isinstance(observation, RuntimeObservation):
            raise ValidationError(
                "RuntimeStateEncoder accepts RuntimeObservation only; it cannot encode a Workload"
            )
        ready_count = len(observation.ready_processes)
        if ready_count < 1:
            raise ValidationError("a scheduling decision requires at least one ready process")
        ready_bin = 0 if ready_count == 1 else (1 if ready_count <= 3 else 2)

        completed_bin = (
            0 if observation.completed_count == 0
            else (1 if observation.completed_count <= 3 else 2)
        )

        median_remaining = float(
            median(process.remaining_burst for process in observation.ready_processes)
        )
        burst_bin = (
            0 if median_remaining <= self.quantum
            else (1 if median_remaining <= 4 * self.quantum else 2)
        )

        arrival_age_bin = (
            0 if observation.mean_ready_arrival_age <= self.quantum
            else (1 if observation.mean_ready_arrival_age <= 4 * self.quantum else 2)
        )
        priority_bin = 0 if observation.observed_priority_spread == 0 else 1
        return ready_bin, completed_bin, burst_bin, arrival_age_bin, priority_bin

    def encode(self, observation: RuntimeObservation) -> int:
        """Map a causal observation to a stable mixed-radix state index."""
        state = 0
        for value, radix in zip(self.feature_bins(observation), self.RADICES):
            state = state * radix + value
        return state

    def __call__(self, observation: RuntimeObservation) -> int:
        return self.encode(observation)
