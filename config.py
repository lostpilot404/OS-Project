"""Central configuration for the whole project.

Every tunable value used anywhere in the simulator, the Q-learning agent, the
experiment pipeline and the plots is declared **once**, here.  No module hard-codes an
experimental value: modules receive their configuration objects explicitly.

Implementation choices that the project brief left unspecified are recorded in
``docs/DESIGN_AND_CHOICES.md``; the values below are the frozen defaults for the
reported experiment.  They are arbitrary-but-declared (no tuning against results was
performed) and every one of them can be overridden by constructing a different
:class:`ExperimentConfig`.

The module deliberately imports nothing from the rest of the project except the
project's exception types, so it can never take part in an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from errors import ConfigurationError

__all__ = [
    "ACTION_FCFS",
    "ACTION_SJF",
    "ACTION_ROUND_ROBIN",
    "ACTION_PRIORITY",
    "ACTION_NAMES",
    "ADAPTIVE_LABEL",
    "SchedulerConfig",
    "WorkloadFamilyConfig",
    "StateConfig",
    "RewardConfig",
    "QLearningConfig",
    "QuantumControllerConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "build_default_config",
]

# --------------------------------------------------------------------------------------
# Action space (fixed by the project brief)
# --------------------------------------------------------------------------------------

ACTION_FCFS: int = 0
ACTION_SJF: int = 1
ACTION_ROUND_ROBIN: int = 2
ACTION_PRIORITY: int = 3

#: Human-readable names, indexed by action.  The order is fixed by the project brief.
ACTION_NAMES: Tuple[str, ...] = ("FCFS", "SJF", "Round Robin", "Priority")

#: Label used for the adaptive (Q-learning) scheduler in tables and plots.
ADAPTIVE_LABEL: str = "Adaptive (Q-Learning)"

#: State variables the encoder knows how to build (see :class:`StateConfig`).
KNOWN_STATE_VARIABLES: Tuple[str, ...] = (
    "burst_profile",
    "offered_load",
    "waiting_pressure",
    "queue_length",
)


# --------------------------------------------------------------------------------------
# Component configuration objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration shared by the four conventional schedulers.

    Attributes:
        round_robin_quantum: Time quantum used by Round Robin, in abstract time units.
        switching_cost: Time consumed by one context switch, in abstract time units.
            ``0`` means switching is costless and merely counted.
        lower_priority_number_is_higher_priority: When ``True`` (the default, and the
            ordering used throughout the documentation) priority ``1`` is the highest
            priority.  When ``False`` the ordering is inverted.
    """

    round_robin_quantum: int = 4
    switching_cost: int = 0
    lower_priority_number_is_higher_priority: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.round_robin_quantum, int) or self.round_robin_quantum < 1:
            raise ConfigurationError(
                "round_robin_quantum must be an integer >= 1, "
                f"got {self.round_robin_quantum!r}"
            )
        if not isinstance(self.switching_cost, int) or self.switching_cost < 0:
            raise ConfigurationError(
                f"switching_cost must be an integer >= 0, got {self.switching_cost!r}"
            )


@dataclass(frozen=True)
class WorkloadFamilyConfig:
    """Generation rule set for one workload condition.

    Every family generates exactly ``num_processes`` synthetic processes.  Arrival
    times and burst times are drawn as described in
    ``docs/DESIGN_AND_CHOICES.md``.

    Attributes:
        name: Short identifier used in tables, plots and file names.
        description: Human-readable description of the condition.
        num_processes: Number of processes generated per workload.
        burst_distribution: ``"uniform"`` for a single burst range, or ``"bimodal"``
            for a short-job mode and a long-job mode.
        burst_time_min: Lower bound (inclusive) of the burst-time range.
        burst_time_max: Upper bound (inclusive) of the burst-time range.
        short_burst_max: Upper bound (inclusive) of the short mode when
            ``burst_distribution == "bimodal"``.
        short_burst_fraction: Probability of drawing from the short mode.
        arrival_pattern: ``"uniform"`` for independent arrival times drawn from
            ``[0, arrival_window]``, or ``"poisson"`` for a Poisson process with rate
            ``arrival_rate`` per time unit.
        arrival_window: Width of the arrival window for the uniform pattern.
        arrival_rate: Mean number of arrivals per time unit for the Poisson pattern.
        priority_pattern: ``"uniform"`` for priorities drawn from
            ``[priority_min, priority_max]``, or ``"high_priority_skewed"`` for a
            mixture that draws ``high_priority_fraction`` of the priorities from
            ``[priority_min, high_priority_cutoff]``.
        priority_min: Lowest priority number used.
        priority_max: Highest priority number used.
        high_priority_cutoff: Upper bound (inclusive) of the high-priority band for the
            skewed pattern.
        high_priority_fraction: Probability of drawing from the high-priority band.
    """

    name: str
    description: str
    num_processes: int = 15
    burst_distribution: str = "uniform"
    burst_time_min: int = 1
    burst_time_max: int = 50
    short_burst_max: int = 5
    short_burst_fraction: float = 0.7
    arrival_pattern: str = "uniform"
    arrival_window: int = 20
    arrival_rate: float = 0.5
    priority_pattern: str = "uniform"
    priority_min: int = 1
    priority_max: int = 5
    high_priority_cutoff: int = 2
    high_priority_fraction: float = 0.7

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("workload family name must be a non-empty string")
        if self.num_processes < 1:
            raise ConfigurationError(
                f"family {self.name!r}: num_processes must be >= 1, got {self.num_processes}"
            )
        if self.burst_distribution not in ("uniform", "bimodal"):
            raise ConfigurationError(
                f"family {self.name!r}: unknown burst_distribution "
                f"{self.burst_distribution!r}"
            )
        if self.arrival_pattern not in ("uniform", "poisson"):
            raise ConfigurationError(
                f"family {self.name!r}: unknown arrival_pattern {self.arrival_pattern!r}"
            )
        if self.priority_pattern not in ("uniform", "high_priority_skewed"):
            raise ConfigurationError(
                f"family {self.name!r}: unknown priority_pattern {self.priority_pattern!r}"
            )
        if self.burst_time_min < 1 or self.burst_time_max < self.burst_time_min:
            raise ConfigurationError(
                f"family {self.name!r}: invalid burst range "
                f"[{self.burst_time_min}, {self.burst_time_max}]"
            )
        if self.burst_distribution == "bimodal" and not (
            self.burst_time_min <= self.short_burst_max < self.burst_time_max
        ):
            raise ConfigurationError(
                f"family {self.name!r}: short_burst_max={self.short_burst_max} must lie in "
                f"[{self.burst_time_min}, {self.burst_time_max})"
            )
        if not 0.0 <= self.short_burst_fraction <= 1.0:
            raise ConfigurationError(
                f"family {self.name!r}: short_burst_fraction must lie in [0, 1]"
            )
        if self.arrival_window < 0:
            raise ConfigurationError(f"family {self.name!r}: arrival_window must be >= 0")
        if self.arrival_pattern == "poisson" and self.arrival_rate <= 0.0:
            raise ConfigurationError(f"family {self.name!r}: arrival_rate must be > 0")
        if self.priority_min < 0 or self.priority_max < self.priority_min:
            raise ConfigurationError(
                f"family {self.name!r}: invalid priority range "
                f"[{self.priority_min}, {self.priority_max}]"
            )
        if self.priority_pattern == "high_priority_skewed":
            if not self.priority_min <= self.high_priority_cutoff <= self.priority_max:
                raise ConfigurationError(
                    f"family {self.name!r}: high_priority_cutoff must lie in "
                    f"[{self.priority_min}, {self.priority_max}]"
                )
            if not 0.0 <= self.high_priority_fraction <= 1.0:
                raise ConfigurationError(
                    f"family {self.name!r}: high_priority_fraction must lie in [0, 1]"
                )


@dataclass(frozen=True)
class StateConfig:
    """Discretisation of the RL state.

    Four state variables are implemented.  The default state uses three of them; see
    ``docs/DESIGN_AND_CHOICES.md`` for the documented decision to exclude
    ``queue_length``.  All variables are *pre-execution* workload characteristics: they
    are computed from the workload description alone, never from a scheduling result.

    Attributes:
        state_variables: Ordered tuple of variables that form the state.
        bins_per_variable: Number of bins per variable (3 -> low / medium / high).
        burst_profile_range: Value range of the burst-profile variable.
        offered_load_range: Value range of the offered-load variable.
        waiting_pressure_range: Value range of the waiting-pressure variable.
        queue_length_range: Value range of the optional queue-length variable.
    """

    state_variables: Tuple[str, ...] = ("burst_profile", "offered_load", "waiting_pressure")
    bins_per_variable: int = 3
    burst_profile_range: Tuple[float, float] = (1.0, 50.0)
    offered_load_range: Tuple[float, float] = (1.0, 60.0)
    waiting_pressure_range: Tuple[float, float] = (1.0, 400.0)
    queue_length_range: Tuple[float, float] = (0.0, 50.0)

    def __post_init__(self) -> None:
        if not self.state_variables:
            raise ConfigurationError("state_variables must not be empty")
        if len(set(self.state_variables)) != len(self.state_variables):
            raise ConfigurationError(f"state_variables contains duplicates: {self.state_variables}")
        unknown = [v for v in self.state_variables if v not in KNOWN_STATE_VARIABLES]
        if unknown:
            raise ConfigurationError(
                f"unknown state variables {unknown}; known: {list(KNOWN_STATE_VARIABLES)}"
            )
        if self.bins_per_variable < 2:
            raise ConfigurationError(
                f"bins_per_variable must be >= 2, got {self.bins_per_variable}"
            )
        for name, value_range in self.value_ranges().items():
            low, high = value_range
            if high <= low:
                raise ConfigurationError(
                    f"{name} range must satisfy high > low, got {value_range}"
                )

    def value_ranges(self) -> dict:
        """Return the configured value range of every implemented state variable."""
        return {
            "burst_profile": self.burst_profile_range,
            "offered_load": self.offered_load_range,
            "waiting_pressure": self.waiting_pressure_range,
            "queue_length": self.queue_length_range,
        }

    @property
    def n_states(self) -> int:
        """Total number of discrete states in the tabular Q-table."""
        return int(self.bins_per_variable ** len(self.state_variables))


@dataclass(frozen=True)
class RewardConfig:
    """Weights used by the composite reward function.

    ``reward = sum(weight_i * normalised_benefit_i) - sum(weight_j * normalised_cost_j)``

    where the benefits are CPU utilisation and throughput, and the costs are mean
    waiting time, mean turnaround time, mean response time and context switches per
    process.  Every metric is normalised by the mean value produced by the four
    conventional policies on the *same* workload, so a reward of ``0`` means "exactly
    as good as the average conventional policy".

    Attributes:
        weight_waiting_time: Cost weight of mean waiting time.
        weight_turnaround_time: Cost weight of mean turnaround time.
        weight_response_time: Cost weight of mean response time.
        weight_context_switches: Cost weight of context switches per process.
        weight_cpu_utilization: Benefit weight of CPU utilisation.
        weight_throughput: Benefit weight of throughput.
        reference_clip: Normalised metric values are clipped to this maximum.
    """

    weight_waiting_time: float = 0.30
    weight_turnaround_time: float = 0.25
    weight_response_time: float = 0.20
    weight_context_switches: float = 0.10
    weight_cpu_utilization: float = 0.075
    weight_throughput: float = 0.075
    reference_clip: float = 2.0

    def __post_init__(self) -> None:
        weights = self.cost_weights() + self.benefit_weights()
        if any(w < 0.0 for w in weights):
            raise ConfigurationError("reward weights must be >= 0")
        total = sum(weights)
        if abs(total - 1.0) > 1e-9:
            raise ConfigurationError(
                f"reward weights must sum to 1.0, got {total!r}"
            )
        if self.reference_clip <= 0.0:
            raise ConfigurationError(
                f"reference_clip must be > 0, got {self.reference_clip!r}"
            )

    def cost_weights(self) -> Tuple[float, ...]:
        """Weights of the metrics that are better when lower, in a fixed order."""
        return (
            self.weight_waiting_time,
            self.weight_turnaround_time,
            self.weight_response_time,
            self.weight_context_switches,
        )

    def benefit_weights(self) -> Tuple[float, ...]:
        """Weights of the metrics that are better when higher, in a fixed order."""
        return (self.weight_cpu_utilization, self.weight_throughput)


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters of the tabular Q-learning agent that selects the policy.

    Attributes:
        learning_rate: Step size alpha of the Q-learning update, in (0, 1].
        discount_factor: Discount gamma applied to the value of the next state.
        epsilon_start: Initial epsilon of the epsilon-greedy exploration.
        epsilon_min: Lower bound epsilon never decays below.
        epsilon_decay_per_episode: Multiplicative epsilon decay applied once per
            training episode.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_per_episode: float = 0.995

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ConfigurationError(
                f"learning_rate must lie in (0, 1], got {self.learning_rate!r}"
            )
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ConfigurationError(
                f"discount_factor must lie in [0, 1], got {self.discount_factor!r}"
            )
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ConfigurationError(
                "epsilon values must satisfy "
                "0 <= epsilon_min <= epsilon_start <= 1, got "
                f"min={self.epsilon_min!r}, start={self.epsilon_start!r}"
            )
        if not 0.0 < self.epsilon_decay_per_episode <= 1.0:
            raise ConfigurationError(
                "epsilon_decay_per_episode must lie in (0, 1], got "
                f"{self.epsilon_decay_per_episode!r}"
            )


@dataclass(frozen=True)
class QuantumControllerConfig:
    """Hyperparameters of the Round-Robin quantum controller (an extension).

    The policy-selection agent chooses *which* of the four policies runs.  Because the
    Round-Robin quantum is itself a scheduling parameter that depends on the workload,
    a second tabular Q-learning agent learns the quantum multiplier for the Round-Robin
    action.  Its episodes are single-step: it sees the same state as the policy agent
    and picks one multiplier for the whole workload.

    The controller is *disabled during evaluation* by default
    (``use_during_evaluation = False``), so the headline comparison uses classic Round
    Robin with the configured fixed quantum.  A separate, explicitly labelled secondary
    measurement reports the effect of enabling it.

    Attributes:
        enabled: Whether the controller is trained at all.
        use_during_evaluation: Whether the learned multiplier is applied to Round Robin
            during evaluation.  ``False`` keeps classic Round Robin in the main
            comparison.
        multipliers: Candidate quantum multipliers (``1.0`` == the configured quantum).
        learning_rate: Step size of the controller's Q-learning update.
        discount_factor: Discount factor of the controller.  Its episodes are
            single-step, so the factor does not affect the values; it is kept for the
            generic update rule and unit-tested.
        epsilon_start: Initial exploration rate of the controller.
        epsilon_min: Lower bound of the controller's exploration rate.
        epsilon_decay_per_episode: Multiplicative decay applied once per training
            episode.
    """

    enabled: bool = True
    use_during_evaluation: bool = False
    multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
    learning_rate: float = 0.15
    discount_factor: float = 0.5
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_per_episode: float = 0.995

    def __post_init__(self) -> None:
        if not self.multipliers:
            raise ConfigurationError("quantum multipliers must not be empty")
        if any(m <= 0.0 for m in self.multipliers):
            raise ConfigurationError("quantum multipliers must be > 0")
        if 1.0 not in tuple(float(m) for m in self.multipliers):
            raise ConfigurationError(
                "quantum multipliers must include 1.0 (the configured quantum)"
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise ConfigurationError(
                f"controller learning_rate must lie in (0, 1], got {self.learning_rate!r}"
            )
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ConfigurationError(
                f"controller discount_factor must lie in [0, 1], got {self.discount_factor!r}"
            )
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ConfigurationError("controller epsilon values must satisfy 0 <= min <= start <= 1")
        if not 0.0 < self.epsilon_decay_per_episode <= 1.0:
            raise ConfigurationError(
                "controller epsilon_decay_per_episode must lie in (0, 1]"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings for the policy-selection agent.

    Attributes:
        episodes: Number of training episodes.  One episode is one workload.
        seed: Master random seed of the training run.
        family_cycle: Workload families visited in round-robin order, one family per
            episode.  Must be a subset of the experiment's families.
    """

    episodes: int = 600
    seed: int = 42
    family_cycle: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ConfigurationError(f"episodes must be >= 1, got {self.episodes}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must be >= 0, got {self.seed}")


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings.

    Attributes:
        repetitions: Number of independent workloads generated per workload family.
        seed: Master random seed of the evaluation workload stream.  It is deliberately
            different from the training seed, so evaluation workloads are unseen.
    """

    repetitions: int = 10
    seed: int = 2024

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ConfigurationError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must be >= 0, got {self.seed}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration binding every component together."""

    name: str
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    families: Tuple[WorkloadFamilyConfig, ...] = ()
    state: StateConfig = field(default_factory=StateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    quantum_controller: QuantumControllerConfig = field(default_factory=QuantumControllerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    results_dir: str = "results"
    figures_dir: str = "figures"

    def validate(self) -> None:
        """Check cross-component consistency.

        Raises:
            ConfigurationError: If the configuration is inconsistent.
        """
        if not self.families:
            raise ConfigurationError("at least one workload family is required")
        names = [f.name for f in self.families]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"workload family names must be unique, got {names}")
        cycle = self.training.family_cycle or tuple(names)
        unknown = [name for name in cycle if name not in names]
        if unknown:
            raise ConfigurationError(
                f"training.family_cycle refers to unknown families: {unknown}"
            )
        burst_upper = max(f.burst_time_max for f in self.families)
        burst_bin_upper = self.state.burst_profile_range[1]
        if burst_upper > burst_bin_upper:
            raise ConfigurationError(
                f"state.burst_profile_range upper bound {burst_bin_upper} does not cover "
                f"the largest configured burst time {burst_upper}; the top bin would "
                "clamp too early"
            )
        if self.state.bins_per_variable ** len(self.state.state_variables) > 10_000:
            raise ConfigurationError(
                "state space larger than 10,000 states; the tabular Q-table would not be "
                "usable with the configured number of episodes"
            )
        if self.training.seed == self.evaluation.seed:
            raise ConfigurationError(
                "training and evaluation seeds must differ so evaluation workloads are unseen"
            )


def build_default_config() -> ExperimentConfig:
    """Build the frozen default configuration used for the reported experiment.

    Returns:
        A validated :class:`ExperimentConfig`.
    """
    families = (
        WorkloadFamilyConfig(
            name="short_jobs",
            description="Short-job-heavy: 80% of bursts drawn from 1-5 time units.",
            burst_distribution="bimodal",
            burst_time_min=1,
            burst_time_max=50,
            short_burst_max=5,
            short_burst_fraction=0.8,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="long_jobs",
            description="Long-job-heavy: only 20% of bursts drawn from 1-5 time units.",
            burst_distribution="bimodal",
            burst_time_min=1,
            burst_time_max=50,
            short_burst_max=5,
            short_burst_fraction=0.2,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="mixed",
            description="Uniform mix of bursts over the full 1-50 range.",
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="cpu_bursty",
            description="CPU-bound: uniformly long bursts (20-50) with dense arrivals.",
            burst_distribution="uniform",
            burst_time_min=20,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=15,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="poisson_arrivals",
            description="Poisson arrivals (rate 0.5 per time unit) with moderate bursts.",
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=20,
            arrival_pattern="poisson",
            arrival_rate=0.5,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="priority_skewed",
            description="Uniform bursts with 70% of processes in the high-priority band 1-2.",
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="high_priority_skewed",
            high_priority_cutoff=2,
            high_priority_fraction=0.7,
        ),
    )
    config = ExperimentConfig(
        name="workload_aware_q_learning_scheduling",
        families=families,
        training=TrainingConfig(
            episodes=600,
            seed=42,
            family_cycle=tuple(f.name for f in families),
        ),
        evaluation=EvaluationConfig(repetitions=10, seed=2024),
    )
    config.validate()
    return config
