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

import hashlib
import math
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
    "KNOWN_STATE_VARIABLES",
    "SchedulerConfig",
    "WorkloadFamilyConfig",
    "StateConfig",
    "RewardConfig",
    "QLearningConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "build_default_config",
    "derive_seed",
]


def derive_seed(base_seed: int, *components: int) -> int:
    """Derive a reproducible child seed from a base seed and integer components.

    Every random draw in the project is seeded through this function, so a run is fully
    determined by its master seeds plus the position of the draw (for example
    ``derive_seed(training_seed, family_index, episode)``).

    The derivation uses SHA-256 rather than Python's built-in ``hash``, which is
    randomised per process, and rather than arithmetic on the seed, which would make
    neighbouring components produce neighbouring streams.

    Args:
        base_seed: Master seed of the run.
        *components: Position identifiers of this particular draw.

    Returns:
        A non-negative 63-bit integer usable as ``numpy.random.default_rng`` seed.
    """
    payload = ":".join(str(int(value)) for value in (base_seed, *components))
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


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
#:
#: All seven are *pre-execution* workload characteristics: CPU utilisation, queue
#: length and waiting time -- named as candidates by the project brief -- are
#: *measured* quantities, so before a workload runs they are zero, and using values
#: measured during execution would leak the outcome of the decision being taken.
#: Their pre-execution analogues are used instead; see ``docs/DESIGN_AND_CHOICES.md``.
KNOWN_STATE_VARIABLES: Tuple[str, ...] = (
    "burst_profile",
    "burst_dispersion",
    "long_job_share",
    "arrival_concentration",
    "offered_load",
    "priority_spread",
    "priority_burst_alignment",
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
    times, burst times and priorities are drawn as described in
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
        long_burst_min: Lower bound (inclusive) of the long mode when
            ``burst_distribution == "bimodal"``.  ``0`` means "unset", in which case the
            long mode starts at ``short_burst_max + 1``.
        arrival_pattern: ``"uniform"`` for independent arrival times drawn from
            ``[0, arrival_window]``, ``"poisson"`` for a Poisson process with rate
            ``arrival_rate`` per time unit, or ``"batch_head"`` for a bimodal burst
            distribution whose long-mode jobs are released at the head of the window
            (``[0, head_window]``) while the short-mode jobs stream in over
            ``[0, arrival_window]``.
        arrival_window: Width of the arrival window for the uniform and batch-head
            patterns.
        head_window: Width of the release window of the long-mode jobs when
            ``arrival_pattern == "batch_head"``.
        arrival_rate: Mean number of arrivals per time unit for the Poisson pattern.
        priority_pattern: ``"uniform"`` for priorities drawn from
            ``[priority_min, priority_max]``, ``"high_priority_skewed"`` for a mixture
            that draws ``high_priority_fraction`` of the priorities from
            ``[priority_min, high_priority_cutoff]``, or ``"burst_aligned"`` to assign
            priorities by burst rank (the shortest jobs receive the highest priority).
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
    long_burst_min: int = 0
    arrival_pattern: str = "uniform"
    arrival_window: int = 20
    head_window: int = 2
    arrival_rate: float = 0.5
    priority_pattern: str = "uniform"
    priority_min: int = 1
    priority_max: int = 5
    high_priority_cutoff: int = 2
    high_priority_fraction: float = 0.7

    def long_mode_min(self) -> int:
        """Lower bound (inclusive) of the long mode of a bimodal burst distribution."""
        if self.long_burst_min:
            return self.long_burst_min
        return self.short_burst_max + 1

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
        if self.arrival_pattern not in ("uniform", "poisson", "batch_head"):
            raise ConfigurationError(
                f"family {self.name!r}: unknown arrival_pattern {self.arrival_pattern!r}"
            )
        if self.priority_pattern not in ("uniform", "high_priority_skewed", "burst_aligned"):
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
        if self.long_burst_min:
            if self.burst_distribution != "bimodal":
                raise ConfigurationError(
                    f"family {self.name!r}: long_burst_min is only meaningful for a "
                    "bimodal burst distribution"
                )
            if not self.short_burst_max < self.long_burst_min <= self.burst_time_max:
                raise ConfigurationError(
                    f"family {self.name!r}: long_burst_min={self.long_burst_min} must lie "
                    f"in ({self.short_burst_max}, {self.burst_time_max}]"
                )
        if self.arrival_window < 0:
            raise ConfigurationError(f"family {self.name!r}: arrival_window must be >= 0")
        if self.head_window < 0:
            raise ConfigurationError(f"family {self.name!r}: head_window must be >= 0")
        if self.arrival_pattern == "batch_head":
            if self.burst_distribution != "bimodal":
                raise ConfigurationError(
                    f"family {self.name!r}: the batch_head arrival pattern needs a bimodal "
                    "burst distribution (it releases the long mode at the head of the window)"
                )
            if self.head_window > self.arrival_window:
                raise ConfigurationError(
                    f"family {self.name!r}: head_window={self.head_window} must not exceed "
                    f"arrival_window={self.arrival_window}"
                )
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

    All variables are *pre-execution* workload characteristics: they are computed from
    the workload description alone, never from a scheduling result.

    Attributes:
        state_variables: Ordered tuple of variables that form the state.
        bins_per_variable: Number of bins per variable (3 -> low / medium / high).
        burst_profile_range: Value range of the median burst time.
        burst_dispersion_range: Value range of the burst-time coefficient of variation.
        long_job_share_range: Value range of the share of the total burst time that
            comes from long jobs.
        arrival_concentration_range: Value range of the fraction of jobs released at
            the modal arrival time.
        offered_load_range: Value range of the offered-load ratio.
        priority_spread_range: Value range of the priority standard deviation.
        priority_burst_alignment_range: Value range of the priority/burst correlation.
        long_burst_threshold: Burst time from which a job counts as *long* when the
            ``long_job_share`` variable is computed.  The default (16) is four times the
            configured Round-Robin quantum (4): a job at least this long occupies the
            CPU for several quanta, which is what makes preemption matter.
    """

    state_variables: Tuple[str, ...] = (
        "burst_profile",
        "burst_dispersion",
        "long_job_share",
        "arrival_concentration",
        "offered_load",
        "priority_spread",
        "priority_burst_alignment",
    )
    bins_per_variable: int = 3
    burst_profile_range: Tuple[float, float] = (1.0, 100.0)
    burst_dispersion_range: Tuple[float, float] = (0.0, 3.0)
    long_job_share_range: Tuple[float, float] = (0.0, 1.0)
    arrival_concentration_range: Tuple[float, float] = (0.0, 1.0)
    offered_load_range: Tuple[float, float] = (0.5, 60.0)
    priority_spread_range: Tuple[float, float] = (0.0, 2.0)
    priority_burst_alignment_range: Tuple[float, float] = (-1.0, 1.0)
    long_burst_threshold: int = 16

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
        if not isinstance(self.long_burst_threshold, int) or self.long_burst_threshold < 1:
            raise ConfigurationError(
                f"long_burst_threshold must be an integer >= 1, got {self.long_burst_threshold!r}"
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
            "burst_dispersion": self.burst_dispersion_range,
            "long_job_share": self.long_job_share_range,
            "arrival_concentration": self.arrival_concentration_range,
            "offered_load": self.offered_load_range,
            "priority_spread": self.priority_spread_range,
            "priority_burst_alignment": self.priority_burst_alignment_range,
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

    These weights are **identical to the ones declared before the first experiment of
    this project** and were deliberately *not* changed when the workload classes and
    the state representation were redesigned (see ``docs/DESIGN_AND_CHOICES.md`` §6):
    the redesign changed *where* the policies differ, not what "better" means.

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
            training episode.  The decay is deliberately slow (0.999): with 5400
            episodes over nine workload classes, a faster decay would drop epsilon to
            its floor after roughly 70 episodes *per class*, which is too few for the
            2-12 states each class occupies -- non-greedy actions would then receive so
            few updates that their Q-values could not resolve near-tied policies (the
            priority-aligned class contains two policies whose rewards differ by less
            than one Q-update step).  With 0.999, epsilon stays above 0.1 until episode
            ~2300 (~256 episodes per class) and still ends at the 0.05 floor, so every
            action receives a comparable number of updates in every visited state while
            the final policy is predominantly greedy.
        initial_value: Value every Q-entry starts at.  Chosen above the reward of a
            merely average policy (a reward of 0 means "as good as the average
            conventional policy"), which makes unexplored actions optimistic and spreads
            early exploration over all four policies.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_per_episode: float = 0.999
    initial_value: float = 0.05

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
        if not math.isfinite(self.initial_value):
            raise ConfigurationError(f"initial_value must be finite, got {self.initial_value!r}")


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings for the policy-selection agent.

    Attributes:
        episodes: Number of training episodes.  One episode is one workload.  5400
            episodes give every one of the nine classes 600 episodes; together with the
            slow epsilon decay this visits every state the classes occupy (the
            evaluation reports zero evaluation states that were never visited during
            training) and gives every action enough updates per state to resolve
            near-tied policies.
        seed: Master random seed of the training run.
        family_cycle: Workload families visited in round-robin order, one family per
            episode.  Must be a subset of the experiment's families.
    """

    episodes: int = 5400
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
        verification_repetitions: Number of independent workloads per family used by
            the *pre-training* class verification (see
            :mod:`experiments.verify_classes`).  The verification answers "which
            conventional policy does this workload class actually favour?" before the
            agent is trained, so that the training result can be judged against a
            measured property of the workload set rather than against an assumption.
        verification_seed: Master random seed of the verification workload stream.  It
            is disjoint from both the training and the evaluation streams.
    """

    repetitions: int = 10
    seed: int = 2024
    verification_repetitions: int = 25
    verification_seed: int = 777

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ConfigurationError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.seed < 0:
            raise ConfigurationError(f"seed must be >= 0, got {self.seed}")
        if self.verification_repetitions < 1:
            raise ConfigurationError(
                f"verification_repetitions must be >= 1, got {self.verification_repetitions}"
            )
        if self.verification_seed < 0:
            raise ConfigurationError(
                f"verification_seed must be >= 0, got {self.verification_seed}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration binding every component together."""

    name: str
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    families: Tuple[WorkloadFamilyConfig, ...] = ()
    state: StateConfig = field(default_factory=StateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
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
        seeds = {
            "training": self.training.seed,
            "evaluation": self.evaluation.seed,
            "verification": self.evaluation.verification_seed,
        }
        if len(set(seeds.values())) != len(seeds):
            raise ConfigurationError(
                "training, evaluation and verification seeds must all differ, so that "
                f"evaluation and verification workloads are unseen: {seeds}"
            )


def build_default_config() -> ExperimentConfig:
    """Build the frozen default configuration used for the reported experiment.

    The nine workload classes span the regimes in which the four conventional
    policies genuinely trade off (see ``docs/DESIGN_AND_CHOICES.md`` §9 and the
    pre-training verification in ``results/class_verification.json``):

    * batch-like conditions (all jobs released together), where non-preemptive SJF is
      optimal on the declared reward;
    * staggered conditions *without* long jobs, where Round Robin degenerates to FCFS
      and SJF still wins;
    * interactive conditions (a small batch of long background jobs released at the
      head of the window plus a stream of very short jobs), where preemption is what
      keeps the short jobs responsive and Round Robin wins the declared reward;
    * priority-sensitive conditions, where priority labels alone do not make Priority
      scheduling optimal under a mean-metric reward.

    Returns:
        A validated :class:`ExperimentConfig`.
    """
    families = (
        WorkloadFamilyConfig(
            name="short_batch",
            description=(
                "Short-job dominated, batch release: 15 jobs of 1-4 time units, all "
                "released together."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=4,
            arrival_pattern="uniform",
            arrival_window=0,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="short_stream",
            description=(
                "Short-job dominated, staggered release: 15 jobs of 1-4 time units "
                "arriving within a 30-unit window (no long jobs)."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=4,
            arrival_pattern="uniform",
            arrival_window=30,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="long_batch",
            description=(
                "Long-job dominated, batch release: 15 jobs of 20-50 time units, all "
                "released together."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=20,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=0,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="interactive",
            description=(
                "Interactive/response-sensitive: a small batch of long background jobs "
                "(60-100 units) released at the head of the window plus a stream of 18 "
                "very short interactive jobs (1-2 units) over 50 time units."
            ),
            num_processes=20,
            burst_distribution="bimodal",
            burst_time_min=1,
            burst_time_max=100,
            short_burst_max=2,
            short_burst_fraction=0.90,
            long_burst_min=60,
            arrival_pattern="batch_head",
            arrival_window=50,
            head_window=2,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="interactive_sparse",
            description=(
                "Interactive with a sparser arrival stream: the same job mix as "
                "'interactive' but the short jobs arrive over 110 time units (lighter "
                "load, slower interactive traffic)."
            ),
            num_processes=20,
            burst_distribution="bimodal",
            burst_time_min=1,
            burst_time_max=100,
            short_burst_max=2,
            short_burst_fraction=0.90,
            long_burst_min=60,
            arrival_pattern="batch_head",
            arrival_window=110,
            head_window=2,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="priority_aligned",
            description=(
                "Priority-sensitive with importance aligned to job size: uniform bursts "
                "1-50, and the shortest jobs carry the highest priority (priority "
                "assigned by burst rank)."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="burst_aligned",
            priority_min=1,
            priority_max=5,
        ),
        WorkloadFamilyConfig(
            name="priority_skewed",
            description=(
                "Priority-sensitive with skewed priority labels uncorrelated with job "
                "size: uniform bursts 1-50 and 70% of the priorities in the band 1-2."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="high_priority_skewed",
            high_priority_cutoff=2,
            high_priority_fraction=0.7,
        ),
        WorkloadFamilyConfig(
            name="quantum_sensitive",
            description=(
                "Quantum-sensitive/preemption-heavy: 20 jobs of 2-12 time units "
                "straddling the Round-Robin quantum (4), arriving within a 30-unit "
                "window; preemption is frequent and costly."
            ),
            num_processes=20,
            burst_distribution="uniform",
            burst_time_min=2,
            burst_time_max=12,
            arrival_pattern="uniform",
            arrival_window=30,
            priority_pattern="uniform",
        ),
        WorkloadFamilyConfig(
            name="mixed",
            description=(
                "Mixed: 15 jobs with bursts uniform over the full 1-50 range, arriving "
                "within a 20-unit window."
            ),
            num_processes=15,
            burst_distribution="uniform",
            burst_time_min=1,
            burst_time_max=50,
            arrival_pattern="uniform",
            arrival_window=20,
            priority_pattern="uniform",
        ),
    )
    config = ExperimentConfig(
        name="workload_aware_q_learning_scheduling",
        families=families,
        training=TrainingConfig(
            episodes=5400,
            seed=42,
            family_cycle=tuple(f.name for f in families),
        ),
        evaluation=EvaluationConfig(
            repetitions=10,
            seed=2024,
            verification_repetitions=25,
            verification_seed=777,
        ),
    )
    config.validate()
    return config
