"""The RL state: a discretised, pre-execution profile of the workload.

State variables
---------------
The project brief lists CPU utilisation, queue length, burst-time characteristics and
waiting-time characteristics as candidate state information.  Because the agent decides
*before* a workload runs (one decision per workload -- see ``docs/DESIGN_AND_CHOICES.md``),
utilisation, queue length and waiting time are all zero or trivial at the decision point,
and values measured during execution would leak the outcome of the very decision being
taken.  The state therefore uses seven pre-execution workload characteristics that
correspond to those candidates and that separate the regimes in which the four
conventional policies trade off:

===================================  ==================================================
Variable                             Definition (see :func:`observe_workload_state`)
===================================  ==================================================
``burst_profile``                    Median burst time: the scale of the jobs relative to
                                     the Round-Robin quantum.
``burst_dispersion``                 Coefficient of variation of the burst times
                                     (standard deviation / mean): whether jobs are of
                                     similar size or a mixture of short and long jobs.
``long_job_share``                   Share of the *total burst time* contributed by long
                                     jobs (burst at least
                                     :attr:`config.StateConfig.long_burst_threshold`):
                                     how much of the demand is concentrated in jobs
                                     that occupy the CPU for several quanta, which is
                                     what makes preemption pay off or not.
``arrival_concentration``            Fraction of jobs released at the modal arrival time:
                                     1 for a batch release, small for a staggered
                                     (interactive) stream.  This is the variable that
                                     separates "all jobs available at once" -- where
                                     non-preemptive SJF is optimal -- from "jobs keep
                                     arriving while long jobs run" -- where Round
                                     Robin's preemption is what keeps short jobs
                                     responsive.
``offered_load``                     Total burst time divided by the arrival span: the
                                     sustained service demand per unit of arrival span,
                                     i.e. the anticipated CPU utilisation pressure.
``priority_spread``                  Standard deviation of the process priorities: how
                                     strongly the workload distinguishes priorities.
``priority_burst_alignment``         Pearson correlation between the priority number and
                                     the burst time: +1 means the priority order agrees
                                     with the shortest-job-first order (importance
                                     tracks size), -1 means it is inverted, 0 means the
                                     labels are uninformative about size.
===================================  ==================================================

The first four variables were chosen from the *declared generation rules* of the
workload classes (before any experiment was run) because they are the dimensions along
which the classes differ and along which the reward-optimal policy changes; the
pre-training verification (``experiments/verify_classes.py``) then measures, on
held-out workloads, that the reward-optimal action is indeed a consistent function of
the encoded state (see ``results/class_verification.json``).

Discretisation
--------------
Every variable is discretised into ``bins_per_variable`` bins.  Quantities whose
interesting variation is proportional (burst times, load ratios) are binned on a
logarithmic scale, where each bin spans the same *factor*; bounded dimensionless
quantities (dispersion, shares, concentrations, correlations) are binned linearly.  The
lowest bin is open at the bottom and the highest bin is open at the top, so no value can
ever fall outside the state space.

The state space is the Cartesian product of the per-variable bins, flattened with the
first configured variable as the most significant digit of a mixed-radix index.

No scheduling outcome ever enters the state: :func:`observe_workload_state` accepts a
workload (and, optionally, the state configuration) and nothing else.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config import KNOWN_STATE_VARIABLES, StateConfig
from errors import ValidationError
from workload.models import Workload

__all__ = ["StateSnapshot", "LogBinner", "LinearBinner", "StateEncoder", "observe_workload_state"]


@dataclass(frozen=True)
class StateSnapshot:
    """Raw (undiscretised) state values observed at the decision point.

    Attributes:
        burst_profile: Median burst time of the workload.
        burst_dispersion: Coefficient of variation of the burst times.
        long_job_share: Share of the total burst time contributed by long jobs.
        arrival_concentration: Fraction of jobs released at the modal arrival time.
        offered_load: Total burst time per unit of arrival span.
        priority_spread: Standard deviation of the process priorities.
        priority_burst_alignment: Correlation between priority number and burst time.
    """

    burst_profile: float
    burst_dispersion: float
    long_job_share: float
    arrival_concentration: float
    offered_load: float
    priority_spread: float
    priority_burst_alignment: float

    def as_dict(self) -> Dict[str, float]:
        """Return the variable values keyed by variable name."""
        return asdict(self)


class _Binner:
    """Shared behaviour of the per-variable discretisers.

    A binner owns ``bins`` bins with ascending interior edges.  Bin ``k`` covers values
    from its lower edge (inclusive) to its upper edge (exclusive), the usual histogram
    convention: a value exactly on an interior edge belongs to the *upper* bin.  The
    lowest bin is open at the bottom and the highest bin is open at the top, so no value
    can fall outside the state space.
    """

    def __init__(self, low: float, high: float, bins: int) -> None:
        if high <= low:
            raise ValidationError(f"binner needs high > low, got low={low}, high={high}")
        if bins < 2:
            raise ValidationError(f"binner needs at least 2 bins, got {bins}")
        self._low = low
        self._high = high
        self._bins = bins
        self._edges: Tuple[float, ...] = tuple(self._interior_edges())

    @property
    def low(self) -> float:
        """Lower bound of the configured value range."""
        return self._low

    @property
    def high(self) -> float:
        """Upper bound of the configured value range."""
        return self._high

    @property
    def bins(self) -> int:
        """Number of bins."""
        return self._bins

    @property
    def interior_edges(self) -> Tuple[float, ...]:
        """Ascending interior bin edges (``bins - 1`` values)."""
        return self._edges

    def _interior_edges(self) -> List[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def index(self, value: float) -> int:
        """Return the bin index of ``value``.

        Values below ``low`` fall into bin 0, values at or above ``high`` fall into the
        last bin, and a value exactly on an interior edge falls into the upper bin.

        Raises:
            ValidationError: If ``value`` is not a real number.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"state value must be a number, got {value!r}")
        return bisect_right(self._edges, float(value))

    def lower_bounds(self) -> Tuple[float, ...]:
        """Lower bound of every bin; the first bin extends downwards without limit."""
        return (float("-inf"),) + self.interior_edges

    def upper_bound_of(self, bin_index: int) -> float:
        """Upper bound of the given bin; the last bin extends upwards without limit."""
        self._check_bin(bin_index)
        if bin_index == self._bins - 1:
            return float("inf")
        return self.interior_edges[bin_index]

    def _check_bin(self, bin_index: int) -> None:
        if not 0 <= bin_index < self._bins:
            raise ValidationError(
                f"bin index must lie in [0, {self._bins - 1}], got {bin_index}"
            )

    def describe(self) -> str:
        """Human-readable description of the bin edges."""
        edges = ", ".join(f"{edge:.4g}" for edge in self.interior_edges)
        return f"{self._bins} bins, edges at [{edges}] (lowest and highest bins are open)"


class LogBinner(_Binner):
    """Bin a positive quantity on a logarithmic scale.

    Each bin spans the same multiplicative factor, which suits quantities such as burst
    times whose interesting variation is proportional rather than additive.
    """

    def _interior_edges(self) -> List[float]:
        ratio = (self._high / self._low) ** (1.0 / self._bins)
        return [self._low * ratio**step for step in range(1, self._bins)]


class LinearBinner(_Binner):
    """Bin a quantity on a linear scale with equally wide bins."""

    def _interior_edges(self) -> List[float]:
        width = (self._high - self._low) / self._bins
        return [self._low + width * step for step in range(1, self._bins)]


#: Which binning scale each implemented state variable uses.
_VARIABLE_SCALES = {
    "burst_profile": "log",
    "burst_dispersion": "linear",
    "long_job_share": "linear",
    "arrival_concentration": "linear",
    "offered_load": "log",
    "priority_spread": "linear",
    "priority_burst_alignment": "linear",
}


class StateEncoder:
    """Turns a :class:`StateSnapshot` into a Q-table index and back.

    The encoder is the single owner of the state-space layout: the tabular Q-learning
    agent only ever sees integers, and the experiment layer only ever sees the raw
    snapshot, so a state cannot be interpreted differently in two places.
    """

    def __init__(self, config: StateConfig) -> None:
        """
        Args:
            config: Discretisation configuration.

        Raises:
            ValidationError: If the configuration is not a :class:`config.StateConfig`.
        """
        if not isinstance(config, StateConfig):
            raise ValidationError(
                f"config must be a StateConfig, got {type(config).__name__}"
            )
        self._config = config
        self._variables = tuple(config.state_variables)
        ranges = config.value_ranges()
        self._binners: Dict[str, _Binner] = {}
        for variable in self._variables:
            if variable not in KNOWN_STATE_VARIABLES:
                raise ValidationError(f"unknown state variable {variable!r}")
            low, high = ranges[variable]
            factory = LogBinner if _VARIABLE_SCALES[variable] == "log" else LinearBinner
            self._binners[variable] = factory(low, high, config.bins_per_variable)

    @property
    def config(self) -> StateConfig:
        """The discretisation configuration."""
        return self._config

    @property
    def variables(self) -> Tuple[str, ...]:
        """The state variables, in significance order (first is most significant)."""
        return self._variables

    @property
    def n_states(self) -> int:
        """Number of states in the state space."""
        return self._config.n_states

    @property
    def n_bins(self) -> int:
        """Number of bins per variable."""
        return self._config.bins_per_variable

    def binner(self, variable: str) -> _Binner:
        """Return the binner of one variable.

        Raises:
            ValidationError: If the variable is not part of the state.
        """
        if variable not in self._binners:
            raise ValidationError(
                f"variable {variable!r} is not part of the state; state uses "
                f"{list(self._variables)}"
            )
        return self._binners[variable]

    def encode(self, snapshot: StateSnapshot) -> int:
        """Encode a snapshot as a Q-table index in ``[0, n_states)``.

        Raises:
            ValidationError: If ``snapshot`` is not a :class:`StateSnapshot`.
        """
        if not isinstance(snapshot, StateSnapshot):
            raise ValidationError(
                f"snapshot must be a StateSnapshot, got {type(snapshot).__name__}"
            )
        values = snapshot.as_dict()
        index = 0
        for variable in self._variables:
            index = index * self.n_bins + self._binners[variable].index(values[variable])
        return index

    def decode(self, index: int) -> Dict[str, int]:
        """Return the bin index of every variable for a flat state index.

        Raises:
            ValidationError: If the index is outside the state space.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError(f"state index must be an integer, got {index!r}")
        if not 0 <= index < self.n_states:
            raise ValidationError(
                f"state index must lie in [0, {self.n_states - 1}], got {index}"
            )
        bins_by_variable: Dict[str, int] = {}
        remainder = index
        for variable in reversed(self._variables):
            bins_by_variable[variable] = remainder % self.n_bins
            remainder //= self.n_bins
        return {variable: bins_by_variable[variable] for variable in self._variables}

    def describe(self) -> Dict[str, str]:
        """Return the bin-edge description of every state variable."""
        return {variable: self._binners[variable].describe() for variable in self._variables}

    def __repr__(self) -> str:
        return (
            f"StateEncoder(variables={list(self._variables)}, "
            f"bins={self.n_bins}, n_states={self.n_states})"
        )


def observe_workload_state(
    workload: Workload, state_config: Optional[StateConfig] = None
) -> StateSnapshot:
    """Compute the pre-execution state of a workload.

    The snapshot is a pure function of the workload description and of the state
    configuration: it uses no scheduling result, no metric and no policy, so it cannot
    leak information about the outcome of a scheduling decision.

    Args:
        workload: The workload the agent must choose a policy for.
        state_config: State configuration supplying the long-job threshold.  ``None``
            uses the default :class:`config.StateConfig`.

    Returns:
        The raw state values.

    Raises:
        ValidationError: If ``workload`` is not a :class:`workload.models.Workload` or
            the state configuration is invalid.
    """
    if not isinstance(workload, Workload):
        raise ValidationError(f"workload must be a Workload, got {type(workload).__name__}")
    if state_config is None:
        state_config = StateConfig()
    if not isinstance(state_config, StateConfig):
        raise ValidationError(
            f"state_config must be a StateConfig, got {type(state_config).__name__}"
        )
    threshold = state_config.long_burst_threshold

    if workload.size:
        bursts = [process.burst_time for process in workload.processes]
        arrivals = [process.arrival_time for process in workload.processes]
        priorities = [process.priority for process in workload.processes]
        mean_burst = workload.mean_burst_time
        burst_dispersion = float(_population_std(bursts) / mean_burst) if mean_burst else 0.0
        total_burst = workload.total_burst_time
        long_burst = sum(burst for burst in bursts if burst >= threshold)
        long_job_share = float(long_burst / total_burst) if total_burst else 0.0
        arrival_concentration = float(max(Counter(arrivals).values()) / len(arrivals))
        priority_spread = float(_population_std(priorities))
        alignment = float(_pearson_correlation(priorities, bursts))
    else:
        burst_dispersion = 0.0
        long_job_share = 0.0
        arrival_concentration = 0.0
        priority_spread = 0.0
        alignment = 0.0

    # A span of zero means every process arrived together; a span of one unit keeps the
    # ratio finite and makes the batch case the maximum-load case, as documented.
    span = max(1, workload.arrival_span)

    return StateSnapshot(
        burst_profile=float(workload.median_burst_time),
        burst_dispersion=burst_dispersion,
        long_job_share=long_job_share,
        arrival_concentration=arrival_concentration,
        offered_load=float(workload.total_burst_time / span),
        priority_spread=priority_spread,
        priority_burst_alignment=alignment,
    )


def _population_std(values: Sequence[float]) -> float:
    """Return the population standard deviation of ``values`` (0 for a single value)."""
    count = len(values)
    if count == 0:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return variance**0.5


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the Pearson correlation of two equally long sequences.

    The correlation is ``0.0`` when either sequence has zero variance (a constant
    priority list or a constant burst list carries no alignment information), which is
    the documented convention for the ``priority_burst_alignment`` state variable.
    """
    count = len(xs)
    if count == 0:
        return 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / count
    variance_x = sum((x - mean_x) ** 2 for x in xs) / count
    variance_y = sum((y - mean_y) ** 2 for y in ys) / count
    if variance_x <= 0.0 or variance_y <= 0.0:
        return 0.0
    return covariance / (variance_x * variance_y) ** 0.5
