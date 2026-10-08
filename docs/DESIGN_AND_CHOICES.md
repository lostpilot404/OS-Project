# Design and experimental protocol

This document defines the project as an **offline workload-aware policy selector**. It
records the scope, simulation rules, workload distributions, state and reward mathematics,
training/evaluation design, and reproducibility contract used for the tracked results.
The complete configuration is also serialized to `results/config.json`.

## 1. Objective and boundary of the claim

The objective is to select one of four conventional single-CPU scheduling policies for a
complete synthetic workload and compare that selection with all four policies on identical
inputs. The decision is made once, before any process is scheduled. The selected policy
runs the entire batch; policies do not change during execution.

The agent sees the complete generated process table. Its pre-execution features include
actual burst lengths and all future arrival times in that table. This is deliberate for a
batch/offline experiment; those features would not be available to an ordinary online OS
scheduler. The implementation is a simulator and has no kernel integration, runtime
scheduler hooks, multicore execution, or claim of production use.

The learned component is single-step episodic tabular Q-learning (a contextual policy
selection problem): one workload state, one action, one terminal reward, and one update.
It is not sequential RL control over a running process queue.

## 2. Conventional schedulers

Actions are ordered **FCFS, SJF, Round Robin, Priority**, matching the Q-table columns and
tie-breaking order.

| Policy | Selection and execution rule |
|---|---|
| FCFS | Non-preemptive. Select the ready process with smallest `(arrival_time, pid)`. |
| SJF | Non-preemptive. Select the smallest `(burst_time, arrival_time, pid)`. Shortest Remaining Time First is not included. |
| Round Robin | Preemptive FIFO queue. Each turn runs for at most the configured quantum; default 4 time units. Arrivals at or before a slice end enter the ready queue before the preempted process. |
| Priority | Non-preemptive static priority. Priority 1 is highest by default; ties use arrival time and PID. The configuration can reverse the priority-number sense. |

The non-preemptive policies share one ready-queue simulation engine. All scheduler ties are
resolved deterministically. The optional context-switch cost is zero in the reported run.

### Context switches and idle time

A context switch is a change of the running process ID in the execution trace. The first
dispatch is not a switch, nor is resuming the same process if no other PID ran. Consecutive
slices of one PID are merged for this definition. If a nonzero switching cost is configured,
it is counted as overhead, not CPU busy time.

CPU idle time is `total_elapsed_time − cpu_busy_time − switching_overhead`. It includes an
initial interval before the first arrival as well as idle gaps between arrivals. There is
no assumption that a workload begins with a process arriving at time zero.

## 3. Workload generation

The default workload set has 15 processes per instance. All time and burst values are
integer abstract time units. Priorities are integers, with 1 highest by default. The first
six conditions below are cycled during training; `poisson_arrivals` is held out from
training and used only for evaluation.

| Family | Burst generation | Arrival generation | Priority generation | Role |
|---|---|---|---|---|
| `short_jobs` | Bimodal: 80% from 1–5, otherwise 6–50 | Independent uniform integers 0–20 | Uniform 1–5 | Training |
| `long_jobs` | Bimodal: 20% from 1–5, otherwise 6–50 | Independent uniform integers 0–20 | Uniform 1–5 | Training |
| `mixed` | Uniform integers 1–50 | Independent uniform integers 0–20 | Uniform 1–5 | Training |
| `cpu_bursty` | Uniform integers 20–50 | Independent uniform integers 0–15 | Uniform 1–5 | Training |
| `priority_skewed` | Uniform integers 1–50 | Independent uniform integers 0–20 | With probability 0.7 draw 1–2; otherwise uniform 1–5 | Training |
| `staggered_interactive` | One long burst drawn 25–50, followed by 14 short bursts drawn 1–3 | First arrival at 0; subsequent integer gaps uniform 4–6 | Fixed priority 1 | Training, synthetic contrast condition |
| `poisson_arrivals` | Uniform integers 1–20 | First arrival at 0; exponential inter-arrivals at rate 0.5, floored to integer times | Uniform 1–5 | Held-out evaluation only |

`staggered_interactive` represents one long-running job alongside staggered short jobs.
It is a synthetic training condition, not a real workload trace or independent validation
set. It makes response- and waiting-time tradeoffs visible under the declared fixed reward;
its inclusion does not modify the reward or force the learned action.

Every generator is deterministic for a family and seed. Process IDs are assigned in
creation order; workloads are stored by `(arrival_time, pid)`. The workload fingerprint is
a deterministic digest of the process table and is carried into every result row.

## 4. State, discretization, and coverage

The state uses four features computed before the selected scheduler runs:

| Feature | Definition | Default configured range | Scale |
|---|---|---:|---|
| `burst_profile` | Conventional median of process burst times | 1–50 | Logarithmic |
| `burst_dispersion` | Population standard deviation of burst times divided by their mean; 0 if mean is 0 | 0–1.5 | Linear |
| `offered_load` | Total burst time divided by `max(1, arrival_span)`, where `arrival_span = latest_arrival − earliest_arrival` | 0.5–60 | Logarithmic |
| `priority_spread` | Population standard deviation of priorities | 0–2 | Linear |

The full process table is available at the offline decision point, so these values use its
actual bursts and arrival times, including future arrivals. No schedule result, CPU
utilization, queue length, or waiting-time outcome enters the state.

Each variable is divided into three bins. Logarithmic variables use equally spaced edges
in log space; linear variables use equally wide bins. For the default ranges, the interior
edges are:

| Feature | Interior edges |
|---|---:|
| `burst_profile` | 3.684031, 13.572088 |
| `burst_dispersion` | 0.5, 1.0 |
| `offered_load` | 2.466212, 12.164404 |
| `priority_spread` | 0.666667, 1.333333 |

Bins are lower-inclusive and upper-exclusive; a value exactly on an interior edge goes to
the higher bin. The outer bins are open-ended and clamp values outside the configured
range. The first configured feature is the most significant digit in the mixed-radix
flattening. Therefore the default state space has `3⁴ = 81` states. Each independent agent
has a Q-table with **81 × 4 = 324 entries**. `results/q_table.json` records all 81 rows per
seed and each row's visit count. Evaluation decisions record whether their state was
visited during that model's training; per-seed and per-condition coverage is reported in
`results/report.md`.

## 5. Reward definition

For a workload `w`, let `reference_m(w)` be the arithmetic mean of metric `m` over exactly
**FCFS, SJF, Round Robin, and Priority**, all run on the same workload. For the policy
selected by the agent, define

```text
ratio_m = 1                                      if reference_m(w) = 0
          min(metric_m / reference_m(w), 2.0)    otherwise

reward = Σ cost metrics    weight_m × (1 − ratio_m)
       + Σ benefit metrics weight_m × (ratio_m − 1)
```

Cost metrics are mean waiting time, mean turnaround time, mean response time, and context
switches per process. Benefit metrics are CPU utilization and throughput. The default
weights are, respectively, 0.30, 0.25, 0.20, 0.10, 0.075, and 0.075; they sum to one.
The reward definition is implemented once in `rl/reward.py` and is used for every training
update. There is no alternate controller reward or substituted reference set.

A zero reward means the selected metrics equal their four-policy means metric-by-metric.
A positive reward indicates a better weighted net under this objective, not necessarily an
improvement in every metric. The sign does not imply that all individual policies must
score at or below zero. Normalized workload-local ratios make the scale less sensitive to
workload difficulty but do not make this hand-selected objective universal.

## 6. Q-learning and exploration

Each training episode schedules one complete workload. The scalar immediate reward is
terminal, so training uses

```text
Q(s, a) ← Q(s, a) + α [reward − Q(s, a)]
```

with default learning rate `α = 0.1`. The generic agent supports a nonterminal discounted
update for unit testing, but the experiment always passes `next_state=None`; therefore
`γ = 0.9` is recorded but unused in this project formulation.

Q-values initialize to **0.0**, a neutral value rather than optimistic initialization. The
action order is FCFS, SJF, Round Robin, Priority. Greedy ties resolve to the lowest action
index. Thus a never-visited state, whose four values remain tied at zero, uses FCFS as an
explicit deterministic fallback. It is counted and identified as untrained rather than
reported as a learned policy choice.

During training, with probability `ε`, the agent samples uniformly from the four actions;
otherwise it chooses the current greedy action. The schedule is
`ε(episode) = max(0.05, 1.0 × 0.995^episode)` by default. The recorded random-exploration
rate counts actual random-branch episodes. It is distinct from epsilon itself and from the
fraction of actions that happen to differ from greedy. Evaluation uses epsilon zero and is
read-only: both Q values and visit counts are compared before and after the evaluation.

## 7. Training and evaluation protocol

### Training

- Five independent training seeds: **42, 43, 44, 45, 46**.
- **1,200 one-workload terminal episodes per seed** (6,000 episodes total).
- The six training families above cycle in configuration order.
- Each workload seed is derived from the training master seed and episode index; the
  agent's random stream is derived separately.
- The held-out `poisson_arrivals` family is absent from every training cycle.

### Evaluation

- Evaluation workload master seed: **2024**, distinct from all training seeds.
- **30 independently seeded workload instances per family × 7 families = 210 unique
  workloads**.
- Each of the five trained agents evaluates the same 210 unique workload instances
  greedily. This gives 1,050 decision rows, not 1,050 unique workloads.
- Four baseline results are repeated per training seed so every learned decision has an
  explicit same-seed pairing. Therefore `metrics.csv` has 4,200 baseline rows plus 1,050
  selector rows (5,250 total); `workloads.csv` has 210 unique-workload rows.
- Training and evaluation fingerprints are checked for overlap. Within every
  `(training_seed, family, repetition)` key, the four baselines and selector must have five
  rows with one fingerprint. Baseline outputs are additionally checked to be identical
  across model seeds.
- The evaluator verifies greedy evaluation and Q-table/visit-count immutability.

The held-out Poisson condition is an explicit, limited test of feature-state overlap and
not a tuning set. No held-out result is used for training or reward/feature selection. When
a held-out evaluation state is unseen, the fallback is reported separately from trained
state decisions.

## 8. Metrics and independent verification

The six reported scheduling metrics are defined once in `evaluation/metrics.py` from a
validated schedule result:

| Metric | Definition |
|---|---|
| Waiting time | `turnaround_time − burst_time`, averaged per process. |
| Turnaround time | `completion_time − arrival_time`, averaged per process. |
| Response time | `first_execution_time − arrival_time`, averaged per process. |
| CPU utilization | `100 × cpu_busy_time / total_elapsed_time`. |
| Throughput | `completed_processes / total_elapsed_time`. |
| Context switches | Changes of the running process in the execution trace. |

`context_switches_per_process = context_switches / num_processes` is a derived value used
by the reward and recorded in the raw metrics file; it is not an additional required
metric. Empty workloads return zero for all metrics. CPU idle time includes time before
the first arrival. Tests verify all six metrics against independent hand calculations,
including idle time before the first process and a Round-Robin trace; the end-to-end
pipeline verifies that all five methods see the same workload fingerprints.

`Workload.percentile_burst_time(p)` uses the nearest-rank rule, with zero-based index
`max(0, ceil(p × n) − 1)`. The median feature is the conventional median, averaging the two
middle values when the number of processes is even.

## 9. Generated artifacts and result interpretation

The default runner writes:

- `results/config.json` — full configuration, bin edges, runtime versions, platform, and
  lockfile checksum;
- `results/workloads.csv` — one row per unique evaluation workload;
- `results/training_history.json` — full histories for each independent seed;
- `results/q_table.json` — all state/action values and training visit counts per seed;
- `results/metrics.csv` — paired baseline and selector metrics;
- `results/decisions.csv` — action, reward, state, training coverage, and epsilon metadata;
- `results/summary.json` — artifact-derived aggregates and variability;
- `results/report.md` — generated, auditable result tables and limitations;
- six optional PNG figures in `figures/`.

The default report distinguishes unique workloads from repeated model-decision and metric
rows, aggregates waiting/turnaround reductions from the reported means, shows selections
per condition and per seed, reports training/evaluation variability, and identifies
unseen-state fallbacks. The README result tables are generated from the same summary JSON.
No result numbers are manually entered into those generated tables.

The `staggered_interactive` and conventional training conditions demonstrate whether
learned action selection changes across covered workload states. The Poisson condition
provides limited held-out evidence; if states are not represented during training, the
fallback is not treated as a learned result. All conclusions remain specific to these
synthetic distributions and the stated reward.

## 10. Software reproducibility

- Direct pins: `requirements.txt`.
- Full resolved dependency pins: `requirements.lock`.
- The checked-in artifacts were produced with Python 3.11.2; runtime package versions and
  the lockfile SHA-256 are serialized with each run.
- Seeds use a SHA-256 derivation (`config.derive_seed`) and NumPy `default_rng`, avoiding
  Python's randomized built-in hash for experiment streams.
- `python main.py experiment` runs the multi-seed experiment. `--no-figures` writes all
  tables and reports without importing Matplotlib.
- `python -m pytest` runs the test suite.

Repeat runs are byte-identical in the same locked software/platform environment. Figure
bytes can differ across platforms because of fonts and rendering libraries.

## 11. Limitations and non-goals

The model is synthetic and single-core, with CPU-only bursts, no I/O blocking, memory,
deadlines, priority aging, real traces, or multicore interference. The agent sees the
complete process table, including future arrivals, so it does not establish feasibility
for an online scheduler. The four-variable, 81-state table is coarse; training covers only
a subset of states, and unseen states fall back to FCFS until updated. Poisson arrivals are
held out, but one held-out family is not broad external validation. `staggered_interactive`
is a deliberately declared synthetic contrast condition, not an external workload trace.
The hand-specified reward encodes one objective and can prefer different schedulers under
different mixes of waiting, turnaround, response, switches, utilization, and throughput.

Kernel integration, multicore scheduling, runtime policy switching, deep RL, workload
traces, and real-time or energy-aware objectives are outside this project.
