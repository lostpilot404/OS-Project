# Offline Workload-Aware CPU Policy Selection with Tabular Q-Learning

A reproducible single-CPU scheduling study that uses tabular Q-learning to choose one
conventional policy for a complete workload. **The selector makes one batch/offline
choice before execution; it is not a runtime OS scheduler and does not switch policies
while a workload runs.**

## Objective and scope

The objective is to measure whether a workload-aware selector can choose among **FCFS,
SJF, Round Robin, and Priority** under declared synthetic workload conditions, using one
consistent reward and paired evaluation against all four policies. Every configured
workload is a finite process list containing process IDs, arrival times, CPU burst times,
and priorities.

The complete list is available at decision time. In particular, state features include
known burst times and even future arrival times from the generated batch. That information
is valid for this offline experiment, but would not be available to a conventional online
OS scheduler. The simulation has one CPU and is not connected to a real operating system.
There is no process-by-process policy switching, runtime hook, learned Round-Robin quantum,
or second controller in the reported experiment.

## Architecture

```text
config.py                 Frozen experiment, scheduler, workload, state, reward and RL settings
workload/
  models.py               Process, complete workload, schedule trace and outcomes
  generator.py            Seeded synthetic workload generation
scheduler/
  fcfs.py                 Non-preemptive First-Come, First-Served
  sjf.py                  Non-preemptive Shortest Job First
  round_robin.py          Preemptive Round Robin with fixed quantum 4
  priority.py             Non-preemptive static Priority
rl/
  state.py                Pre-execution features and mixed-radix discretization
  reward.py               One reward against exactly the four conventional policies
  q_learning.py           Tabular action values and epsilon-greedy exploration
  adaptive.py             Offline selector: one policy choice per complete workload
evaluation/
  metrics.py              Six required scheduling metrics and derived reward input
  comparison.py           Paired comparisons, variability and unique-workload summaries
experiments/
  train.py                One terminal workload decision and Q update per episode
  evaluate.py             Read-only paired evaluation of one independently trained model
  run_experiment.py       Multi-seed run, artifacts, metadata and generated report
  report.py               Results report and README tables rendered from artifacts
visualization/plots.py    Six optional figures; imported only when figures are requested
tests/                     Unit, independent-metric, reproducibility and end-to-end tests
results/                   Tracked machine-readable outputs and generated results report
figures/                   Tracked figures from the default experiment
```

## Conventional scheduling policies

The action order is fixed and is also the deterministic tie order: **FCFS, SJF, Round
Robin, Priority**.

| Policy | Simulation rule |
|---|---|
| FCFS | Non-preemptive; among ready processes, earliest arrival then lowest PID. |
| SJF | Non-preemptive; shortest burst, then earliest arrival, then lowest PID. SRTF is not implemented. |
| Round Robin | FIFO ready queue; each turn runs for at most the configured quantum (4 time units by default). Arrivals at or before the end of a turn are enqueued before a preempted process. |
| Priority | Non-preemptive static priority; priority 1 is highest by default, then arrival time and PID. |

Tie-breaking is deterministic. Context switches count changes of the running PID: the first
dispatch is not a switch, and a process that resumes without another PID running is not a
switch. An optional nonzero switching cost is configurable, but the reported default is
zero.

## RL formulation: a one-step episodic selector

One training episode is one complete workload:

1. Compute four pre-execution workload features from the complete process list.
2. Encode the features as one of **81 discrete states**.
3. Choose one action using epsilon-greedy action selection.
4. Run that conventional policy to completion. Run the four conventional policies on the
   same workload to calculate the reward reference.
5. Apply one terminal tabular Q update. There is no next workload state in the episode.

This is single-step episodic tabular Q-learning, equivalently a contextual policy-selection
problem. It is not sequential scheduling control. With terminal transitions,
`Q(s,a) ← Q(s,a) + α [r − Q(s,a)]`; the configured discount factor γ has no effect in this
experiment because `next_state=None` is used for every training update.

### State and decision-time availability

| State variable | Definition from the full workload | Default range and scale |
|---|---|---|
| `burst_profile` | Conventional median CPU burst | 1–50, logarithmic |
| `burst_dispersion` | Population standard deviation of bursts divided by their mean (coefficient of variation) | 0–1.5, linear |
| `offered_load` | Total burst time divided by `max(1, latest_arrival − earliest_arrival)` | 0.5–60, logarithmic |
| `priority_spread` | Population standard deviation of process priorities | 0–2, linear |

These values use only the generated workload description, not a simulated schedule or
metric. The full workload includes future arrivals, so this state is appropriate only for
the documented batch/offline setting. CPU utilization, ready-queue length, and waiting time
are outcomes of execution and are not treated as pre-execution observations.

Each variable uses 3 bins, giving `3⁴ = 81` states and a Q-table of shape **81 × 4 = 324
values per training seed**. The default interior edges are:

| Variable | Interior edges | Binning |
|---|---:|---|
| `burst_profile` | 3.684031, 13.572088 | logarithmic, equal multiplicative factors |
| `burst_dispersion` | 0.5, 1.0 | linear, equal widths |
| `offered_load` | 2.466212, 12.164404 | logarithmic, equal multiplicative factors |
| `priority_spread` | 0.666667, 1.333333 | linear, equal widths |

A value exactly on an edge enters the upper bin. Values outside the configured range are
clamped to the lowest or highest open-ended bin. The first listed state variable is the
most significant digit when the four bin indices are flattened into a Q-table row.

Every Q value starts at **0.0** (neutral, not optimistic). For an unvisited state, all four
actions remain tied; deterministic `argmax` selects the lowest action index, FCFS. Such a
decision is an **untrained fallback**, not evidence of learning. Evaluation rows mark
`state_seen_in_training`; state coverage and fallback counts are reported explicitly.

Exploration uses `ε = max(ε_min, ε_start × decay^episode)` with default ε start 1.0,
minimum 0.05, and multiplicative decay 0.995. With probability ε the agent samples
uniformly from the four actions; otherwise it selects the greedy action. The reported
exploration rate is the observed fraction of episodes that used the random-action branch;
it is not the fraction of actions differing from greedy. Evaluation uses ε = 0, is
repeatable, and is checked not to mutate either Q values or visit counts.

## Reward and comparison objective

For each metric `m`, divide the chosen policy's value by the arithmetic mean across exactly
**FCFS, SJF, Round Robin, and Priority on the same workload**. Clip the ratio above at 2.0;
when the reference is zero, define the ratio as 1.0. Lower is better for mean waiting,
turnaround and response time and context switches per process. Higher is better for CPU
utilization and throughput. The one implementation used by both training and reporting is:

```text
reward = Σ cost_weight    × (1 − clipped(chosen / four-policy-mean))
       + Σ benefit_weight × (clipped(chosen / four-policy-mean) − 1)
```

| Reward term | Default weight |
|---|---:|
| Mean waiting time (cost) | 0.30 |
| Mean turnaround time (cost) | 0.25 |
| Mean response time (cost) | 0.20 |
| Context switches per process (cost) | 0.10 |
| CPU utilization (benefit) | 0.075 |
| Throughput (benefit) | 0.075 |

Weights sum to one. A reward of zero means the selected policy matches the four-policy
reference mean on every metric. A positive reward denotes a better weighted net under this
objective; it does not imply every metric improved or that all policies must score
non-positive. The reward is workload-relative, which avoids rewarding a policy merely
because a workload is easy. The reward itself is not claimed to be a universal OS objective.

## Metrics

All four baselines and the selector are measured on the same workload object. Definitions
are implemented in one place and independently checked against hand-computed schedules.

| Metric | Definition |
|---|---|
| Waiting time | Per-process `turnaround − burst`, averaged over processes. |
| Turnaround time | Per-process `completion − arrival`, averaged over processes. |
| Response time | Per-process `first execution − arrival`, averaged over processes. |
| CPU utilization | `100 × CPU busy time / total elapsed time` (percent). |
| Throughput | Completed processes / total elapsed time. |
| Context switches | Changes of the running process in the execution trace. |

The derived `context_switches_per_process` is also recorded and is the reward's normalized
context-switch input; it is not a seventh required metric. Idle time includes intervals
before the first process arrives. Empty workloads yield zero metrics. The generated
`results/report.md` and `results/metrics.csv` provide measured results and per-seed
variability; no headline result is transcribed manually here.

## Experiment design and checked-in results

The default experiment trains **five independent agents** with seeds 42–46, 1,200 terminal
episodes per seed, cycling through six declared workload conditions. Each evaluation run
uses the same independent evaluation seed stream (master seed 2024) and evaluates 30 unique
workloads per condition across all seven conditions. `poisson_arrivals` is excluded from
training and is a held-out evaluation condition. The pipeline checks fingerprint
separation, exact one-to-one workload pairing, deterministic greedy evaluation, and
Q-table/visit-count immutability.

Baseline metric rows are repeated per model seed to keep every selector decision explicitly
paired with all four baselines. This does **not** increase the unique-workload count. The
summary and report distinguish unique workloads, model-decision rows, and metric rows.

The tables below are generated from the tracked `results/summary.json`; the detailed report
is [`results/report.md`](results/report.md), and each CSV/JSON can be inspected independently.

<!-- BEGIN GENERATED RESULTS -->
**Run design:** 210 unique workloads, 1050 model decisions across 5 independently trained seeds (42, 43, 44, 45, 46); 30 workloads per condition. Repeated model decisions are not counted as additional workloads.

Mean ± sample SD. Baseline SD is across unique workloads; selector SD is across training-seed × workload pairs.

| Method | Waiting time | Turnaround time | Response time | CPU utilization (%) | Throughput | Context switches | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 123.39 ± 80.71 | 142.34 ± 91.49 | 123.39 ± 80.71 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| SJF | 88.10 ± 67.11 | 107.06 ± 77.80 | 88.10 ± 67.11 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| Round Robin | 167.57 ± 135.06 | 186.52 ± 145.75 | 19.38 ± 8.22 | 97.84 ± 6.19 | 0.08 ± 0.07 | 75.40 ± 40.91 | 210 |
| Priority | 122.22 ± 81.44 | 141.17 ± 92.25 | 122.22 ± 81.44 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| Offline Policy Selector (Q-Learning) | 88.66 ± 67.89 | 107.61 ± 78.54 | 88.11 ± 68.47 | 97.84 ± 6.17 | 0.08 ± 0.07 | 15.53 ± 4.04 | 1050 |

**Selected policies by workload condition** (counts across independent models):

| Condition | Unique workloads | Model decisions | FCFS | SJF | Round Robin | Priority | Unseen-state fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_jobs | 30 | 150 | 3 | 117 | 30 | 0 | 2 |
| long_jobs | 30 | 150 | 5 | 145 | 0 | 0 | 4 |
| mixed | 30 | 150 | 0 | 150 | 0 | 0 | 0 |
| cpu_bursty | 30 | 150 | 0 | 150 | 0 | 0 | 0 |
| priority_skewed | 30 | 150 | 4 | 144 | 0 | 2 | 0 |
| staggered_interactive | 30 | 150 | 0 | 0 | 150 | 0 | 0 |
| poisson_arrivals | 30 | 150 | 46 | 104 | 0 | 0 | 32 |

Full generated results, including per-seed decisions, held-out coverage, variability and reproducibility metadata: [`results/report.md`](results/report.md).
<!-- END GENERATED RESULTS -->

### Generated files

- `results/config.json` — complete configuration, Q/state details, exact software versions,
  platform metadata and dependency-lock checksum.
- `results/workloads.csv` — one row per unique generated workload, with state features.
- `results/training_history.json` — episode-level data for each training seed.
- `results/q_table.json` — all 81 state rows for each seed, including visit coverage.
- `results/metrics.csv` — paired baseline/selector metric rows.
- `results/decisions.csv` — one decision per workload per trained model, including state
  coverage and greedy/random-action metadata.
- `results/summary.json` and `results/report.md` — generated summaries, variability,
  workload reductions, per-condition selections, and held-out limitations.
- `figures/` — six optional plots from the default run.

## Installation and use

The fully resolved environment is recorded in `requirements.lock`; direct exact dependency
pins are also listed in `requirements.txt`. The checked-in run was produced with Python
3.11.2. For a reproducible local environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

Run tests and the full experiment:

```bash
python -m pytest
python main.py experiment
```

Skip the optional plotting dependency at import/use time while still writing all tables and
the report:

```bash
python main.py experiment --no-figures
```

Train only the configured independent seeds, without evaluation:

```bash
python main.py train
```

The CLI accepts `--results-dir` and `--figures-dir` for alternate output locations. A run
using the repository's default `results/` directory regenerates the marked result tables in
this README directly from `summary.json`.

## Reproducibility and limitations

All synthetic workloads and exploration streams use a SHA-256 child-seed derivation and
NumPy's `default_rng`. The exact dependencies are pinned in `requirements.lock`; the
runtime package versions and lock checksum are recorded in each generated run. Repeated
runs are byte-identical in the same software/platform environment; plot bytes may depend on
platform fonts and rendering libraries.

This is a synthetic single-CPU model: it omits I/O blocking, multicore execution, memory,
real traces, deadlines, priority aging, and OS-level overhead except for the optional
configured context-switch cost. The state space is coarse and many of its 81 states are not
visited during a training run. A held-out Poisson family probes limited generalization, but
its states may be unseen and use the declared FCFS tie fallback; results do not support a
claim of broad out-of-distribution generalization. `staggered_interactive` is a synthetic
training condition designed to represent one long-running task plus staggered short jobs;
it makes a useful contrast for the fixed reward but is not a real trace or independent
external validation set. All conclusions are limited to the declared workload generator,
state discretization, scheduler variants, and reward.

For implementation rationale, exact workload parameters, state equations, boundary
conventions, and further limitations, see [`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md).
