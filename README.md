# Event-Driven Runtime-Adaptive CPU Scheduling

This repository now contains a **single-CPU, event-driven runtime scheduler**. At every
ready-queue dispatch or Round-Robin quantum boundary, a controller chooses FCFS, SJF,
Round Robin, or Priority for the next segment of the same evolving workload. It preserves
one ready queue, remaining bursts, arrivals, RR FIFO/re-enqueue behavior, and per-process
context-switch accounting. The default command trains, validates, and tests a sequential
Q-learning controller on separate seeded workloads.

The older whole-workload policy selector remains available as a clearly separate **legacy
offline reference**. It is not the runtime scheduler and is not treated as a causal
baseline.

## Runtime architecture and causal boundary

```text
scheduler/runtime.py           Event-driven single-CPU environment and immutable observations
scheduler/{fcfs,sjf,round_robin,priority}.py
                               Preserved standalone fixed-policy baselines
rl/runtime_state.py             Five-feature, causal state encoder (162 tabular states)
rl/runtime_controller.py        Sequential Q-learning; visits, masking, deterministic fallback
rl/runtime_heuristic.py         Predeclared deterministic non-RL adaptive rule
experiments/runtime_config.py   Frozen seeds, split sizes, switch cost and Q settings
experiments/runtime_experiment.py
                               Training, validation, final test, paired metrics and reports
workload/models.py              Process/workload models and schedule-trace validation
results/runtime/                Reproducible runtime-experiment tables, audit trail and report
rl/adaptive.py                  Explicitly legacy offline whole-workload selector
experiments/run_experiment.py   Explicitly legacy offline result pipeline
```

The environment privately owns the complete event queue. At a decision epoch its
controller receives only a frozen `RuntimeObservation` containing:

- current simulated time;
- processes that have arrived and are currently ready, with their known burst,
  remaining burst, arrival time and priority;
- counts of arrived and completed processes;
- aggregates over **arrived work only** (observed mean burst and priority spread);
- mean age of the current ready queue and the previous selected policy.

The controller API accepts an observation, not a `Workload`; the state encoder likewise
accepts only that observation. Unarrived process IDs/attributes, future arrivals/bursts,
workload-wide summaries, and counterfactual schedule metrics are not passed to the
controller. Exact burst lengths for arrived processes are assumed known on arrival, matching
the information assumption already required by SJF. This is a simulator assumption, not a
claim about real operating-system burst prediction.

Causality tests change hidden future arrivals, bursts and priorities while keeping the
observable history identical; the corresponding observation/action prefixes remain
identical. The simulator computes rewards from actual waiting-time increments between
events; it does not pass hypothetical policy outcomes to the learner.

## Runtime execution semantics

- **FCFS:** smallest `(arrival_time, pid)`, non-preemptive.
- **SJF:** smallest `(remaining_burst, arrival_time, pid)`, non-preemptive.
- **Round Robin:** FIFO; execute `min(quantum, remaining_burst)`. Arrivals at or before
  the quantum endpoint are enqueued before the preempted process is put at the tail.
- **Priority:** static non-preemptive priority, lower number first by default, then arrival
  time and PID.
- A controller policy switch never resets or reconstructs the ready queue. A non-RR
  selection always runs its process to completion; RR can yield only at its quantum
  boundary.
- A context-switch cost is charged only when the running PID changes. The first dispatch
  and redispatch of the same PID are free. The default runtime experiment uses quantum 4
  and switch cost 1 time unit. The original standalone schedulers remain unchanged; a
  corrected trace validator recognizes that a switch interval can follow idle time spent
  waiting for an arrival.

The fixed-action runtime simulator is tested against each standalone implementation on
hand-built and seeded random traces (all six metric outcomes, context switches, idle time,
and trace slices).

## Online learning and predeclared heuristic

The causal encoder discretizes five current/history-only features into `3 × 3 × 3 × 3 × 2
= 162` states: ready-queue size (1, 2–3, 4+), completed count (0, 1–3, 4+), median ready
remaining burst (`<=q`, `<=4q`, `>4q`), mean ready age (`<=q`, `<=4q`, `>4q`), and whether
priority spread among arrived jobs is zero. It does not use total workload size or any
unarrived values.

A training **episode is a sequential trace**, not a whole-workload selection. The
controller repeatedly observes, chooses among the four policies, runs the next segment,
receives a nonterminal/terminal reward, and updates the tabular Q values. For an observed
waiting-time increment `ΔW_t`, reward is `r_t = -ΔW_t / max(1,q)`. At `gamma = 1`, the
undiscounted reward telescopes exactly to negative total waiting time divided by the
quantum. The default `gamma = 0.95` intentionally discounts later waiting increments; it
therefore affects nonterminal Q targets and is not a decorative parameter.

The update is `Q(s,a) += alpha * (r + gamma * max_known_a' Q(s',a') - Q(s,a))`; terminal
transitions omit the bootstrap. State-action visit counts are explicit. Training uses an
episode-level epsilon schedule and forces a random action on a state’s first visit;
greedy training/evaluation and bootstrapping mask unvisited actions. Evaluation is
deterministic and read-only. In a wholly unseen state, the runtime Q controller reports an
explicit fallback to the causal heuristic; the offline selector’s FCFS tie fallback is not
silently reused.

A predeclared non-RL baseline applies the same observation boundary and these fixed rules:

1. Round Robin if at least three processes are ready and their mean ready age is at least
   one quantum.
2. Otherwise Priority if at least two ready processes have different priorities.
3. Otherwise SJF if the largest visible remaining burst is at least twice the smallest.
4. Otherwise FCFS.

No test-set tuning or complete-workload features are used by this heuristic.

## Default experiment and measured final-test results

Run `python main.py runtime-experiment` (or use `--results-dir`) to generate five
independently trained agents (seeds 7101–7105), 1,200 sequential training episodes per
agent, 140 validation workloads and 210 untouched final-test workloads. Training covers
six declared families; validation and final test cover all seven, including the
training-held-out Poisson-arrival condition. The final workloads are generated only after
training and validation. The four standalone fixed schedulers, the heuristic, and all five
Q agents receive each identical final workload. Train/validation/test fingerprint overlap
is checked and rejected.

The current checked-in runtime test set is 30 workloads per family. All time metrics below
are equal-weight means across the 210 test workloads; Q-learning rows average five
independent learned models. Costless scheduler metrics are not assumed: these results use
the declared one-unit switch cost.

| Method | Mean wait | Mean turnaround | Mean response | CPU utilization % | Throughput | Context switches | Policy changes / trace |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 130.412 | 149.525 | 130.412 | 92.545 | 0.0766 | 14.000 | 0.000 |
| SJF | 94.471 | 113.585 | 94.471 | 92.545 | 0.0766 | 14.000 | 0.000 |
| Round Robin | 217.840 | 236.954 | 26.209 | 78.278 | 0.0671 | 76.081 | 0.000 |
| Priority | 129.577 | 148.690 | 129.577 | 92.545 | 0.0766 | 14.000 | 0.000 |
| Causal heuristic | 206.494 | 225.608 | 40.882 | 80.701 | 0.0703 | 68.662 | 3.091 |
| Runtime Q-learning | 184.648 | 203.761 | 41.251 | 81.794 | 0.0705 | 63.154 | 5.657 |

These are descriptive results, not a claim of universal superiority. On this test suite,
SJF has the lowest mean waiting time; runtime Q-learning has higher mean waiting time than
SJF and FCFS, while it improves on the predeclared heuristic in waiting time and context
switches. The main demonstration is genuine **within-trace learned adaptation**, not a
claim that adaptation beats every fixed scheduler. The checked-in learned example makes 16
policy changes over 60 dispatch/quantum decisions on one held-out priority-skewed workload;
all 60 actions came from training-seen states (zero heuristic fallbacks). Its actions,
observations, trace, seed and fingerprint are recorded in
`results/runtime/runtime_learned_switch_demo.json`.

Uncertainty is reported as a family-stratified crossed bootstrap interval, resampling test
workloads within each family and independently resampling learned-model seeds. The
intervals are conditional on the declared workload families and seeds; no p-values or
blanket significance claims are made. The paired Q-learning minus SJF waiting-time
contrast is **+90.176** with 95% interval **[+79.148, +99.409]**; the Q-learning minus
heuristic contrast is **−21.846** with interval **[−32.933, −12.648]**. See the generated
report and CSV for all six paired metrics and comparisons.

The five learned models covered 255–265 of 648 possible state-action pairs each
(39.4–40.9%). On final evaluation they encountered 75–77 distinct states per model;
2–8 decisions per model used the explicit unseen-state fallback. Remaining unvisited
actions are reported separately in the coverage table.

In the recorded run, mean observation construction/action selection took **26.937 /
29.789 microseconds per Q decision**; a Q update took **17.525 microseconds per training
transition**. These are host-specific Python timings, not simulated CPU costs. The generated
report also records total per-trace timings and simulator wall time separately.

## Artifacts

`results/runtime/` contains:

- `runtime_training_metrics.csv` — sequential training outcomes and separate timing fields;
- `runtime_validation_metrics.csv` and `runtime_final_test_metrics.csv` — all six existing
  metrics, policy-switch counts, decision counts and runtime overhead;
- `runtime_validation_decisions.csv` and `runtime_final_test_decisions.csv` — causal
  observation summaries, action choices and exact policy-switch times;
- `runtime_workload_manifest.csv` — split, generator seeds and workload fingerprints;
- `runtime_state_action_coverage.csv` and `runtime_q_table.csv` — state/action counts,
  Q values, unseen-state fallback counts and unvisited-action exposure;
- `runtime_paired_comparisons.csv`, summary tables, `runtime_summary.json` and the
  generated `runtime_report.md`;
- `runtime_learned_switch_demo.json` — verified learned policy changes within one held-out
  trace.

Observation construction, action selection, Q updates and total simulator wall time are
measured separately and are not charged to simulated time. Controller timing is host
specific; schedule outcomes and policy decisions are seeded and deterministic.

## Legacy offline reference

`python main.py experiment` (and `python main.py train`) still runs the prior offline
whole-workload selector. That selector sees complete-workload features and uses
counterfactual policy metrics for its reward. Its historical outputs under `results/`
remain explicitly offline and are not mixed into the causal runtime comparison. The
runtime experiment does not claim a fair head-to-head comparison against that
information-leaking reference.

## Installation, commands and tests

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
python main.py runtime-experiment
```

The event-driven simulator is synthetic and single-core. It omits I/O/blocking, multicore
interference, deadlines, priority aging, actual burst-time prediction error, kernel
integration, and physical switch latency. The state abstraction is coarse, so a substantial
fraction of state-action pairs remain unvisited. All conclusions are limited to the
specified generated workload families, fixed quantum/action set, switch-cost model,
heuristic and waiting-time reward.
