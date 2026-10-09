# Causal runtime-adaptation experiment

This is the sequential, event-driven runtime experiment. It is separate from the
legacy offline selector: no controller receives a `Workload`, unarrived process,
future arrival/burst, or counterfactual schedule metric. The environment privately
advances its event queue and exposes an immutable arrived-work observation at each
dispatch/quantum decision epoch.

## Design and split integrity

- Independent Q-learning seeds: 7101, 7102, 7103, 7104, 7105.
- Training: 1200 sequential episodes/model across 6 families; 6000 total model-episodes.
- Validation: 140 distinct workloads; used for reporting only, not tuning.
- Untouched final test: 210 distinct workloads, generated/evaluated only after training and validation.
- Final test and validation workload fingerprints overlap: False.
- Round Robin quantum / per-PID-change switch cost: 4 / 1 time units.
- Exact burst lengths are assumed known when a process arrives (also required by SJF); no unarrived process details are exposed.

## Sequential learning and causal heuristic

At each decision epoch, the Q learner encodes five coarse causal features, chooses
FCFS, SJF, Round Robin, or Priority, executes the chosen policy's next dispatch or
quantum segment, and observes the next causal state. It updates with
`Q(s,a) <- Q(s,a) + alpha * (r + gamma max_known_a' Q(s',a') - Q(s,a))`; terminal
updates omit the bootstrap. With `r = -delta_wait/q`, gamma=1 telescopes to negative
total waiting time per quantum, while configured gamma<1 explicitly discounts
later waiting increments. The state-action visit table masks unvisited actions for
greedy evaluation and bootstrapping; wholly unseen states use the documented causal
heuristic fallback. Evaluation is deterministic and read-only.

The predeclared non-RL heuristic selects RR for at least three ready jobs with mean
ready age at least one quantum; otherwise Priority if ready priorities differ;
otherwise SJF if the largest visible remaining burst is at least twice the smallest;
otherwise FCFS. Fixed baselines are the four preserved standalone implementations.

## Final held-out test means

| Method | Mean wait | Mean turnaround | Mean response | CPU util. % | Throughput | Context switches | Policy switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 130.412 | 149.525 | 130.412 | 92.545 | 0.077 | 14.000 | 0.000 |
| SJF | 94.471 | 113.585 | 94.471 | 92.545 | 0.077 | 14.000 | 0.000 |
| Round Robin | 217.840 | 236.954 | 26.209 | 78.278 | 0.067 | 76.081 | 0.000 |
| Priority | 129.577 | 148.690 | 129.577 | 92.545 | 0.077 | 14.000 | 0.000 |
| Causal heuristic | 206.494 | 225.608 | 40.882 | 80.701 | 0.070 | 68.662 | 3.090 |
| Runtime Q-learning | 184.648 | 203.761 | 41.251 | 81.794 | 0.071 | 63.154 | 5.657 |

Q-learning rows average across the five independently trained models and the same paired test workloads; fixed and heuristic rows contain one deterministic result per workload.

## Paired uncertainty

Intervals below use a 1000-replicate, family-stratified crossed bootstrap: resample test workloads within each family and independently resample training-model seeds. Intervals quantify uncertainty across these specified seeds/workloads; they do not justify claims about all operating systems or workload populations.
A difference is target minus reference; negative is favorable for waiting/turnaround/response/context-switch costs, positive for utilization/throughput. No p-values or blanket significance claims are reported.

| Target | Reference | Metric | Difference | 95% interval |
|---|---|---|---:|---:|
| Causal heuristic | SJF | avg_waiting_time | 112.023 | [108.681, 115.650] |
| Causal heuristic | SJF | avg_turnaround_time | 112.023 | [108.334, 115.723] |
| Causal heuristic | SJF | avg_response_time | -53.590 | [-56.062, -51.212] |
| Causal heuristic | SJF | cpu_utilization | -11.844 | [-12.063, -11.636] |
| Causal heuristic | SJF | throughput | -0.006 | [-0.006, -0.006] |
| Causal heuristic | SJF | context_switches | 54.662 | [53.243, 56.143] |
| FCFS | SJF | avg_waiting_time | 35.940 | [34.100, 37.784] |
| FCFS | SJF | avg_turnaround_time | 35.940 | [34.272, 37.953] |
| FCFS | SJF | avg_response_time | 35.940 | [34.129, 37.682] |
| FCFS | SJF | cpu_utilization | 0.000 | [0.000, 0.000] |
| FCFS | SJF | throughput | 0.000 | [0.000, 0.000] |
| FCFS | SJF | context_switches | 0.000 | [0.000, 0.000] |
| Priority | SJF | avg_waiting_time | 35.105 | [33.221, 37.054] |
| Priority | SJF | avg_turnaround_time | 35.105 | [33.062, 37.084] |
| Priority | SJF | avg_response_time | 35.105 | [33.145, 37.081] |
| Priority | SJF | cpu_utilization | 0.000 | [0.000, 0.000] |
| Priority | SJF | throughput | 0.000 | [0.000, 0.000] |
| Priority | SJF | context_switches | 0.000 | [0.000, 0.000] |
| Round Robin | SJF | avg_waiting_time | 123.369 | [119.354, 127.309] |
| Round Robin | SJF | avg_turnaround_time | 123.369 | [119.378, 127.218] |
| Round Robin | SJF | avg_response_time | -68.263 | [-70.720, -65.570] |
| Round Robin | SJF | cpu_utilization | -14.267 | [-14.449, -14.068] |
| Round Robin | SJF | throughput | -0.009 | [-0.010, -0.009] |
| Round Robin | SJF | context_switches | 62.081 | [60.519, 63.392] |
| Runtime Q-learning | SJF | avg_waiting_time | 90.176 | [79.148, 99.409] |
| Runtime Q-learning | SJF | avg_turnaround_time | 90.176 | [78.767, 100.029] |
| Runtime Q-learning | SJF | avg_response_time | -53.220 | [-57.535, -48.534] |
| Runtime Q-learning | SJF | cpu_utilization | -10.751 | [-11.741, -9.366] |
| Runtime Q-learning | SJF | throughput | -0.006 | [-0.007, -0.005] |
| Runtime Q-learning | SJF | context_switches | 49.154 | [43.216, 53.362] |
| Runtime Q-learning | Causal heuristic | avg_waiting_time | -21.846 | [-32.933, -12.648] |
| Runtime Q-learning | Causal heuristic | avg_turnaround_time | -21.846 | [-33.045, -12.251] |
| Runtime Q-learning | Causal heuristic | avg_response_time | 0.370 | [-3.713, 3.980] |
| Runtime Q-learning | Causal heuristic | cpu_utilization | 1.093 | [0.072, 2.394] |
| Runtime Q-learning | Causal heuristic | throughput | 0.000 | [-0.001, 0.001] |
| Runtime Q-learning | Causal heuristic | context_switches | -5.508 | [-11.363, -1.639] |

## Coverage and controller overhead

Per-model state/action visit counts, unseen-state fallbacks, unvisited-action exposure,
policy decision sequences/times, and all six scheduler metrics are in the CSV outputs.
Observation construction, action selection, Q updates, and total simulator wall time
are recorded separately; timing is host-dependent and is not charged to simulated time.

- Learned same-trace policy-change demonstration: `runtime_learned_switch_demo.json` (16 policy changes across 60 decisions; training seed 7102, priority_skewed repetition 11).
- Mean Q-controller observation/selection overhead per evaluation trace: 1.738843 ms / 1.922951 ms.
- Mean Q-controller observation/action-selection time per decision: 26.937 us / 29.789 us.
- Mean Q training update time: 17.525 us per transition.

## Limitations and legacy reference

This is a single-CPU synthetic simulator, not a kernel scheduler. It assumes exact
arrived burst lengths, uses a small hand-binned state abstraction, and has no I/O,
multicore contention, deadlines, or hardware latency model. The historical offline
selector in `rl/adaptive.py` remains separately available for reproducibility, but
it sees complete-workload features and counterfactual schedule metrics; it is not a
causal runtime baseline and is not included in these fair paired comparisons.

## Artifact map

- `runtime_training_metrics.csv`: per-episode sequential training outcomes.
- `runtime_validation_metrics.csv` / `runtime_final_test_metrics.csv`: paired metrics.
- `runtime_validation_decisions.csv` / `runtime_final_test_decisions.csv`: causal state/action event records and policy-switch timing.
- `runtime_workload_manifest.csv`: split seeds and fingerprints for regeneration.
- `runtime_state_action_coverage.csv` / `runtime_q_table.csv`: state/action visits and learned values.
- `runtime_summary.json`: configuration, software, split audit, aggregate results, and intervals.
