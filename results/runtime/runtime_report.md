# Causal runtime-adaptation experiment

This is the sequential, event-driven runtime experiment. It is separate from the
legacy offline selector: no controller receives a `Workload`, unarrived process,
future arrival/burst, or counterfactual schedule metric. The environment privately
advances its event queue and exposes an immutable arrived-work observation at each
dispatch/quantum decision epoch.

## Design and split integrity

- Independent Q-learning seeds: 7101, 7102, 7103, 7104, 7105.
- Training: 1200 sequential episodes/model across 6 families; 6000 total model-episodes.
- Validation: 140 distinct workloads; used for reporting and the predeclared illustrative-demo selection rule, not model/hyperparameter tuning.
- Untouched final test: 210 distinct workloads, generated only after the validation-based demo was selected.
- Final test and validation workload fingerprints overlap: False.
- Round Robin quantum / per-PID-change switch cost: 4 / 1 time units.
- Exact burst lengths are assumed known when a process arrives (also required by SJF); no unarrived process details are exposed.

## Sequential learning and causal heuristic

At each decision epoch, the Q learner encodes five coarse causal features, chooses
FCFS, SJF, Round Robin, or Priority, executes the chosen policy's next dispatch or
quantum segment, and observes the next causal state. It updates with
`Q(s,a) <- Q(s,a) + alpha * (r + gamma max_known_a' Q(s',a') - Q(s,a))`; terminal
updates omit the bootstrap. This run uses gamma=1: since each finite workload is an
episode and `r = -delta_wait/q`, the undiscounted return is exactly negative total
waiting time divided by the Round-Robin quantum. There is no per-decision or
simulated-time discount. Unvisited actions are masked for greedy evaluation and
bootstrapping; wholly unseen states use the documented causal heuristic fallback.
Evaluation is deterministic and read-only.

## Ready-queue contract

The live queue contains arrived, unfinished, non-running processes. Dispatch removes
one process; completion removes it permanently. FCFS and RR select the current FIFO
head. RR requeues an unfinished process at the tail after admitting endpoint arrivals.
SJF and Priority select by their primary key, with ties preserving current queue order.
A policy change never rebuilds the queue or resets remaining bursts. Simultaneous
arrivals are admitted by `(arrival_time, pid)` before RR requeue at a service endpoint.

The predeclared non-RL heuristic selects RR for at least three ready jobs with mean
arrival age at least one quantum; otherwise Priority if ready priorities differ;
otherwise SJF if the largest visible remaining burst is at least twice the smallest;
otherwise FCFS. Fixed baselines are the four preserved standalone implementations.

## Final held-out test means

| Method | Mean wait | Mean turnaround | Mean response | CPU util. % | Throughput | Context switches | Policy switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 127.797 | 146.612 | 127.797 | 92.237 | 0.077 | 14.000 | 0.000 |
| SJF | 93.345 | 112.161 | 93.345 | 92.237 | 0.077 | 14.000 | 0.000 |
| Round Robin | 215.729 | 234.544 | 26.220 | 78.216 | 0.068 | 74.767 | 0.000 |
| Priority | 129.207 | 148.022 | 129.207 | 92.237 | 0.077 | 14.000 | 0.000 |
| Causal heuristic | 203.404 | 222.220 | 40.070 | 80.481 | 0.071 | 67.371 | 3.210 |
| Runtime Q-learning | 104.789 | 123.604 | 86.341 | 90.836 | 0.076 | 17.322 | 4.729 |

Q-learning rows average across the five independently trained models and the same paired test workloads; fixed and heuristic rows contain one deterministic result per workload.

## Paired uncertainty

Intervals below use a 1000-replicate, family-stratified crossed bootstrap: resample test workloads within each family and independently resample training-model seeds. Intervals quantify uncertainty across these specified seeds/workloads; they do not justify claims about all operating systems or workload populations.
A difference is target minus reference; negative is favorable for waiting/turnaround/response/context-switch costs, positive for utilization/throughput. No p-values or blanket significance claims are reported.

| Target | Reference | Metric | Difference | 95% interval |
|---|---|---|---:|---:|
| Causal heuristic | SJF | avg_waiting_time | 110.059 | [106.219, 114.032] |
| Causal heuristic | SJF | avg_turnaround_time | 110.059 | [106.042, 113.971] |
| Causal heuristic | SJF | avg_response_time | -53.276 | [-56.026, -50.873] |
| Causal heuristic | SJF | cpu_utilization | -11.756 | [-11.992, -11.518] |
| Causal heuristic | SJF | throughput | -0.006 | [-0.006, -0.006] |
| Causal heuristic | SJF | context_switches | 53.371 | [51.666, 55.191] |
| FCFS | SJF | avg_waiting_time | 34.451 | [32.640, 36.540] |
| FCFS | SJF | avg_turnaround_time | 34.451 | [32.548, 36.377] |
| FCFS | SJF | avg_response_time | 34.451 | [32.605, 36.440] |
| FCFS | SJF | cpu_utilization | 0.000 | [0.000, 0.000] |
| FCFS | SJF | throughput | 0.000 | [0.000, 0.000] |
| FCFS | SJF | context_switches | 0.000 | [0.000, 0.000] |
| Priority | SJF | avg_waiting_time | 35.862 | [33.987, 37.684] |
| Priority | SJF | avg_turnaround_time | 35.862 | [34.127, 37.907] |
| Priority | SJF | avg_response_time | 35.862 | [33.956, 37.823] |
| Priority | SJF | cpu_utilization | 0.000 | [0.000, 0.000] |
| Priority | SJF | throughput | 0.000 | [0.000, 0.000] |
| Priority | SJF | context_switches | 0.000 | [0.000, 0.000] |
| Round Robin | SJF | avg_waiting_time | 122.383 | [117.619, 127.315] |
| Round Robin | SJF | avg_turnaround_time | 122.383 | [118.040, 126.896] |
| Round Robin | SJF | avg_response_time | -67.126 | [-70.219, -64.285] |
| Round Robin | SJF | cpu_utilization | -14.021 | [-14.241, -13.760] |
| Round Robin | SJF | throughput | -0.009 | [-0.010, -0.009] |
| Round Robin | SJF | context_switches | 60.767 | [58.995, 62.491] |
| Runtime Q-learning | SJF | avg_waiting_time | 11.443 | [5.022, 19.297] |
| Runtime Q-learning | SJF | avg_turnaround_time | 11.443 | [4.941, 18.899] |
| Runtime Q-learning | SJF | avg_response_time | -7.004 | [-9.541, -4.305] |
| Runtime Q-learning | SJF | cpu_utilization | -1.402 | [-1.843, -1.005] |
| Runtime Q-learning | SJF | throughput | -0.002 | [-0.002, -0.001] |
| Runtime Q-learning | SJF | context_switches | 3.322 | [2.209, 4.555] |
| Runtime Q-learning | Causal heuristic | avg_waiting_time | -98.616 | [-106.097, -90.192] |
| Runtime Q-learning | Causal heuristic | avg_turnaround_time | -98.616 | [-106.286, -90.115] |
| Runtime Q-learning | Causal heuristic | avg_response_time | 46.272 | [43.472, 49.100] |
| Runtime Q-learning | Causal heuristic | cpu_utilization | 10.354 | [9.890, 10.834] |
| Runtime Q-learning | Causal heuristic | throughput | 0.005 | [0.004, 0.005] |
| Runtime Q-learning | Causal heuristic | context_switches | -50.050 | [-51.929, -48.066] |

## Coverage and controller overhead

Per-model state/action visit counts, unseen-state fallbacks, unvisited-action exposure,
policy decision sequences/times, and all six scheduler metrics are in the CSV outputs.
Observation construction, action selection, Q updates, and total simulator wall time
are recorded separately; timing is host-dependent and is not charged to simulated time.

- Illustrative validation-split Q trace (not representative): `runtime_learned_switch_demo.json` (15 policy changes across 23 decisions; training seed 7103, staggered_interactive repetition 0).
- Demo selection rule (validation only): validation only: maximize policy switches among Q traces with zero unseen-state fallbacks; if none qualify, use all validation Q traces; break ties by ascending training seed, family, then repetition.
- Mean Q-controller observation/selection overhead per evaluation trace: 0.253699 ms / 0.393512 ms.
- Mean Q-controller observation/action-selection time per decision: 13.693 us / 21.239 us.
- Mean Q training update time: 18.124 us per transition.

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
- `runtime_validation_decisions.csv` / `runtime_final_test_decisions.csv`: large causal state/action event logs, regenerated by the command above and intentionally excluded from version control.
- `runtime_workload_manifest.csv`: split seeds and fingerprints for regeneration.
- `runtime_state_action_coverage.csv` / `runtime_q_table.csv`: state/action visits and learned values.
- `runtime_summary.json`: configuration, software, split audit, aggregate results, and intervals.
