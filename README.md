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
- mean arrival age of the currently ready jobs (not accumulated ready-queue waiting), and
  the policy selected at the immediately preceding decision epoch (`None` on the first call).

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

The ready queue is one persistent FIFO of arrived, unfinished, non-running processes:

- A process enters at arrival time; simultaneous arrivals are appended by `(arrival_time,
  pid)`. Dispatch removes the selected process from the queue. It is not in the queue while
  running or while completing.
- **FCFS:** dispatch the current queue head, non-preemptively. Thus it follows current FIFO
  insertion order, not the process's original arrival time after an RR requeue.
- **Round Robin:** dispatch the queue head for `min(quantum, remaining_burst)`. If it is
  unfinished at the quantum boundary, admit all arrivals at or before that endpoint in
  `(arrival_time, pid)` order, then append the yielded process at the tail. A completed
  process is never re-enqueued.
- **SJF:** choose the smallest remaining burst, non-preemptively; ties preserve current
  ready-queue order.
- **Priority:** choose the highest configured static priority, non-preemptively; equal
  priorities preserve current ready-queue order (lower numeric value is higher by default).
- A policy switch changes only the next selection rule and service length. It does not
  rebuild/reorder the queue or reset remaining bursts. Arrivals during switch overhead are
  appended, but the dispatch already selected before that overhead is not reconsidered.
- Completion/quantum-expiration endpoints are settled before the next decision: endpoint
  arrivals are queued first; a completed process leaves permanently, while an unfinished RR
  process goes to the tail after those arrivals.
- A context-switch cost is charged only when the running PID changes. The first dispatch
  and redispatch of the same PID are free. The default runtime experiment uses quantum 4
  and switch cost 1 time unit. The standalone fixed policies retain their original
  selection rules; under an unchanging policy, initial queue insertion order yields the
  same FCFS/SJF/Priority tie order and RR FIFO behavior.

The fixed-action runtime simulator is tested against each standalone implementation on
hand-built and seeded random traces (all six metric outcomes, context switches, idle time,
and trace slices).

## Online learning and predeclared heuristic

The causal encoder discretizes five current/history-only features into `3 × 3 × 3 × 3 × 2
= 162` states: ready-queue size (1, 2–3, 4+), completed count (0, 1–3, 4+), median ready
remaining burst (`<=q`, `<=4q`, `>4q`), mean arrival age of currently ready jobs
(`<=q`, `<=4q`, `>4q`), and whether priority spread among arrived jobs is zero. Arrival age
is `current_time - arrival_time`; it includes prior CPU service and is not accumulated
ready-queue waiting. The encoder does not use total workload size or unarrived values.

A training **episode is a sequential trace**, not a whole-workload selection. The
controller repeatedly observes, chooses among the four policies, runs the next segment,
receives a nonterminal/terminal reward, and updates the tabular Q values. For an observed
waiting-time increment `ΔW_t`, reward is `r_t = -ΔW_t / max(1,q)`. The runtime experiment
default uses `gamma = 1`: for a finite completed workload, the undiscounted episode return
is exactly `-total_waiting_time / q`. There is no per-decision or simulated-time discount.
Rewards are scaled by the positive quantum to keep their numeric magnitude smaller; gamma 1
can still produce larger cumulative values on longer traces, but episodes are finite and
Python floating-point Q values are used. A custom `gamma < 1` would optimize discounted
per-decision waiting increments instead and would not be the stated total-waiting objective.

The update is `Q(s,a) += alpha * (r + gamma * max_known_a' Q(s',a') - Q(s,a))`; terminal
transitions omit the bootstrap. State-action visit counts are explicit. Training uses an
episode-level epsilon schedule and forces a random action on a state’s first visit;
greedy training/evaluation and bootstrapping mask unvisited actions. Evaluation is
deterministic and read-only. In a wholly unseen state, the runtime Q controller reports an
explicit fallback to the causal heuristic; the offline selector’s FCFS tie fallback is not
silently reused.

A predeclared non-RL baseline applies the same observation boundary and these fixed rules:

1. Round Robin if at least three processes are ready and their mean arrival age is at least
   one quantum; this is an age-based proxy, not measured accumulated ready-queue wait.
2. Otherwise Priority if at least two ready processes have different priorities.
3. Otherwise SJF if the largest visible remaining burst is at least twice the smallest.
4. Otherwise FCFS.

No test-set tuning or complete-workload features are used by this heuristic.

## Default experiment and measured final-test results

Run `python main.py runtime-experiment` to train five independent agents (seeds 7101–7105),
each for 1,200 sequential episodes. For an isolated regeneration, use
`python main.py runtime-experiment --results-dir /tmp/os-project-runtime`. The design uses
140 validation workloads and 210 final-test workloads across seven families; six families
are used for
training. The validation split is used for reporting and a predeclared illustrative-demo
selection rule, not for model or hyperparameter tuning. The demo is selected before the
final-test workloads are generated. Train/validation/test fingerprint overlap is checked
and rejected.

The prior final test (master seed 9301) was examined in the independent audit and informed
the queue/reward methodology changes below. It is retired as a final claim. The revised,
untouched final set uses master seed **19301**. All means below are equal-weight across its
210 workloads; Q-learning averages five trained models. The declared one-unit switch cost
is included.

| Method | Mean wait | Mean turnaround | Mean response | CPU utilization % | Throughput | Context switches | Policy changes / trace |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 127.797 | 146.612 | 127.797 | 92.237 | 0.077 | 14.000 | 0.000 |
| SJF | 93.345 | 112.161 | 93.345 | 92.237 | 0.077 | 14.000 | 0.000 |
| Round Robin | 215.729 | 234.544 | 26.220 | 78.216 | 0.068 | 74.767 | 0.000 |
| Priority | 129.207 | 148.022 | 129.207 | 92.237 | 0.077 | 14.000 | 0.000 |
| Causal heuristic | 203.404 | 222.220 | 40.070 | 80.481 | 0.071 | 67.371 | 3.210 |
| Runtime Q-learning | 104.789 | 123.604 | 86.341 | 90.836 | 0.076 | 17.322 | 4.729 |

These generator-specific results do not show Q-learning beating SJF on mean waiting time:
Q minus SJF is **+11.443** (95% crossed-bootstrap interval **[+5.022, +19.297]**). Q
has lower mean waiting time than the heuristic by **98.616** (**[−106.097, −90.192]**),
but its mean response time is higher than the heuristic by **46.272** (**[+43.472,
+49.100]**). Q also has fewer context switches than the heuristic (−50.050) but more
than SJF (+3.322). These are conditional comparisons over the declared workloads and five
model seeds, not claims about general OS workloads.

The illustrative example is selected from validation, not final test: training seed 7103,
`staggered_interactive` repetition 0, fingerprint `275409d7458331f6`, with 15 policy
changes across 23 decisions. It is selected by the documented validation-only maximum-switch
rule, generated by the actual controller, and explicitly **not representative** of typical
performance. Its process definition, observations, actions, switch times, and trace are in
`results/runtime/runtime_learned_switch_demo.json`.

With `gamma=1`, the runtime learner uses the undiscounted episodic waiting-cost objective:
its summed reward is exactly negative total waiting time divided by the RR quantum. The
five models visited 267–276 of 648 possible state/action pairs (41.2–42.6%). Across final
evaluation, 1–3 decisions per model used the explicit unseen-state fallback. Per-trace
controller timings are host-dependent and are reported in `runtime_report.md`, not treated
as simulated CPU costs.

## Artifacts

`results/runtime/` contains:

- `runtime_training_metrics.csv` — sequential training outcomes and separate timing fields;
- `runtime_validation_metrics.csv` and `runtime_final_test_metrics.csv` — all six existing
  metrics, policy-switch counts, decision counts and runtime overhead;
- `runtime_validation_decisions.csv` and `runtime_final_test_decisions.csv` — large causal
  event logs generated by the CLI but intentionally gitignored; all decision logs are
  reproducible from the recorded configuration and seeds;
- `runtime_workload_manifest.csv` — split, generator seeds and workload fingerprints;
- `runtime_state_action_coverage.csv` and `runtime_q_table.csv` — state/action counts,
  Q values, unseen-state fallback counts and unvisited-action exposure;
- `runtime_paired_comparisons.csv`, summary tables, `runtime_summary.json` and the
  generated `runtime_report.md`;
- `runtime_learned_switch_demo.json` — an illustrative validation-split trace selected by
  a documented validation-only rule; it records the workload definition/fingerprint, model
  seed, visible observations, actions, switch times and execution trace. It is not
  representative of typical performance.

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

## Review 2 report PDF

`Review_2_Report.pdf` (with a copy at `docs/Review_2_Report.pdf`) is the Review 2
deliverable for *Methodology and Partial Implementation*. It follows the review rubric
item by item, and nothing else is included:

1. Detailed methodology of the proposed system (pipeline, causal observation contract,
   state abstraction, learning objective, methodological controls)
2. Overall conceptual, architectural and pipeline diagram
3. Description of the major modules
4. Workflow diagram for each major module
5. Dataset details and data collection procedure
6. Tools, technologies, algorithms and frameworks used
7. Experimental plan and evaluation metrics
8. Implementation progress (completed modules demonstrated with valid outputs)
9. Challenges encountered and plan for completing the remaining work

The PDF is generated **from the committed artifacts only** — it never re-runs the
experiment, so the reported numbers cannot drift from `results/`. Two things *are*
recomputed at build time: `pytest` is re-run so the reported test count is measured on the
build host, and the dataset statistics in §5.4 / Figure 5.1 are rebuilt with the project's
own seeded generator from the declared seeds. Only comparisons that the experiment itself
produced are reported, so no interval is recomputed at build time.

```bash
python -m pip install -r requirements-report.txt   # adds reportlab
python tools/build_review2_report.py               # rewrites both PDF copies + figures/review2/
python tools/build_review2_report.py --skip-tests  # skip the pytest re-run
```

`tools/build_review2_report.py` reads `results/runtime/*.csv|json` and `results/*.json`,
draws the figures with Matplotlib into `figures/review2/`, and lays out the document with
ReportLab. Useful switches: `--output`, `--docs-copy`, `--figures-dir`, `--root`.

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
