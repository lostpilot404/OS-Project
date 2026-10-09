# Runtime scheduler design and experimental protocol

This document describes the event-driven, causal runtime-adaptation experiment now exposed
by default through `python main.py runtime-experiment`. The historical whole-workload
selector remains as a separate offline reference and is described at the end; it must not
be confused with the runtime controller.

## 1. Objective, system boundary and information contract

The runtime task is a discrete-event, single-CPU simulation. At each dispatch/quantum
epoch a controller chooses one of four scheduling policies—FCFS, non-preemptive SJF,
Round Robin, or non-preemptive Priority—for the next segment of the *same evolving trace*.
The controller does not choose one policy for a whole workload and the simulator does not
run four complete schedules to select a winner.

The simulator privately owns the complete event list so it can admit future processes at
their arrival times. This is an environment privilege, not a controller input. A
`RuntimeObservation` contains only:

- current simulated time and the previous selected policy;
- currently ready processes that have already arrived, with PID, arrival, known burst,
  remaining burst and priority;
- completed-process and arrived-process counts;
- mean burst and priority spread across processes that have arrived so far (including
  previously completed processes);
- mean waiting age of the current ready queue.

No `Workload` or reference to it crosses the controller API. The observation/state
encoder do not accept unarrived process data, the number of future processes, future
arrival/burst summaries, or completed-schedule/counterfactual metrics. A decision reward
is based only on the actual waiting-time increments accumulated as the trace advances.

**Burst-information assumption:** exact burst lengths are known at arrival. This is
consistent with the project's standalone SJF model, but is an idealized simulator
assumption; no process burst predictor is included. Arrival time, burst, and priority are
exposed for a process only after it is ready.

Causality is checked two ways. The controller and encoder signatures accept observations,
not workloads. A paired hidden-future test changes a not-yet-arrived process's arrival,
burst and priority while preserving the observable prefix; the observation and action
prefixes must be exactly equal. The deterministic and randomized fixed-policy tests also
verify that event advancement has not broken the existing scheduler semantics.

## 2. Event-driven execution semantics

The ready queue persists for the entire trace; changing the selected rule never clears,
rebuilds or reorders it on behalf of the controller. On a decision epoch:

1. admit all arrivals at or before the epoch in canonical `(arrival_time, pid)` order;
2. construct an immutable observation from arrived/currently ready/completed work;
3. ask the controller for one action;
4. apply that action to the current ready set;
5. run the selected process's next service segment, charge any PID-change switch cost,
   account for observed waiting-time increments, and admit arrivals at the segment end;
6. for RR only, append a still-incomplete process to the existing queue tail *after*
   admitting all arrivals at or before the quantum endpoint; otherwise a selected process
   runs to completion;
7. expose the next causal observation and learn from the transition if this is a training
   run.

Policy selection rules are total and deterministic:

| Action | Selection/execution rule |
|---|---|
| FCFS | Smallest `(arrival_time, pid)`, non-preemptive. |
| SJF | Smallest `(remaining_burst, arrival_time, pid)`, non-preemptive. |
| Round Robin | FIFO queue; run for `min(configured_quantum, remaining_burst)`. Boundary arrivals precede re-enqueue of the yielded process. |
| Priority | Smallest configured priority key, then arrival time and PID; non-preemptive. Lower numeric priority is higher by default. |

The first dispatch has no switch charge. A charge is added only when the next PID differs
from the last running PID; changing policy alone is not a context switch. A process that
resumes after a consecutive same-PID dispatch incurs no PID-change switch cost. CPU busy,
idle and switch overhead remain distinct in `ScheduleResult` and all six existing metrics.
The trace validator checks work conservation outside the actual switch interval, including
when an idle wait for an arrival precedes a switch. This corrects a validator edge case; it
does not alter the standalone policies' selection or execution rules.

The default runtime experiment uses RR quantum 4 and switching cost 1. The cost is applied
in both adaptive and fixed-policy schedules. Fixed FCFS/SJF/RR/Priority references are run
through their preserved standalone classes—not a learned or adaptive proxy. A randomized
seeded test compares event-driven constant-action schedules against all four standalone
implementations across varying arrivals, priorities, quantum and switch costs.

## 3. Causal state encoding

`RuntimeStateEncoder` accepts only a `RuntimeObservation`. Its fixed, mixed-radix encoding
has five features:

| Feature | Bins |
|---|---|
| Current ready count | 1, 2–3, 4+ |
| Completed process count | 0, 1–3, 4+ |
| Median remaining burst among ready jobs | `<= q`, `<= 4q`, `> 4q` |
| Mean ready waiting age | `<= q`, `<= 4q`, `> 4q` |
| Priority spread among arrived work | zero, non-zero |

The state space is `3 × 3 × 3 × 3 × 2 = 162`; there are 648 state-action entries for the
four actions. Values outside the categories fall into the outer bins. There is no
workload-size feature. Exact arrived process data is visible to the action implementation,
but the encoder reduces it to these documented coarse bins. The previous action is
included in the raw observation (for auditing) but not in the tabular state.

Aggregates are recomputed from the environment's *arrived list only*. Changing any
unarrived process leaves them unchanged. When a process arrives, its data may legitimately
change the state; completed-process count and observed priority/burst aggregates preserve
some causal history without revealing future workload composition.

## 4. Sequential Q-learning

A training episode is one complete runtime trace. Each decision is an MDP step. If
`ΔW_t` is the actual total waiting-time area accumulated since the prior decision and `q`
is the RR quantum, the reward is

```text
r_t = - ΔW_t / max(1, q)
```

`ΔW_t` includes waiting while the selected PID incurs a switch cost, time spent in the
next CPU segment by other ready/arriving jobs, and no service interval for jobs arriving
exactly at an interval endpoint. The simulator checks that the sum of all interval
increments equals the validated schedule's total per-process waiting time. Thus for
`gamma=1`, `sum_t r_t` is exactly `-total_waiting_time/q`; configured `gamma=0.95` instead
optimizes an explicitly discounted sum of later waiting increments.

For nonterminal transitions:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_{a' in visited(s')} Q(s',a') - Q(s,a))
```

For terminal transitions the bootstrap is omitted. A unit test uses different gamma values
and verifies they produce different nonterminal Q targets. The runtime uses the generic
`QLearningAgent` with explicit state-action counts; the old single-step offline selector
continues to use terminal transitions.

Exploration is epsilon-greedy, with epsilon scheduled by training episode using the
configured `epsilon_start`, `epsilon_min` and multiplicative decay. The first occurrence of
a state forces a uniformly random action so an unseen zero-initialized row is not treated
as evidence. Thereafter the greedy branch considers only actions with a positive
state-action visit count; epsilon exploration may sample any of the four actions.
Q-learning bootstraps only from actions visited in the next state. Evaluation is greedy,
deterministic, and read-only. If a state has no learned action, the controller invokes the
explicit causal-heuristic fallback and records `unseen-state-causal-fallback`. If only some
actions are unvisited, evaluation considers known actions only and reports the remaining
unvisited count. Both state and state-action visit counts are exported.

## 5. Predeclared non-RL adaptive heuristic

The heuristic uses the same `RuntimeObservation` and no fitted parameters. Its rule is
fixed before the final test:

1. RR if at least 3 processes are ready and mean ready age is at least one quantum;
2. else Priority if at least 2 ready processes have differing priority values;
3. else SJF if the largest visible remaining burst is at least twice the smallest;
4. else FCFS.

Priority ordering follows `SchedulerConfig` in the simulator (lower number is higher by
default). The heuristic is a causal comparison method, not an oracle over the complete
workload.

## 6. Independent workload splits and fairness

The default runtime study declares:

- **Training:** five independent agent seeds `7101–7105`, 1,200 sequential workload
  episodes per seed. Six families cycle across episodes; `poisson_arrivals` is held out
  from training.
- **Validation:** 20 instances per each of 7 families = 140 unique workloads, generated
  from master seed 8201. Used for reporting/health checks only; there is no hyperparameter
  selection in this pipeline.
- **Final test:** 30 instances per each of 7 families = 210 unique workloads, generated
  from master seed 9301. Generated only after all training and validation runs; it is not
  used to change the model, heuristic or configuration.

SHA-256-derived workload seeds use separate split tags and family/repetition coordinates;
agent exploration streams are also derived separately. Fingerprints are checked for
training/validation/test overlap. The workload manifest saves split, family, repetition,
seed and fingerprint so workloads can be regenerated exactly.

All six evaluated methods receive identical test workload instances: four unchanged
standalone schedulers, the predeclared heuristic, and each of the five learned Q agents.
Fixed-policy metrics are recorded once per workload; Q rows are repeated for each
independent model seed. Q results are averaged across the model seeds without treating the
five rows as five unique workloads.

The historical offline selector is intentionally excluded from the runtime comparison. It
uses full-workload features and computes counterfactual complete schedules to reward each
choice. Keeping it as an explicitly labeled legacy result is useful for repository
reproducibility, but presenting it as a causal online baseline would be misleading.

## 7. Metrics, coverage and uncertainty

The six established scheduling metrics remain centralized in `evaluation/metrics.py`:

| Metric | Definition |
|---|---|
| Mean waiting time | Per-process `turnaround − burst`, averaged. |
| Mean turnaround time | `completion − arrival`, averaged. |
| Mean response time | `first execution − arrival`, averaged. |
| CPU utilization | `100 × CPU busy time / total elapsed time`. |
| Throughput | Completed processes / total elapsed time. |
| Context switches | Changes of running PID in the trace. |

The runtime experiment adds within-trace policy-switch counts and action-change decision
times, state/state-action visit coverage, unseen-state fallback and unvisited-action
counts, decision counts, and separately measured observation-construction, action-selection,
Q-update and simulator wall times. These CPU timings are host-dependent, do not enter
simulated time, and are not confused with training or scheduling rewards.

Paired differences use workload fingerprints and family/repetition coordinates. Reported
95% intervals use a family-stratified crossed bootstrap: independently resample training
model seeds and resample workload repetitions within each family. The intervals are
conditional on the selected seed and family design; they do not establish significance or
universal workload-population behavior. The report includes raw aggregate means and paired
intervals for all six metrics; no significance claim is inferred solely from interval
exclusion of zero.

## 8. Current measured default result

With the configuration above, the currently checked-in run reports:

| Method | Wait | Turnaround | Response | Utilization % | Throughput | Context switches | Policy switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 130.412 | 149.525 | 130.412 | 92.545 | 0.0766 | 14.000 | 0.000 |
| SJF | 94.471 | 113.585 | 94.471 | 92.545 | 0.0766 | 14.000 | 0.000 |
| Round Robin | 217.840 | 236.954 | 26.209 | 78.278 | 0.0671 | 76.081 | 0.000 |
| Priority | 129.577 | 148.690 | 129.577 | 92.545 | 0.0766 | 14.000 | 0.000 |
| Causal heuristic | 206.494 | 225.608 | 40.882 | 80.701 | 0.0703 | 68.662 | 3.091 |
| Runtime Q-learning | 184.648 | 203.761 | 41.251 | 81.794 | 0.0705 | 63.154 | 5.657 |

This run does **not** establish that the learner dominates fixed policies: SJF has the
lowest mean waiting time and Q-learning's mean wait is higher than both SJF and FCFS. The
learner did improve on the selected predeclared heuristic in mean waiting time and
context-switch count in this test set. These are descriptive, generator-specific results.
The paired Q-learning minus SJF waiting-time difference is +90.176 with a 95% interval
[+79.148, +99.409]; Q-learning minus heuristic is -21.846 with interval [-32.933,
-12.648]. Complete tables and intervals are generated by `runtime_report.md` and
`runtime_paired_comparisons.csv`, not copied into computations.

Each model visited 255–265 of 648 state/action pairs (39.4–40.9%). On final test, each
encountered 75–77 distinct states; 2–8 individual decisions per model took the explicit
causal fallback. Coverage is meaningful and visibly incomplete; unseen estimates are not
presented as learned actions.

The learned demonstration is selected from actual held-out Q-controller runs only after
learning. The checked-in case is priority-skewed repetition 11, training seed 7102: 16
policy changes over 60 dispatch/quantum decisions. Its per-epoch causal observations,
selected actions, switch times, execution trace, seed and fingerprint are saved in
`results/runtime/runtime_learned_switch_demo.json`. Selection of the case is by highest
observed action-switch count among the evaluated learned traces; the action sequence itself
is not specified in code or scripted.

## 9. Artifact map, command and verification

The default command is:

```bash
python main.py runtime-experiment
```

`results/runtime/` contains training and validation/test metric tables, per-decision causal
audit rows, workload manifest, per-model Q tables and state/action coverage, paired
bootstrap comparisons, summary JSON, report Markdown, and the learned switch demonstration.
The timing columns are inherently machine-dependent; seeds, workload fingerprints, action
sequences, schedule traces, schedule metrics, and bootstrap results are deterministic.

Run the full test suite with:

```bash
python -m pytest -q
```

Tests cover fixed-scheduler equivalence, randomized schedules, RR endpoint arrivals,
switch-cost accounting, reward-to-total-wait identity, causal hidden-future invariance,
nonterminal gamma dependence, state-action visit counts, deterministic read-only
evaluation, unseen-state fallback reporting, split integrity, reproducibility and the real
learned within-trace switch artifact.

## 10. Limitations and legacy offline selector

The runtime experiment is still a synthetic single-CPU simulator. It omits I/O/blocking,
multicore contention, deadlines, priority aging, burst prediction error, kernel integration,
real scheduling latency and physical CPU measurements. Its state abstraction is coarse and
many state-action pairs remain unvisited. Exact burst knowledge after arrival is an idealized
assumption. No policy is claimed to be universally optimal.

The original offline code remains under `rl/adaptive.py`, `rl/state.py`, and
`experiments/run_experiment.py`; its `python main.py experiment` command and historical
`results/` artifacts are preserved. It makes a single pre-execution action using the full
workload and one terminal counterfactual reward. Its old design document/artifacts are
historical offline outputs only, not evidence about causal runtime adaptation.
