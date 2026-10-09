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
- mean arrival age (`current_time - arrival_time`) of currently ready jobs; this includes
  any earlier CPU service and is not accumulated ready-queue waiting.

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

The simulator maintains one FIFO ready queue for the entire trace. It contains arrived,
unfinished processes that are not currently selected/running. Arrival appends a process in
canonical `(arrival_time, pid)` order. Dispatch removes the selected process; completion
removes it permanently. An RR process that remains unfinished after its quantum is
reinserted at the tail. In particular, FCFS after an RR decision uses this live insertion
order—not the process's original arrival time—and SJF/Priority ties also preserve it.

At a decision epoch:

1. admit all arrivals at or before the epoch in canonical `(arrival_time, pid)` order;
2. construct an immutable observation (its `previous_policy` is the action from the
   immediately preceding decision, or `None` at the first decision);
3. ask the controller for one action and remove the selected process from the ready queue;
4. commit to that dispatch before any switch overhead; arrivals during the overhead are
   appended but do not retroactively change the selected PID;
5. run the selected process's next service segment and account for waiting-time increments;
6. at the service endpoint, admit simultaneous arrivals before resolving the running
   process: a completed process leaves permanently; an unfinished RR process is appended
   at the tail; non-RR service runs to completion;
7. expose the next observation and learn from the transition if this is a training run.

Policy selection rules are deterministic:

| Action | Selection/execution rule |
|---|---|
| FCFS | Current ready-queue head, non-preemptive. Initial arrivals are already ordered by `(arrival_time, pid)`. |
| SJF | Smallest remaining burst, non-preemptive; ties preserve current queue order. |
| Round Robin | Current ready-queue head; run for `min(configured_quantum, remaining_burst)`. Endpoint arrivals precede tail re-enqueue. |
| Priority | Highest configured static priority, non-preemptive; ties preserve current queue order. Lower numeric value is higher by default. |

A policy change alters only the next selection rule/service length. Queue membership/order,
remaining bursts, arrivals, and completed history are not reset. Thus an RR requeue remains
behind jobs already waiting if the next action is FCFS; SJF/Priority can select another job
only because their primary key differs, with queue order resolving ties. Under an
unchanging policy, initial queue insertion order preserves standalone FCFS/SJF/Priority tie
behavior, and RR has the same endpoint-arrival ordering; fixed-policy equivalence tests
remain required.

The first dispatch has no switch charge. A charge is added only when the next PID differs
from the last running PID; changing policy alone is not a context switch. CPU busy, idle and
switch overhead remain distinct in `ScheduleResult` and all six existing metrics. The trace
validator checks work conservation outside the actual switch interval, including when an
idle wait for an arrival precedes a switch.

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
| Mean arrival age of currently ready jobs | `<= q`, `<= 4q`, `> 4q`; includes earlier service, not accumulated queue wait |
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
increments equals the validated schedule's total per-process waiting time. The runtime
experiment default sets `gamma=1`, so `sum_t r_t` is exactly
`-total_waiting_time/max(1,q)` for each completed finite trace. This is the undiscounted
episodic objective of minimizing total waiting time (the positive scale factor does not
change the minimizing policy). There is no discount by decision count or simulated time.
Reward normalization by the quantum limits the per-interval magnitude; gamma 1 can still
produce larger cumulative Q values for longer traces. Episodes are finite, and Q values use
floating-point arithmetic. A custom `gamma<1` changes the objective to discounting by
decision epoch, not by elapsed simulated duration; such a setting must not be described as
optimizing total waiting time.

For nonterminal transitions:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_{a' in visited(s')} Q(s',a') - Q(s,a))
```

For terminal transitions the bootstrap is omitted. Tests verify the undiscounted
nonterminal target with `gamma=1`, verify terminal updates omit future value, and retain a
generic check that changing gamma changes a nonterminal target. The runtime uses the generic
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

1. RR if at least 3 processes are ready and mean arrival age of those jobs is at least
   one quantum (an age proxy, not accumulated ready-queue waiting);
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
  from master seed 8201. Used for reporting and for the predeclared illustrative-demo
  selection rule only; no model or hyperparameter tuning is performed.
- **Final test:** 30 instances per each of 7 families = 210 unique workloads, generated
  from fresh master seed 19301 after training, validation reporting and demo selection. Its
  metrics do not influence model, heuristic, configuration or demonstration choice.

The previous final set (master seed 9301) was examined in the independent audit and
therefore informed the queue/reward remediation. It is retired as a final claim; seed 19301
is the new held-out set for the revised result.

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

The previous final set (master seed 9301) was examined in the independent audit and
informed the queue/reward methodology changes. It is retired as a final claim. The revised
result below uses the new held-out master seed 19301, declared before its run; 30 instances
per family produce 210 final-test workloads.

| Method | Wait | Turnaround | Response | Utilization % | Throughput | Context switches | Policy switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 127.797 | 146.612 | 127.797 | 92.237 | 0.077 | 14.000 | 0.000 |
| SJF | 93.345 | 112.161 | 93.345 | 92.237 | 0.077 | 14.000 | 0.000 |
| Round Robin | 215.729 | 234.544 | 26.220 | 78.216 | 0.068 | 74.767 | 0.000 |
| Priority | 129.207 | 148.022 | 129.207 | 92.237 | 0.077 | 14.000 | 0.000 |
| Causal heuristic | 203.404 | 222.220 | 40.070 | 80.481 | 0.071 | 67.371 | 3.210 |
| Runtime Q-learning | 104.789 | 123.604 | 86.341 | 90.836 | 0.076 | 17.322 | 4.729 |

Q-learning does **not** beat SJF on mean waiting time: Q minus SJF is +11.443 with a
95% crossed-bootstrap interval [+5.022, +19.297]. It improves on the heuristic's mean
waiting time by 98.616 (interval [-106.097, -90.192]), but the heuristic has substantially
lower mean response time (Q minus heuristic +46.272, interval [+43.472, +49.100]). Q uses
fewer context switches than the heuristic (-50.050) but more than SJF (+3.322). These are
conditional, generator-specific comparisons over five learned seeds; they do not imply
performance on general OS workloads. Complete intervals for all metrics remain in the
generated `runtime_report.md` and `runtime_paired_comparisons.csv`.

Each model visited 267–276 of 648 state/action pairs (41.2–42.6%). On final test, 1–3
decisions per model used the explicit unseen-state fallback. Coverage remains incomplete;
unseen estimates are not presented as learned actions.

The illustrative demo is selected from validation before final-test generation. It is the
validation trace with the most policy changes among no-fallback Q traces, with deterministic
tie-breaking by ascending training seed, family and repetition; if no trace qualifies, the
rule falls back to all validation Q traces. The selected case is seed 7103,
`staggered_interactive` repetition 0, fingerprint `275409d7458331f6`, with 15 changes over
23 decisions. It is an illustration, not a representative performance sample. The real
controller supplies every decision; the artifact records workload definition/fingerprint,
observations, actions, switch times and execution trace.

## 9. Artifact map, command and verification

Regenerate the default run into a separate directory with:

```bash
python main.py runtime-experiment --results-dir /tmp/os-project-runtime
```

The declared seeds and configuration are in `experiments/runtime_config.py`; the command
also regenerates large per-decision CSV logs in that output directory. The normal default
command writes to `results/runtime/`.

`results/runtime/` tracks compact training and validation/test metric tables, workload
manifest, per-model Q tables and state/action coverage, paired bootstrap comparisons,
summary JSON, report Markdown, and the validation-split demonstration. Large per-decision
CSV logs are generated by the same command but intentionally gitignored; the deterministic
seeded generation procedure plus compact raw per-workload metrics are retained for
verification. Timing columns are inherently machine-dependent; workload fingerprints,
actions, schedule metrics and bootstrap results are deterministic.

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
