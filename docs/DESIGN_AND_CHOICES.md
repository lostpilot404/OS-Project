# Design and documented choices

This document records every implementation decision that the project brief left open, so
that nothing in the codebase rests on a silent assumption.  Each entry states the
decision, the reason, and where it can be changed.  All values live in
[`config.py`](../config.py); nothing is hard-coded in the algorithms.

The order follows the project phases: scheduler semantics, workload model, metrics, state,
reward, learning, experiment protocol.

---

## 1. Decision model: one policy per workload, chosen before execution

**Decision.** The Q-learning agent observes a workload, selects one of the four policies,
and that policy then schedules the whole workload.  One episode is one workload.

**Why.** The brief's conceptual flow is *Workload → Observe State → Agent → Select Policy →
Execute → Measure → Reward → Update*.  This is the reading under which the state describes
the *workload*, which is the only reading that is both well defined and free of outcome
leakage: the alternatives (re-selecting at every arrival, completion or quantum) would have
to define what happens to a running process when the policy changes, and would make the
state partly a product of the very decisions being evaluated.  The brief also requires that
"no future information leaks into the decision state"; a decision taken once, before any
execution, satisfies this by construction (see §5).

**Consequence for the state.** CPU utilisation, queue length and waiting time are *measured*
quantities: at the decision point they are zero or trivial.  §5 therefore uses their
pre-execution analogues.

**Change it.** The decision point is fixed in `rl/adaptive.py`.  Supporting mid-execution
re-selection would require an additional rule for preempting a running process and is a
scope decision, not a configuration change.

---

## 2. Scheduler variants

| Policy | Variant | Justification |
|---|---|---|
| FCFS | Non-preemptive | Definitional. |
| SJF | Non-preemptive (no SRTF) | The brief states the standard non-preemptive interpretation unless the project specification requires otherwise; none does. |
| Round Robin | Preemptive, quantum from configuration | Definitional; the quantum is `SchedulerConfig.round_robin_quantum` (default 4). |
| Priority | Non-preemptive, lower number = higher priority | Same reasoning as SJF. `SchedulerConfig.lower_priority_number_is_higher_priority` inverts the sense. |

Tie-breaking is a total order, so results are deterministic for deterministic input:

* FCFS — `(arrival_time, pid)`
* SJF — `(burst_time, arrival_time, pid)`
* Priority — `(priority, arrival_time, pid)`
* Round Robin — FIFO; simultaneous arrivals are enqueued in `pid` order

The three non-preemptive policies share one simulation engine and differ only in this key
(`scheduler/base.py::NonPreemptiveReadyQueuePolicy`), so no scheduling rule is duplicated.

---

## 3. Context switches, switching cost and Round-Robin boundaries

**Context switch definition.** A change of the *running process* in the execution trace.
The first dispatch of a run is not a switch, and a process that keeps the CPU across a
quantum boundary is not switched out.

**Quantum boundaries.** Round Robin re-queues a process whose quantum expired.  If no other
process is ready, the process immediately continues; the two slices merge in the trace
(`scheduler/base.py::Timeline`) and *no* context switch is counted.  A burst equal to the
quantum finishes inside one turn; a burst of two quanta returns to the queue once.

**Arrival convention.** When a process is preempted, the processes that arrived at or
before the end of its slice are placed *ahead* of it.  A process arriving exactly at the end
of a slice is therefore ready at that instant.  This is documented in
`scheduler/round_robin.py` and covered by tests.

**Switching cost.** `SchedulerConfig.switching_cost` defaults to `0`: switching is costless
and merely counted.  A non-zero value models a per-switch time overhead, which shifts every
subsequent start time and adds to the makespan; the test suite exercises both settings.  The
default keeps the six metric definitions directly comparable with the textbook formulas.

---

## 4. Metrics

| Metric | Definition implemented |
|---|---|
| Waiting time | `turnaround_time - burst_time`, averaged over processes |
| Turnaround time | `completion_time - arrival_time`, averaged over processes |
| Response time | `first_execution_time - arrival_time`, averaged over processes |
| CPU utilisation | `100 * cpu_busy_time / total_elapsed_time` |
| Throughput | `completed_processes / total_elapsed_time` |
| Context switches | number of changes of the running process in the trace |

Derived in exactly one place (`workload/models.py::ProcessOutcome`, `evaluation/metrics.py`);
no other module re-derives them.  For an empty workload every metric is `0`.

**Note on makespan.** With a work-conserving single CPU and `switching_cost = 0`, the
makespan is the total burst time plus any interval in which no process was pending.
Measured consequence: in this study all schedulers produce the *same* CPU utilisation and
throughput on a given workload (the ratios in `results/summary.json` are exactly 1.0000),
because no evaluated workload ever leaves the CPU without pending work.

---

## 5. State representation

The brief lists CPU utilisation, queue length, burst-time characteristics and waiting-time
characteristics as candidate state information.  Given the one-decision-per-workload model
(§1), the first, second and fourth of these are measured quantities and cannot be observed
before execution.  The state therefore uses four pre-execution workload characteristics —
the documented substitution:

| Variable | Definition | Scale |
|---|---|---|
| `burst_profile` | median burst time | logarithmic, range `(1, 50)` |
| `burst_dispersion` | coefficient of variation of burst times (population σ / mean) | linear, range `(0.0, 1.5)` |
| `offered_load` | total burst time ÷ `max(1, arrival_span)` | logarithmic, range `(0.5, 60)` |
| `priority_spread` | population σ of the process priorities | linear, range `(0.0, 2.0)` |

`bins_per_variable = 3` (low / medium / high) gives 3⁴ = **81 states**.  Values are encoded
with the first configured variable as the most significant digit of a mixed-radix index.
Bin `k` covers values from its lower edge (inclusive) to its upper edge (exclusive); the
lowest and highest bins are open, so no value can fall outside the state space.

**Why these four.** They are the dimensions along which the configured workload conditions
actually differ — job-size scale, job-size heterogeneity, demand versus arrival window, and
how strongly priorities distinguish jobs — and they were chosen from the families'
*declared* generation rules, not from results.  The dispersion and priority variables were
added after measuring that median burst alone collapsed four of the six conditions into the
same cell; that calibration used only generator parameters, before the experiment was run.

**Offline-view limitation.** The profile variables describe the whole job list, which is
legitimate for the batch view in which a workload is submitted as a unit, but a scheduler
that does not know future arrivals could not compute them.  This is stated as a limitation
in the README.  It is also why the state contains no measured quantity and no policy
outcome: two workloads with identical job lists produce identical states, whatever any
scheduler does with them.

**Substituting variables.** `KNOWN_STATE_VARIABLES` in `config.py` lists what the encoder can
build; `StateConfig(state_variables=...)` selects any subset, with per-variable ranges.

---

## 6. Reward

```
term_m       = min(clip, metric_m / reference_m)          reference_m = mean of metric m
                                                          over the four conventional policies
reward       = Σ_benefits w_m · (term_m − 1)              (CPU utilisation, throughput)
             + Σ_costs    w_m · (1 − term_m)              (waiting, turnaround, response,
                                                           switches per process)
clip         = 2.0        reference = 0 ⇒ term = 1 (neutral)
```

Zero means "exactly as good as the average conventional policy on the same workload".
Normalising against the four-policy mean on the *same* workload is what makes rewards
comparable across workload conditions, and it means the agent can never be rewarded for a
workload being easy.

**Weights** (`RewardConfig`): waiting 0.30, turnaround 0.25, response 0.20, switches 0.10,
utilisation 0.075, throughput 0.075; they sum to 1.  These weights were declared before the
experiment and were **not** adjusted afterwards; the reason for ranking the three time
metrics first is that they are the classical objectives of CPU scheduling, while context
switches are a secondary cost and utilisation/throughput are structural in this model (§4).

**Context switches** enter per process (`context_switches_per_process`) because a raw count
is not comparable across workload sizes.

**Quantum controller reward** (`compute_reward_against_reference`) uses the same equation
and weights, but its reference is classic Round Robin on the same workload, so the
controller optimises "better than the configured quantum" rather than "better than the
average policy".

---

## 7. Learning

Tabular Q-learning, one table of shape `(81 states × 4 actions)`:

| Setting | Value | Note |
|---|---|---|
| α | 0.1 | `QLearningConfig.learning_rate` |
| γ | 0.9 | unused in value terms: every episode is a single terminal transition |
| ε | 1.0 → 0.05, multiplicative decay 0.995 per episode | `EpsilonSchedule` |
| Q₀ | 0.05 | optimistic: above the best achievable reward, so unexplored actions get tried |
| update | `Q(s,a) += α(reward + γ·max Q(s′) − Q(s,a))`, `s′ = None` | one update per episode |

Because one episode is one workload, the episode is a single-step bandit-like transition;
γ and the terminal-update rule are still implemented and unit-tested so that the code
follows the general rule rather than a special case.

**Evaluation** calls `QLearningAgent.select_action(state, epsilon=0.0)` through
`AdaptiveScheduler.run_evaluation`, which never calls `update`.  The evaluation pipeline
additionally compares the Q-table before and after the whole run and raises if it changed
(`experiments/evaluate.py::_check_q_table_untouched`).

---

## 8. Round-Robin quantum controller (documented extension)

**What.** A second tabular Q-learning agent learns a multiplier applied to the configured
quantum for the Round-Robin action.  Multipliers: `(0.5, 1.0, 2.0)`; α = 0.15; ε 1.0 → 0.05
with decay 0.995; single-step episodes; unvisited states fall back to 1.0 (classic Round
Robin) rather than to an arbitrary action.

**Why it is included and how it is kept out of the headline.** The project brief requires
dynamic policy selection but leaves the quantum as a fixed parameter, and the one-decision-
per-workload model would otherwise make the quantum untunable.  The controller is therefore
reported *separately*: `use_during_evaluation` defaults to `False`, so the headline
comparison uses classic Round Robin at the configured quantum, and the controller is
measured directly by running Round Robin with the learned multiplier on the same workloads
(regime `round_robin_learned_quantum` in `results/metrics.csv`).

The published concept is the five learned policies of Hazarika, Bora, Bora and Singh,
"Workload Aware Dynamic Scheduling Algorithm for Multi-core Systems", ACM SIGSOFT Software
Engineering Notes 43(4), 2018.  This implementation is the same idea on a single simulated
CPU, and it is **not** claimed to be the original method.

**Change it.** `config.QuantumControllerConfig(enabled=False)` removes it entirely from
training and evaluation; `use_during_evaluation=True` folds it into the adaptive path.

---

## 9. Workload generation

Six conditions (`build_default_config`), 15 processes each, on a single CPU:

| Condition | Bursts | Arrivals | Priorities |
|---|---|---|---|
| `short_jobs` | bimodal 1–50, 80 % from 1–5 | uniform over [0, 20] | uniform 1–5 |
| `long_jobs` | bimodal 1–50, 20 % from 1–5 | uniform over [0, 20] | uniform 1–5 |
| `mixed` | uniform 1–50 | uniform over [0, 20] | uniform 1–5 |
| `cpu_bursty` | uniform 20–50 | uniform over [0, 15] | uniform 1–5 |
| `poisson_arrivals` | uniform 1–20 | Poisson process, rate 0.5 (first arrival at 0) | uniform 1–5 |
| `priority_skewed` | uniform 1–50 | uniform over [0, 20] | 70 % from {1, 2}, else uniform 1–5 |

All values are integers in abstract time units; priorities are integers with 1 = highest.
Identifiers are assigned in generation order and the model stores processes ordered by
`(arrival_time, pid)`, so attribute order never depends on input order.

**Seeds.** Every draw uses `numpy.random.default_rng(derive_seed(master_seed, tag, ...))`,
where `derive_seed` is a SHA-256-based child-seed derivation (`config.py`).  Training and
evaluation use different master seeds, and the evaluation pipeline verifies that no
evaluation workload appeared in training.

**Change it.** Add or edit `WorkloadFamilyConfig` entries; generation rule codes
(`burst_distribution`, `arrival_pattern`, `priority_pattern`) are validated at construction.

---

## 10. Experiment protocol

| Item | Value | Where |
|---|---|---|
| Training episodes | 600, families cycled in configuration order | `TrainingConfig` |
| Training seed | 42 (workloads: `derive_seed(42, 0, episode)`; agent RNG: `derive_seed(42, 1)`; controller RNG: `derive_seed(42, 2)`) | `TrainingConfig` |
| Evaluation | 10 repetitions × 6 conditions = 60 workloads | `EvaluationConfig` |
| Evaluation seed | 2024, disjoint master seed | `EvaluationConfig` |
| Quantum | 4 time units | `SchedulerConfig` |
| Switching cost | 0 | `SchedulerConfig` |
| Baselines | FCFS, SJF, Round Robin, Priority on the *same workload object* | `experiments/evaluate.py` |
| Adaptive regimes | classic quantum (headline) + learned quantum (secondary) | `experiments/evaluate.py` |

Three integrity checks are enforced in code and raise instead of warning: evaluation
workloads are disjoint from training workloads; the Q-tables are unchanged by evaluation;
every scheduler of a workload saw the same fingerprint, and every adaptive decision was
greedy.

**Reproduce with** `python3 main.py experiment` (see the README).  The run writes
`results/config.json`, `results/training_history.json`, `results/q_table.json`,
`results/workloads.csv`, `results/metrics.csv`, `results/decisions.csv`,
`results/summary.json` and the figures.

---

## 11. Deliberate non-goals

Not implemented, and not claimed anywhere: Linux kernel integration, multi-core or
distributed scheduling, real-time or energy-aware scheduling, deep RL (DQN/PPO/actor-critic),
cloud/edge deployment, databases, REST APIs, web front ends, Docker/Kubernetes, additional
scheduling algorithms, and any claim that the result is a production scheduler.
